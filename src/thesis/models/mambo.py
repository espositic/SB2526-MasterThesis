"""
XLSR-MamBo (Ng et al., "XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone
for Audio Deepfake Detection", arXiv 2601.02944, ACL 2026 Findings).

Port in PyTorch puro della configurazione migliore del paper, MamBo-3-Hydra-N3:
XLS-R 300M → proiezione a d_model=128 → L=5 unità [3×Hydra, MLP, MHA, MLP]
→ attention pooling con gate SwiGLU → lineare a 2 classi.
Codice originale: https://github.com/saki-ciallo/MamBo-for-ADD (licenza MIT).

Differenze rispetto all'originale, tutte dovute alla 3060 / Windows:
  - XLS-R caricato da Hugging Face (transformers) invece che da torchaudio;
    stessi pesi, normalizzazione della forma d'onda fatta qui.
  - Hydra, RMSNorm e MHA riscritti in PyTorch puro (niente mamba-ssm / Triton).
  - opzionali: gradient checkpointing di XLS-R e congelamento dei primi layer.

Convenzione del progetto: uscita [logit bonafide, logit spoof]
(label 0 = bonafide, 1 = spoof); punteggio = logit_bonafide − logit_spoof.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

from thesis.models.hydra import Hydra, RMSNorm

XLSR_300M = "facebook/wav2vec2-xls-r-300m"


@dataclass
class MamBoConfig:
    d_model: int = 128
    d_state: int = 64
    d_conv: int = 7          # Hydra usa d_conv=7
    expand: int = 2
    headdim: int = 64        # l'originale non passa headdim a Hydra: vale il default 64
    mha_heads: int = 4
    mha_d_conv: int = 4
    n_layers: int = 5        # L nel paper
    n_hydra: int = 3         # N nel paper
    freeze_ssl: bool = False           # True = XLS-R congelato
    freeze_ssl_layers: int = 0         # congela solo i primi k layer transformer di XLS-R
    gradient_checkpointing: bool = True


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden = int(8 * d_model / 3)
        self.w_in = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.w_in.weight, mode="fan_in", nonlinearity="linear")
        nn.init.kaiming_normal_(self.w_out.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x):
        gate, value = self.w_in(x).chunk(2, dim=-1)
        return self.dropout(self.w_out(F.silu(gate) * value))


class MHA(nn.Module):
    """MHA di mamba-ssm: conv1d causale depthwise su qkv, attenzione non causale."""

    def __init__(self, d_model: int, n_heads: int, d_conv: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.conv1d = nn.Conv1d(3 * d_model, 3 * d_model, kernel_size=d_conv, padding=d_conv - 1, groups=3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        qkv = self.in_proj(x)
        qkv = self.conv1d(qkv.transpose(1, 2))[..., :-(self.d_conv - 1)].transpose(1, 2)
        q, k, v = (t.unflatten(-1, (self.n_heads, -1)).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).flatten(-2))


class PreNormBlock(nn.Module):
    """
    Struttura "Add → Norm → Mixer" di mamba-ssm: il residuo viaggia separato
    (in float32) e viene sommato all'uscita del blocco successivo.
    """

    def __init__(self, d_model: int, mixer: nn.Module):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = mixer

    def forward(self, hidden, residual):
        residual = hidden.float() if residual is None else residual + hidden.float()
        return self.mixer(self.norm(residual.to(hidden.dtype))), residual


class GatedAttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(8 * d_model // 3)
        self.gate_value = nn.Linear(d_model, 2 * hidden, bias=False)
        self.to_score = nn.Linear(hidden, 1, bias=False)
        self.dropout = nn.Dropout(0.1)
        nn.init.kaiming_normal_(self.gate_value.weight, mode="fan_in", nonlinearity="linear")
        nn.init.kaiming_normal_(self.to_score.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x):
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        weights = F.softmax(self.to_score(F.silu(gate) * value).squeeze(-1), dim=1)
        return self.dropout(torch.bmm(weights.unsqueeze(1), x).squeeze(1))


class XLSRMamBo(nn.Module):
    def __init__(self, cfg: MamBoConfig):
        super().__init__()
        self.cfg = cfg

        self.ssl = Wav2Vec2Model.from_pretrained(XLSR_300M)
        self.ssl.config.mask_time_prob = 0.0      # niente masking SpecAugment interno
        self.ssl.config.layerdrop = 0.0
        self.ssl.freeze_feature_encoder()          # CNN iniziale sempre congelata
        if cfg.freeze_ssl:
            self.ssl.requires_grad_(False)
        for layer in self.ssl.encoder.layers[:cfg.freeze_ssl_layers]:
            layer.requires_grad_(False)
        if cfg.gradient_checkpointing and not cfg.freeze_ssl:
            self.ssl.gradient_checkpointing_enable()

        d = cfg.d_model
        self.input_proj = nn.Linear(self.ssl.config.hidden_size, d, bias=False)
        self.input_norm = RMSNorm(d)

        blocks = []
        for _ in range(cfg.n_layers):
            blocks += [PreNormBlock(d, Hydra(d, cfg.d_state, cfg.d_conv, cfg.expand, cfg.headdim))
                       for _ in range(cfg.n_hydra)]
            blocks += [PreNormBlock(d, SwiGLU(d)),
                       PreNormBlock(d, MHA(d, cfg.mha_heads, cfg.mha_d_conv)),
                       PreNormBlock(d, SwiGLU(d))]
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(d)

        self.pool = GatedAttentionPool(d)
        self.classifier = nn.Linear(d, 2)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: (b, campioni) a 16 kHz → logit (b, 2) = [bonafide, spoof]"""
        # Normalizzazione a media 0 e varianza 1 per clip, come per XLS-R in torchaudio
        wave = (wave - wave.mean(dim=1, keepdim=True)) / torch.sqrt(wave.var(dim=1, keepdim=True) + 1e-7)
        x = self.ssl(wave).last_hidden_state                  # (b, frame, 1024)
        x = F.silu(self.input_norm(self.input_proj(x)))

        hidden, residual = x, None
        for block in self.blocks:
            hidden, residual = block(hidden, residual)
        x = self.final_norm((residual + hidden.float()).to(hidden.dtype))

        return self.classifier(self.pool(x))


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"totali": total, "addestrabili": trainable}
