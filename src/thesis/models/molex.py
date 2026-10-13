"""
MoLEx: WavLM congelato con un mix di esperti LoRA (Pan, Sailor, Wu, "MoLEx:
Mixture of LoRA Experts in Speech Self-Supervised Models for Audio Deepfake
Detection", arXiv 2509.09175, Interspeech 2025 secondo il README del codice).

Re-implementazione dal paper: il codice ufficiale (pandarialTJU/MOLEx-ORLoss)
non ha licenza, quindi qui non ne è copiata nessuna riga. Da lì vengono solo
gli iperparametri della config ASVspoof 5 (molex_ssl.conf) e due scelte che il
paper non descrive, segnalate sotto.

Struttura:
  - WavLM-Large ridotto ai primi 12 layer, tutto congelato;
  - negli ultimi 6 di questi 12, accanto al feed-forward, un mix di 12 esperti
    LoRA (rango 32) scelti da un gate "noisy top-k" (4 esperti per frame);
  - attentive merging dei 12 layer (Pan et al., Interspeech 2024, [42] in tesi)
    e classificatore LSTM (192 unità) con media sul tempo;
  - loss di ortogonalità sugli esperti, peso 0,01.

Scelte prese dal codice ufficiale perché il paper non le specifica:
  - in training WavLM resta in modalità train, quindi con il suo dropout,
    anche se i pesi sono congelati (serve anche al gradient checkpointing);
  - ogni esperto, oltre alla parte a basso rango A·B, ha una matrice piena
    allenabile (expert_full_weight=True); con False si ottiene un LoRA puro;
  - la loss di ortogonalità è l'errore quadratico medio tra la matrice di Gram
    di A·B e l'identità, sommata su tutti gli esperti.

Differenze: WavLM da Hugging Face (microsoft/wavlm-large); loss e ricetta
comuni del progetto (focal loss, warmup + coseno) invece di cross-entropy
pesata e decadimento esponenziale. Learning rate come nel codice: 1e-5 per gli
esperti (lr della config di training) e 1e-4 per merging e classificatore (lr_head).

Convenzione del progetto: uscita [logit bonafide, logit spoof].
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.models.ssl_frontend import WAVLM_LARGE, load_ssl, ssl_layers


@dataclass
class MoLExConfig:
    ssl_name: str = WAVLM_LARGE
    n_layers: int = 12             # primi 12 layer di WavLM-Large
    n_moe_layers: int = 6          # esperti negli ultimi 6 di questi
    n_experts: int = 12
    top_k: int = 4
    rank: int = 32
    expert_full_weight: bool = True
    orth_weight: float = 0.01
    lstm_hidden: int = 192
    gradient_checkpointing: bool = True


class LoRAExpert(nn.Module):
    """Proiezione dim → dim: W·x + A·B·x (+ bias), con A·B di rango r."""

    def __init__(self, dim: int, rank: int, full_weight: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim)) if full_weight else None
        if full_weight:
            nn.init.xavier_uniform_(self.weight)
        self.lora_a = nn.Parameter(torch.empty(dim, rank))   # "up"
        self.lora_b = nn.Parameter(torch.zeros(rank, dim))   # "down", a zero: all'inizio A·B = 0
        nn.init.kaiming_uniform_(self.lora_a, a=0.01)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        out = F.linear(x, self.lora_a @ self.lora_b, self.bias)
        return out if self.weight is None else out + F.linear(x, self.weight)

    def orthogonality_loss(self) -> torch.Tensor:
        """
        mean((ΔΔᵀ − I)²) con Δ = A·B (dim × dim), calcolata con matrici rango × rango:
        ‖A M Aᵀ‖² = tr(M G M G) e tr(A M Aᵀ) = tr(M G), con G = AᵀA e M = B·Bᵀ.
        """
        a, b = self.lora_a.float(), self.lora_b.float()
        dim = a.shape[0]
        g, m = a.T @ a, b @ b.T
        mg = m @ g
        return (torch.trace(mg @ mg) - 2 * torch.trace(mg) + dim) / dim**2


class NoisyTopKRouter(nn.Module):
    """Gate di Shazeer et al. (2017): rumore appreso in training, softmax sui soli top-k."""

    def __init__(self, dim: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.logits = nn.Linear(dim, n_experts)
        self.noise = nn.Linear(dim, n_experts)

    def forward(self, x):
        logits = self.logits(x)
        if self.training:
            logits = logits + torch.randn_like(logits) * F.softplus(self.noise(x))
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)
        sparse = torch.full_like(logits, float("-inf")).scatter(-1, top_idx, top_vals)
        return F.softmax(sparse, dim=-1)                  # (…, n_experts), zero fuori dai top-k


class MixtureOfLoRAExperts(nn.Module):
    def __init__(self, dim: int, n_experts: int, top_k: int, rank: int, full_weight: bool):
        super().__init__()
        self.router = NoisyTopKRouter(dim, n_experts, top_k)
        self.experts = nn.ModuleList([LoRAExpert(dim, rank, full_weight) for _ in range(n_experts)])

    def forward(self, x):
        gates = self.router(x)
        # Calcolo denso: un esperto non scelto ha peso 0 e quindi né uscita né gradiente
        return sum(gates[..., i:i + 1] * expert(x) for i, expert in enumerate(self.experts))


class FeedForwardWithExperts(nn.Module):
    """Sostituisce il feed_forward di un layer WavLM: FFN(x) + MoLEx(x), stesso ingresso normalizzato."""

    def __init__(self, ffn: nn.Module, moe: MixtureOfLoRAExperts):
        super().__init__()
        self.ffn = ffn
        self.moe = moe

    def forward(self, x):
        return self.ffn(x) + self.moe(x)


class AttentiveMerging(nn.Module):
    """
    Attentive merging dei layer (Pan et al., Interspeech 2024): ogni layer riceve un
    peso in (0, 1) calcolato dalla sua media sul tempo (stile squeeze-and-excitation),
    poi i layer pesati vengono concatenati sul canale e proiettati a dim.
    """

    def __init__(self, n_layers: int, dim: int):
        super().__init__()
        self.score = nn.Parameter(torch.randn(dim, 1))
        self.squeeze = nn.Parameter(torch.randn(n_layers, n_layers // 2))
        self.excite = nn.Parameter(torch.randn(n_layers // 2, n_layers))
        hidden = n_layers * dim // 4
        self.proj = nn.Sequential(nn.Linear(n_layers * dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, layers: torch.Tensor) -> torch.Tensor:
        """layers: (b, L, frame, dim) → (b, frame, dim)"""
        s = F.silu(layers.mean(dim=2) @ self.score).squeeze(-1)          # (b, L)
        w = torch.sigmoid(F.silu(s @ self.squeeze) @ self.excite)         # (b, L)
        weighted = layers * w[:, :, None, None]
        b, n_layers, t, d = weighted.shape
        return self.proj(weighted.permute(0, 2, 1, 3).reshape(b, t, n_layers * d))


class WavLMMoLEx(nn.Module):
    def __init__(self, cfg: MoLExConfig):
        super().__init__()
        self.cfg = cfg
        self.ssl, self.normalize = load_ssl(cfg.ssl_name, freeze=True, n_layers=cfg.n_layers,
                                            gradient_checkpointing=cfg.gradient_checkpointing)
        dim = self.ssl.config.hidden_size
        self.moe_layers = []
        for layer in self.ssl.encoder.layers[cfg.n_layers - cfg.n_moe_layers:]:
            moe = MixtureOfLoRAExperts(dim, cfg.n_experts, cfg.top_k, cfg.rank, cfg.expert_full_weight)
            layer.feed_forward = FeedForwardWithExperts(layer.feed_forward, moe)  # gli esperti sono allenabili
            self.moe_layers.append(moe)
        self.merge = AttentiveMerging(cfg.n_layers, dim)
        self.lstm = nn.LSTM(dim, cfg.lstm_hidden, batch_first=True)
        self.classifier = nn.Linear(cfg.lstm_hidden, 2)
        self.aux_loss = None

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: (b, campioni) a 16 kHz → logit (b, 2) = [bonafide, spoof]"""
        layers = ssl_layers(self.ssl, wave, self.normalize, frozen=False)[:, 1:]  # solo i layer transformer
        x = self.merge(layers)
        with torch.autocast(device_type=x.device.type, enabled=False):          # LSTM in float32
            x, _ = self.lstm(x.float())
        logits = self.classifier(x.mean(dim=1))
        if self.training and self.cfg.orth_weight > 0:
            # letto da train.py e sommato alla loss di classificazione
            self.aux_loss = self.cfg.orth_weight * sum(
                e.orthogonality_loss() for moe in self.moe_layers for e in moe.experts)
        return logits
