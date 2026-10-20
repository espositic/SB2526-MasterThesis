# XLSR-MamBo (Ng et al., "XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for
# Audio Deepfake Detection", Findings of ACL 2026), configurazione MamBo-3-Hydra-N3.
# Port di github.com/saki-ciallo/MamBo-for-ADD (licenza MIT).
#
# XLS-R 300M -> proiezione a 128 -> 5 unità [3 Hydra, MLP, attenzione, MLP]
# -> attention pooling -> 2 classi [bonafide, spoof].
#
# Differenze dall'originale (per la 3060 e Windows): XLS-R da Hugging Face invece che
# da torchaudio (stessi pesi), Hydra in PyTorch puro, gradient checkpointing di XLS-R.
# I nomi dei moduli sono gli stessi del codice vecchio, così i checkpoint vecchi si ricaricano.

import torch
import torch.nn as nn
import torch.nn.functional as F

from tesi.modelli.hydra import Hydra, RMSNorm
from tesi.modelli.ssl import XLSR, carica_ssl, normalizza_wave


class SwiGLU(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        nascosto = int(8 * d / 3)
        self.w_in = nn.Linear(d, 2 * nascosto, bias=False)
        self.w_out = nn.Linear(nascosto, d, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.w_in.weight, mode="fan_in", nonlinearity="linear")
        nn.init.kaiming_normal_(self.w_out.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x):
        gate, valore = self.w_in(x).chunk(2, dim=-1)
        return self.dropout(self.w_out(F.silu(gate) * valore))


class MHA(nn.Module):
    """Attenzione multi-testa di mamba-ssm, con una convoluzione causale su q, k e v."""

    def __init__(self, d, n_teste=4, d_conv=4):
        super().__init__()
        self.n_teste = n_teste
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d, 3 * d)
        self.conv1d = nn.Conv1d(3 * d, 3 * d, kernel_size=d_conv, padding=d_conv - 1, groups=3 * d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x):
        qkv = self.in_proj(x)
        qkv = self.conv1d(qkv.transpose(1, 2))[..., :-(self.d_conv - 1)].transpose(1, 2)
        q, k, v = (t.unflatten(-1, (self.n_teste, -1)).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).flatten(-2))


class PreNormBlock(nn.Module):
    """Norm -> blocco; il residuo viaggia a parte in float32, come in mamba-ssm."""

    def __init__(self, d, mixer):
        super().__init__()
        self.norm = RMSNorm(d)
        self.mixer = mixer

    def forward(self, hidden, residuo):
        residuo = hidden.float() if residuo is None else residuo + hidden.float()
        return self.mixer(self.norm(residuo.to(hidden.dtype))), residuo


class GatedAttentionPool(nn.Module):
    """Media pesata dei frame, con pesi calcolati da un gate SwiGLU."""

    def __init__(self, d):
        super().__init__()
        nascosto = int(8 * d // 3)
        self.gate_value = nn.Linear(d, 2 * nascosto, bias=False)
        self.to_score = nn.Linear(nascosto, 1, bias=False)
        self.dropout = nn.Dropout(0.1)
        nn.init.kaiming_normal_(self.gate_value.weight, mode="fan_in", nonlinearity="linear")
        nn.init.kaiming_normal_(self.to_score.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x):
        gate, valore = self.gate_value(x).chunk(2, dim=-1)
        pesi = F.softmax(self.to_score(F.silu(gate) * valore).squeeze(-1), dim=1)
        return self.dropout(torch.bmm(pesi.unsqueeze(1), x).squeeze(1))


class XLSRMamBo(nn.Module):
    def __init__(self, d=128, n_layer=5, n_hydra=3):
        super().__init__()
        self.ssl, self.normalizza = carica_ssl(XLSR)
        self.input_proj = nn.Linear(self.ssl.config.hidden_size, d, bias=False)
        self.input_norm = RMSNorm(d)

        blocchi = []
        for _ in range(n_layer):
            blocchi += [PreNormBlock(d, Hydra(d)) for _ in range(n_hydra)]
            blocchi += [PreNormBlock(d, SwiGLU(d)), PreNormBlock(d, MHA(d)), PreNormBlock(d, SwiGLU(d))]
        self.blocks = nn.ModuleList(blocchi)
        self.final_norm = RMSNorm(d)
        self.pool = GatedAttentionPool(d)
        self.classifier = nn.Linear(d, 2)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, wave):
        """wave: (batch, campioni) a 16 kHz -> logit (batch, 2) = [bonafide, spoof]"""
        wave = normalizza_wave(wave)
        x = self.ssl(wave).last_hidden_state                 # (batch, frame, 1024)
        x = F.silu(self.input_norm(self.input_proj(x)))
        hidden, residuo = x, None
        for blocco in self.blocks:
            hidden, residuo = blocco(hidden, residuo)
        x = self.final_norm((residuo + hidden.float()).to(hidden.dtype))
        return self.classifier(self.pool(x))
