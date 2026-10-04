"""
WavLM-Large + Nes2Net-X (Liu et al., "Nes2Net: A Lightweight Nested Architecture
for Foundation Model Driven Speech Anti-Spoofing", IEEE TIFS 2025, arXiv 2504.05657).

Port del back-end da https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW
(ramo asvspoof5, models/wavlm_nes2net.py; licenza Apache 2.0 dichiarata nel README).
Il back-end è un Res2Net annidato: le 1024 dimensioni del front-end sono divise in
8 gruppi elaborati in cascata da blocchi Res2Net (a loro volta divisi in 8), con
somma pesata appresa dei rami (la "X" di Nes2Net-X) e squeeze-and-excitation.
Nessuno strato di riduzione di dimensione: circa 0,5 M parametri.

Front-end: WavLM-Large fine-tunato, uscita dell'ultimo strato, come nel codice
originale. Differenze, dovute alla 3060 da 8 GB / Windows:
  - WavLM-Large da Hugging Face (microsoft/wavlm-large) invece del checkpoint .pt
    di unilm (stessi pesi); CNN iniziale congelata e gradient checkpointing.

Uscita: logit (b, 2) = [bonafide, spoof]; punteggio = logit0 − logit1 (alto = bonafide).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import WavLMModel

WAVLM_LARGE = "microsoft/wavlm-large"


@dataclass
class Nes2NetConfig:
    nes_ratio: tuple = (8, 8)      # (gruppi esterni, scala dei Res2Net interni)
    se_ratio: int = 1
    dilation: int = 1              # config ASVspoof5 del repo
    pool: str = "mean"             # "mean" | "astp"
    freeze_ssl: bool = False
    gradient_checkpointing: bool = True


class SEModule(nn.Module):
    def __init__(self, channels: int, ratio: int):
        super().__init__()
        bottleneck = max(channels // ratio, 2)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, bottleneck, 1), nn.ReLU(),
                                nn.Conv1d(bottleneck, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class Bottle2neck(nn.Module):
    """Blocco Res2Net: ogni ramo vede la concatenazione dei precedenti, combinata con pesi appresi."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, scale: int, se_ratio: int):
        super().__init__()
        width = channels // scale
        self.width, self.nums = width, scale - 1
        self.conv1 = nn.Conv1d(channels, width * scale, 1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        pad = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList(nn.Conv2d(width, width, (kernel_size, 1), dilation=(dilation, 1), padding=(pad, 0))
                                   for _ in range(self.nums))
        self.bns = nn.ModuleList(nn.BatchNorm2d(width) for _ in range(self.nums))
        self.weighted_sum = nn.ParameterList(nn.Parameter(torch.full((1, 1, 1, i + 2), 1 / (i + 2)))
                                             for i in range(self.nums))
        self.conv3 = nn.Conv1d(width * scale, channels, 1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.se = SEModule(channels, se_ratio)

    def forward(self, x):
        out = self.bn1(self.relu(self.conv1(x))).unsqueeze(-1)    # (b, c, t, 1)
        spx = torch.split(out, self.width, 1)
        sp, outs = spx[self.nums], []
        for i in range(self.nums):
            sp = torch.cat((sp, spx[i]), -1)                        # accumula i rami sull'ultimo asse
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            outs.append((sp * self.weighted_sum[i]).sum(dim=-1))
        out = torch.cat(outs + [spx[self.nums].squeeze(-1)], 1)
        out = self.se(self.bn3(self.relu(self.conv3(out))))
        return out + x


class ASTP(nn.Module):
    """Attentive statistics pooling (media e deviazione standard pesate)."""

    def __init__(self, dim: int, bottleneck: int = 128):
        super().__init__()
        self.linear1 = nn.Conv1d(dim, bottleneck, 1)
        self.linear2 = nn.Conv1d(bottleneck, dim, 1)

    def forward(self, x):
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x))), dim=2)
        mean = (alpha * x).sum(dim=2)
        std = ((alpha * x ** 2).sum(dim=2) - mean ** 2).clamp(min=1e-10).sqrt()
        return torch.cat([mean, std], dim=1)


class NestedRes2NetTDNN(nn.Module):
    def __init__(self, cfg: Nes2NetConfig, input_dim: int = 1024):
        super().__init__()
        n_outer, scale = cfg.nes_ratio
        assert input_dim % n_outer == 0
        self.c = input_dim // n_outer
        self.n_outer = n_outer
        self.blocks = nn.ModuleList(Bottle2neck(self.c, 3, cfg.dilation, scale, cfg.se_ratio) for _ in range(n_outer - 1))
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.c) for _ in range(n_outer - 1))
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.pool = ASTP(input_dim) if cfg.pool == "astp" else None
        self.fc = nn.Linear(2 * input_dim if self.pool else input_dim, 2)

    def forward(self, x):                                          # (b, 1024, t)
        spx = torch.split(x, self.c, 1)
        outs, sp = [], None
        for i in range(self.n_outer - 1):
            sp = spx[i] if i == 0 else sp + spx[i]                  # ogni gruppo riceve anche il precedente
            sp = self.bns[i](self.relu(self.blocks[i](sp)))
            outs.append(sp)
        out = self.relu(self.bn(torch.cat(outs + [spx[-1]], 1)))
        out = self.pool(out) if self.pool else out.mean(dim=-1)
        return self.fc(out)


class WavLMNes2Net(nn.Module):
    def __init__(self, cfg: Nes2NetConfig):
        super().__init__()
        self.cfg = cfg
        self.ssl = WavLMModel.from_pretrained(WAVLM_LARGE)
        self.ssl.config.mask_time_prob = 0.0       # niente masking interno (l'originale non lo usa)
        self.ssl.config.layerdrop = 0.0
        self.ssl.freeze_feature_encoder()
        if cfg.freeze_ssl:
            self.ssl.requires_grad_(False)
        elif cfg.gradient_checkpointing:
            self.ssl.gradient_checkpointing_enable()
        self.backend = NestedRes2NetTDNN(cfg, self.ssl.config.hidden_size)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: (b, campioni) a 16 kHz, senza normalizzazione (come nell'originale)."""
        x = self.ssl(wave).last_hidden_state                       # (b, t, 1024)
        return self.backend(x.transpose(1, 2))
