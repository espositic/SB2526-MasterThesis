# Nes2Net-X (Liu, Truong, Das, Lee, Li, "Nes2Net: A Lightweight Nested Architecture for
# Foundation Model Driven Speech Anti-spoofing", arXiv 2504.05657, 2025).
# Port del back-end di github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW (branch asvspoof5,
# models/wavlm_nes2net.py, licenza Apache 2.0) con la configurazione usata su ASVspoof 5:
# Nes_ratio [8, 8], dilatazione 1, pooling medio, SE_ratio 1.
#
# WavLM Large (ultimo layer, 1024 canali) -> Nested Res2Net TDNN -> media sul tempo -> 2 classi.
# Differenza dall'originale: WavLM da Hugging Face invece che dal checkpoint di Microsoft.

import math

import torch
import torch.nn as nn

from tesi.modelli.ssl import WAVLM_LARGE, carica_ssl, normalizza_wave


class SEModule(nn.Module):
    """Squeeze-and-excitation: ripesa i canali con pesi calcolati dalla media sul tempo."""

    def __init__(self, canali, se_ratio=1):
        super().__init__()
        bottleneck = max(canali // se_ratio, 2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(canali, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(bottleneck, canali, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Bottle2neck(nn.Module):
    """Blocco Res2Net interno: i canali sono divisi in gruppi e ogni gruppo vede anche i precedenti."""

    def __init__(self, canali, kernel_size=3, dilation=1, scale=8, se_ratio=1):
        super().__init__()
        self.width = canali // scale
        self.nums = scale - 1
        self.conv1 = nn.Conv1d(canali, self.width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(self.width * scale)
        # somma pesata dei gruppi già calcolati, pesi inizializzati alla media
        self.weighted_sum = nn.ParameterList([nn.Parameter(torch.ones(1, 1, 1, i + 2) / (i + 2)) for i in range(self.nums)])
        pad = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=(kernel_size, 1), dilation=(dilation, 1), padding=(pad, 0))
            for _ in range(self.nums)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(self.nums)])
        self.conv3 = nn.Conv1d(self.width * scale, canali, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(canali)
        self.relu = nn.ReLU()
        self.se = SEModule(canali, se_ratio)

    def forward(self, x):
        residuo = x
        out = self.bn1(self.relu(self.conv1(x))).unsqueeze(-1)    # (batch, canali, frame, 1)
        gruppi = torch.split(out, self.width, 1)
        sp = gruppi[self.nums]
        uscite = []
        for i in range(self.nums):
            sp = torch.cat((sp, gruppi[i]), -1)
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            uscite.append(torch.sum(sp * self.weighted_sum[i], dim=-1))
        out = torch.cat(uscite + [gruppi[self.nums].squeeze(-1)], 1)
        out = self.bn3(self.relu(self.conv3(out)))
        return self.se(out) + residuo


class NestedRes2NetTDNN(nn.Module):
    """Livello esterno: anche i 1024 canali di WavLM sono divisi in 8 gruppi, in cascata."""

    def __init__(self, canali=1024, nes_ratio=(8, 8), dilation=1, se_ratio=1):
        super().__init__()
        self.n_gruppi = nes_ratio[0]
        self.C = canali // self.n_gruppi
        self.blocchi = nn.ModuleList([Bottle2neck(self.C, 3, dilation, nes_ratio[1], se_ratio)
                                      for _ in range(self.n_gruppi - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.C) for _ in range(self.n_gruppi - 1)])
        self.bn = nn.BatchNorm1d(canali)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(canali, 2)

    def forward(self, x):
        """x: (batch, 1024, frame) -> logit (batch, 2)"""
        gruppi = torch.split(x, self.C, 1)
        uscite = []
        sp = gruppi[0]
        for i in range(self.n_gruppi - 1):
            if i > 0:
                sp = sp + gruppi[i]     # ogni gruppo riceve anche l'uscita del blocco precedente
            sp = self.bns[i](self.relu(self.blocchi[i](sp)))
            uscite.append(sp)
        out = torch.cat(uscite + [gruppi[-1]], 1)
        out = self.relu(self.bn(out))
        return self.fc(out.mean(dim=-1))


class WavLMNes2NetX(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl, self.normalizza = carica_ssl(WAVLM_LARGE)
        self.backend = NestedRes2NetTDNN(self.ssl.config.hidden_size)

    def forward(self, wave):
        """wave: (batch, campioni) a 16 kHz -> logit (batch, 2) = [bonafide, spoof]"""
        if self.normalizza:
            wave = normalizza_wave(wave)
        x = self.ssl(wave).last_hidden_state          # (batch, frame, 1024)
        return self.backend(x.transpose(1, 2))
