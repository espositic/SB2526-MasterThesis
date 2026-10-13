"""
TFPARN ("Transformer-based Focal-Pairwise Attentive Ranking Network"): Transformer
su log-Mel con focal loss e ranking loss a coppie (Yin e Zhao, "A training-efficient
transformer-based anti-spoofing network for logical access in ASVspoof 5", arXiv
2606.02980), modello di confronto per la famiglia "spettrogramma + rete neurale".
Codice originale: https://github.com/SomeB1oody/TFPARN (licenza MIT, vedi
THIRD_PARTY_NOTICES.md), con AASIST e RawNet2 nello stesso repository.

Port di Model/model.py con gli iperparametri di Model/main_train.py
(quelli usati su ASVspoof 5):
forma d'onda normalizzata al picco → STFT (1024 punti, hop 160) → 160 bande Mel
in dB, compresse in [−8, 0] → LayerNorm → lineare a 256 + codifica posizionale
→ 6 layer Transformer (8 teste, FF 1024, dropout 0,3) → attentive pooling sul
tempo → MLP a 2 classi.

La ranking loss a coppie (hinge con margine 1, peso 0,3) è in train.py
(pairwise_weight, pairwise_margin), così vale per qualunque modello.

Differenze rispetto all'originale:
  - classi nell'ordine del progetto, [logit bonafide, logit spoof] (l'originale
    ha 0 = spoof); la ranking loss usa il punteggio del progetto
    (logit bonafide − logit spoof) invece del solo logit bonafide;
  - normalizzazione al picco fatta nel modello, cioè dopo RawBoost;
  - niente test-time augmentation (l'originale media più crop in validazione);
  - niente pooling top-k (resta "mean" come alternativa all'attention);
  - ricetta del progetto: warmup + coseno per step invece che per epoca.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TFPARNConfig:
    n_mels: int = 160
    n_fft: int = 1024
    hop_length: int = 160
    sample_rate: int = 16_000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.3
    activation: str = "relu"
    pooling: str = "attention"     # "attention" o "mean"


def mel_filterbank(n_fft: int, n_mels: int, sample_rate: int) -> torch.Tensor:
    """Filtri triangolari tra 0 e sample_rate/2, costruiti come nell'originale: (n_mels, n_fft/2 + 1)."""
    hz_to_mel = lambda hz: 2595 * math.log10(1 + hz / 700)
    mel_points = torch.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long().tolist()
    fbank = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            fbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            fbank[m - 1, k] = (right - k) / (right - center)
    return fbank


class TFPARN(nn.Module):
    def __init__(self, cfg: TFPARNConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mel_basis", mel_filterbank(cfg.n_fft, cfg.n_mels, cfg.sample_rate), persistent=False)
        self.register_buffer("window", torch.hann_window(cfg.n_fft), persistent=False)
        self.input_norm = nn.LayerNorm(cfg.n_mels)
        self.projection = nn.Linear(cfg.n_mels, cfg.d_model)

        # Codifica posizionale sinusoidale standard (fino a 5000 frame, ~50 s)
        pos = torch.arange(5000)[:, None]
        div = torch.exp(torch.arange(0, cfg.d_model, 2) * (-math.log(10000.0) / cfg.d_model))
        pe = torch.zeros(1, 5000, cfg.d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe, persistent=False)
        self.pos_dropout = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout,
                                           activation=cfg.activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        assert cfg.pooling in ("attention", "mean"), f"pooling sconosciuto: {cfg.pooling}"
        if cfg.pooling == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2), nn.Tanh(), nn.Linear(cfg.d_model // 2, 1))
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 2))

        # Inizializzazione dell'originale: Xavier ovunque, guadagno 0,5 sulla proiezione
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5 if name.startswith("projection") else 1.0)

    def log_mel(self, wave: torch.Tensor) -> torch.Tensor:
        """(b, campioni) → (b, frame, n_mels), in float32 anche sotto autocast."""
        with torch.autocast(device_type=wave.device.type, enabled=False):
            wave = wave.float()
            peak = wave.abs().amax(dim=1, keepdim=True)
            wave = torch.where(peak > 0, wave / peak.clamp(min=1e-12), wave)   # picco a 1
            spec = torch.stft(wave, self.cfg.n_fft, self.cfg.hop_length, window=self.window,
                              center=True, return_complex=True).abs() ** 2
            mel = (self.mel_basis @ spec).clamp(min=1e-10)
            return (10 * torch.log10(mel)).clamp(-80, 0).div(10).transpose(1, 2)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: (b, campioni) a 16 kHz → logit (b, 2) = [bonafide, spoof]"""
        x = self.projection(self.input_norm(self.log_mel(wave)))
        x = self.pos_dropout(x + self.pos_enc[:, :x.shape[1]].to(x.dtype))
        x = self.encoder(x)
        if self.cfg.pooling == "attention":
            weights = torch.softmax(self.attention_pooling(x), dim=1)    # (b, frame, 1)
            pooled = (x * weights).sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        return self.classifier(pooled)
