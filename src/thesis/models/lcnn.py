"""
LCNN (Light CNN con Max-Feature-Map) su spettrogramma, baseline classica di
ASVspoof: Lavrentyeva et al., "STC Antispoofing Systems for the ASVspoof2019
Challenge", Interspeech 2019. Poche centinaia di migliaia di parametri: gira su CPU.

Componenti attivabili da config, per un'ablation:
  - pooling="lstm_attn": BiLSTM sul tempo + attention pooling invece della
    media (come la baseline LFCC-LCNN-LSTM di ASVspoof 2021);
  - specaug=True: SpecAugment (Park et al., Interspeech 2019), maschere in
    tempo e frequenza, solo in training;
  - loss="ocsoftmax": One-Class Softmax (Zhang, Jiang, Duan, IEEE SPL 2021)
    al posto della cross-entropy binaria.

Ingresso: (b, 1, n_mels, frame). Uscita: logit (b, 2) = [bonafide, spoof];
con OC-Softmax i "logit" sono [s, -s] con s = α·cos/2, così
punteggio = logit0 − logit1 = α·cos (alto = bonafide) come per gli altri modelli.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LCNNConfig:
    n_mels: int = 128
    dropout: float = 0.75
    pooling: str = "mean"          # "mean" | "lstm_attn"
    loss: str = "ce"               # "ce" | "ocsoftmax"
    specaug: bool = False
    freq_mask: int = 20            # larghezza massima delle maschere (bin Mel)
    time_mask: int = 20            # (frame)
    n_masks: int = 2               # maschere per asse
    oc_r_real: float = 0.9         # margini e scala di OC-Softmax (valori del paper)
    oc_r_fake: float = 0.2
    oc_alpha: float = 20.0


class MFM(nn.Module):
    """Max-Feature-Map: dimezza i canali tenendo il massimo tra le due metà."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.maximum(a, b)


def conv_mfm(c_in, c_out, k):
    return nn.Sequential(nn.Conv2d(c_in, 2 * c_out, k, padding=k // 2), MFM())


def spec_augment(x: torch.Tensor, cfg: LCNNConfig) -> torch.Tensor:
    """Maschere a zero in frequenza e tempo, diverse per ogni esempio del batch."""
    b, _, n_freq, n_time = x.shape
    mask = torch.zeros(b, 1, n_freq, n_time, dtype=torch.bool, device=x.device)
    for size, n_axis, axis in ((cfg.freq_mask, n_freq, 2), (cfg.time_mask, n_time, 3)):
        for _ in range(cfg.n_masks):
            width = torch.randint(1, size + 1, (b,), device=x.device)
            start = (torch.rand(b, device=x.device) * (n_axis - width)).long()
            pos = torch.arange(n_axis, device=x.device)
            band = (pos[None] >= start[:, None]) & (pos[None] < (start + width)[:, None])  # (b, n_axis)
            mask |= band[:, None, :, None] if axis == 2 else band[:, None, None, :]
    assert mask.any(dim=(1, 2, 3)).all(), "SpecAugment non ha mascherato nulla"
    return x.masked_fill(mask, 0.0)


class AttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, x):                       # (b, t, d)
        w = F.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)


class LCNN(nn.Module):
    def __init__(self, cfg: LCNNConfig):
        super().__init__()
        assert cfg.pooling in ("mean", "lstm_attn") and cfg.loss in ("ce", "ocsoftmax")
        self.cfg = cfg
        self.features = nn.Sequential(
            conv_mfm(1, 32, 5), nn.MaxPool2d(2),
            conv_mfm(32, 32, 1), nn.BatchNorm2d(32),
            conv_mfm(32, 48, 3), nn.MaxPool2d(2), nn.BatchNorm2d(48),
            conv_mfm(48, 48, 1), nn.BatchNorm2d(48),
            conv_mfm(48, 64, 3), nn.MaxPool2d(2),
            conv_mfm(64, 64, 1), nn.BatchNorm2d(64),
            conv_mfm(64, 32, 3), nn.BatchNorm2d(32),
            conv_mfm(32, 32, 1), nn.BatchNorm2d(32),
            conv_mfm(32, 32, 3), nn.MaxPool2d(2),
            nn.Dropout(cfg.dropout),
        )
        d = 32 * (cfg.n_mels // 16)             # canali × frequenze dopo 4 max-pool
        if cfg.pooling == "lstm_attn":
            self.lstm = nn.LSTM(d, d // 2, num_layers=2, batch_first=True, bidirectional=True)
            self.pool = AttentionPool(d)
        self.embed = nn.Sequential(nn.Linear(d, 2 * 80), MFM(), nn.BatchNorm1d(80))
        if cfg.loss == "ce":
            self.out = nn.Linear(80, 2)
        else:
            self.center = nn.Parameter(torch.randn(80))  # direzione della classe bonafide

    def forward(self, x):
        if self.training and self.cfg.specaug:
            x = spec_augment(x, self.cfg)
        x = self.features(x)                    # (b, 32, n_mels/16, frame/16)
        x = x.flatten(1, 2).transpose(1, 2)     # (b, frame/16, d)
        if self.cfg.pooling == "lstm_attn":
            x = self.pool(x + self.lstm(x)[0])  # residuo attorno alla BiLSTM
        else:
            x = x.mean(dim=1)
        emb = self.embed(x)
        if self.cfg.loss == "ce":
            return self.out(emb)
        cos = F.normalize(emb, dim=1) @ F.normalize(self.center, dim=0)
        s = self.cfg.oc_alpha * cos / 2
        return torch.stack([s, -s], dim=1)

    def loss_fn(self, logits, label):
        """Loss propria del modello (None = usa quella di default del training)."""
        if self.cfg.loss == "ce":
            return None
        cos = 2 * logits[:, 0].float() / self.cfg.oc_alpha
        # bonafide (0): cos deve superare r_real; spoof (1): cos deve stare sotto r_fake
        margin = torch.where(label == 0, self.cfg.oc_r_real - cos, cos - self.cfg.oc_r_fake)
        return F.softplus(self.cfg.oc_alpha * margin).mean()
