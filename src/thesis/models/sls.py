"""
XLS-R 300M + SLS (Sensitive Layer Selection; Zhang, Wen, Hu, "Audio Deepfake
Detection with Self-Supervised XLS-R and SLS Classifier", ACM Multimedia 2024).

Reimplementazione dal paper: il repo ufficiale (QiShanZhang/SLSforADD) non
dichiara una licenza, quindi il codice non è copiato. Struttura:
  1. si prendono gli hidden state di tutti i 24 strati transformer di XLS-R;
  2. per ogni strato: media sul tempo → lineare 1024→1 → sigmoide = peso dello
     strato (diverso per ogni clip);
  3. somma degli strati pesati → (frame, 1024);
  4. BatchNorm + SELU + max-pool 3×3 → appiattimento → lineare a 1024 → SELU → 2 classi.
I pesi degli strati restano disponibili in `last_layer_weights` (b, 24): servono
per la parte di spiegabilità (quali strati di XLS-R usa il modello).

Front-end come per XLSR-MamBo: Hugging Face, CNN congelata, gradient checkpointing.
Uscita: logit (b, 2) = [bonafide, spoof]; punteggio = logit0 − logit1 (alto = bonafide).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

XLSR_300M = "facebook/wav2vec2-xls-r-300m"


def xlsr_frames(n_samples: int) -> int:
    """Frame prodotti dalla CNN di wav2vec2 (finestra 400, passo 320 campioni)."""
    return (n_samples - 400) // 320 + 1


@dataclass
class SLSConfig:
    n_samples: int = 66_800        # serve per dimensionare il primo lineare (come nel paper)
    freeze_ssl: bool = False
    gradient_checkpointing: bool = True


class XLSRSLS(nn.Module):
    def __init__(self, cfg: SLSConfig):
        super().__init__()
        self.cfg = cfg
        self.ssl = Wav2Vec2Model.from_pretrained(XLSR_300M)
        self.ssl.config.mask_time_prob = 0.0
        self.ssl.config.layerdrop = 0.0
        self.ssl.freeze_feature_encoder()
        if cfg.freeze_ssl:
            self.ssl.requires_grad_(False)
        elif cfg.gradient_checkpointing:
            self.ssl.gradient_checkpointing_enable()
        d = self.ssl.config.hidden_size

        self.layer_score = nn.Linear(d, 1)                 # peso di ogni strato
        self.bn = nn.BatchNorm2d(1)
        t, f = xlsr_frames(cfg.n_samples) // 3, d // 3     # dopo max-pool 3×3
        self.fc1 = nn.Linear(t * f, 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.last_layer_weights = None

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        out = self.ssl(wave, output_hidden_states=True)
        layers = torch.stack(out.hidden_states[1:], dim=1)       # (b, 24, frame, 1024), senza gli embedding CNN
        w = torch.sigmoid(self.layer_score(layers.mean(dim=2)))  # (b, 24, 1)
        self.last_layer_weights = w.squeeze(-1).detach()
        x = (layers * w.unsqueeze(-1)).sum(dim=1)                # (b, frame, 1024)
        x = F.selu(self.bn(x.unsqueeze(1)))
        x = F.max_pool2d(x, (3, 3)).flatten(1)
        assert x.shape[1] == self.fc1.in_features, \
            f"lunghezza clip diversa da quella della config ({x.shape[1]} contro {self.fc1.in_features})"
        x = F.selu(self.fc1(x))
        return self.fc2(x)
