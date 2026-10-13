"""
WavLM + MHFA (Multi-Head Factorized Attentive pooling), come nel sistema EPITA
per ASVspoof 5: Stourbe, Miara, Lepage, Dehak, "Exploring WavLM Back-ends for
Speech Spoofing and Deepfake Detection", arXiv 2409.05032 (Track 1, condizione aperta).
Il back-end MHFA è di Peng, Stafylakis, Gu, Plchot, Černocký, "An attention-based
backend allowing efficient fine-tuning of transformer models for speaker
verification", IEEE SLT 2022.

Re-implementazione dal paper: il codice EPITA non è pubblico e il repository di
MHFA (JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling) non ha licenza.

Idea: due combinazioni pesate di tutti i layer di WavLM, una per le chiavi e
una per i valori, compresse a pochi canali; ogni testa di attenzione fa un
attentive pooling sul tempo dei valori e le teste vengono concatenate.

Differenze rispetto al paper EPITA:
  - WavLM da Hugging Face (microsoft/wavlm-base, gli stessi pesi);
  - niente regolarizzazione L2 verso i pesi pre-addestrati né decadimento del
    learning rate per layer; lr del back-end con lr_head nella config di training;
  - loss e ricetta comuni del progetto (focal loss, subset bilanciati) invece
    della cross-entropy pesata 9:1.

Convenzione del progetto: uscita [logit bonafide, logit spoof].
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.models.ssl_frontend import WAVLM_BASE, load_ssl, ssl_layers


@dataclass
class MHFAConfig:
    ssl_name: str = WAVLM_BASE
    n_heads: int = 32              # 32 teste nel sistema EPITA
    compression_dim: int = 128     # canali di chiavi e valori dopo la compressione
    embed_dim: int = 256
    freeze_ssl: bool = False       # True = WavLM congelato, si allena solo MHFA
    gradient_checkpointing: bool = True


class MHFA(nn.Module):
    def __init__(self, n_layers: int, in_dim: int, n_heads: int, compression_dim: int, embed_dim: int):
        super().__init__()
        # Pesi per layer: zero → softmax uniforme all'inizio
        self.layer_weights_k = nn.Parameter(torch.zeros(n_layers))
        self.layer_weights_v = nn.Parameter(torch.zeros(n_layers))
        self.compress_k = nn.Linear(in_dim, compression_dim)
        self.compress_v = nn.Linear(in_dim, compression_dim)
        self.head_scores = nn.Linear(compression_dim, n_heads)
        self.out = nn.Linear(n_heads * compression_dim, embed_dim)

    def forward(self, layers: torch.Tensor) -> torch.Tensor:
        """layers: (b, L, frame, dim) → embedding (b, embed_dim)"""
        k = torch.einsum("bltd,l->btd", layers, F.softmax(self.layer_weights_k, dim=0))
        v = torch.einsum("bltd,l->btd", layers, F.softmax(self.layer_weights_v, dim=0))
        k, v = self.compress_k(k), self.compress_v(v)
        attn = F.softmax(self.head_scores(k), dim=1)          # (b, frame, teste), somma 1 sul tempo
        pooled = torch.einsum("bth,btc->bhc", attn, v)        # (b, teste, compression_dim)
        return self.out(pooled.flatten(1))


class WavLMMHFA(nn.Module):
    def __init__(self, cfg: MHFAConfig):
        super().__init__()
        self.cfg = cfg
        self.ssl, self.normalize = load_ssl(cfg.ssl_name, freeze=cfg.freeze_ssl,
                                            gradient_checkpointing=cfg.gradient_checkpointing and not cfg.freeze_ssl)
        n_layers = self.ssl.config.num_hidden_layers + 1      # uscita della CNN + layer transformer
        self.backend = MHFA(n_layers, self.ssl.config.hidden_size, cfg.n_heads, cfg.compression_dim, cfg.embed_dim)
        self.classifier = nn.Linear(cfg.embed_dim, 2)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_ssl:
            self.ssl.eval()  # front-end congelato: niente dropout, feature deterministiche
        return self

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: (b, campioni) a 16 kHz → logit (b, 2) = [bonafide, spoof]"""
        layers = ssl_layers(self.ssl, wave, self.normalize, frozen=self.cfg.freeze_ssl)
        return self.classifier(self.backend(layers))
