# WavLM Base + MHFA (Multi-Head Factorized Attentive pooling), come nel sistema EPITA per
# ASVspoof 5: Stourbe et al., "Exploring WavLM Back-ends for Speech Spoofing and Deepfake
# Detection", ASVspoof 2024 Workshop. MHFA è di Peng et al., IEEE SLT 2022.
# Riscritto dal paper: il codice EPITA non è pubblico e quello di MHFA non ha licenza.
#
# Due medie pesate di tutti i layer di WavLM (una per le chiavi, una per i valori),
# compresse a 128 canali; ogni testa fa una media pesata sul tempo dei valori.

import torch
import torch.nn as nn
import torch.nn.functional as F

from tesi.modelli.ssl import WAVLM_BASE, carica_ssl, normalizza_wave


class MHFA(nn.Module):
    def __init__(self, n_layer, dim, n_teste=32, dim_compressa=128, dim_embedding=256):
        super().__init__()
        # pesi dei layer a zero -> all'inizio la softmax dà la media semplice
        self.layer_weights_k = nn.Parameter(torch.zeros(n_layer))
        self.layer_weights_v = nn.Parameter(torch.zeros(n_layer))
        self.compress_k = nn.Linear(dim, dim_compressa)
        self.compress_v = nn.Linear(dim, dim_compressa)
        self.head_scores = nn.Linear(dim_compressa, n_teste)
        self.out = nn.Linear(n_teste * dim_compressa, dim_embedding)

    def forward(self, layer):
        """layer: (batch, n_layer, frame, dim) -> embedding (batch, dim_embedding)"""
        k = torch.einsum("bltd,l->btd", layer, F.softmax(self.layer_weights_k, dim=0))
        v = torch.einsum("bltd,l->btd", layer, F.softmax(self.layer_weights_v, dim=0))
        k, v = self.compress_k(k), self.compress_v(v)
        attenzione = F.softmax(self.head_scores(k), dim=1)          # (batch, frame, teste), somma 1 sul tempo
        per_testa = torch.einsum("bth,btc->bhc", attenzione, v)     # (batch, teste, dim_compressa)
        return self.out(per_testa.flatten(1))


class WavLMMHFA(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl, self.normalizza = carica_ssl(WAVLM_BASE)
        n_layer = self.ssl.config.num_hidden_layers + 1          # uscita della CNN + layer transformer
        self.backend = MHFA(n_layer, self.ssl.config.hidden_size)
        self.classifier = nn.Linear(256, 2)

    def forward(self, wave):
        """wave: (batch, campioni) a 16 kHz -> logit (batch, 2) = [bonafide, spoof]"""
        if self.normalizza:
            wave = normalizza_wave(wave)
        layer = torch.stack(self.ssl(wave, output_hidden_states=True).hidden_states, dim=1)
        return self.classifier(self.backend(layer))
