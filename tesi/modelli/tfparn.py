# TFPARN: Transformer su spettrogramma log-Mel, allenato con focal loss + ranking loss a coppie
# (Yin e Zhao, "A training-efficient transformer-based anti-spoofing network for logical access
# in ASVspoof 5", arXiv 2606.02980). È il modello di confronto senza SSL.
# Port di Model/model.py di github.com/SomeB1oody/TFPARN (licenza MIT), con gli iperparametri
# usati su ASVspoof 5. La ranking loss è in tesi/training.py.
#
# forma d'onda -> STFT (1024 punti, hop 160) -> 160 bande Mel in dB -> LayerNorm -> lineare a 256
# + codifica posizionale -> 6 layer Transformer -> attention pooling -> MLP a 2 classi.
#
# Differenze dall'originale: classi nell'ordine del progetto [bonafide, spoof]; niente
# test-time augmentation; SpecAugment opzionale (specaugment=True), solo durante il training.

import math

import torch
import torch.nn as nn


def filtri_mel(n_fft=1024, n_mel=160, freq=16000):
    """Filtri triangolari tra 0 e freq/2, costruiti come nell'originale."""
    hz_a_mel = lambda hz: 2595 * math.log10(1 + hz / 700)
    punti_mel = torch.linspace(hz_a_mel(0), hz_a_mel(freq / 2), n_mel + 2)
    punti_hz = 700 * (10 ** (punti_mel / 2595) - 1)
    bin_ = torch.floor((n_fft + 1) * punti_hz / freq).long().tolist()
    filtri = torch.zeros(n_mel, n_fft // 2 + 1)
    for m in range(1, n_mel + 1):
        sinistra, centro, destra = bin_[m - 1], bin_[m], bin_[m + 1]
        for k in range(sinistra, centro):
            filtri[m - 1, k] = (k - sinistra) / (centro - sinistra)
        for k in range(centro, destra):
            filtri[m - 1, k] = (destra - k) / (destra - centro)
    return filtri


def spec_augment(mel, n_maschere=2, max_freq=20, max_tempo=40):
    """
    SpecAugment (Park et al., 2019): copre alcune bande di frequenza e alcuni intervalli di tempo
    con il valore medio della clip.
    """
    mel = mel.clone()
    b, n_frame, n_mel = mel.shape
    for i in range(b):
        media = mel[i].mean()
        for _ in range(n_maschere):
            f = int(torch.randint(0, max_freq + 1, (1,)))
            f0 = int(torch.randint(0, n_mel - f + 1, (1,)))
            mel[i, :, f0:f0 + f] = media
            t = int(torch.randint(0, min(max_tempo, n_frame) + 1, (1,)))
            t0 = int(torch.randint(0, n_frame - t + 1, (1,)))
            mel[i, t0:t0 + t, :] = media
    return mel


class TFPARN(nn.Module):
    def __init__(self, specaugment=False, n_mel=160, d=256, n_teste=8, n_layer=6, dim_ff=1024, dropout=0.3):
        super().__init__()
        self.specaugment = specaugment
        self.register_buffer("mel_basis", filtri_mel(n_mel=n_mel), persistent=False)
        self.register_buffer("window", torch.hann_window(1024), persistent=False)
        self.input_norm = nn.LayerNorm(n_mel)
        self.projection = nn.Linear(n_mel, d)

        # codifica posizionale sinusoidale (fino a 5000 frame, ~50 s)
        pos = torch.arange(5000)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(1, 5000, d)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe, persistent=False)
        self.pos_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(d, n_teste, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.attention_pooling = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.classifier = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d // 2, 2))

        # inizializzazione dell'originale: Xavier ovunque, guadagno 0,5 sulla proiezione
        for nome, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5 if nome.startswith("projection") else 1.0)

    def log_mel(self, wave):
        """(batch, campioni) -> (batch, frame, n_mel), sempre in float32."""
        with torch.autocast(device_type=wave.device.type, enabled=False):
            wave = wave.float()
            picco = wave.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            wave = wave / picco
            spettro = torch.stft(wave, 1024, 160, window=self.window, center=True, return_complex=True).abs() ** 2
            mel = (self.mel_basis @ spettro).clamp(min=1e-10)
            return (10 * torch.log10(mel)).clamp(-80, 0).div(10).transpose(1, 2)

    def forward(self, wave):
        """wave: (batch, campioni) a 16 kHz -> logit (batch, 2) = [bonafide, spoof]"""
        mel = self.log_mel(wave)
        if self.specaugment and self.training:
            mel = spec_augment(mel)
        x = self.projection(self.input_norm(mel))
        x = self.pos_dropout(x + self.pos_enc[:, :x.shape[1]].to(x.dtype))
        x = self.encoder(x)
        pesi = torch.softmax(self.attention_pooling(x), dim=1)     # (batch, frame, 1)
        return self.classifier((x * pesi).sum(dim=1))
