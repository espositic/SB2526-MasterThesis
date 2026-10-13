"""
Front-end auto-supervisionati (WavLM, XLS-R) caricati da Hugging Face,
in comune tra i modelli che usano gli stati di tutti i layer (MHFA, MoLEx)
o solo l'ultimo (Conformer + TCM).

Scelte uguali a XLSR-MamBo (models/mambo.py):
  - niente masking SpecAugment interno né layerdrop;
  - CNN iniziale sempre congelata;
  - normalizzazione della forma d'onda come nel feature extractor ufficiale
    del modello (WavLM-Large e XLS-R sì, WavLM Base no).
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

WAVLM_BASE = "microsoft/wavlm-base"
WAVLM_LARGE = "microsoft/wavlm-large"
XLSR_300M = "facebook/wav2vec2-xls-r-300m"


def load_ssl(name: str, freeze: bool, gradient_checkpointing: bool, n_layers: int = 0) -> tuple[nn.Module, bool]:
    """
    Restituisce (modello, normalize). n_layers > 0 tiene solo i primi n_layers
    layer transformer (MoLEx usa i primi 12 di WavLM-Large).
    """
    config = AutoConfig.from_pretrained(name, mask_time_prob=0.0, mask_feature_prob=0.0, layerdrop=0.0)
    ssl = AutoModel.from_pretrained(name, config=config)
    if n_layers > 0:
        assert n_layers <= len(ssl.encoder.layers), f"{name} ha solo {len(ssl.encoder.layers)} layer"
        ssl.encoder.layers = ssl.encoder.layers[:n_layers]
        ssl.config.num_hidden_layers = n_layers
    ssl.freeze_feature_encoder()
    if freeze:
        ssl.requires_grad_(False)
    if gradient_checkpointing:
        # use_reentrant=False: serve quando i primi layer sono congelati e
        # l'ingresso di un layer con parametri allenabili non richiede gradiente
        ssl.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    normalize = bool(getattr(AutoFeatureExtractor.from_pretrained(name), "do_normalize", True))
    return ssl, normalize


def normalize_wave(wave: torch.Tensor) -> torch.Tensor:
    """Media 0 e varianza 1 per clip."""
    return (wave - wave.mean(dim=1, keepdim=True)) / torch.sqrt(wave.var(dim=1, keepdim=True) + 1e-7)


def ssl_layers(ssl: nn.Module, wave: torch.Tensor, normalize: bool, frozen: bool) -> torch.Tensor:
    """
    Stati nascosti di tutti i layer: (b, L+1, frame, dim). L'indice 0 è l'uscita
    della CNN dopo proiezione e convoluzione posizionale, gli altri sono i layer transformer.
    """
    if normalize:
        wave = normalize_wave(wave)
    with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
        hidden = ssl(wave, output_hidden_states=True).hidden_states
    return torch.stack(hidden, dim=1)
