# Modelli auto-supervisionati (SSL) presi da Hugging Face, in comune tra MamBo, Nes2Net-X e MHFA.
#
# Stesse scelte per tutti:
#   - niente masking interno (SpecAugment di wav2vec2) e niente layerdrop;
#   - la CNN iniziale resta congelata, il resto si allena;
#   - gradient checkpointing per stare negli 8 GB della 3060.

import torch
from transformers import AutoFeatureExtractor, AutoModel

XLSR = "facebook/wav2vec2-xls-r-300m"
WAVLM_BASE = "microsoft/wavlm-base"
WAVLM_LARGE = "microsoft/wavlm-large"


def carica_ssl(nome, congelato=False, checkpointing=True):
    """
    Restituisce (modello, normalizza). normalizza dice se il modello vuole la forma d'onda
    a media 0 e varianza 1 (XLS-R e WavLM Large sì, WavLM Base no).
    """
    ssl = AutoModel.from_pretrained(nome)
    # la config si cambia dopo il caricamento, come nei primi esperimenti:
    # così i checkpoint vecchi di MamBo hanno gli stessi parametri e si possono ricaricare
    ssl.config.mask_time_prob = 0.0
    ssl.config.mask_feature_prob = 0.0
    ssl.config.layerdrop = 0.0
    ssl.freeze_feature_encoder()
    if congelato:
        ssl.requires_grad_(False)
    elif checkpointing:
        ssl.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    normalizza = getattr(AutoFeatureExtractor.from_pretrained(nome), "do_normalize", True)
    return ssl, normalizza


def normalizza_wave(wave):
    """Media 0 e varianza 1 per ogni clip."""
    return (wave - wave.mean(dim=1, keepdim=True)) / torch.sqrt(wave.var(dim=1, keepdim=True) + 1e-7)
