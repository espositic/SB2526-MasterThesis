# I quattro modelli della tesi e i loro iperparametri di training.
#
#   "mambo"    XLSR-MamBo        XLS-R 300M + Mamba/attenzione          (Ng et al., 2026)
#   "nes2net"  Nes2Net-X         WavLM Large + Nested Res2Net           (Liu et al., 2025)
#   "mhfa"     WavLM Base + MHFA                                          (Stourbe et al., 2024)
#   "tfparn"   TFPARN            Transformer su log-Mel, senza SSL      (Yin e Zhao, 2026)
#
# Tutti prendono la forma d'onda (batch, campioni) a 16 kHz e restituiscono
# due logit [bonafide, spoof].

from tesi.modelli.mambo import XLSRMamBo
from tesi.modelli.mhfa import WavLMMHFA
from tesi.modelli.nes2net import WavLMNes2NetX
from tesi.modelli.tfparn import TFPARN

# Ricetta comune (quella dei primi esperimenti con MamBo e Nes2Net-X):
# AdamW, warmup 10% + coseno, focal loss, batch 8 con 4 passi di accumulo (= 32),
# al massimo 20 epoche, stop dopo 3 epoche senza miglioramenti dell'EER sul dev.
RICETTA = {
    "batch": 8,
    "accumulo": 4,
    "lr": 1e-5,
    "lr_backend": None,      # None = stesso lr per tutto il modello
    "weight_decay": 0.05,
    "epoche": 20,
    "pazienza": 3,
    "warmup": 0.1,
    "peso_ranking": 0.0,     # ranking loss a coppie, solo per TFPARN
}

IPERPARAMETRI = {
    "mambo": RICETTA,
    "nes2net": RICETTA,
    # MHFA: lr del paper EPITA (2e-5 per WavLM, 5e-3 per il back-end).
    # Se la loss esplode nelle prime epoche, provare lr_backend 1e-3.
    "mhfa": {**RICETTA, "lr": 2e-5, "lr_backend": 5e-3},
    # TFPARN: iperparametri del paper, è piccolo e si allena da zero
    "tfparn": {**RICETTA, "batch": 64, "accumulo": 1, "lr": 6.67e-5, "weight_decay": 0.01,
               "epoche": 100, "pazienza": 15, "warmup": 0.05, "peso_ranking": 0.3},
}


def crea_modello(nome, specaugment=False):
    if nome == "mambo":
        return XLSRMamBo()
    if nome == "nes2net":
        return WavLMNes2NetX()
    if nome == "mhfa":
        return WavLMMHFA()
    if nome == "tfparn":
        return TFPARN(specaugment=specaugment)
    raise ValueError(f"modello sconosciuto: {nome}")


def conta_parametri(modello):
    """Numero di parametri in milioni (totali, allenabili)."""
    totali = sum(p.numel() for p in modello.parameters())
    allenabili = sum(p.numel() for p in modello.parameters() if p.requires_grad)
    return round(totali / 1e6, 1), round(allenabili / 1e6, 1)
