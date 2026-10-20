# Metriche di ASVspoof 5: EER, minDCF, actDCF e Cllr, più EER per gruppo e intervallo di confidenza.
#
# Convenzione: label 0 = bonafide, 1 = spoof; punteggio alto = bonafide.
# Il punteggio di un modello è logit_bonafide - logit_spoof.

import numpy as np
import pandas as pd

# parametri del DCF di ASVspoof 5
P_SPOOF, C_MISS, C_FA = 0.05, 1, 10


def tassi(label, punteggi):
    """FRR e FAR per ogni soglia possibile (bonafide se il punteggio supera la soglia)."""
    label, punteggi = np.asarray(label), np.asarray(punteggi, dtype=float)
    bonafide = label[np.argsort(punteggi, kind="mergesort")] == 0
    frr = np.concatenate([[0], np.cumsum(bonafide) / bonafide.sum()])
    far = np.concatenate([[1], 1 - np.cumsum(~bonafide) / (~bonafide).sum()])
    return frr, far


def eer(label, punteggi):
    """Equal Error Rate: il punto in cui FRR e FAR sono uguali."""
    frr, far = tassi(label, punteggi)
    i = np.argmin(np.abs(frr - far))
    return (frr[i] + far[i]) / 2


def min_dcf(label, punteggi):
    """DCF normalizzato alla soglia migliore."""
    frr, far = tassi(label, punteggi)
    dcf = C_MISS * (1 - P_SPOOF) * frr + C_FA * P_SPOOF * far
    return dcf.min() / min(C_MISS * (1 - P_SPOOF), C_FA * P_SPOOF)


def act_dcf(label, punteggi):
    """DCF alla soglia di Bayes: misura anche la calibrazione dei punteggi."""
    label, punteggi = np.asarray(label), np.asarray(punteggi, dtype=float)
    soglia = -np.log(C_MISS * (1 - P_SPOOF) / (C_FA * P_SPOOF))
    p_miss = np.mean(punteggi[label == 0] < soglia)
    p_fa = np.mean(punteggi[label == 1] >= soglia)
    dcf = C_MISS * (1 - P_SPOOF) * p_miss + C_FA * P_SPOOF * p_fa
    return dcf / min(C_MISS * (1 - P_SPOOF), C_FA * P_SPOOF)


def cllr(label, punteggi):
    """Costo log-likelihood ratio in bit (0 = perfetto, 1 = come tirare a caso)."""
    label, punteggi = np.asarray(label), np.asarray(punteggi, dtype=float)
    bona, spoof = punteggi[label == 0], punteggi[label == 1]
    return 0.5 * (np.mean(np.logaddexp(0, -bona)) + np.mean(np.logaddexp(0, spoof))) / np.log(2)


def tutte(label, punteggi):
    return {
        "EER (%)": float(100 * eer(label, punteggi)),
        "minDCF": float(min_dcf(label, punteggi)),
        "actDCF": float(act_dcf(label, punteggi)),
        "Cllr": float(cllr(label, punteggi)),
    }


def eer_per_gruppo(label, punteggi, gruppi):
    """
    EER (%) per ogni attacco o codec: gli spoof sono quelli del gruppo,
    i bonafide sono sempre tutti.
    """
    label, punteggi, gruppi = np.asarray(label), np.asarray(punteggi, dtype=float), np.asarray(gruppi)
    bona = label == 0
    risultato = {}
    for g in sorted(set(gruppi[~bona])):
        tieni = bona | (gruppi == g)
        risultato[str(g)] = float(100 * eer(label[tieni], punteggi[tieni]))
    return pd.Series(risultato)


def intervallo_eer(label, punteggi, n=1000, seed=0):
    """Intervallo di confidenza al 95% dell'EER (%) con il bootstrap, ricampionando dentro ogni classe."""
    label, punteggi = np.asarray(label), np.asarray(punteggi, dtype=float)
    rng = np.random.default_rng(seed)
    i_bona, i_spoof = np.flatnonzero(label == 0), np.flatnonzero(label == 1)
    valori = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(i_bona, len(i_bona)), rng.choice(i_spoof, len(i_spoof))])
        valori.append(100 * eer(label[idx], punteggi[idx]))
    return np.percentile(valori, [2.5, 97.5])
