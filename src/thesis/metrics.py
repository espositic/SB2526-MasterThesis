"""
Metriche ASVspoof 5: EER e minDCF.

Convenzione: label 0 = bonafide, 1 = spoof; punteggio alto = bonafide.
"""

import numpy as np

# Parametri DCF di ASVspoof 5
P_SPOOF, C_MISS, C_FA = 0.05, 1.0, 10.0


def _rates(labels, scores):
    """FRR e FAR per ogni soglia possibile (si accetta come bonafide se score > soglia)."""
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    is_bonafide = labels[np.argsort(scores, kind="mergesort")] == 0
    n_bonafide, n_spoof = is_bonafide.sum(), (~is_bonafide).sum()
    frr = np.concatenate([[0], np.cumsum(is_bonafide) / n_bonafide])
    far = np.concatenate([[1], 1 - np.cumsum(~is_bonafide) / n_spoof])
    return frr, far


def eer(labels, scores) -> float:
    frr, far = _rates(labels, scores)
    i = np.argmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2)


def min_dcf(labels, scores) -> float:
    frr, far = _rates(labels, scores)
    dcf = C_MISS * (1 - P_SPOOF) * frr + C_FA * P_SPOOF * far
    return float(dcf.min() / min(C_MISS * (1 - P_SPOOF), C_FA * P_SPOOF))
