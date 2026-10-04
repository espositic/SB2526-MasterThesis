"""
Metriche ASVspoof 5: EER, minDCF, actDCF, Cllr (più EER per gruppo e bootstrap).

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


def act_dcf(labels, scores) -> float:
    """
    DCF alla soglia di Bayes, come nel kit di valutazione di ASVspoof 5:
    richiede punteggi interpretabili come log-likelihood ratio (misura la calibrazione).
    """
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    threshold = -np.log(C_MISS * (1 - P_SPOOF) / (C_FA * P_SPOOF))
    p_miss = np.mean(scores[labels == 0] < threshold)
    p_fa = np.mean(scores[labels == 1] >= threshold)
    dcf = C_MISS * (1 - P_SPOOF) * p_miss + C_FA * P_SPOOF * p_fa
    return float(dcf / min(C_MISS * (1 - P_SPOOF), C_FA * P_SPOOF))


def cllr(labels, scores) -> float:
    """Costo log-likelihood ratio in bit (0 = perfetto, 1 = non informativo)."""
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    bona, spoof = scores[labels == 0], scores[labels == 1]
    return float(0.5 * (np.mean(np.logaddexp(0, -bona)) + np.mean(np.logaddexp(0, spoof))) / np.log(2))


def all_metrics(labels, scores) -> dict:
    return {
        "eer": eer(labels, scores),
        "min_dcf": min_dcf(labels, scores),
        "act_dcf": act_dcf(labels, scores),
        "cllr": cllr(labels, scores),
        "n": int(len(labels)),
    }


def eer_by_group(labels, scores, groups) -> dict:
    """
    EER per gruppo (es. attacco o codec): i bonafide sono gli stessi per tutti
    i gruppi, gli spoof sono solo quelli del gruppo.
    """
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    bona = labels == 0
    out = {}
    for g in sorted(set(groups[~bona])):
        keep = bona | (groups == g)
        out[str(g)] = eer(labels[keep], scores[keep])
    return out


def fit_fusion(labels, scores: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """
    Fusione lineare di più sistemi: llr = w·s + b, stimata con regressione
    logistica a classi bilanciate (bonafide positivo). scores: (n, k), un
    punteggio per sistema. Ritorna [w_1..w_k, b]. Con k = 1 è la calibrazione.
    """
    labels = np.asarray(labels).ravel()
    s = np.asarray(scores, dtype=float).reshape(len(labels), -1)
    t = (labels == 0).astype(float)
    w = np.where(t == 1, 0.5 / t.mean(), 0.5 / (1 - t.mean()))
    X = np.hstack([s, np.ones((len(s), 1))])
    theta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-(X @ theta)))
        step = np.linalg.solve((X * (w * p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1]),
                               X.T @ (w * (p - t)))
        theta -= step
        if np.abs(step).max() < 1e-10:
            break
    return theta


def apply_fusion(scores: np.ndarray, theta: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1, len(theta) - 1)
    return s @ theta[:-1] + theta[-1]


def fit_calibration(labels, scores, n_iter: int = 50) -> tuple[float, float]:
    """
    Calibrazione lineare dei punteggi: llr = a·score + b, stimata con regressione
    logistica (bonafide = classe positiva) e classi pesate in modo bilanciato.
    Con pesi bilanciati il logit stimato corrisponde a un prior 50/50, quindi
    l'uscita si può usare come log-likelihood ratio (alto = bonafide).
    Stima con Newton-Raphson: due soli parametri, nessuna dipendenza esterna.
    """
    labels = np.asarray(labels).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    t = (labels == 0).astype(float)                       # 1 = bonafide
    w = np.where(t == 1, 0.5 / t.mean(), 0.5 / (1 - t.mean()))
    X = np.stack([s, np.ones_like(s)], axis=1)
    theta = np.zeros(2)
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-(X @ theta)))
        grad = X.T @ (w * (p - t))
        hess = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        theta -= step
        if np.abs(step).max() < 1e-10:
            break
    a, b = theta
    assert a > 0, "calibrazione con pendenza non positiva: i punteggi del dev non separano le classi"
    return float(a), float(b)


def apply_calibration(scores, a: float, b: float) -> np.ndarray:
    return a * np.asarray(scores, dtype=float) + b


def bootstrap_eer(labels, scores, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Intervallo di confidenza al 95% dell'EER (ricampionamento stratificato per classe)."""
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    idx_b, idx_s = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(idx_b, len(idx_b)), rng.choice(idx_s, len(idx_s))])
        values.append(eer(labels[idx], scores[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)
