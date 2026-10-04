"""
RawBoost: data augmentation sulla forma d'onda (Tak et al., "RawBoost: A Raw Data
Boosting and Augmentation Method applied to Automatic Speaker Verification
Anti-Spoofing", ICASSP 2022). Port dell'implementazione originale
(https://github.com/TakHemlata/RawBoost-antispoofing, licenza MIT) nella versione
usata da XLSR-MamBo (MamBo-for-ADD/RawBoost.py, licenza MIT), con gli stessi
parametri di default e lo stesso algoritmo 5 che MamBo usa per ASVspoof LA.

Differenza: un generatore casuale passato esplicitamente invece di np.random
globale, così ogni worker del DataLoader produce rumori diversi e riproducibili.

Algoritmi (numerazione originale):
  1 rumore convolutivo lineare e non lineare (LnL)
  2 rumore impulsivo dipendente dal segnale (ISD)
  3 rumore additivo colorato stazionario (SSI)
  4 = 1+2+3 · 5 = 1+2 · 6 = 1+3 · 7 = 2+3 (in serie) · 8 = 1 || 2 (in parallelo)
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class RawBoostConfig:
    algo: int = 5            # 5 (LnL + ISD) è quello usato per ASVspoof LA
    n_bands: int = 5
    min_f: float = 20
    max_f: float = 8000
    min_bw: float = 100
    max_bw: float = 1000
    min_coeff: int = 10
    max_coeff: int = 100
    min_g: float = 0
    max_g: float = 0
    min_bias_lin_nonlin: float = 5
    max_bias_lin_nonlin: float = 20
    n_f: int = 5             # ordine delle non linearità
    p: float = 10            # % massima di campioni con rumore impulsivo
    g_sd: float = 2
    snr_min: float = 10
    snr_max: float = 40


def _norm(x, always=False):
    peak = np.max(np.abs(x))
    return x / peak if always or peak > 1 else x


def _notch_coeffs(c: RawBoostConfig, min_g, max_g, fs, rng):
    b = 1
    for _ in range(c.n_bands):
        fc = rng.uniform(c.min_f, c.max_f)
        bw = rng.uniform(c.min_bw, c.max_bw)
        n = int(rng.uniform(c.min_coeff, c.max_coeff))
        n += 1 if n % 2 == 0 else 0
        f1, f2 = max(fc - bw / 2, 1e-3), min(fc + bw / 2, fs / 2 - 1e-3)
        b = np.convolve(signal.firwin(n, [f1, f2], window="hamming", fs=fs), b)
    # Come np.random.uniform dell'originale, che accetta anche min > max (succede
    # per la componente non lineare: min_g=-5, max_g=-20); Generator.uniform no.
    g = min_g + (max_g - min_g) * rng.random()
    _, h = signal.freqz(b, 1, fs=fs)
    return 10 ** (g / 20) * b / np.max(np.abs(h))


def _fir(x, b):
    n = b.shape[0] + 1
    y = signal.lfilter(b, 1, np.pad(x, (0, n)))
    return y[n // 2: y.shape[0] - n // 2]


def lnl_convolutive(x, c: RawBoostConfig, fs, rng):
    y = np.zeros_like(x)
    min_g, max_g = c.min_g, c.max_g
    for i in range(c.n_f):
        if i == 1:
            min_g, max_g = min_g - c.min_bias_lin_nonlin, max_g - c.max_bias_lin_nonlin
        y = y + _fir(np.power(x, i + 1), _notch_coeffs(c, min_g, max_g, fs, rng))[: len(x)]
    return _norm(y - y.mean())


def isd_additive(x, c: RawBoostConfig, rng):
    y = x.copy()
    n = int(len(x) * rng.uniform(0, c.p) / 100)
    idx = rng.permutation(len(x))[:n]
    f_r = (2 * rng.random(n) - 1) * (2 * rng.random(n) - 1)
    y[idx] = x[idx] + c.g_sd * x[idx] * f_r
    return _norm(y)


def ssi_additive(x, c: RawBoostConfig, fs, rng):
    noise = _norm(_fir(rng.normal(0, 1, len(x)), _notch_coeffs(c, c.min_g, c.max_g, fs, rng))[: len(x)], always=True)
    snr = rng.uniform(c.snr_min, c.snr_max)
    noise *= np.sqrt(np.dot(x, x) / np.dot(noise, noise)) * 10 ** (-0.05 * snr)
    return x + noise


def rawboost(x: np.ndarray, c: RawBoostConfig, fs: int, rng: np.random.Generator) -> np.ndarray:
    x = x.astype(np.float64)
    lnl = lambda v: lnl_convolutive(v, c, fs, rng)
    isd = lambda v: isd_additive(v, c, rng)
    ssi = lambda v: ssi_additive(v, c, fs, rng)
    chains = {1: [lnl], 2: [isd], 3: [ssi], 4: [lnl, isd, ssi], 5: [lnl, isd], 6: [lnl, ssi], 7: [isd, ssi]}
    if c.algo == 8:
        y = _norm(lnl(x) + isd(x))
    else:
        assert c.algo in chains, f"algoritmo RawBoost sconosciuto: {c.algo}"
        y = x
        for f in chains[c.algo]:
            y = f(y)
    assert y.shape == x.shape and np.isfinite(y).all(), "RawBoost ha prodotto un segnale non valido"
    return y.astype(np.float32)
