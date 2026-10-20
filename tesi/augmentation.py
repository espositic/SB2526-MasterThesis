# Data augmentation sulla forma d'onda, applicata solo alle clip di train.
#
# Le quattro condizioni da confrontare:
#   "nessuna"   nessuna augmentation
#   "rawboost"  RawBoost, algoritmo 5 (Tak et al., ICASSP 2022), su ogni clip
#   "rumore"    rumore, parlato o musica di MUSAN oppure riverbero (RIR), su metà delle clip
#   "codec"     compressione MP3, AAC oppure Opus con ffmpeg, su metà delle clip
# Per TFPARN c'è anche SpecAugment, che però lavora sullo spettrogramma ed è dentro il modello.
#
# Ogni augmentation è una classe che si chiama come una funzione: augmentation(wave, rng),
# dove rng è un generatore numpy (np.random.default_rng). Il Dataset ne passa uno diverso
# per ogni clip.

import os
import subprocess

import numpy as np
import soundfile as sf
from scipy import signal

from tesi import percorsi
from tesi.dati import FREQ, taglia


def crea_augmentation(nome):
    """Restituisce l'augmentation da passare ad AudioDataset (None = nessuna)."""
    if nome in ("nessuna", "specaugment"):   # SpecAugment è dentro TFPARN, non qui
        return None
    if nome == "rawboost":
        return RawBoost()
    if nome == "rumore":
        return RumoreRiverbero()
    if nome == "codec":
        return Codec()
    raise ValueError(f"augmentation sconosciuta: {nome}")


# ---------------------------------------------------------------------------
# RawBoost
# Port dell'implementazione originale (github.com/TakHemlata/RawBoost-antispoofing,
# licenza MIT) nella versione di XLSR-MamBo, con gli stessi parametri.
# Algoritmo 5 = rumore convolutivo lineare e non lineare (LnL) + rumore impulsivo (ISD).
# ---------------------------------------------------------------------------

class RawBoost:
    # parametri di default dell'originale
    n_bande = 5
    min_f, max_f = 20, 8000
    min_bw, max_bw = 100, 1000
    min_coeff, max_coeff = 10, 100
    min_g, max_g = 0, 0
    min_bias, max_bias = 5, 20
    n_f = 5                  # ordine delle non linearità
    p = 10                   # percentuale massima di campioni con rumore impulsivo
    g_sd = 2

    def __call__(self, wave, rng):
        return self.trasforma(wave, rng)

    def trasforma(self, wave, rng):
        x = wave.astype(np.float64)
        x = self.lnl(x, rng)
        x = self.isd(x, rng)
        return x.astype(np.float32)

    def filtro_notch(self, min_g, max_g, rng):
        """Cascata di n_bande filtri passa-banda con frequenza e larghezza casuali."""
        b = 1
        for _ in range(self.n_bande):
            fc = rng.uniform(self.min_f, self.max_f)
            bw = rng.uniform(self.min_bw, self.max_bw)
            n = int(rng.uniform(self.min_coeff, self.max_coeff))
            if n % 2 == 0:
                n += 1
            f1, f2 = max(fc - bw / 2, 1e-3), min(fc + bw / 2, FREQ / 2 - 1e-3)
            b = np.convolve(signal.firwin(n, [f1, f2], window="hamming", fs=FREQ), b)
        # come np.random.uniform dell'originale, che funziona anche con min_g > max_g
        g = min_g + (max_g - min_g) * rng.random()
        _, h = signal.freqz(b, 1, fs=FREQ)
        return 10 ** (g / 20) * b / np.max(np.abs(h))

    def lnl(self, x, rng):
        """Rumore convolutivo lineare e non lineare."""
        y = np.zeros_like(x)
        min_g, max_g = self.min_g, self.max_g
        for i in range(self.n_f):
            if i == 1:
                min_g, max_g = min_g - self.min_bias, max_g - self.max_bias
            b = self.filtro_notch(min_g, max_g, rng)
            y = y + applica_fir(np.power(x, i + 1), b)[: len(x)]
        return normalizza(y - y.mean())

    def isd(self, x, rng):
        """Rumore impulsivo dipendente dal segnale su una parte dei campioni."""
        y = x.copy()
        n = int(len(x) * rng.uniform(0, self.p) / 100)
        idx = rng.permutation(len(x))[:n]
        f_r = (2 * rng.random(n) - 1) * (2 * rng.random(n) - 1)
        y[idx] = x[idx] + self.g_sd * x[idx] * f_r
        return normalizza(y)


def applica_fir(x, b):
    n = b.shape[0] + 1
    y = signal.lfilter(b, 1, np.pad(x, (0, n)))
    return y[n // 2: y.shape[0] - n // 2]


def normalizza(x):
    """Riporta il picco a 1, solo se lo supera."""
    picco = np.max(np.abs(x))
    return x / picco if picco > 1 else x


# ---------------------------------------------------------------------------
# Rumore e riverbero: MUSAN (Snyder et al., 2015) e RIR simulate (Ko et al., 2017),
# come nel codice di Nes2Net (musan.py, rir.py) e in ECAPA-TDNN.
# ---------------------------------------------------------------------------

class RumoreRiverbero:
    # categoria di MUSAN: (SNR minimo, SNR massimo, numero di file da sommare)
    CATEGORIE = {
        "noise": (0, 15, (1, 1)),
        "speech": (13, 20, (3, 8)),   # più persone che parlano in sottofondo
        "music": (5, 25, (1, 1)),
    }

    def __init__(self, prob=0.5):
        self.prob = prob
        self.musan = {c: sorted((percorsi.MUSAN / c).rglob("*.wav")) for c in self.CATEGORIE}
        self.rir = sorted((percorsi.RIR / "simulated_rirs").rglob("*.wav"))
        for nome, file in list(self.musan.items()) + [("RIR", self.rir)]:
            if len(file) == 0:
                raise FileNotFoundError(f"nessun file per {nome}: hai scaricato MUSAN e RIRS_NOISES in data/?")

    def __call__(self, wave, rng):
        if rng.random() < self.prob:
            return self.trasforma(wave, rng)
        return wave

    def trasforma(self, wave, rng):
        # metà delle volte rumore di MUSAN, metà riverbero
        if rng.random() < 0.5:
            return self.aggiungi_musan(wave, rng, categoria=rng.choice(list(self.CATEGORIE)))
        return self.aggiungi_riverbero(wave, rng)

    def aggiungi_musan(self, wave, rng, categoria):
        snr_min, snr_max, (n_min, n_max) = self.CATEGORIE[categoria]
        n_file = rng.integers(n_min, n_max + 1)
        scelti = rng.choice(len(self.musan[categoria]), size=n_file, replace=False)
        rumori = [leggi_pezzo(self.musan[categoria][i], len(wave), rng) for i in scelti]
        rumore = np.mean(rumori, axis=0)
        snr = rng.uniform(snr_min, snr_max)
        # scala il rumore per avere il rapporto segnale/rumore scelto (in dB)
        p = decibel(wave) - decibel(rumore) - snr
        return (wave + np.sqrt(10 ** (p / 10)) * rumore).astype(np.float32)

    def aggiungi_riverbero(self, wave, rng):
        rir, _ = sf.read(self.rir[rng.integers(len(self.rir))], dtype="float32")
        if rir.ndim > 1:
            rir = rir[:, 0]
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.fftconvolve(wave, rir)[: len(wave)].astype(np.float32)


def leggi_pezzo(file, n, rng):
    """Un pezzo casuale di n campioni da un file di rumore (ripetuto se il file è più corto)."""
    totale = sf.info(file).frames
    if totale <= n:
        rumore, _ = sf.read(file, dtype="float32")
        return taglia(rumore, n)
    inizio = rng.integers(0, totale - n)
    rumore, _ = sf.read(file, start=inizio, stop=inizio + n, dtype="float32")
    return rumore


def decibel(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-4)


# ---------------------------------------------------------------------------
# Codec: la clip viene compressa e decompressa con ffmpeg, a un bitrate a caso.
# Bitrate bassi = qualità peggiore, come nelle chiamate al telefono o sui social.
# ---------------------------------------------------------------------------

class Codec:
    # codec: (encoder di ffmpeg, formato del file compresso, bitrate in kbit/s)
    CODEC = {
        "mp3": ("libmp3lame", "mp3", [16, 24, 32, 48, 64, 96, 128]),
        "aac": ("aac", "adts", [16, 24, 32, 48, 64]),
        "opus": ("libopus", "ogg", [8, 12, 16, 24, 32]),
    }

    def __init__(self, prob=0.5):
        self.prob = prob
        self.ffmpeg = trova_ffmpeg()

    def __call__(self, wave, rng):
        if rng.random() < self.prob:
            return self.trasforma(wave, rng)
        return wave

    def trasforma(self, wave, rng):
        codec = rng.choice(list(self.CODEC))
        bitrate = rng.choice(self.CODEC[codec][2])
        return comprimi(wave, codec, bitrate, self.ffmpeg)


def trova_ffmpeg():
    """ffmpeg del pacchetto imageio-ffmpeg (pip install imageio-ffmpeg), altrimenti quello di sistema."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import shutil
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("ffmpeg non trovato: pip install imageio-ffmpeg")
        return "ffmpeg"


def comprimi(wave, codec, bitrate, ffmpeg="ffmpeg"):
    """Comprime e decomprime una clip passando i dati a ffmpeg in memoria, senza file temporanei."""
    encoder, formato, _ = Codec.CODEC[codec]
    opzioni = {"capture_output": True, "check": True}
    if os.name == "nt":
        opzioni["creationflags"] = subprocess.CREATE_NO_WINDOW   # niente finestre nere su Windows
    pcm = ["-f", "f32le", "-ar", str(FREQ), "-ac", "1"]
    compresso = subprocess.run(
        [ffmpeg, "-loglevel", "error", *pcm, "-i", "pipe:0", "-c:a", encoder, "-b:a", f"{bitrate}k", "-f", formato, "pipe:1"],
        input=wave.astype(np.float32).tobytes(), **opzioni).stdout
    decompresso = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-i", "pipe:0", *pcm, "pipe:1"],
        input=compresso, **opzioni).stdout
    y = np.frombuffer(decompresso, dtype=np.float32)
    # il codec aggiunge qualche campione all'inizio o alla fine: si torna alla lunghezza originale
    if len(y) < len(wave):
        y = np.pad(y, (0, len(wave) - len(y)))
    return y[: len(wave)].copy()
