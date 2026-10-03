"""
Dataset PyTorch su forma d'onda grezza, a partire da un manifest.

Ogni clip viene portata a lunghezza fissa come in XLSR-MamBo: se è più lunga
si tengono i primi `n_samples` campioni, se è più corta si ripete in loop.
Label: 0 = bonafide, 1 = spoof.
"""

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from thesis.datasets.base import LABEL_TO_INT, BaseDataset

SAMPLE_RATE = 16_000


def fix_length(wave: np.ndarray, n_samples: int) -> np.ndarray:
    assert len(wave) > 0, "clip audio vuota"
    if len(wave) >= n_samples:
        return wave[:n_samples]
    return np.tile(wave, -(-n_samples // len(wave)))[:n_samples]


class WaveformDataset(Dataset):
    def __init__(self, ds: BaseDataset, subset: pd.DataFrame, n_samples: int):
        self.paths = [str(ds.abs_path(p)) for p in subset["audio_path"]]
        self.labels = subset["label"].map(LABEL_TO_INT).to_numpy()
        assert not np.isnan(self.labels.astype(float)).any(), "label sconosciute nel manifest"
        self.n_samples = n_samples

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        wave, sr = sf.read(self.paths[i], dtype="float32")
        assert sr == SAMPLE_RATE, f"{self.paths[i]}: frequenza {sr} Hz, attesa {SAMPLE_RATE}"
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        return torch.from_numpy(fix_length(wave, self.n_samples)), int(self.labels[i])
