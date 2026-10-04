"""
Dataset PyTorch su forma d'onda grezza, a partire da un manifest.

Ogni clip viene portata a lunghezza fissa come in XLSR-MamBo: se è più lunga
si tengono i primi `n_samples` campioni, se è più corta si ripete in loop.
Label: 0 = bonafide, 1 = spoof.
"""

from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from thesis.augment import RawBoostConfig, rawboost
from thesis.datasets.base import LABEL_TO_INT, BaseDataset

SAMPLE_RATE = 16_000


def fix_length(wave: np.ndarray, n_samples: int) -> np.ndarray:
    assert len(wave) > 0, "clip audio vuota"
    if len(wave) >= n_samples:
        return wave[:n_samples]
    return np.tile(wave, -(-n_samples // len(wave)))[:n_samples]


class WaveformDataset(Dataset):
    """augment: config RawBoost da applicare (solo per il train), None = nessuna augmentation."""

    def __init__(self, ds: BaseDataset, subset: pd.DataFrame, n_samples: int,
                 augment: Optional[RawBoostConfig] = None):
        self.paths = [str(ds.abs_path(p)) for p in subset["audio_path"]]
        self.labels = subset["label"].map(LABEL_TO_INT).to_numpy()
        assert not np.isnan(self.labels.astype(float)).any(), "label sconosciute nel manifest"
        self.n_samples = n_samples
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        wave, sr = sf.read(self.paths[i], dtype="float32")
        assert sr == SAMPLE_RATE, f"{self.paths[i]}: frequenza {sr} Hz, attesa {SAMPLE_RATE}"
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if self.augment is not None:
            # seme preso da torch: diverso per ogni worker e riproducibile con il seed del run
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            wave = rawboost(wave, self.augment, SAMPLE_RATE, rng)
        return torch.from_numpy(fix_length(wave, self.n_samples)), int(self.labels[i])
