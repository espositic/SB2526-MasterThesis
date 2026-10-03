"""
Mel-spettrogrammi (scala dB) salvati come .npy, uno per file audio.

Layout su disco:
    ARTIFACTS_ROOT/features/<dataset>/<config.tag>/<file_name>.npy
    ARTIFACTS_ROOT/features/<dataset>/<config.tag>/index_<subset_name>.csv

Il tag codifica i parametri, quindi configurazioni diverse non si
sovrascrivono a vicenda. Gli spettrogrammi hanno lunghezza variabile
(dipende dalla durata): padding/crop si fanno a valle, nel DataLoader.
"""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from thesis.config import ARTIFACTS_ROOT
from thesis.datasets.base import LABEL_TO_INT, BaseDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 16_000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512

    @property
    def tag(self) -> str:
        return f"mel_sr{self.sample_rate}_m{self.n_mels}_fft{self.n_fft}_hop{self.hop_length}"

    def compute(self, audio_path: Path) -> np.ndarray:
        """Mel-spettrogramma in dB, shape (n_mels, n_frames), float32."""
        y, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length,
        )
        # ref=np.max: normalizzazione per file (0 dB = picco del file).
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def features_dir(dataset: BaseDataset, config: MelConfig) -> Path:
    return ARTIFACTS_ROOT / "features" / dataset.name / config.tag


class MelDataset(Dataset):
    """
    Dataset PyTorch sugli spettrogrammi già estratti (non serve l'audio).
    Ogni spettrogramma viene portato a `n_frames` come per la forma d'onda:
    primi n_frames se è più lungo, ripetuto in loop se è più corto.
    """

    def __init__(self, feat_dir: Path, subset: pd.DataFrame, n_frames: int):
        self.paths = [str(feat_dir / f"{name}.npy") for name in subset["file_name"]]
        missing = [p for p in self.paths if not os.path.exists(p)]
        assert not missing, f"{len(missing)} spettrogrammi mancanti in {feat_dir} (primo: {missing[0]})"
        self.labels = subset["label"].map(LABEL_TO_INT).to_numpy()
        self.n_frames = n_frames

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        mel = np.load(self.paths[i])
        if mel.shape[1] < self.n_frames:
            mel = np.tile(mel, (1, -(-self.n_frames // mel.shape[1])))
        return torch.from_numpy(mel[None, :, :self.n_frames].copy()), int(self.labels[i])


def _process_one(args: tuple[MelConfig, Path, Path]) -> tuple[Optional[int], Optional[str]]:
    """Worker (top-level per essere serializzabile dai processi). Ritorna (n_frames, errore)."""
    config, audio_path, out_path = args
    try:
        if out_path.exists():
            return np.load(out_path, mmap_mode="r").shape[1], None
        mel = config.compute(audio_path)
        tmp = out_path.with_suffix(".tmp.npy")
        np.save(tmp, mel)
        os.replace(tmp, out_path)  # scrittura atomica: niente .npy troncati se si interrompe
        return mel.shape[1], None
    except Exception as exc:
        return None, f"{audio_path.name}: {exc}"


def extract_mel(
    dataset: BaseDataset,
    subset: pd.DataFrame,
    subset_name: str,
    config: MelConfig = MelConfig(),
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calcola (o riusa, se già presenti) gli spettrogrammi del subset in parallelo.

    Args:
        dataset:      dataset da cui risolvere i percorsi audio.
        subset:       DataFrame con [file_name, label, audio_path] (es. un manifest).
        subset_name:  nome dell'indice da scrivere (di solito lo stem del manifest).

    Returns:
        Indice [file_name, label, feature_path, n_frames]; label è 0=bonafide,
        1=spoof; feature_path è relativo alla cartella delle feature.
    """
    out_dir = features_dir(dataset, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    max_workers = max_workers or os.cpu_count() or 4
    jobs = [
        (config, dataset.abs_path(row.audio_path), out_dir / f"{row.file_name}.npy")
        for row in subset.itertuples(index=False)
    ]
    logger.info("Estrazione %s per %d campioni (workers=%d) → %s", config.tag, len(jobs), max_workers, out_dir)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(_process_one, jobs, chunksize=16),
            total=len(jobs), desc="Estrazione spettrogrammi", unit="file",
        ))

    errors = [err for _, err in results if err is not None]
    for err in errors[:10]:
        logger.error("Errore: %s", err)
    if errors:
        logger.error("%d file non elaborati.", len(errors))

    index = pd.DataFrame({
        "file_name": subset["file_name"].values,
        "label": subset["label"].map(LABEL_TO_INT).values,
        "feature_path": [f"{name}.npy" for name in subset["file_name"]],
        "n_frames": [n for n, _ in results],
    })
    index = index[index["n_frames"].notna()].astype({"n_frames": int})
    index.to_csv(out_dir / f"index_{subset_name}.csv", index=False)
    logger.info("Indice salvato: %s (%d righe)", out_dir / f"index_{subset_name}.csv", len(index))
    return index
