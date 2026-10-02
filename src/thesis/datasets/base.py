"""
Interfaccia comune ai dataset di anti-spoofing.

Ogni dataset concreto implementa:
  - prepare(split):        download + estrazione, idempotente;
  - load_protocol(split):  DataFrame con almeno [file_name, label, audio_path],
                           dove label ∈ {"bonafide", "spoof"} e audio_path è
                           relativo alla radice del dataset (POSIX).

La selezione di un sottoinsieme bilanciato e la gestione dei manifest sono
invece comuni e vivono qui.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from thesis.config import DATA_ROOT, DEFAULT_SEED, MANIFESTS_DIR

logger = logging.getLogger(__name__)

LABELS = ("bonafide", "spoof")
LABEL_TO_INT = {"bonafide": 0, "spoof": 1}
MANIFEST_COLUMNS = ["file_name", "label", "audio_path", "duration"]


def audio_duration(path: Path) -> float:
    """Durata in secondi letta dall'header (nessuna decodifica). 0.0 se illeggibile."""
    try:
        return sf.info(str(path)).duration
    except Exception:
        return 0.0


class BaseDataset(ABC):
    name: str  # identificativo breve, usato per cartelle e nomi dei manifest

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else DATA_ROOT / self.name

    # ------------------------------------------------------------------
    # Da implementare nei dataset concreti
    # ------------------------------------------------------------------

    @abstractmethod
    def prepare(self, split: str) -> None:
        """Rende disponibili su disco protocollo e audio dello split."""

    @abstractmethod
    def load_protocol(self, split: str) -> pd.DataFrame:
        """Protocollo dello split: [file_name, label, audio_path, ...metadati]."""

    # ------------------------------------------------------------------
    # Percorsi
    # ------------------------------------------------------------------

    def abs_path(self, rel_path: str) -> Path:
        return self.root / rel_path

    def manifest_path(
        self, split: str, n_per_class: int, seed: int,
        min_duration: float, max_duration: float,
    ) -> Path:
        return MANIFESTS_DIR / (
            f"{self.name}_{split}_n{n_per_class}_seed{seed}"
            f"_dur{min_duration:g}-{max_duration:g}.csv"
        )

    # ------------------------------------------------------------------
    # Sottoinsieme bilanciato + manifest
    # ------------------------------------------------------------------

    def balanced_subset(
        self,
        split: str,
        n_per_class: int,
        min_duration: float,
        max_duration: float,
        seed: int = DEFAULT_SEED,
    ) -> pd.DataFrame:
        """
        Restituisce n_per_class campioni per classe con durata in
        [min_duration, max_duration].

        Se il manifest per questa configurazione esiste lo ricarica (e verifica
        che tutti i file siano su disco); altrimenti lo genera e lo salva.
        """
        path = self.manifest_path(split, n_per_class, seed, min_duration, max_duration)
        if path.exists():
            return self.load_manifest(path)

        subset = self._select_balanced(split, n_per_class, min_duration, max_duration, seed)
        self.save_manifest(subset, path, params={
            "dataset": self.name,
            "split": split,
            "n_per_class": n_per_class,
            "seed": seed,
            "min_duration": min_duration,
            "max_duration": max_duration,
        })
        return subset

    def _select_balanced(
        self, split: str, n_per_class: int,
        min_duration: float, max_duration: float, seed: int,
    ) -> pd.DataFrame:
        df = self.load_protocol(split)
        logger.info("Righe nel protocollo: %d — %s", len(df), df["label"].value_counts().to_dict())

        df = df[df["audio_path"].map(lambda p: self.abs_path(p).exists())]
        logger.info("File audio presenti su disco: %d", len(df))
        if df.empty:
            raise RuntimeError(f"Nessun file audio trovato per lo split '{split}'. Hai eseguito prepare()?")

        # Mescolamento deterministico (stessa procedura del progetto originale,
        # così a parità di seed e di file su disco si ottiene la stessa selezione).
        rng = np.random.default_rng(seed)
        df = df.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

        # Scansione con early stopping: la durata si legge solo finché serve.
        buckets: dict[str, list[dict]] = {label: [] for label in LABELS}
        scanned = 0
        with tqdm(total=n_per_class * len(LABELS), desc="Ricerca campioni validi", unit="campione") as pbar:
            for row in df.itertuples(index=False):
                if all(len(b) >= n_per_class for b in buckets.values()):
                    break
                scanned += 1
                bucket = buckets.get(row.label)
                if bucket is None or len(bucket) >= n_per_class:
                    continue
                duration = audio_duration(self.abs_path(row.audio_path))
                if min_duration <= duration <= max_duration:
                    bucket.append({
                        "file_name": row.file_name,
                        "label": row.label,
                        "audio_path": row.audio_path,
                        "duration": round(duration, 4),
                    })
                    pbar.update(1)
                    pbar.set_postfix({**{k: len(v) for k, v in buckets.items()}, "scansionati": scanned})

        for label, found in buckets.items():
            if len(found) < n_per_class:
                logger.warning(
                    "Classe '%s': richiesti %d campioni, trovati solo %d (su %d scansionati).",
                    label, n_per_class, len(found), scanned,
                )

        return pd.DataFrame(
            [s for label in LABELS for s in buckets[label]],
            columns=MANIFEST_COLUMNS,
        )

    def save_manifest(self, subset: pd.DataFrame, path: Path, params: dict) -> None:
        """
        Prima riga: commento JSON con i parametri; poi CSV con MANIFEST_COLUMNS.
        I percorsi audio sono relativi alla radice del dataset, così il
        manifest resta valido su qualunque macchina.
        """
        params = {
            **params,
            "total_samples": len(subset),
            **{label: int((subset["label"] == label).sum()) for label in LABELS},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write(f"# {json.dumps(params)}\n")
            subset[MANIFEST_COLUMNS].to_csv(f, index=False)
        logger.info("Manifest salvato: %s (%d campioni)", path.name, len(subset))

    def load_manifest(self, path: Path) -> pd.DataFrame:
        """
        Carica un manifest e verifica che tutti i file audio esistano.
        Un manifest caricato solo in parte darebbe un esperimento diverso da
        quello registrato, quindi in quel caso si solleva un errore.
        """
        params = read_manifest_params(path)
        logger.info("Manifest trovato: %s — %s", path.name, params)
        subset = pd.read_csv(path, skiprows=1 if params else 0)

        missing = [p for p in subset["audio_path"] if not self.abs_path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(subset)} file del manifest {path.name} non trovati "
                f"sotto {self.root} (primo: {missing[0]}). "
                "Esegui prepare() sullo split corretto o verifica DATA_ROOT."
            )

        logger.info("Subset caricato: %d campioni — %s", len(subset), subset["label"].value_counts().to_dict())
        return subset


def read_manifest_params(path: Path) -> dict:
    """Parametri JSON salvati nella prima riga del manifest ({} se assenti)."""
    with Path(path).open("r", encoding="utf-8") as f:
        first = f.readline()
    if first.startswith("#"):
        try:
            return json.loads(first[1:].strip())
        except json.JSONDecodeError:
            pass
    return {}
