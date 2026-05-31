"""
Pipeline di estrazione feature per ASVspoof 5.

Flusso:
  1. Estrae i TAR dei protocolli e dell'audio (se non già fatto).
  2. Seleziona un sottoinsieme bilanciato (bonafide / spoof) con filtro per durata.
     → Se esiste già un manifest per la stessa configurazione, lo ricarica
       direttamente garantendo la riproducibilità dell'esperimento.
  3. Calcola Mel-Spettrogrammi con librosa.
  4. Salva gli array NumPy e aggiorna il CSV delle feature.
"""

import csv
import json
import logging
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Percorsi di progetto
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "Dataset_full" / "ASVspoof5"
EXPERIMENT_DIR = PROJECT_ROOT / "experiment" / "Dataset" / "ASVSpoof5Spectogram"
FEATURES_CSV = PROJECT_ROOT / "experiment" / "Dataset" / "ASVSpoof5_features.csv"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

# Colonne del TSV di protocollo (spazio come separatore)
TSV_COLUMNS = [
    "SPEAKER_ID", "FLAC_FILE_NAME", "SPEAKER_GENDER",
    "CODEC", "CODEC_Q", "CODEC_SEED",
    "ATTACK_TAG", "ATTACK_LABEL", "KEY", "TMP",
]

# Mapping split → (pattern TAR, cartella audio estratta, file TSV)
SPLIT_INFO: dict[str, dict] = {
    "train": {
        "tar_pattern": "flac_T_*.tar",
        "audio_dir": "flac_T",
        "tsv_name": "ASVspoof5.train.tsv",
    },
    "dev": {
        "tar_pattern": "flac_D_*.tar",
        "audio_dir": "flac_D",
        "tsv_name": "ASVspoof5.dev.track_1.tsv",
    },
    "eval": {
        "tar_pattern": "flac_E_*.tar",
        "audio_dir": "flac_E_eval",
        "tsv_name": "ASVspoof5.eval.track_1.tsv",
    },
}

PROTOCOLS_TAR = "ASVspoof5_protocols.tar"


# ---------------------------------------------------------------------------
# Dataclass per un singolo campione
# ---------------------------------------------------------------------------
@dataclass
class AudioSample:
    file_name: str          # es. "T_0000001"
    label: str              # "bonafide" o "spoof"
    audio_path: Path
    feature_path: Optional[Path] = field(default=None)


# ---------------------------------------------------------------------------
# Utilità per l'estrazione dei TAR
# ---------------------------------------------------------------------------

def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """Estrae un singolo archivio TAR in dest_dir."""
    logger.info("Estrazione %s → %s ...", tar_path.name, dest_dir)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=dest_dir)
    logger.info("Estrazione completata: %s", tar_path.name)


def ensure_protocols_extracted(dataset_dir: Path = DATASET_DIR) -> Path:
    """
    Estrae ASVspoof5_protocols.tar se i file TSV non sono ancora presenti.
    Restituisce la cartella in cui i TSV sono stati estratti.
    """
    tsv_check = dataset_dir / "ASVspoof5.train.tsv"
    if tsv_check.exists():
        logger.info("Protocolli già estratti in %s", dataset_dir)
        return dataset_dir

    tar_path = dataset_dir / PROTOCOLS_TAR
    if not tar_path.exists():
        raise FileNotFoundError(
            f"Archivio protocolli non trovato: {tar_path}\n"
            "Assicurati che il download sia stato completato."
        )
    _extract_tar(tar_path, dataset_dir)
    return dataset_dir


def ensure_audio_extracted(split: str, dataset_dir: Path = DATASET_DIR) -> Path:
    """
    Estrae i TAR audio per lo split richiesto se la cartella non esiste ancora.
    Restituisce il percorso della cartella audio.
    """
    info = SPLIT_INFO[split]
    audio_dir = dataset_dir / info["audio_dir"]

    if audio_dir.exists() and any(audio_dir.rglob("*.flac")):
        logger.info("Audio già estratto in %s", audio_dir)
        return audio_dir

    tar_files = sorted(dataset_dir.glob(info["tar_pattern"]))
    if not tar_files:
        raise FileNotFoundError(
            f"Nessun TAR audio trovato con pattern '{info['tar_pattern']}' in {dataset_dir}."
        )

    logger.info(
        "Trovati %d archivi TAR per lo split '%s'. Inizio estrazione...",
        len(tar_files), split,
    )
    for tar_path in tqdm(tar_files, desc=f"Estrazione TAR [{split}]", unit="file"):
        _extract_tar(tar_path, dataset_dir)

    return audio_dir


# ---------------------------------------------------------------------------
# Manifest — riproducibilità degli esperimenti
# ---------------------------------------------------------------------------

def _manifest_path(split: str, n_per_class: int, seed: int) -> Path:
    """Restituisce il percorso canonico del manifest per una data configurazione."""
    return MANIFESTS_DIR / f"subset_{split}_n{n_per_class}_seed{seed}.csv"


def save_manifest(
    samples: "list[AudioSample]",
    split: str,
    n_per_class: int,
    seed: int,
    min_duration: float,
    max_duration: float,
) -> Path:
    """
    Salva la lista di campioni selezionati in un CSV leggero (solo nomi e label).
    Il file viene scritto in manifests/ ed è destinato al commit su git per
    garantire la riproducibilità dell'esperimento.

    Formato colonne: file_name, label, audio_path
    Header aggiuntivo (commento JSON nella prima riga) con i parametri usati.
    """
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(split, n_per_class, seed)

    params = {
        "split": split,
        "n_per_class": n_per_class,
        "seed": seed,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "total_samples": len(samples),
        "bonafide": sum(1 for s in samples if s.label == "bonafide"),
        "spoof": sum(1 for s in samples if s.label == "spoof"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with path.open("w", newline="", encoding="utf-8") as f:
        # Prima riga: metadati dell'esperimento come commento JSON
        f.write(f"# {json.dumps(params)}\n")
        writer = csv.writer(f)
        writer.writerow(["file_name", "label", "audio_path"])
        for s in samples:
            writer.writerow([s.file_name, s.label, str(s.audio_path)])

    logger.info(
        "Manifest salvato: %s (%d campioni)",
        path.relative_to(PROJECT_ROOT), len(samples),
    )
    return path


def load_manifest(
    split: str,
    n_per_class: int,
    seed: int,
) -> "Optional[list[AudioSample]]":
    """
    Tenta di caricare il manifest per la configurazione richiesta.
    Restituisce la lista di AudioSample se il file esiste e tutti i path
    audio sono ancora validi su disco; None altrimenti.
    """
    path = _manifest_path(split, n_per_class, seed)
    if not path.exists():
        return None

    logger.info("Manifest trovato: %s — carico campioni pre-selezionati...", path.name)
    samples: list[AudioSample] = []
    missing: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):          # riga metadati JSON
                try:
                    params = json.loads(line[1:].strip())
                    logger.info("Parametri manifest: %s", params)
                except json.JSONDecodeError:
                    pass
                continue
            break                             # fine intestazione

        reader = csv.DictReader(f, fieldnames=["file_name", "label", "audio_path"])
        next(reader, None)                   # salta riga header CSV
        for row in reader:
            audio_path = Path(row["audio_path"])
            if not audio_path.exists():
                missing.append(row["file_name"])
                continue
            samples.append(AudioSample(
                file_name=row["file_name"],
                label=row["label"],
                audio_path=audio_path,
            ))

    if missing:
        logger.warning(
            "%d file del manifest non trovati su disco — "
            "potrebbe essere necessario ri-estrarre i TAR audio. "
            "Primo assente: %s",
            len(missing), missing[0],
        )

    if not samples:
        logger.warning("Manifest vuoto o tutti i file mancanti. Rigenero il subset.")
        return None

    logger.info(
        "Subset caricato dal manifest: %d campioni (%d bonafide, %d spoof).",
        len(samples),
        sum(1 for s in samples if s.label == "bonafide"),
        sum(1 for s in samples if s.label == "spoof"),
    )
    return samples


# ---------------------------------------------------------------------------
# Subsetting bilanciato
# ---------------------------------------------------------------------------

def _get_duration_seconds(audio_path: Path) -> float:
    """Restituisce la durata in secondi usando soundfile (veloce, no decodifica)."""
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        return 0.0


def select_balanced_subset(
    split: str = "train",
    n_per_class: int = 5_000,
    min_duration: float = 1.0,
    max_duration: float = 10.0,
    dataset_dir: Path = DATASET_DIR,
    seed: int = 42,
) -> list[AudioSample]:
    """
    Legge il TSV di protocollo, filtra per durata e restituisce un sottoinsieme
    bilanciato di n_per_class campioni per classe (bonafide e spoof).

    Se esiste già un manifest per la stessa configurazione (split, n_per_class,
    seed) lo carica direttamente, saltando la scansione delle durate e
    garantendo la riproducibilità dell'esperimento.

    Args:
        split:          "train", "dev" o "eval".
        n_per_class:    Numero di campioni per classe.
        min_duration:   Durata minima in secondi.
        max_duration:   Durata massima in secondi.
        dataset_dir:    Radice del dataset.
        seed:           Seed per la riproducibilità del campionamento.

    Returns:
        Lista di AudioSample con percorso audio e label.
    """
    # --- Tentativo di caricamento dal manifest (percorso veloce) ----------
    cached = load_manifest(split, n_per_class, seed)
    if cached is not None:
        return cached
    # ----------------------------------------------------------------------

    tsv_name = SPLIT_INFO[split]["tsv_name"]
    tsv_path = dataset_dir / tsv_name

    if not tsv_path.exists():
        raise FileNotFoundError(
            f"File di protocollo non trovato: {tsv_path}\n"
            "Chiama prima ensure_protocols_extracted()."
        )

    logger.info("Lettura metadati da %s ...", tsv_path.name)
    # NOTA: sep=r"\s+" (engine="python") collassa qualsiasi sequenza di
    # whitespace in un unico delimitatore, evitando colonne fantasma che
    # si creano con sep=" " quando nel file ci sono doppie spaziature o
    # spazi di allineamento — il che causerebbe lo slittamento del mapping
    # e la lettura errata della colonna KEY.
    df = pd.read_csv(
        tsv_path,
        sep=r"\s+",
        engine="python",
        names=TSV_COLUMNS,
        header=None,
    )
    logger.info("Righe totali nel TSV: %d", len(df))

    # Diagnostica: mostra i valori unici nella colonna KEY per verificare
    # che il parsing sia corretto prima di proseguire.
    key_counts = df["KEY"].value_counts().to_dict()
    logger.info("Valori unici in colonna KEY → %s", key_counts)

    audio_dir = dataset_dir / SPLIT_INFO[split]["audio_dir"]

    # Costruisci i percorsi assoluti
    df["audio_path"] = df["FLAC_FILE_NAME"].apply(
        lambda name: audio_dir / f"{name}.flac"
    )
    # Tieni solo i file esistenti
    df = df[df["audio_path"].apply(lambda p: p.exists())]
    logger.info("File audio presenti su disco: %d", len(df))

    if df.empty:
        raise RuntimeError(
            "Nessun file audio trovato. "
            "Assicurati che i TAR audio siano stati estratti con ensure_audio_extracted()."
        )

    # Mescola casualmente per garantire un campionamento uniforme prima dello scan
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    # Scansione riga per riga con early stopping:
    # ci fermiamo non appena entrambe le classi raggiungono n_per_class campioni.
    logger.info(
        "Ricerca dei primi %d campioni per classe nel filtro durata [%.1f s – %.1f s] ...",
        n_per_class, min_duration, max_duration,
    )

    buckets: dict[str, list[AudioSample]] = {"bonafide": [], "spoof": []}
    target = n_per_class * 2          # totale campioni cercati
    scanned = 0

    with tqdm(
        total=target,
        desc="Ricerca campioni validi",
        unit="campione",
        dynamic_ncols=True,
    ) as pbar:
        for _, row in df.iterrows():
            # Stop anticipato se entrambi i bucket sono pieni
            if all(len(v) >= n_per_class for v in buckets.values()):
                break

            label: str = row["KEY"]
            if label not in buckets or len(buckets[label]) >= n_per_class:
                scanned += 1
                continue

            audio_path: Path = row["audio_path"]
            duration = _get_duration_seconds(audio_path)
            scanned += 1

            if min_duration <= duration <= max_duration:
                buckets[label].append(AudioSample(
                    file_name=row["FLAC_FILE_NAME"],
                    label=label,
                    audio_path=audio_path,
                ))
                pbar.update(1)
                pbar.set_postfix(
                    bonafide=len(buckets["bonafide"]),
                    spoof=len(buckets["spoof"]),
                    scansionati=scanned,
                )

    # Avviso se una classe ha meno campioni del richiesto
    for label, found in buckets.items():
        if len(found) < n_per_class:
            logger.warning(
                "Classe '%s': richiesti %d campioni, trovati solo %d "
                "(su %d file scansionati).",
                label, n_per_class, len(found), scanned,
            )

    samples: list[AudioSample] = buckets["bonafide"] + buckets["spoof"]
    logger.info(
        "Subset finale: %d campioni (%d bonafide, %d spoof) — %d file scansionati.",
        len(samples),
        len(buckets["bonafide"]),
        len(buckets["spoof"]),
        scanned,
    )

    # Salva il manifest per garantire la riproducibilità dei run futuri
    save_manifest(
        samples=samples,
        split=split,
        n_per_class=n_per_class,
        seed=seed,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    return samples


# ---------------------------------------------------------------------------
# Estrattore di Feature
# ---------------------------------------------------------------------------

class AudioFeatureExtractor:
    """
    Calcola Mel-Spettrogrammi da file .flac e li salva come array NumPy.

    Args:
        output_dir:     Cartella di destinazione per i file .npy.
        sample_rate:    Frequenza di campionamento target (default 16000 Hz).
        n_mels:         Numero di bande Mel (default 128).
        n_fft:          Dimensione della FFT (default 1024).
        hop_length:     Passo tra finestre (default 512).
        max_workers:    Thread paralleli per l'estrazione (default: CPU count).
    """

    def __init__(
        self,
        output_dir: Path = EXPERIMENT_DIR,
        sample_rate: int = 16_000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        max_workers: Optional[int] = None,
    ) -> None:
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_workers = max_workers or os.cpu_count() or 4
        self.output_dir.mkdir(parents=True, exist_ok=True)
        FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Metodi interni
    # ------------------------------------------------------------------

    def _compute_mel_spectrogram(self, audio_path: Path) -> np.ndarray:
        """Carica un file audio e calcola il Mel-Spettrogramma (dB scale)."""
        y, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float32)

    def _process_sample(self, sample: AudioSample) -> Optional[dict]:
        """
        Elabora un singolo AudioSample: calcola lo spettrogramma e lo salva.
        Restituisce un dizionario con i metadati o None in caso di errore.
        """
        out_path = self.output_dir / f"{sample.file_name}.npy"

        # Skip se già elaborato
        if out_path.exists():
            return {
                "file_name": sample.file_name,
                "label": 1 if sample.label == "spoof" else 0,
                "feature_path": str(out_path),
            }

        try:
            mel = self._compute_mel_spectrogram(sample.audio_path)
            np.save(str(out_path), mel)
            return {
                "file_name": sample.file_name,
                "label": 1 if sample.label == "spoof" else 0,
                "feature_path": str(out_path),
            }
        except Exception as exc:
            logger.error("Errore su %s: %s", sample.file_name, exc)
            return None

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def extract(self, samples: list[AudioSample]) -> pd.DataFrame:
        """
        Estrae le feature da una lista di AudioSample in parallelo.

        Args:
            samples: Lista prodotta da select_balanced_subset().

        Returns:
            DataFrame con colonne [file_name, label, feature_path].
        """
        logger.info(
            "Inizio estrazione feature per %d campioni "
            "(workers=%d, n_mels=%d, n_fft=%d, hop=%d) ...",
            len(samples), self.max_workers, self.n_mels, self.n_fft, self.hop_length,
        )

        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_sample, s): s for s in samples}
            with tqdm(total=len(futures), desc="Estrazione spettrogrammi", unit="file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    pbar.update(1)

        df = pd.DataFrame(results, columns=["file_name", "label", "feature_path"])
        self._update_csv(df)
        logger.info(
            "Estrazione completata: %d/%d campioni salvati in %s",
            len(df), len(samples), self.output_dir,
        )
        return df

    def _update_csv(self, new_rows: pd.DataFrame) -> None:
        """Aggiunge (o crea) le righe nel CSV delle feature, evitando duplicati."""
        if FEATURES_CSV.exists():
            existing = pd.read_csv(FEATURES_CSV)
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=["file_name"], keep="last")
        else:
            combined = new_rows

        combined.to_csv(FEATURES_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logger.info("CSV aggiornato: %s (%d righe totali)", FEATURES_CSV, len(combined))
