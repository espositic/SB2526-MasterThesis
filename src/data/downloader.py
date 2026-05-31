import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "Dataset_full" / "ASVspoof5"


class ASVSpoofDownloader:
    def __init__(self, local_dir: Path = DATASET_DIR) -> None:
        self.local_dir = local_dir
        self.repo_id = "jungjee/asvspoof5"

    def download(self) -> None:
        logger.info("Avvio download del dataset ASVspoof 5 da Hugging Face...")
        logger.info("Destinazione: %s", self.local_dir)
        try:
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=str(self.local_dir),
            )
            logger.info("Download completato.")
        except Exception as e:
            logger.error("Errore durante il download: %s", e)
            sys.exit(1)

    def verify_dataset_integrity(self) -> bool:
        logger.info("Verifica dataset in corso...")

        if not self.local_dir.exists():
            logger.warning("Cartella dataset non trovata: %s", self.local_dir)
            return False

        metadata_files = list(self.local_dir.rglob("*.txt")) + list(self.local_dir.rglob("*.csv"))
        if not metadata_files:
            logger.warning("Nessun file di metadati (.txt/.csv) trovato.")
            return False
        logger.info("Trovati %d file di metadati.", len(metadata_files))

        audio_files = list(self.local_dir.rglob("*.flac")) + list(self.local_dir.rglob("*.wav"))
        count = len(audio_files)
        if count == 0:
            logger.warning("Nessun file audio trovato. Il dataset potrebbe essere incompleto.")
            return False

        logger.info("Trovati %d file audio, dataset apparentemente integro.", count)
        return True

    def ensure_dataset_ready(self) -> None:
        if not self.verify_dataset_integrity():
            logger.info("Dataset assente o incompleto. Avvio del download...")
            self.download()
            self.verify_dataset_integrity()
