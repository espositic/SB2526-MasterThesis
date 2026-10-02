"""
ASVspoof 5 (Hugging Face: jungjee/asvspoof5).

Il repository HF contiene i protocolli in ASVspoof5_protocols.tar e l'audio
in più TAR per split (flac_T_*.tar, flac_D_*.tar, flac_E_*.tar).
prepare(split) scarica solo i TAR dello split richiesto, uno alla volta, e
li estrae nella radice del dataset.
"""

import logging
import tarfile
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, get_token, hf_hub_download
from tqdm import tqdm  # barre testuali: i widget di tqdm.auto non si vedono in VS Code

from thesis.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

HF_REPO_ID = "jungjee/asvspoof5"
PROTOCOLS_TAR = "ASVspoof5_protocols.tar"

# Colonne del TSV di protocollo (separatore: spazi)
TSV_COLUMNS = [
    "SPEAKER_ID", "FLAC_FILE_NAME", "SPEAKER_GENDER",
    "CODEC", "CODEC_Q", "CODEC_SEED",
    "ATTACK_TAG", "ATTACK_LABEL", "KEY", "TMP",
]

SPLIT_INFO: dict[str, dict] = {
    "train": {"tar_pattern": "flac_T_*.tar", "audio_dir": "flac_T", "tsv_name": "ASVspoof5.train.tsv"},
    "dev": {"tar_pattern": "flac_D_*.tar", "audio_dir": "flac_D", "tsv_name": "ASVspoof5.dev.track_1.tsv"},
    "eval": {"tar_pattern": "flac_E_*.tar", "audio_dir": "flac_E_eval", "tsv_name": "ASVspoof5.eval.track_1.tsv"},
}


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """
    Estrae un TAR e lascia un marker accanto: così un'estrazione interrotta
    viene ripresa invece di essere considerata completa.
    """
    marker = tar_path.with_name(f".{tar_path.name}.extracted")
    if marker.exists():
        return
    with tarfile.open(tar_path, "r:*") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=dest_dir, filter="data")
        else:
            tar.extractall(path=dest_dir)
    marker.touch()


class ASVspoof5(BaseDataset):
    name = "asvspoof5"

    # ------------------------------------------------------------------
    # Download ed estrazione
    # ------------------------------------------------------------------

    def _tsv_path(self, split: str) -> Path:
        tsv_name = SPLIT_INFO[split]["tsv_name"]
        direct = self.root / tsv_name
        if direct.exists():
            return direct
        found = next(self.root.rglob(tsv_name), None) if self.root.exists() else None
        if found is None:
            raise FileNotFoundError(f"Protocollo {tsv_name} non trovato sotto {self.root}. Esegui prepare('{split}').")
        return found

    def _audio_dir(self, split: str) -> Path:
        return self.root / SPLIT_INFO[split]["audio_dir"]

    def is_ready(self, split: str) -> bool:
        try:
            self._tsv_path(split)
        except FileNotFoundError:
            return False
        audio_dir = self._audio_dir(split)
        return audio_dir.exists() and next(audio_dir.glob("*.flac"), None) is not None

    def _remote_files(self, split: str) -> list[str]:
        """TAR da scaricare per lo split: prima i protocolli, poi l'audio in ordine."""
        patterns = [PROTOCOLS_TAR, SPLIT_INFO[split]["tar_pattern"]]
        files = HfApi().list_repo_files(HF_REPO_ID, repo_type="dataset")
        return sorted(f for f in files if any(fnmatch(f, p) for p in patterns))

    def _split_marker(self, split: str) -> Path:
        return self.root / f".{split}.complete"

    def prepare(self, split: str, delete_tars: bool = False) -> None:
        """
        Scarica ed estrae i TAR dello split **uno alla volta** (download → estrazione
        → file successivo), così la memoria resta bassa e si vede l'avanzamento.

        Idempotente e riprendibile: i TAR già estratti vengono saltati, anche se
        sono stati cancellati. Con delete_tars=True ogni TAR viene rimosso subito
        dopo l'estrazione (dimezza lo spazio su disco necessario).
        """
        if self._split_marker(split).exists():
            logger.info("Split '%s' già pronto in %s", split, self.root)
            return

        if get_token() is None:
            logger.warning(
                "Nessun token HF trovato: download con limiti ridotti. "
                "In locale esegui `hf auth login`; su Colab aggiungi il secret HF_TOKEN."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        files = self._remote_files(split)
        logger.info("Split '%s': %d file da %s → %s", split, len(files), HF_REPO_ID, self.root)

        for i, filename in enumerate(files, 1):
            tar_path = self.root / filename
            if tar_path.with_name(f".{filename}.extracted").exists():
                logger.info("[%d/%d] %s già estratto, salto.", i, len(files), filename)
                continue
            logger.info("[%d/%d] Download %s ...", i, len(files), filename)
            hf_hub_download(
                repo_id=HF_REPO_ID, filename=filename, repo_type="dataset",
                local_dir=str(self.root), tqdm_class=tqdm,
            )
            logger.info("[%d/%d] Estrazione %s ...", i, len(files), filename)
            _extract_tar(tar_path, self.root)
            if delete_tars:
                tar_path.unlink()

        self._split_marker(split).touch()
        logger.info("Split '%s' pronto.", split)

    # ------------------------------------------------------------------
    # Protocollo
    # ------------------------------------------------------------------

    def load_raw_protocol(self, split: str) -> pd.DataFrame:
        """TSV originale con le colonne TSV_COLUMNS."""
        # sep=r"\s+" collassa spazi multipli: con sep=" " si creano colonne
        # fantasma che fanno slittare il mapping (e KEY viene letto male).
        return pd.read_csv(
            self._tsv_path(split), sep=r"\s+", engine="python",
            names=TSV_COLUMNS, header=None,
        )

    def load_protocol(self, split: str) -> pd.DataFrame:
        raw = self.load_raw_protocol(split)
        audio_dir = SPLIT_INFO[split]["audio_dir"]
        return pd.DataFrame({
            "file_name": raw["FLAC_FILE_NAME"],
            "label": raw["KEY"],
            "audio_path": audio_dir + "/" + raw["FLAC_FILE_NAME"] + ".flac",
            "speaker_id": raw["SPEAKER_ID"],
            "speaker_gender": raw["SPEAKER_GENDER"],
            "codec": raw["CODEC"],
            "attack_label": raw["ATTACK_LABEL"],
        })
