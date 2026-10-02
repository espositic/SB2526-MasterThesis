"""
Percorsi di progetto, validi sia in locale sia su Google Colab.

Ordine di precedenza per ogni radice:
  1. variabile d'ambiente (THESIS_DATA_ROOT, THESIS_ARTIFACTS_ROOT);
  2. default per l'ambiente rilevato:
       - locale: <repo>/data e <repo>/artifacts
       - Colab:  /content/drive/MyDrive/Tesi/data e .../artifacts

DATA_ROOT      → dataset grezzi (download, TAR, audio estratto).
ARTIFACTS_ROOT → tutto ciò che è prodotto dagli esperimenti (feature, modelli, risultati).
MANIFESTS_DIR  → resta dentro il repo: i manifest si committano per la riproducibilità.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

IN_COLAB = "google.colab" in sys.modules
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/Tesi")

DEFAULT_SEED = 42


def _root(env_var: str, local_default: Path, colab_default: Path) -> Path:
    if env_var in os.environ:
        return Path(os.environ[env_var]).expanduser().resolve()
    return colab_default if IN_COLAB else local_default


DATA_ROOT = _root("THESIS_DATA_ROOT", PROJECT_ROOT / "data", COLAB_DRIVE_ROOT / "data")
ARTIFACTS_ROOT = _root("THESIS_ARTIFACTS_ROOT", PROJECT_ROOT / "artifacts", COLAB_DRIVE_ROOT / "artifacts")


def describe() -> str:
    """Riepilogo leggibile della configurazione, da stampare all'inizio dei notebook."""
    return "\n".join([
        f"Ambiente:       {'Colab' if IN_COLAB else 'locale'}",
        f"PROJECT_ROOT:   {PROJECT_ROOT}",
        f"DATA_ROOT:      {DATA_ROOT}",
        f"ARTIFACTS_ROOT: {ARTIFACTS_ROOT}",
        f"MANIFESTS_DIR:  {MANIFESTS_DIR}",
    ])
