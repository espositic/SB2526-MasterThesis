import logging

from src.data.downloader import ASVSpoofDownloader
from src.features.feature_extractor import (
    AudioFeatureExtractor,
    ensure_audio_extracted,
    ensure_protocols_extracted,
    load_manifest,
    save_manifest,
    select_balanced_subset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parametri configurabili
# ---------------------------------------------------------------------------
SPLIT = "train"          # "train" | "dev" | "eval"
N_PER_CLASS = 5_000      # campioni per classe (bonafide e spoof)
MIN_DURATION = 1.0       # secondi
MAX_DURATION = 15.0      # secondi
MAX_WORKERS = None       # None = usa tutti i core disponibili


def main() -> None:
    # 1. Download / verifica dataset
    downloader = ASVSpoofDownloader()
    downloader.ensure_dataset_ready()

    # 2. Estrai i file di protocollo (TSV) se necessario
    ensure_protocols_extracted()

    # 3. Estrai i TAR audio per lo split scelto se necessario
    ensure_audio_extracted(split=SPLIT)

    # 4. Selezione sottoinsieme bilanciato con filtro per durata
    samples = select_balanced_subset(
        split=SPLIT,
        n_per_class=N_PER_CLASS,
        min_duration=MIN_DURATION,
        max_duration=MAX_DURATION,
    )

    # 5. Estrazione Mel-Spettrogrammi in parallelo
    extractor = AudioFeatureExtractor(max_workers=MAX_WORKERS)
    features_df = extractor.extract(samples)

    logger.info(
        "Pipeline completata. %d feature salvate.",
        len(features_df),
    )


if __name__ == "__main__":
    main()
