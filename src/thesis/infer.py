"""
Inferenza di un modello già addestrato sull'eval COMPLETO di ASVspoof 5
(tutte le clip del protocollo ufficiale), per avere numeri confrontabili con la letteratura.

    python -m thesis.infer --config configs/nes2net_x_rawboost_asvspoof5.json --seed 1

Usa model_best.pt del run e tratta le clip esattamente come nella valutazione
sul subset (stessa lunghezza fissa, nessuna augmentation). Lavora a blocchi:
ogni blocco di punteggi è salvato su file appena calcolato, quindi
un'interruzione costa al massimo un blocco e rilanciando si riprende.
Le metriche si calcolano solo quando tutti i blocchi ci sono.
Output in runs/<nome>/seed<k>/eval_completo/.
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from thesis.datasets import ASVspoof5
from thesis.train import ARTIFACTS_ROOT, MODELS, TrainConfig, predict, split_metrics
from thesis.waveform import WaveformDataset

logger = logging.getLogger(__name__)
BLOCCO = 20_000  # clip per blocco (circa 18 minuti)


def main():
    parser = argparse.ArgumentParser(description="Inferenza sull'eval completo di ASVspoof 5")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    for noisy in ("httpx", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = TrainConfig.from_json(args.config)
    _, model_cls, input_kind = MODELS[cfg.model_type]
    assert input_kind == "waveform", "inferenza sull'eval completo prevista solo per i modelli su forma d'onda"
    run = ARTIFACTS_ROOT / "runs" / cfg.name / f"seed{args.seed}"
    out = run / "eval_completo"
    out.mkdir(exist_ok=True)

    ds = ASVspoof5()
    proto = ds.load_protocol("eval")[["file_name", "label", "audio_path", "attack_label", "codec"]]
    logger.info("Eval completo: %d clip — %s", len(proto), proto["label"].value_counts().to_dict())
    n_blocks = -(-len(proto) // BLOCCO)

    device = torch.device("cuda")
    model = model_cls(cfg.model).to(device)
    model.load_state_dict(torch.load(run / "model_best.pt", map_location=device))

    for b in range(n_blocks):
        path = out / f"scores_{b:03d}.csv"
        if path.exists():
            continue
        part = proto.iloc[b * BLOCCO:(b + 1) * BLOCCO]
        loader = DataLoader(WaveformDataset(ds, part, cfg.n_samples), batch_size=2 * cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        scores, _ = predict(model, loader, device)
        assert len(scores) == len(part)
        tmp = path.with_suffix(".tmp")
        part.drop(columns="audio_path").assign(score=scores).to_csv(tmp, index=False)
        os.replace(tmp, path)
        logger.info("Blocco %d/%d salvato", b + 1, n_blocks)

    # Tutti i blocchi presenti: punteggi completi e metriche, con lo stesso codice del subset
    scores = pd.concat([pd.read_csv(out / f"scores_{b:03d}.csv") for b in range(n_blocks)], ignore_index=True)
    assert len(scores) == len(proto) and (scores["file_name"].values == proto["file_name"].values).all()
    scores.to_csv(out / "scores_eval_completo.csv", index=False)
    m = split_metrics(scores, scores["score"].to_numpy())
    (out / "metrics.json").write_text(json.dumps({"seed": args.seed, "eval_completo": m}, indent=2), encoding="utf-8")
    logger.info("Eval completo: EER %.4f, minDCF %.4f", m["eer"], m["min_dcf"])


if __name__ == "__main__":
    main()
