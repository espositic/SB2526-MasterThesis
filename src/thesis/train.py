"""
Training e valutazione dei detector: XLSR-MamBo su forma d'onda (GPU) e
LCNN sui Mel-spettrogrammi già estratti (anche solo CPU).

Entry point unico, pensato per girare in background senza Jupyter:
    python -m thesis.train --config configs/mambo_asvspoof5.json --seed 1
    python -m thesis.train --config configs/lcnn_asvspoof5.json --seed 1 2 3

Protocollo: train sul manifest di train, scelta dell'epoca migliore SOLO sul
dev (EER), test una volta sola sull'eval con il modello migliore.
Tutto finisce in ARTIFACTS_ROOT/runs/<nome>/seed<k>/:
  config.json, log.csv (una riga per epoca), model_best.pt,
  scores_dev.csv, scores_eval.csv, metrics.json (scritto per ultimo).
Un seed con metrics.json già presente viene saltato; un seed interrotto riparte
dall'ultima epoca completata grazie a checkpoint_last.pt (cancellato a fine seed).
"""

import argparse
import json
import logging
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from thesis import metrics
from thesis.augment import RawBoostConfig
from thesis.config import ARTIFACTS_ROOT, MANIFESTS_DIR, PROJECT_ROOT
from thesis.datasets import ASVspoof5
from thesis.features import MelDataset
from thesis.models import LCNN, LCNNConfig, MamBoConfig, XLSRMamBo, count_parameters
from thesis.waveform import WaveformDataset

logger = logging.getLogger(__name__)


MODELS = {  # model_type → (classe della config, classe del modello, ingresso)
    "mambo": (MamBoConfig, XLSRMamBo, "waveform"),
    "lcnn": (LCNNConfig, LCNN, "mel"),
}


@dataclass
class TrainConfig:
    name: str = "mambo_asvspoof5"
    model_type: str = "mambo"
    device: str = "cuda"             # "cpu" per i PC senza GPU
    num_threads: int = 0             # thread CPU di PyTorch (0 = default)
    train_manifest: str = "asvspoof5_train_n5000_seed42_dur1-15.csv"
    dev_manifest: str = "asvspoof5_dev_n2500_seed42_dur1-15.csv"
    eval_manifest: str = "asvspoof5_eval_n5000_seed42_dur1-15.csv"
    n_samples: int = 66_800          # forma d'onda: 4.175 s a 16 kHz, come nel paper MamBo
    n_frames: int = 128              # Mel: 128 frame × 32 ms ≈ 4.1 s, stessa durata
    mel_tag: str = "mel_sr16000_m128_fft1024_hop512"
    batch_size: int = 8
    accum_steps: int = 4             # batch effettivo 32, come nel paper
    lr: float = 1e-5
    weight_decay: float = 0.05
    epochs: int = 20
    patience: int = 7                # early stopping sull'EER del dev
    warmup_ratio: float = 0.1        # warmup lineare 10% + coseno fino al 10% del lr
    focal_gamma: float = 2.0
    focal_alpha: tuple = (0.5, 0.5)  # [bonafide, spoof]: i nostri subset sono bilanciati
    num_workers: int = 4
    subset_frac: float = 1.0         # < 1 solo per lo smoke test
    rawboost: Optional[RawBoostConfig] = None  # solo forma d'onda, solo train; None = disattivato
    model: object = field(default_factory=MamBoConfig)

    @classmethod
    def from_json(cls, path) -> "TrainConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        model_type = raw.get("model_type", "mambo")
        assert model_type in MODELS, f"model_type sconosciuto: {model_type}"
        model = MODELS[model_type][0](**raw.pop("model", {}))
        if raw.get("rawboost") is not None:
            raw["rawboost"] = RawBoostConfig(**raw["rawboost"])
        unknown = set(raw) - set(cls.__dataclass_fields__)
        assert not unknown, f"chiavi sconosciute nella config: {unknown}"
        if "focal_alpha" in raw:
            raw["focal_alpha"] = tuple(raw["focal_alpha"])
        return cls(**raw, model=model)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def focal_loss(logits, target, gamma: float, alpha: tuple) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
    weight = torch.tensor(alpha, device=logits.device)[target]
    return (-weight * (1 - logp.exp()) ** gamma * logp).mean()


def lr_lambda(step: int, total: int, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "sconosciuto"


def save_atomic(obj, path: Path) -> None:
    """Salva su un file temporaneo e poi rinomina: un'interruzione non lascia file a metà."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_split(ds: ASVspoof5, split: str, manifest: str, frac: float, seed: int,
               check_audio: bool = True) -> pd.DataFrame:
    """Manifest + metadati del protocollo (attacco, codec) per l'analisi per gruppo."""
    subset = ds.load_manifest(MANIFESTS_DIR / manifest, check_audio=check_audio)
    if frac < 1:
        subset = subset.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)
    meta = ds.load_protocol(split)[["file_name", "attack_label", "codec"]]
    subset = subset.merge(meta, on="file_name", how="left", validate="one_to_one")
    assert subset["attack_label"].notna().all(), f"file del manifest {manifest} assenti dal protocollo"
    return subset.reset_index(drop=True)


def autocast(device: torch.device):
    """BF16 solo su GPU; su CPU si resta in float32."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, float]:
    """Punteggi (logit bonafide − logit spoof) e loss media di cross-entropy."""
    model.eval()
    scores, loss_sum, n = [], 0.0, 0
    for wave, label in tqdm(loader, desc="valutazione", leave=False):
        wave, label = wave.to(device, non_blocking=True), label.to(device)
        with autocast(device):
            logits = model(wave)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, label, reduction="sum").item()
        n += len(label)
        scores.append((logits[:, 0] - logits[:, 1]).cpu().numpy())
    return np.concatenate(scores), loss_sum / n


def save_scores(path: Path, subset: pd.DataFrame, scores: np.ndarray) -> None:
    assert len(scores) == len(subset), f"{len(scores)} punteggi per {len(subset)} file"
    out = subset[["file_name", "label", "attack_label", "codec"]].copy()
    out["score"] = scores
    out.to_csv(path, index=False)


def split_metrics(subset: pd.DataFrame, scores: np.ndarray) -> dict:
    y = (subset["label"] == "spoof").astype(int).to_numpy()
    m = metrics.all_metrics(y, scores)
    m["eer_ci95"] = metrics.bootstrap_eer(y, scores)
    m["eer_per_attacco"] = metrics.eer_by_group(y, scores, subset["attack_label"])
    m["eer_per_codec"] = metrics.eer_by_group(y, scores, subset["codec"])
    return m


def run(cfg: TrainConfig, seed: int) -> dict:
    out_dir = ARTIFACTS_ROOT / "runs" / cfg.name / f"seed{seed}"
    if (out_dir / "metrics.json").exists():
        logger.info("%s esiste già: seed %d saltato", out_dir / "metrics.json", seed)
        return json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device(cfg.device)
    assert device.type != "cuda" or torch.cuda.is_available(), "config con device=cuda ma GPU non disponibile"
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    _, model_cls, input_kind = MODELS[cfg.model_type]
    ds = ASVspoof5()
    data = {
        split: load_split(ds, split, manifest, cfg.subset_frac, seed, check_audio=input_kind == "waveform")
        for split, manifest in (("train", cfg.train_manifest), ("dev", cfg.dev_manifest), ("eval", cfg.eval_manifest))
    }
    assert cfg.rawboost is None or input_kind == "waveform", "RawBoost si applica solo alla forma d'onda"
    if input_kind == "waveform":
        make_ds = lambda subset, augment=None: WaveformDataset(ds, subset, cfg.n_samples, augment)
    else:
        feat_dir = ARTIFACTS_ROOT / "features" / ds.name / cfg.mel_tag
        make_ds = lambda subset, augment=None: MelDataset(feat_dir, subset, cfg.n_frames)
    train_ds = make_ds(data["train"], cfg.rawboost)  # augmentation solo sul train
    if cfg.rawboost is not None:
        # Controllo esplicito: la clip aumentata deve differire da quella originale
        raw_wave, aug_wave = make_ds(data["train"])[0][0], train_ds[0][0]
        diff = float((raw_wave - aug_wave).abs().mean())
        assert diff > 0, "RawBoost attivo ma la forma d'onda non è cambiata"
        logger.info("RawBoost attivo (algoritmo %d): differenza media su una clip di prova %.4f",
                    cfg.rawboost.algo, diff)
    loader_args = dict(num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                       persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=True, **loader_args)
    eval_loaders = {s: DataLoader(make_ds(data[s]), batch_size=2 * cfg.batch_size,
                                  shuffle=False, **loader_args) for s in ("dev", "eval")}

    model = model_cls(cfg.model).to(device)
    n_params = count_parameters(model)
    logger.info("Parametri: %s", n_params)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader) // cfg.accum_steps
    total_steps = steps_per_epoch * cfg.epochs
    assert steps_per_epoch > 0, "troppi pochi dati per batch_size × accum_steps"
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, total_steps, int(cfg.warmup_ratio * total_steps)))

    (out_dir / "config.json").write_text(json.dumps({
        **asdict(cfg), "seed": seed, "git_commit": git_commit(), "parametri": n_params,
        "n_campioni": {s: len(d) for s, d in data.items()},
    }, indent=2), encoding="utf-8")

    log_path = out_dir / "log.csv"
    ckpt_path = out_dir / "checkpoint_last.pt"
    best_eer, best_epoch, waited, start_epoch = float("inf"), 0, 0, 1
    if ckpt_path.exists():
        # Ripresa dopo un'interruzione: si riparte dall'epoca successiva all'ultima completata
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # in RAM: in GPU non ci sta
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_eer, best_epoch, waited = ckpt["best_eer"], ckpt["best_epoch"], ckpt["waited"]
        start_epoch = ckpt["epoch"] + 1
        torch.set_rng_state(ckpt["rng_cpu"])
        if ckpt["rng_cuda"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state(ckpt["rng_cuda"])
        # Le righe di log.csv oltre l'ultima epoca salvata appartengono all'epoca interrotta
        if log_path.exists():
            log = pd.read_csv(log_path)
            log[log["epoca"] <= ckpt["epoch"]].to_csv(log_path, index=False)
        logger.info("Ripresa dal checkpoint: epoca %d completata, migliore %d (EER dev %.4f)",
                    ckpt["epoch"], best_epoch, best_eer)
        del ckpt
    elif log_path.exists():
        log_path.unlink()  # log di un tentativo interrotto prima della fine della prima epoca

    on_gpu = device.type == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    stop = waited >= cfg.patience
    for epoch in range(start_epoch, cfg.epochs + 1):
        if stop:
            break
        model.train()
        t0, loss_sum, n_batches = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        batches = tqdm(train_loader, desc=f"epoca {epoch}", leave=False)
        for i, (wave, label) in enumerate(batches, 1):
            wave, label = wave.to(device, non_blocking=True), label.to(device)
            with autocast(device):
                logits = model(wave)
            loss = model.loss_fn(logits, label) if hasattr(model, "loss_fn") else None
            if loss is None:  # loss di default: focal (con gamma=0 è la cross-entropy pesata)
                loss = focal_loss(logits, label, cfg.focal_gamma, cfg.focal_alpha)
            assert torch.isfinite(loss), f"loss non finita all'epoca {epoch}, batch {i}"
            (loss / cfg.accum_steps).backward()
            if i % cfg.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item()
            n_batches += 1
            batches.set_postfix(loss=f"{loss_sum / n_batches:.4f}")
        train_time = time.time() - t0

        dev_scores, dev_loss = predict(model, eval_loaders["dev"], device)
        y_dev = (data["dev"]["label"] == "spoof").astype(int).to_numpy()
        dev_eer = metrics.eer(y_dev, dev_scores)
        row = {"epoca": epoch, "train_loss": loss_sum / n_batches, "dev_loss": dev_loss, "dev_eer": dev_eer,
               "lr": scheduler.get_last_lr()[0], "rawboost": cfg.rawboost.algo if cfg.rawboost else 0,
               "sec_train": round(train_time),
               "sec_totale": round(time.time() - t0),
               "gpu_max_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if on_gpu else 0.0}
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        logger.info("Epoca %d: %s", epoch, row)

        if dev_eer < best_eer:
            best_eer, best_epoch, waited = dev_eer, epoch, 0
            save_atomic(model.state_dict(), out_dir / "model_best.pt")
        else:
            waited += 1
            if waited >= cfg.patience:
                logger.info("Early stopping all'epoca %d (migliore: %d)", epoch, best_epoch)
                stop = True

        save_atomic({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_eer": best_eer, "best_epoch": best_epoch,
            "waited": waited, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state() if on_gpu else None,
        }, ckpt_path)

    # Valutazione finale con il modello scelto sul dev
    model.load_state_dict(torch.load(out_dir / "model_best.pt", map_location=device))
    results = {"seed": seed, "epoca_migliore": best_epoch}
    for split in ("dev", "eval"):
        scores, _ = predict(model, eval_loaders[split], device)
        save_scores(out_dir / f"scores_{split}.csv", data[split], scores)
        results[split] = split_metrics(data[split], scores)

    tmp = out_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    os.replace(tmp, out_dir / "metrics.json")
    ckpt_path.unlink()  # serve solo per riprendere un seed interrotto (~4 GB)
    logger.info("Seed %d: dev EER %.4f, eval EER %.4f", seed, results["dev"]["eer"], results["eval"]["eer"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Training di un detector su forma d'onda")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    for noisy in ("httpx", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = TrainConfig.from_json(args.config)
    for seed in args.seed:
        run(cfg, seed)


if __name__ == "__main__":
    main()
