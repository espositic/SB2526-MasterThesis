"""
Monitoraggio dei training in corso e tabelle dei risultati, letti solo dai file
in ARTIFACTS_ROOT/runs/<nome>/ (train.log, seed<k>/log.csv, seed<k>/metrics.json).
Usato dai notebook: non tocca mai il processo di training.
"""

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from thesis.config import ARTIFACTS_ROOT, PROJECT_ROOT

TQDM = re.compile(r"epoca (\d+):\s+\d+%.*?\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.?]+)(s/it|it/s).*?loss=([\w.+-]+)")
VAL = re.compile(r"valutazione:.*?\|\s*(\d+)/(\d+) \[[\d:]+<([\d:?]+)")
ERRORI = re.compile(r"Traceback|Error|ERROR|AssertionError|out of memory")


def run_dir(name: str) -> Path:
    return ARTIFACTS_ROOT / "runs" / name


def processo_attivo(config_file: str) -> bool:
    """C'è un processo `python -m thesis.train` lanciato con questa config?"""
    for p in psutil.process_iter(["cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        if "thesis.train" in cmd and config_file in cmd:
            return True
    return False


def lancia(configs: list[Path], seeds: list[int], log_name: str) -> int:
    """
    Avvia in background, uno dopo l'altro, i training delle config date.
    Il processo è staccato dal kernel: continua anche chiudendo VS Code.
    I seed già completati vengono saltati, quelli interrotti ripresi.
    """
    log_path = ARTIFACTS_ROOT / "runs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    seed_args = " ".join(map(str, seeds))
    comandi = [f'"{sys.executable}" -m thesis.train --config "{c}" --seed {seed_args}' for c in configs]
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
        shell_cmd = ["cmd", "/c", " & ".join(comandi)]
    else:
        shell_cmd = ["sh", "-c", " ; ".join(comandi)]
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(shell_cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    return proc.pid


def _coda(path: Path, n_byte: int = 30_000) -> list[str]:
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - n_byte))
        testo = f.read().decode("utf-8", errors="replace")
    return [r.strip() for r in re.split(r"[\r\n]+", testo) if r.strip()]


def stato(name: str, config_file: str, seeds: list[int], log_file: Path, log_fermo_min: int = 10) -> None:
    """Stampa lo stato di un run e un riquadro ATTENZIONE se qualcosa non va."""
    rd = run_dir(name)
    print(f"[{name}]")
    avvisi = []
    attivo = processo_attivo(config_file)
    completati = [s for s in seeds if (rd / f"seed{s}" / "metrics.json").exists()]

    for s in seeds:
        sd = rd / f"seed{s}"
        if (sd / "metrics.json").exists():
            r = json.loads((sd / "metrics.json").read_text(encoding="utf-8"))
            print(f"  seed {s}: COMPLETATO — epoca migliore {r['epoca_migliore']}, "
                  f"EER dev {100 * r['dev']['eer']:.2f}%, EER eval {100 * r['eval']['eer']:.2f}%")
        elif (sd / "log.csv").exists():
            log = pd.read_csv(sd / "log.csv")
            best = log.loc[log.dev_eer.idxmin()]
            print(f"  seed {s}: {len(log)} epoche finite — EER dev ultima {100 * log.dev_eer.iloc[-1]:.2f}%, "
                  f"migliore {100 * best.dev_eer:.2f}% (epoca {int(best.epoca)}), "
                  f"{log.sec_totale.mean() / 60:.1f} min/epoca")
            losses = log[["train_loss", "dev_loss"]].to_numpy(dtype=float)
            if not np.isfinite(losses).all():
                avvisi.append(f"loss NaN/inf nel log.csv del seed {s}")
        elif (sd / "config.json").exists():
            print(f"  seed {s}: in corso, prima epoca non ancora finita")
        else:
            print(f"  seed {s}: non ancora iniziato")

    if attivo and log_file.exists():
        righe = _coda(log_file)
        ultima = next((r for r in reversed(righe) if r.startswith(("epoca", "valutazione"))), "")
        barre = [m for r in righe if (m := TQDM.search(r))]
        corrente = next((s for s in seeds if s not in completati), None)
        if barre and ultima.startswith("epoca"):
            ep, fatti, tot, _, resto, vel, unita, loss = barre[-1].groups()
            bps = float(vel) if unita == "it/s" else 1 / float(vel)
            print(f"  in corso: seed {corrente}, epoca {ep}, batch {fatti}/{tot} ({100 * int(fatti) / int(tot):.0f}%), "
                  f"{bps:.2f} batch/s, loss {loss}, fine epoca tra ~{resto}")
            if loss.lower() in ("nan", "inf", "-inf"):
                avvisi.append(f"loss {loss} nell'epoca {ep}")
        elif (m := VAL.search(ultima)):
            print(f"  in corso: seed {corrente}, valutazione (dev a fine epoca, eval a fine seed): "
                  f"batch {m.group(1)}/{m.group(2)}, mancano ~{m.group(3)}")
        fermo = (time.time() - log_file.stat().st_mtime) / 60
        if fermo > log_fermo_min:
            avvisi.append(f"{log_file.name} non si aggiorna da {fermo:.0f} minuti: training bloccato?")
        errori = [r for r in righe if ERRORI.search(r)]
        if errori:
            avvisi.append("errori nel log:\n      " + "\n      ".join(errori[-5:]))

    iniziato = any((rd / f"seed{s}" / "config.json").exists() for s in seeds)
    if len(completati) == len(seeds):
        print("  tutti i seed completati")
    elif not attivo and not iniziato:
        print("  non ancora lanciato (o in coda dietro a un'altra config)")
    elif not attivo:
        avvisi.append("nessun training attivo con questa config ma seed non tutti completati "
                      "(crash, riavvio del PC o interruzione): rilancia, riprende dall'ultima epoca")

    if avvisi:
        print("  " + "!" * 66)
        for a in avvisi:
            print("  ATTENZIONE:", a)
        print("  " + "!" * 66)
        if log_file.exists():
            print("  ultime righe di", log_file.name)
            for r in _coda(log_file)[-6:]:
                print("    ", r[:200])
    else:
        print("  nessun problema rilevato")


def risultati(names: list[str], seeds: list[int]) -> pd.DataFrame:
    """Una riga per (run, seed, split) con le metriche principali; solo i seed completati."""
    rows = []
    for name in names:
        for s in seeds:
            path = run_dir(name) / f"seed{s}" / "metrics.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text(encoding="utf-8"))
            for split in ("dev", "eval"):
                m = r[split]
                rows.append({"run": name, "seed": s, "split": split, "epoca": r["epoca_migliore"],
                             "EER %": 100 * m["eer"], "EER IC95 %": f"{100 * m['eer_ci95'][0]:.2f}–{100 * m['eer_ci95'][1]:.2f}",
                             "minDCF": m["min_dcf"], "actDCF": m["act_dcf"], "Cllr": m["cllr"]})
    return pd.DataFrame(rows)


def riepilogo(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Media ± deviazione standard sui seed, con il numero di seed usati."""
    g = per_seed.groupby(["run", "split"])
    out = g[["EER %", "minDCF", "actDCF", "Cllr"]].agg(["mean", "std"]).round(4)
    out[("seed", "n")] = g.size()
    return out


def eer_per_gruppo(name: str, seeds: list[int], key: str, split: str = "eval") -> pd.DataFrame:
    """EER (%) per attacco o codec ("eer_per_attacco" / "eer_per_codec"), colonne = seed + media."""
    cols = {}
    for s in seeds:
        path = run_dir(name) / f"seed{s}" / "metrics.json"
        if path.exists():
            cols[f"seed {s}"] = json.loads(path.read_text(encoding="utf-8"))[split][key]
    df = 100 * pd.DataFrame(cols)
    return df.assign(media=df.mean(axis=1)).round(2)


def calibra(name: str, seeds: list[int]) -> pd.DataFrame:
    """
    Per ogni seed: calibrazione lineare stimata sui punteggi del dev e applicata
    all'eval. Scrive scores_<split>_calibrati.csv e calibrazione.json accanto ai
    file originali (che non vengono toccati) e ritorna actDCF e Cllr prima/dopo.
    EER e minDCF non possono cambiare (trasformazione monotona): fanno da controllo.
    """
    from thesis import metrics

    rows = []
    for s in seeds:
        sd = run_dir(name) / f"seed{s}"
        dev = pd.read_csv(sd / "scores_dev.csv")
        y_dev = (dev["label"] == "spoof").astype(int).to_numpy()
        a, b = metrics.fit_calibration(y_dev, dev["score"])
        (sd / "calibrazione.json").write_text(json.dumps({"a": a, "b": b, "stimata_su": "dev"}, indent=2),
                                              encoding="utf-8")
        for split in ("dev", "eval"):
            df = pd.read_csv(sd / f"scores_{split}.csv")
            y = (df["label"] == "spoof").astype(int).to_numpy()
            cal = metrics.apply_calibration(df["score"], a, b)
            df.assign(score=cal).to_csv(sd / f"scores_{split}_calibrati.csv", index=False)
            prima, dopo = metrics.all_metrics(y, df["score"]), metrics.all_metrics(y, cal)
            assert abs(prima["eer"] - dopo["eer"]) < 1e-9 and abs(prima["min_dcf"] - dopo["min_dcf"]) < 1e-9, \
                "la calibrazione ha cambiato EER/minDCF: non dovrebbe succedere"
            rows.append({"seed": s, "split": split, "a": a, "b": b, "EER %": 100 * prima["eer"],
                         "actDCF prima": prima["act_dcf"], "actDCF dopo": dopo["act_dcf"],
                         "Cllr prima": prima["cllr"], "Cllr dopo": dopo["cllr"]})
    return pd.DataFrame(rows)


def ogni_minuto(funzione, minuti: int) -> None:
    """Richiama funzione() ogni 60 s per `minuti` minuti, ripulendo l'output (interrompibile)."""
    from IPython.display import clear_output

    fine = time.time() + 60 * minuti
    while True:
        clear_output(wait=True)
        print(f"Aggiornato alle {datetime.now():%H:%M:%S}\n")
        funzione()
        if time.time() > fine:
            break
        time.sleep(60)
