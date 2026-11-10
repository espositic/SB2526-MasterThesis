# Estrae le feature SSL congelate di HuBERT Large (hubert-large-ll60k) su tutte le liste, con
# estrai_feature_ssl.py, dopo che è finita l'estrazione di WavLM Large in corso (PID passato come
# argomento), così la GPU non è mai usata da due processi insieme.
# Le feature vanno sul disco D: artifacts\feature_ssl\hubert-large-ll60k è una junction verso
# D:\feature_ssl\hubert-large-ll60k (circa 78 GB, su C: non ci starebbero insieme a WavLM Large).
#
#   python lancia_hubert_large.py <PID di WavLM Large>
import ctypes
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
LINK = REPO / "artifacts" / "feature_ssl" / "hubert-large-ll60k"
DESTINAZIONE = Path(r"D:\feature_ssl\hubert-large-ll60k")

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def vivo(pid):
    righe = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in righe


if __name__ == "__main__":
    pid = int(sys.argv[1])
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # niente sospensione finché gira
    DESTINAZIONE.mkdir(parents=True, exist_ok=True)
    if not LINK.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(LINK), str(DESTINAZIONE)], check=True)
    print(f"{LINK} -> {DESTINAZIONE}")
    print(f"Aspetto la fine di WavLM Large (PID {pid}, controllo ogni 5 minuti)...")
    while vivo(pid):
        time.sleep(300)
    print(f"PID {pid} terminato alle {time.strftime('%H:%M')}, parte HuBERT Large")
    esito = subprocess.run([sys.executable, "-u", "estrai_feature_ssl.py", "hubert-large-ll60k"], cwd=REPO)
    print(f"estrai_feature_ssl.py terminato alle {time.strftime('%H:%M')} con codice {esito.returncode}")
