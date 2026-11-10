# Estrae le feature SSL congelate di ASVspoof 5 train intero e dev (asvspoof5intero_train / _dev)
# per WavLM Large e HuBERT Large, con estrai_feature_ssl.py, dopo che è finita la coda in corso
# (PID passato come argomento), così la GPU non è mai usata da due processi insieme.
# Le feature di HuBERT Large finiscono su D: (la sua cartella è una junction, vedi lancia_hubert_large.py).
#
#   python lancia_large_asv5train.py <PID da aspettare>
import ctypes
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def vivo(pid):
    righe = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in righe


if __name__ == "__main__":
    pid = int(sys.argv[1])
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # niente sospensione finché gira
    print(f"Aspetto la fine della coda in corso (PID {pid}, controllo ogni 5 minuti)...")
    while vivo(pid):
        time.sleep(300)
    print(f"PID {pid} terminato alle {time.strftime('%H:%M')}, parte l'estrazione di ASVspoof 5 train/dev")
    comando = [sys.executable, "-u", "estrai_feature_ssl.py", "wavlm-large", "hubert-large-ll60k",
               "asvspoof5intero_train", "asvspoof5intero_dev"]
    esito = subprocess.run(comando, cwd=REPO)
    print(f"estrai_feature_ssl.py terminato alle {time.strftime('%H:%M')} con codice {esito.returncode}")
