# Test cross-dataset di MamBo e MHFA allenati su ASVspoof 5 intero con RawBoost (seed 1): finora
# sono testati solo su ASVspoof 5 eval completo, qui si aggiungono In-the-Wild, 2019 LA eval,
# 2021 LA e 2021 DF. Parte dopo la coda in corso (PID passato come argomento), così la GPU non
# è mai usata da due processi insieme.
#
#   python lancia_test_asv5intero.py <PID da aspettare>
import ctypes
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8")

from coda_tabella3 import testa_e_stampa
from tesi import percorsi

ESPERIMENTI = ["mambo_asvspoof5intero_rawboost", "mhfa_asvspoof5intero_rawboost"]
LISTE = ["inthewild", "asvspoof2019_eval", "asvspoof2021la_eval", "asvspoof2021df_eval"]   # dalla più piccola


def vivo(pid):
    righe = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in righe


if __name__ == "__main__":
    pid = int(sys.argv[1])
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # niente sospensione finché gira
    print(f"Aspetto la fine della coda in corso (PID {pid}, controllo ogni 5 minuti)...")
    while vivo(pid):
        time.sleep(300)
    print(f"PID {pid} terminato alle {time.strftime('%H:%M')}, partono i test")
    for esperimento in ESPERIMENTI:
        cartella = percorsi.ESPERIMENTI / esperimento / "seed1"
        for lista in LISTE:
            testa_e_stampa(cartella, lista)
            print(f"   finito alle {time.strftime('%H:%M')}")
    print("Tutto finito.")
