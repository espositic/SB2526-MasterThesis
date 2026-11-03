# Protocollo dei paper su ASVspoof 2019 LA, seed 1: MamBo, Nes2Net-X e MHFA con RawBoost.
# 1) training su asvspoof2019_train, scelta dell'epoca su asvspoof2019_dev (ricetta IPERPARAMETRI così com'è)
# 2) test su In-the-Wild, 2021 LA eval e 2019 LA eval
# 3) per ultimo il test su 2021 DF eval, il più lungo
# Se un training va in errore si passa al modello successivo, senza ritentare. I passi già fatti vengono saltati.
import ctypes
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8")

from tesi import metriche, training

MODELLI = ["mambo", "nes2net", "mhfa"]
TEST = ["inthewild", "asvspoof2021la_eval", "asvspoof2019_eval"]


def testa_e_stampa(cartella, nome_lista):
    print(f"== test {cartella.parent.name}/{cartella.name} su {nome_lista}")
    try:
        p = training.testa(cartella, nome_lista)
    except Exception:
        traceback.print_exc()
        print("ERRORE nel test, passo al successivo")
        return
    print(nome_lista, metriche.tutte((p["label"] == "spoof").astype(int), p["punteggio"]))


if __name__ == "__main__":
    # impedisce a Windows di andare in sospensione finché lo script gira
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    allenati = []
    for m in MODELLI:
        print(f"== training {m} su asvspoof2019 con rawboost, seed 1")
        try:
            allenati.append(training.addestra(m, "asvspoof2019", "rawboost", seed=1))
        except Exception:
            traceback.print_exc()
            print(f"ERRORE nel training di {m}: passo al modello successivo")

    for cartella in allenati:
        for nome in TEST:
            testa_e_stampa(cartella, nome)
    for cartella in allenati:
        testa_e_stampa(cartella, "asvspoof2021df_eval")
    print("Tutto finito.")
