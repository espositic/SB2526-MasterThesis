# Replica della Tabella 3 di XLSR-MamBo (Sez. 5.3 del paper) con i 4 modelli della tesi.
#
# Protocollo del paper: training su ASVspoof 2019 LA intero con RawBoost algoritmo 5 (LnL + ISD),
# test senza riallenare su 2019 LA eval, 2021 LA, 2021 DF e In-the-Wild. Noi in più testiamo
# su ASVspoof 5 eval completo e alleniamo anche su ASVspoof 5 intero (stessa ricetta).
# (Il paper usa un secondo modello con RawBoost algoritmo 3 per il 2021 DF: noi no, lo dichiariamo.)
#
# Il lavoro è diviso in due parti, che possono girare nello stesso momento:
#   python coda_tabella3.py fisso           TFPARN su 2019 LA (gli altri 3 sono già fatti) e test su ASVspoof 5
#   python coda_tabella3.py colab [modelli] training su ASVspoof 5 intero (di default solo MamBo),
#                                           da notebooks/05_colab.ipynb; i test si fanno poi sul fisso
# I passi già fatti vengono saltati, quindi se si interrompe basta rilanciare lo stesso comando.
# Se un passo va in errore si passa al successivo, senza ritentare.
import os
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8")

from tesi import dati, metriche, percorsi, training

MODELLI = ["mambo", "nes2net", "mhfa", "tfparn"]


def prepara_lista(nome):
    """Crea la lista intera di un test se non c'è ancora (sono troppo grandi per stare su git)."""
    if (percorsi.LISTE / f"{nome}.csv").exists():
        return
    crea = {
        "asvspoof2019_eval": lambda: dati.lista_asvspoof2019("eval"),
        "asvspoof2021la_eval": lambda: dati.lista_asvspoof2021("LA"),
        "asvspoof2021df_eval": lambda: dati.lista_asvspoof2021("DF"),
        "inthewild": dati.lista_in_the_wild,
        "asvspoof5_eval_completo": lambda: dati.lista_asvspoof5("eval"),
        # train intero di ASVspoof 5; per scegliere l'epoca si usa lo stesso dev da 5.000 clip di sempre
        "asvspoof5intero_train": lambda: dati.lista_asvspoof5("train"),
        "asvspoof5intero_dev": lambda: dati.carica_lista("asvspoof5_dev"),
    }
    dati.salva_lista(crea[nome](), nome)


def allena(modello, augmentation, dataset="asvspoof2019", num_workers=4):
    print(f"== training {modello} su {dataset} con {augmentation}, seed 1")
    try:
        return training.addestra(modello, dataset, augmentation, seed=1, num_workers=num_workers)
    except Exception:
        traceback.print_exc()
        print(f"ERRORE nel training di {modello}: passo al successivo")
        return None


def testa_e_stampa(cartella, nome_lista):
    if cartella is None:
        return
    print(f"== test {cartella.parent.name}/{cartella.name} su {nome_lista}")
    try:
        prepara_lista(nome_lista)
        p = training.testa(cartella, nome_lista)
    except Exception:
        traceback.print_exc()
        print("ERRORE nel test, passo al successivo")
        return
    print(nome_lista, metriche.tutte((p["label"] == "spoof").astype(int), p["punteggio"]))


def parte_fisso():
    # modelli LA (RawBoost algoritmo 5): MamBo, Nes2Net-X e MHFA ci sono già dalla coda del 28-30/10
    cartelle_la = [allena(m, "rawboost") for m in MODELLI]
    for nome in ["asvspoof2019_eval", "asvspoof2021la_eval", "inthewild"]:
        testa_e_stampa(cartelle_la[-1], nome)   # solo TFPARN, gli altri sono già testati
    for cartella in cartelle_la:
        testa_e_stampa(cartella, "asvspoof5_eval_completo")


def parte_colab(modelli):
    # training su ASVspoof 5 intero (circa 180.000 clip), uno dopo l'altro
    prepara_lista("asvspoof5intero_train")
    prepara_lista("asvspoof5intero_dev")
    for m in modelli:
        # su Colab ci sono più CPU: più worker, perché RawBoost gira sulla CPU
        allena(m, "rawboost", dataset="asvspoof5intero", num_workers=min(8, os.cpu_count()))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("fisso", "colab"):
        sys.exit("uso: python coda_tabella3.py fisso   oppure   python coda_tabella3.py colab [mambo nes2net ...]")
    if sys.platform == "win32":
        import ctypes
        # impedisce a Windows di andare in sospensione finché lo script gira
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if sys.argv[1] == "fisso":
        parte_fisso()
    else:
        parte_colab(sys.argv[2:] or ["mambo"])
    print("Tutto finito.")
