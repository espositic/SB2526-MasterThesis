# Replica della Tabella 3 di XLSR-MamBo (Sez. 5.3 del paper) con i 4 modelli della tesi.
#
# Il paper allena su ASVspoof 2019 LA due modelli diversi (Sez. 4.2):
#   - modello "LA": RawBoost algoritmo 5 (LnL + ISD), testato su 2021 LA
#   - modello "DF": RawBoost algoritmo 3 (SSI), testato su 2021 DF
# Noi in più testiamo il modello LA su 2019 LA eval, In-the-Wild e ASVspoof 5 eval completo.
#
# Il lavoro è diviso in due parti, che possono girare nello stesso momento:
#   python coda_tabella3.py fisso   modello LA di TFPARN (gli altri 3 sono già fatti) e test su ASVspoof 5
#   python coda_tabella3.py colab   i 4 modelli DF (RawBoost SSI) e il loro test su 2021 DF
# I passi già fatti vengono saltati, quindi se si interrompe basta rilanciare lo stesso comando.
# Se un passo va in errore si passa al successivo, senza ritentare.
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
    }
    dati.salva_lista(crea[nome](), nome)


def allena(modello, augmentation):
    print(f"== training {modello} su asvspoof2019 con {augmentation}, seed 1")
    try:
        return training.addestra(modello, "asvspoof2019", augmentation, seed=1)
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


def parte_colab():
    # modelli DF (RawBoost algoritmo 3), ciascuno testato subito su 2021 DF
    for m in MODELLI:
        testa_e_stampa(allena(m, "rawboost_ssi"), "asvspoof2021df_eval")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("fisso", "colab"):
        sys.exit("uso: python coda_tabella3.py fisso   oppure   python coda_tabella3.py colab")
    if sys.platform == "win32":
        import ctypes
        # impedisce a Windows di andare in sospensione finché lo script gira
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    parte_fisso() if sys.argv[1] == "fisso" else parte_colab()
    print("Tutto finito.")
