# Estrazione delle feature SSL congelate per la baseline con classificatori leggeri (tesi/baseline_ssl.py).
#
# Per ogni modello SSL e ogni lista intera del protocollo della Tabella 3 (training su
# ASVspoof 2019 LA, test su 2019 LA, 2021 LA e DF, In-the-Wild, ASVspoof 5 eval completo) salva in
# artifacts/feature_ssl/<ssl>/ due file:
#   <lista>.npy        medie sul tempo di ogni layer, (n_clip, n_layer, dim) in float16
#   <lista>_label.npy  etichette, 0 = bonafide, 1 = spoof
# Poi i classificatori girano solo sulla CPU (anche su un altro PC: basta copiare la cartella).
#
#   python estrai_feature_ssl.py                      tutti e tre i modelli, uno dopo l'altro
#   python estrai_feature_ssl.py wavlm-large hubert-large-ll60k   i modelli Large (solo se indicati)
#   python estrai_feature_ssl.py wavlm-base hubert-base-ls960   solo quelli indicati
#   python estrai_feature_ssl.py asvspoof5intero_train asvspoof5intero_dev   solo le liste indicate
# Gli argomenti che non sono nomi di modelli sono nomi di liste (senza liste: quelle di LISTE).
# Le liste già fatte vengono saltate, quindi se si interrompe basta rilanciare.
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8")

from coda_tabella3 import prepara_lista
from tesi import baseline_ssl
from tesi.modelli.ssl import HUBERT_BASE, HUBERT_LARGE, WAVLM_BASE, WAVLM_LARGE, XLSR

MODELLI = [WAVLM_BASE, HUBERT_BASE, XLSR]   # quelli di default
ALTRI = [WAVLM_LARGE, HUBERT_LARGE]          # circa 78 GB ciascuno su tutte le liste: solo se indicati
# train e dev di 2019 LA per allenare e scegliere il layer, poi i test, dal più piccolo al più grande
LISTE = ["asvspoof2019_train", "asvspoof2019_dev", "inthewild", "asvspoof2019_eval",
         "asvspoof2021la_eval", "asvspoof2021df_eval", "asvspoof5_eval_completo"]

if __name__ == "__main__":
    nomi_modelli = {baseline_ssl.nome_cartella(m): m for m in MODELLI + ALTRI}
    scelti = [m for nome, m in nomi_modelli.items() if nome in sys.argv[1:]] or MODELLI
    liste = [a for a in sys.argv[1:] if a not in nomi_modelli] or LISTE
    if sys.platform == "win32":
        import ctypes
        # impedisce a Windows di andare in sospensione finché lo script gira
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    for lista in liste:
        if lista not in LISTE[:2]:
            prepara_lista(lista)   # le liste intere, se mancano (train e dev di 2019 ci sono già)
    for nome_ssl in scelti:
        print(f"== {nome_ssl}")
        for lista in liste:
            try:
                medie = baseline_ssl.estrai_medie(lista, nome_ssl)
                print(lista, medie.shape)
            except Exception:
                traceback.print_exc()
                print(f"ERRORE su {lista} con {nome_ssl}: passo alla prossima")
    print("Tutto finito.")
