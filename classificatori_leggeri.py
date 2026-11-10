# Classificatori leggeri (shallow learning) sulle feature SSL congelate estratte da estrai_feature_ssl.py.
#
# Protocollo come la Tabella 3: training su ASVspoof 2019 LA train, scelte su 2019 LA dev,
# test senza riallenare su In-the-Wild, 2019 LA eval, 2021 LA, 2021 DF e ASVspoof 5 eval completo.
#   1. per ogni layer: StandardScaler + regressione logistica su train, EER su dev -> layer migliore
#   2. sul layer migliore: 7 classificatori classici di scikit-learn
#   3. test su tutte le liste, lette a blocchi (un layer alla volta, mmap) per stare in poca RAM
# Gira solo sulla CPU, quindi anche sul T470: basta copiare artifacts/feature_ssl/<ssl>/.
#
#   python classificatori_leggeri.py wavlm-base          esecuzione completa
#   python classificatori_leggeri.py wavlm-base 3000     prova veloce: al massimo 3000 clip per lista
#
# Risultati in artifacts/classificatori_leggeri/<ssl>/ (risultati.csv, risultati.md, layer_dev.csv).
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from tesi import metriche, percorsi

TRAIN, DEV = "asvspoof2019_train", "asvspoof2019_dev"
TEST = ["inthewild", "asvspoof2019_eval", "asvspoof2021la_eval", "asvspoof2021df_eval", "asvspoof5_eval_completo"]
BLOCCO = 50_000   # clip lette insieme dalle liste grandi
N_JOB = -1        # tutti i core; nella prova se ne usano 2 per non disturbare altro


def classificatori(n_job):
    """I classificatori da confrontare. class_weight='balanced' dove c'è: nel train di 2019 LA
    gli spoof sono circa 9 volte i bonafide."""
    return {
        "Regressione logistica": LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        "SVM lineare": LinearSVC(C=0.1, class_weight="balanced", max_iter=5000),
        "Albero di decisione": DecisionTreeClassifier(max_depth=15, min_samples_leaf=20, class_weight="balanced", random_state=0),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=n_job, random_state=0),
        "kNN (PCA 128)": make_pipeline(PCA(n_components=128, random_state=0), KNeighborsClassifier(n_neighbors=15, n_jobs=n_job)),
        "Naive Bayes gaussiano": GaussianNB(),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced",
                                                            early_stopping=True, random_state=0),
    }


def carica(cartella, lista, n_max):
    """Feature (memmap, non lette subito) ed etichette (0 = bonafide, 1 = spoof)."""
    x = np.load(cartella / f"{lista}.npy", mmap_mode="r")
    y = np.load(cartella / f"{lista}_label.npy")
    if n_max:   # prova: un sottoinsieme fisso, con entrambe le classi
        scelte = np.sort(np.random.default_rng(0).permutation(len(y))[:n_max])
        return x, y[scelte], scelte
    return x, y, None


def layer(x, l, righe=None, inizio=0, fine=None):
    """Un layer (float32) di un pezzo della lista."""
    if righe is not None:
        return np.asarray(x[righe[inizio:fine], l], dtype=np.float32)
    return np.asarray(x[inizio:fine, l], dtype=np.float32)


def punteggio(modello, x):
    """Punteggio alto = bonafide: log-odds di bonafide contro spoof (con i pesi bilanciati
    è vicino a un log-likelihood ratio, quindi actDCF e Cllr hanno senso)."""
    if hasattr(modello, "decision_function"):
        return -modello.decision_function(x)   # la classe "positiva" è 1 = spoof
    p = np.clip(modello.predict_proba(x), 1e-6, 1 - 1e-6)
    return np.log(p[:, 0]) - np.log(p[:, 1])


def punteggi_lista(modelli, x, righe, l):
    """Punteggi di tutti i modelli su una lista, un blocco alla volta: così ogni blocco
    si legge dal disco una volta sola."""
    n = len(righe) if righe is not None else len(x)
    punteggi = {nome: [] for nome in modelli}
    for i in range(0, n, BLOCCO):
        pezzo = layer(x, l, righe, i, i + BLOCCO)
        for nome, modello in modelli.items():
            punteggi[nome].append(punteggio(modello, pezzo))
    return {nome: np.concatenate(p) for nome, p in punteggi.items()}


def tabella_markdown(df):
    righe = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        righe.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(righe) + "\n"


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("uso: python classificatori_leggeri.py <ssl> [n_max]   (ssl: wavlm-base, hubert-base-ls960, wav2vec2-xls-r-300m)")
    ssl = sys.argv[1]
    n_max = int(sys.argv[2]) if len(sys.argv) == 3 else None
    n_job = 2 if n_max else N_JOB
    cartella = percorsi.PROGETTO / "artifacts" / "feature_ssl" / ssl
    uscita = percorsi.PROGETTO / "artifacts" / "classificatori_leggeri" / (ssl + ("_prova" if n_max else ""))
    uscita.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr, r_tr = carica(cartella, TRAIN, n_max)
    x_dev, y_dev, r_dev = carica(cartella, DEV, n_max)
    n_layer = x_tr.shape[1]
    print(f"{ssl}: {n_layer} layer da {x_tr.shape[2]}, train {len(y_tr)} clip, dev {len(y_dev)} clip")

    # 1. scelta del layer con la regressione logistica (veloce), EER sul dev
    eer_dev = []
    for l in range(n_layer):
        modello = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        modello.fit(layer(x_tr, l, r_tr), y_tr)
        eer_dev.append(100 * metriche.eer(y_dev, punteggio(modello, layer(x_dev, l, r_dev))))
        print(f"layer {l:2d}: EER dev {eer_dev[-1]:.2f}%")
    migliore = int(np.argmin(eer_dev))
    pd.DataFrame({"layer": range(n_layer), "EER dev (%)": eer_dev}).to_csv(uscita / "layer_dev.csv", index=False)
    print(f"layer migliore: {migliore} (EER dev {eer_dev[migliore]:.2f}%)")

    # 2. classificatori sul layer migliore, scelta e controllo sul dev
    xl_tr, xl_dev = layer(x_tr, migliore, r_tr), layer(x_dev, migliore, r_dev)
    modelli, righe_tabella = {}, {}
    for nome, clf in classificatori(n_job).items():
        t = time.time()
        modelli[nome] = make_pipeline(StandardScaler(), clf).fit(xl_tr, y_tr)
        righe_tabella[nome] = {"classificatore": nome, "layer": migliore,
                               "minuti training": round((time.time() - t) / 60, 2),
                               "EER dev (%)": float(100 * metriche.eer(y_dev, punteggio(modelli[nome], xl_dev)))}
        print(righe_tabella[nome])

    # 3. test: ogni lista si legge una volta e si danno i punteggi con tutti i modelli
    for lista in TEST:
        t = time.time()
        x, y, righe = carica(cartella, lista, n_max)
        for nome, p in punteggi_lista(modelli, x, righe, migliore).items():
            m = metriche.tutte(y, p)
            righe_tabella[nome][f"{lista} EER (%)"] = m["EER (%)"]
            righe_tabella[nome][f"{lista} minDCF"] = m["minDCF"]
        print(f"{lista}: {len(y)} clip, {(time.time() - t) / 60:.1f} minuti")
        df = pd.DataFrame(list(righe_tabella.values()))   # salvata a ogni lista, così non si perde niente
        df.to_csv(uscita / "risultati.csv", index=False)
        (uscita / "risultati.md").write_text(tabella_markdown(df), encoding="utf-8")
    print(df[["classificatore"] + [c for c in df.columns if "EER" in c]].round(2).to_string(index=False))
    print("Tutto finito:", uscita)
