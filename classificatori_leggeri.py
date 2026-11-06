# Classificatori leggeri (shallow learning) sulle feature SSL congelate estratte da estrai_feature_ssl.py.
#
# Protocollo come la Tabella 3: training su ASVspoof 2019 LA train, scelte su 2019 LA dev,
# test senza riallenare su In-the-Wild, 2019 LA eval, 2021 LA, 2021 DF e ASVspoof 5 eval completo.
#   1. per ogni layer: StandardScaler + regressione logistica su train, EER su dev -> layer migliore
#      (a parità di EER dev, entro SPAREGGIO punti, vince la log-loss dev più bassa)
#   2. sul layer migliore: classificatori classici di scikit-learn (StandardScaler + classificatore),
#      più le varianti (due GMM, C scelto sul dev, voting, SelectKBest, sottocampionamento,
#      concatenazione di 3 layer, somma pesata appresa di tutti i layer + MLP come in Spoof-SUPERB,
#      e la configurazione di Pascu et al. 2023: ultimo layer, regressione logistica C=1e6 senza scaler): stesso train, stesso dev, stesse liste di test e stesse metriche,
#      quindi tutte nella stessa tabella (risultati.md), con la famiglia di ognuna.
#      Di scorta, fuori tabella: anomaly detection solo sui bonafide (risultati_scorta.md)
#      e clustering k-means (clustering.csv), che non è un classificatore
#   3. test su tutte le liste, lette a blocchi (mmap) per stare in poca RAM
# Gira solo sulla CPU, quindi anche sul T470: basta copiare artifacts/feature_ssl/<ssl>/.
#
#   python classificatori_leggeri.py wavlm-base          esecuzione completa
#   python classificatori_leggeri.py wavlm-base 3000     prova veloce: al massimo 3000 clip per lista
#   python classificatori_leggeri.py wavlm-base --train asvspoof5intero_train --dev asvspoof5intero_dev
#                                                        training e scelte su ASVspoof 5 intero
#                                                        (risultati in <ssl>_train-<lista train>)
#
# Risultati in artifacts/classificatori_leggeri/<ssl>/ (risultati.csv/.md,
# risultati_scorta.csv/.md, clustering.csv, layer_dev.csv), modelli allenati in modelli/ (.joblib, .pt per la
# somma pesata; indice.csv dice quale layer usa ognuno) e punteggi per clip in punteggi/<lista>.npz
# (nomi, punteggi (n_clip, n_modelli), label; utili per fusione e curve DET).
import argparse
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              IsolationForest, RandomForestClassifier)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDOneClassSVM
from sklearn.metrics import adjusted_rand_score, log_loss
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from tesi import metriche, percorsi

TRAIN, DEV = "asvspoof2019_train", "asvspoof2019_dev"   # default: training e scelte su 2019 LA
SPAREGGIO = 0.05  # punti di EER dev entro cui due layer sono considerati pari
TEST = ["inthewild", "asvspoof2019_eval", "asvspoof2021la_eval", "asvspoof2021df_eval", "asvspoof5_eval_completo"]
BLOCCO = 20_000   # clip lette insieme dalle liste grandi (con tutti i layer di XLS-R sono già 2 GB)
N_JOB = -1        # tutti i core; nella prova se ne usano 2 per non disturbare altro
VALORI_C = [0.01, 0.1, 1, 10, 100]   # C della regressione logistica, scelto sul dev
VALORI_K = [64, 128, 256]            # feature tenute da SelectKBest, scelte sul dev
# con train grandi (ASVspoof 5 intero, 182k clip) SVC esatta e kNN nel test durerebbero ore:
# si allenano su al massimo MAX_LENTI clip prese a caso (stratificate), la taglia del train di 2019 LA
LENTI, MAX_LENTI = ["SVM RBF (esatta)", "kNN (PCA 128)"], 25_380
MAX_RAM = 30_000   # oltre queste clip la somma pesata legge il train dal disco un batch alla volta


def classificatori(n_job, rapporto):
    """I classificatori della pipeline standard (fit su tutto il train), con la loro famiglia.
    class_weight='balanced' dove c'è: nel train di 2019 LA gli spoof sono circa 9 volte i bonafide
    (rapporto = n_bonafide / n_spoof, per XGBoost che usa scale_pos_weight)."""
    return {
        "Regressione logistica": ("lineare", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
        "SVM lineare": ("lineare", LinearSVC(C=0.1, class_weight="balanced", max_iter=5000)),
        "LDA": ("lineare", LinearDiscriminantAnalysis()),
        "Albero di decisione": ("alberi", DecisionTreeClassifier(max_depth=15, min_samples_leaf=20, class_weight="balanced",
                                                                 random_state=0)),
        "Random forest": ("alberi", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                           n_jobs=n_job, random_state=0)),
        "Extra trees": ("alberi", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                       n_jobs=n_job, random_state=0)),
        "Gradient boosting": ("alberi", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced",
                                                                       early_stopping=True, random_state=0)),
        "XGBoost": ("alberi", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                                            scale_pos_weight=rapporto, n_jobs=n_job, random_state=0)),
        "AdaBoost": ("alberi", AdaBoostClassifier(n_estimators=200, random_state=0)),
        "kNN (PCA 128)": ("vicini", make_pipeline(PCA(n_components=128, random_state=0),
                                                  KNeighborsClassifier(n_neighbors=15, n_jobs=n_job))),
        "Naive Bayes gaussiano": ("probabilistico", GaussianNB()),
        "QDA (PCA 128)": ("probabilistico", make_pipeline(PCA(n_components=128, random_state=0),
                                                          QuadraticDiscriminantAnalysis(reg_param=0.1))),
        "SVM RBF (Nystroem 2000)": ("kernel", make_pipeline(Nystroem(n_components=2000, random_state=0),
                                                            LinearSVC(C=0.1, class_weight="balanced", max_iter=5000))),
        "SVM RBF (esatta)": ("kernel", SVC(C=1.0, gamma="scale", class_weight="balanced", cache_size=2000)),
        "MLP (256)": ("rete", MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=0)),
    }


class Sottoinsieme:
    """Allena il modello su al massimo n clip prese a caso, con le stesse proporzioni di bonafide e spoof."""
    def __init__(self, modello, n):
        self.modello, self.n = modello, n

    def fit(self, x, y):
        rng = np.random.default_rng(0)
        tieni = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), round(self.n * np.mean(y == c)), replace=False)
                                        for c in (0, 1)]))
        self.modello.fit(x[tieni], y[tieni])
        return self

    def punteggio(self, x):
        return punteggio(self.modello, x)


class DueGMM:
    """Una GMM per i bonafide e una per gli spoof (PCA a 128, diagonali, 64 componenti):
    punteggio = log-verosimiglianza bonafide - log-verosimiglianza spoof (un LLR vero e proprio)."""
    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        self.pca = PCA(n_components=128, random_state=0).fit(self.scaler.transform(x))
        z = self.pca.transform(self.scaler.transform(x))
        self.gmm = [GaussianMixture(64, covariance_type="diag", reg_covar=1e-3, random_state=0).fit(z[y == c]) for c in (0, 1)]
        return self

    def punteggio(self, x):
        z = self.pca.transform(self.scaler.transform(x))
        return self.gmm[0].score_samples(z) - self.gmm[1].score_samples(z)


class SoloBonafide:
    """Anomaly detection: si allena solo sui bonafide del train, punteggio = normalità
    (decision_function: più alto = più simile ai bonafide visti)."""
    def __init__(self, modello):
        self.modello = make_pipeline(StandardScaler(), modello)

    def fit(self, x, y):
        self.modello.fit(x[y == 0])
        return self

    def punteggio(self, x):
        return self.modello.decision_function(x)


class Sottocampionato:
    """Invece di class_weight: si tengono tutti i bonafide e altrettanti spoof presi a caso (1:1)."""
    def __init__(self, modello):
        self.modello = make_pipeline(StandardScaler(), modello)

    def fit(self, x, y):
        bona, spoof = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
        spoof = np.random.default_rng(0).choice(spoof, size=len(bona), replace=False)
        tieni = np.sort(np.concatenate([bona, spoof]))
        self.modello.fit(x[tieni], y[tieni])
        return self

    def punteggio(self, x):
        return punteggio(self.modello, x)


class Media:
    """Voting: media dei punteggi standardizzati (media e deviazione sul train) di modelli già allenati."""
    def __init__(self, modelli, x_tr):
        self.modelli = modelli
        p = [punteggio(m, x_tr) for m in modelli]
        self.media, self.dev_std = [q.mean() for q in p], [q.std() + 1e-9 for q in p]

    def punteggio(self, x):
        return np.mean([(punteggio(m, x) - mu) / sd for m, mu, sd in zip(self.modelli, self.media, self.dev_std)], axis=0)


class SommaPesata:
    """Come Spoof-SUPERB: somma pesata di tutti i layer (pesi appresi, softmax) + MLP piccolo, in PyTorch
    sulla CPU. Ogni layer è standardizzato con media e deviazione del train. L'epoca si sceglie sul dev
    (EER più basso). Punteggio = logit di bonafide."""
    def __init__(self, n_layer, epoche=30):
        self.n_layer, self.epoche = n_layer, epoche   # con train grandi le epoche calano (vedi fit)

    def _tensore(self, x):
        x = torch.from_numpy(x).view(len(x), self.n_layer, -1)
        return (x - self.media) / self.dev_std

    def _logit(self, x):
        pesi = torch.softmax(self.pesi, dim=0)
        return self.mlp((x * pesi[None, :, None]).sum(dim=1)).squeeze(1)

    def fit(self, x, y, x_dev, y_dev, righe=None):
        """x: le feature della lista (memmap, tutti i layer); righe: le clip da usare (None = tutte)."""
        torch.manual_seed(0)
        tutti = list(range(self.n_layer))
        if len(y) <= MAX_RAM:   # train piccolo (2019 LA): tutto in RAM
            x = torch.from_numpy(layer(x, tutti, righe)).view(len(y), self.n_layer, -1)
            self.media, self.dev_std = x.mean(dim=0), x.std(dim=0) + 1e-5
            x.sub_(self.media).div_(self.dev_std)   # sul posto: con tutti i layer di XLS-R il train occupa 2,6 GB
            prendi = lambda i: x[i]
        else:                   # train grande (ASVspoof 5 intero): un batch alla volta dal disco
            righe = np.arange(len(y)) if righe is None else righe
            campione = np.sort(np.random.default_rng(0).choice(righe, MAX_RAM, replace=False))
            c = torch.from_numpy(layer(x, tutti, campione)).view(MAX_RAM, self.n_layer, -1)
            self.media, self.dev_std = c.mean(dim=0), c.std(dim=0) + 1e-5   # statistiche su 30k clip a caso
            del c
            prendi = lambda i: self._tensore(layer(x, tutti, righe[np.sort(i.numpy())]))
            ordina = lambda i: i[np.argsort(righe[i.numpy()], kind="stable")]   # stesso ordine di prendi
            self.epoche = max(3, round(self.epoche * MAX_RAM / len(y)))   # circa gli stessi passi di ottimizzazione
        bona = torch.from_numpy((y == 0).astype(np.float32))   # bersaglio 1 = bonafide
        self.pesi = torch.nn.Parameter(torch.zeros(self.n_layer))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.media.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                       torch.nn.Linear(256, 1))
        ottimizzatore = torch.optim.Adam([self.pesi, *self.mlp.parameters()], lr=1e-3, weight_decay=1e-4)
        perdita = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y == 1).sum() / (y == 0).sum()))  # classi bilanciate
        migliore = (np.inf, None)
        for epoca in range(self.epoche):
            self.mlp.train()
            for i in torch.randperm(len(y)).split(256):
                if len(y) > MAX_RAM:
                    i = ordina(i)   # dal disco si legge in ordine crescente
                ottimizzatore.zero_grad()
                perdita(self._logit(prendi(i)), bona[i]).backward()
                ottimizzatore.step()
            e = 100 * metriche.eer(y_dev, self.punteggio(x_dev))
            if e < migliore[0]:
                migliore = (e, epoca, {k: v.clone() for k, v in self.mlp.state_dict().items()}, self.pesi.detach().clone())
        self.eer_dev, self.epoca = migliore[0], migliore[1]
        self.mlp.load_state_dict(migliore[2])
        self.pesi = torch.nn.Parameter(migliore[3])
        print(f"   epoca scelta sul dev: {self.epoca + 1}, pesi dei layer: {np.round(torch.softmax(self.pesi, 0).detach().numpy(), 3)}")
        return self, f"epoca {self.epoca + 1}"

    def salva(self, file):
        torch.save({"n_layer": self.n_layer, "pesi": self.pesi.detach(), "mlp": self.mlp.state_dict(),
                    "media": self.media, "dev_std": self.dev_std, "epoca": self.epoca}, file)

    @staticmethod
    def da_file(file):
        """Ricarica un modello salvato con salva()."""
        d = torch.load(file)
        m = SommaPesata(d["n_layer"])
        m.media, m.dev_std, m.epoca, m.pesi = d["media"], d["dev_std"], d["epoca"], torch.nn.Parameter(d["pesi"])
        m.mlp = torch.nn.Sequential(torch.nn.Linear(m.media.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                    torch.nn.Linear(256, 1))
        m.mlp.load_state_dict(d["mlp"])
        return m

    def punteggio(self, x):
        self.mlp.eval()
        with torch.no_grad():
            return np.concatenate([self._logit(self._tensore(np.ascontiguousarray(x[i:i + 4096]))).numpy()
                                   for i in range(0, len(x), 4096)])


def carica(cartella, lista, n_max):
    """Feature (memmap, non lette subito) ed etichette (0 = bonafide, 1 = spoof)."""
    x = np.load(cartella / f"{lista}.npy", mmap_mode="r")
    y = np.load(cartella / f"{lista}_label.npy")
    if n_max:   # prova: un sottoinsieme fisso, con entrambe le classi
        scelte = np.sort(np.random.default_rng(0).permutation(len(y))[:n_max])
        return x, y[scelte], scelte
    return x, y, None


def layer(x, l, righe=None, inizio=0, fine=None):
    """Un layer (float32) di un pezzo della lista; con una lista di layer, la loro concatenazione."""
    pezzo = x[righe[inizio:fine]] if righe is not None else x[inizio:fine]
    return np.asarray(pezzo[:, l], dtype=np.float32).reshape(len(pezzo), -1)


def punteggio(modello, x):
    """Punteggio alto = bonafide: log-odds di bonafide contro spoof (con i pesi bilanciati
    è vicino a un log-likelihood ratio, quindi actDCF e Cllr hanno senso)."""
    if hasattr(modello, "punteggio"):          # le classi qui sopra
        return modello.punteggio(x)
    if hasattr(modello, "decision_function"):
        return -modello.decision_function(x)   # la classe "positiva" è 1 = spoof
    p = np.clip(modello.predict_proba(x), 1e-6, 1 - 1e-6)
    return np.log(p[:, 0]) - np.log(p[:, 1])


def scegli_su_dev(candidati, x_tr, y_tr, x_dev, y_dev):
    """Allena ogni candidato {valore: modello} e tiene quello con l'EER dev più basso."""
    migliore = None
    for valore, modello in candidati.items():
        modello.fit(x_tr, y_tr)
        e = 100 * metriche.eer(y_dev, punteggio(modello, x_dev))
        print(f"   {valore}: EER dev {e:.2f}%")
        if migliore is None or e < migliore[2]:
            migliore = (valore, modello, e)
    return migliore[1], migliore[0]


def punteggi_lista(modelli, quali_layer, x, righe):
    """Punteggi di tutti i modelli su una lista, un blocco alla volta: così ogni blocco
    si legge dal disco una volta sola (e ogni layer si estrae una volta sola per blocco)."""
    n = len(righe) if righe is not None else len(x)
    punteggi = {nome: [] for nome in modelli}
    for i in range(0, n, BLOCCO):
        pezzi = {}
        for nome, modello in modelli.items():
            chiave = str(quali_layer[nome])
            if chiave not in pezzi:
                pezzi[chiave] = layer(x, quali_layer[nome], righe, i, i + BLOCCO)
            punteggi[nome].append(punteggio(modello, pezzi[chiave]))
    return {nome: np.concatenate(p) for nome, p in punteggi.items()}


def salva_modello(cartella, nome, modello):
    """Salva un modello allenato (pipeline intera, scaler e PCA compresi) e restituisce il nome del file."""
    base = re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")
    if isinstance(modello, SommaPesata):
        modello.salva(cartella / f"{base}.pt")
        return f"{base}.pt"
    joblib.dump(modello, cartella / f"{base}.joblib", compress=3)
    return f"{base}.joblib"


def salva_punteggi(file, punteggi, y, righe):
    """Punteggi per clip di tutti i modelli su una lista: matrice (n_clip, n_modelli) + nomi e label."""
    extra = {} if righe is None else {"righe": righe}
    np.savez(file, nomi=np.array(list(punteggi)), punteggi=np.stack(list(punteggi.values()), axis=1).astype(np.float32),
             label=y, **extra)


def purezza(vere, cluster):
    """Frazione di clip che cadono nel gruppo più frequente del proprio cluster."""
    m = contingency_matrix(vere, cluster)
    return m.max(axis=0).sum() / m.sum()


def tabella_markdown(df):
    righe = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        righe.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(righe) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ssl", help="wavlm-base, hubert-base-ls960, wav2vec2-xls-r-300m, ...")
    parser.add_argument("n_max", nargs="?", type=int, help="prova veloce: al massimo n_max clip per lista")
    parser.add_argument("--train", default=TRAIN, help="lista di training (default 2019 LA train)")
    parser.add_argument("--dev", default=DEV, help="lista per le scelte (default 2019 LA dev)")
    arg = parser.parse_args()
    ssl, n_max = arg.ssl, arg.n_max
    n_job = 2 if n_max else N_JOB
    cartella = percorsi.PROGETTO / "artifacts" / "feature_ssl" / ssl
    nome_uscita = ssl + ("" if arg.train == TRAIN else f"_train-{arg.train}") + ("_prova" if n_max else "")
    uscita = percorsi.PROGETTO / "artifacts" / "classificatori_leggeri" / nome_uscita
    uscita.mkdir(parents=True, exist_ok=True)
    inizio_tutto = time.time()

    x_tr, y_tr, r_tr = carica(cartella, arg.train, n_max)
    x_dev, y_dev, r_dev = carica(cartella, arg.dev, n_max)
    n_layer = x_tr.shape[1]
    print(f"{ssl}: {n_layer} layer da {x_tr.shape[2]}, train {arg.train} {len(y_tr)} clip, dev {arg.dev} {len(y_dev)} clip")

    # 1. scelta del layer con la regressione logistica (veloce), EER sul dev;
    #    a parità (entro SPAREGGIO punti) vince la log-loss dev più bassa
    eer_dev, ll_dev = [], []
    for l in range(n_layer):
        modello = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        modello.fit(layer(x_tr, l, r_tr), y_tr)
        xd = layer(x_dev, l, r_dev)
        eer_dev.append(100 * metriche.eer(y_dev, punteggio(modello, xd)))
        ll_dev.append(log_loss(y_dev, modello.predict_proba(xd)))
        print(f"layer {l:2d}: EER dev {eer_dev[-1]:.2f}%, log-loss dev {ll_dev[-1]:.4f}")
    pari = [l for l in range(n_layer) if eer_dev[l] <= min(eer_dev) + SPAREGGIO]
    migliore = min(pari, key=lambda l: ll_dev[l])
    tre_migliori = sorted(range(n_layer), key=lambda l: (eer_dev[l], ll_dev[l]))[:3]
    pd.DataFrame({"layer": range(n_layer), "EER dev (%)": eer_dev, "log-loss dev": ll_dev}).to_csv(uscita / "layer_dev.csv", index=False)
    print(f"layer migliore: {migliore} (EER dev {eer_dev[migliore]:.2f}%, log-loss dev {ll_dev[migliore]:.4f}; "
          f"layer pari entro {SPAREGGIO} punti: {pari}); 3 migliori: {tre_migliori}")

    # 2. classificatori sul layer migliore, scelta e controllo sul dev
    xl_tr, xl_dev = layer(x_tr, migliore, r_tr), layer(x_dev, migliore, r_dev)
    modelli, quali_layer, righe_tabella, punteggi_dev, file_modelli = {}, {}, {}, {}, {}
    cartella_modelli, cartella_punteggi = uscita / "modelli", uscita / "punteggi"
    cartella_modelli.mkdir(exist_ok=True)
    cartella_punteggi.mkdir(exist_ok=True)

    def aggiungi(nome, famiglia, allena, quale=migliore, scelta="-", scorta=False):
        """Allena (allena() restituisce il modello già allenato, ed eventualmente la scelta fatta sul dev),
        misura il tempo e l'EER dev, e mette il modello nella lista da testare."""
        t = time.time()
        print(f"{nome} ...")
        risultato = allena()
        modello, scelta = risultato if isinstance(risultato, tuple) else (risultato, scelta)
        xd = xl_dev if quale == migliore else layer(x_dev, quale, r_dev)
        modelli[nome], quali_layer[nome] = modello, quale
        punteggi_dev[nome] = punteggio(modello, xd)
        file_modelli[nome] = salva_modello(cartella_modelli, nome, modello)
        righe_tabella[nome] = {"tabella": "scorta" if scorta else "standard", "classificatore": nome, "famiglia": famiglia,
                               "layer": quale if isinstance(quale, int) else
                                        "tutti" if len(quale) == n_layer else "+".join(map(str, quale)),
                               "scelta dev": scelta, "minuti training": round((time.time() - t) / 60, 2),
                               "EER dev (%)": float(100 * metriche.eer(y_dev, punteggi_dev[nome]))}
        print(righe_tabella[nome])

    # A) classificatori con iperparametri fissi
    for nome, (famiglia, clf) in classificatori(n_job, (y_tr == 0).sum() / (y_tr == 1).sum()).items():
        if nome in LENTI and len(y_tr) > MAX_LENTI:
            aggiungi(nome, famiglia, lambda: Sottoinsieme(make_pipeline(StandardScaler(), clf), MAX_LENTI).fit(xl_tr, y_tr),
                     scelta=f"train {MAX_LENTI} clip")
        else:
            aggiungi(nome, famiglia, lambda: make_pipeline(StandardScaler(), clf).fit(xl_tr, y_tr))
    aggiungi("Due GMM (PCA 128, LLR)", "probabilistico", lambda: DueGMM().fit(xl_tr, y_tr))
    aggiungi("Regressione logistica (C sul dev)", "lineare", lambda: scegli_su_dev(
        {f"C={c}": make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))
         for c in VALORI_C}, xl_tr, y_tr, xl_dev, y_dev))
    aggiungi("Voting (RF + GB + SVM lineare)", "ensemble", lambda: Media(
        [modelli["Random forest"], modelli["Gradient boosting"], modelli["SVM lineare"]], xl_tr))

    # B) di scorta, fuori tabella: anomaly detection, solo bonafide del train
    aggiungi("Isolation forest (solo bonafide)", "anomaly", lambda: SoloBonafide(
        IsolationForest(n_estimators=300, n_jobs=n_job, random_state=0)).fit(xl_tr, y_tr), scorta=True)
    aggiungi("One-class SVM (Nystroem, solo bonafide)", "anomaly", lambda: SoloBonafide(make_pipeline(
        Nystroem(n_components=2000, random_state=0), SGDOneClassSVM(nu=0.1, random_state=0))).fit(xl_tr, y_tr), scorta=True)

    # C) selezione delle feature: le k più discriminanti (test F), k scelto sul dev
    aggiungi("SelectKBest + regressione logistica", "selezione feature", lambda: scegli_su_dev(
        {f"k={k}": make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k),
                                 LogisticRegression(max_iter=2000, class_weight="balanced"))
         for k in VALORI_K if k < xl_tr.shape[1]}, xl_tr, y_tr, xl_dev, y_dev))

    # D) sottocampionamento degli spoof (1:1) invece di class_weight
    aggiungi("Regressione logistica (sottocampionata)", "campionamento",
             lambda: Sottocampionato(LogisticRegression(max_iter=2000)).fit(xl_tr, y_tr))
    aggiungi("Random forest (sottocampionata)", "campionamento", lambda: Sottocampionato(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=n_job, random_state=0)).fit(xl_tr, y_tr))

    # E) multi-layer: concatenazione dei 3 layer migliori sul dev
    aggiungi("Regressione logistica (3 layer)", "multi-layer", lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(layer(x_tr, tre_migliori, r_tr), y_tr),
        quale=tre_migliori)
    # F) somma pesata appresa di tutti i layer + MLP (Spoof-SUPERB), epoca scelta sul dev
    tutti = list(range(n_layer))
    aggiungi("Somma pesata dei layer + MLP", "multi-layer", lambda: SommaPesata(n_layer).fit(
        x_tr, y_tr, layer(x_dev, tutti, r_dev), y_dev, righe=r_tr), quale=tutti)

    # G) riferimento dalla letteratura (Pascu et al. 2023): ultimo layer, regressione logistica
    #    quasi senza regolarizzazione (C=1e6) e senza scaler; qui con il solo train, come le altre righe
    ultimo = n_layer - 1
    aggiungi("Regressione logistica (Pascu: ultimo layer, C=1e6)", "lineare", lambda: LogisticRegression(
        C=1e6, max_iter=1000).fit(layer(x_tr, ultimo, r_tr), y_tr), quale=ultimo)
    pd.DataFrame({"classificatore": list(file_modelli), "file": list(file_modelli.values()),
                  "layer": [str(quali_layer[n]) for n in file_modelli]}).to_csv(cartella_modelli / "indice.csv", index=False)
    salva_punteggi(cartella_punteggi / "dev.npz", punteggi_dev, y_dev, r_dev)
    print(f"training finito in {(time.time() - inizio_tutto) / 60:.1f} minuti")

    # di scorta: F) clustering (analisi, non classificatore): k-means sul train, layer migliore
    x_cl = StandardScaler().fit_transform(xl_tr)
    vere = {"label": y_tr}
    manifest = percorsi.PROGETTO / "manifests" / f"{arg.train}.csv"
    if manifest.exists():   # gli attacchi ci sono solo se c'è il manifest della lista di train
        attacco = pd.read_csv(manifest)["attacco"].to_numpy()
        if len(attacco) == len(x_tr):
            vere["attacco"] = attacco if r_tr is None else attacco[r_tr]
    k_attacchi = len(set(vere["attacco"])) if "attacco" in vere else 2   # attacchi + bonafide
    risultati_cl = []
    for k in sorted({2, k_attacchi}):
        cluster = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x_cl)
        for nome_vere, v in vere.items():
            risultati_cl.append({"k": k, "rispetto a": nome_vere, "purezza": float(purezza(v, cluster)),
                                 "ARI": float(adjusted_rand_score(v, cluster))})
            print(risultati_cl[-1])
    pd.DataFrame(risultati_cl).to_csv(uscita / "clustering.csv", index=False)

    # 3. test: ogni lista si legge una volta e si danno i punteggi con tutti i modelli
    for lista in TEST:
        if not (cartella / f"{lista}.npy").exists():   # feature non ancora estratte: si salta
            print(f"{lista}: feature mancanti, saltata")
            continue
        t = time.time()
        x, y, righe = carica(cartella, lista, n_max)
        punteggi = punteggi_lista(modelli, quali_layer, x, righe)
        salva_punteggi(cartella_punteggi / f"{lista}.npz", punteggi, y, righe)
        for nome, p in punteggi.items():
            m = metriche.tutte(y, p)
            righe_tabella[nome][f"{lista} EER (%)"] = m["EER (%)"]
            righe_tabella[nome][f"{lista} minDCF"] = m["minDCF"]
        print(f"{lista}: {len(y)} clip, {(time.time() - t) / 60:.1f} minuti")
        df = pd.DataFrame(list(righe_tabella.values()))   # salvate a ogni lista, così non si perde niente
        standard = df[df["tabella"] == "standard"].drop(columns="tabella")
        scorta = df[df["tabella"] == "scorta"].drop(columns="tabella")
        standard.to_csv(uscita / "risultati.csv", index=False)
        (uscita / "risultati.md").write_text(tabella_markdown(standard), encoding="utf-8")
        scorta.to_csv(uscita / "risultati_scorta.csv", index=False)
        (uscita / "risultati_scorta.md").write_text(tabella_markdown(scorta), encoding="utf-8")
    colonne = ["classificatore", "famiglia"] + [c for c in df.columns if "EER" in c]
    print("Tabella:\n" + standard[colonne].round(2).to_string(index=False))
    print("Di scorta:\n" + scorta[colonne].round(2).to_string(index=False))
    print(f"Tutto finito in {(time.time() - inizio_tutto) / 60:.1f} minuti:", uscita)
