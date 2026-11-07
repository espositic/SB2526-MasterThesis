# Intervalli di confidenza al 95% dell'EER (bootstrap, ricampionando dentro ogni classe) per i
# classificatori leggeri (training su 2019 LA) e per i modelli fine-tunati (2019 LA + RawBoost),
# più il bootstrap appaiato della differenza tra il miglior leggero e ogni fine-tunato: con gli
# stessi indici ricampionati per i due sistemi, la differenza è significativa se l'intervallo non
# contiene lo 0.
#
#   python intervalli.py          (n = 1000 ricampionamenti)
#
# Leggeri: per ogni SSL il classificatore con l'EER medio più basso sui 5 test, più il migliore di
# ogni lista (scelto guardando il test: è ottimistico, va detto). Fine-tunati: punteggi_<lista>.csv
# copiati dal fisso in artifacts/punteggi_finetunati/<esperimento>/.
# Risultato in artifacts/classificatori_leggeri/intervalli.csv e intervalli.md.
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tesi import metriche, percorsi

CARTELLA = percorsi.PROGETTO / "artifacts" / "classificatori_leggeri"
FINETUNATI = {"MamBo": "mambo_asvspoof2019_rawboost", "Nes2Net-X": "nes2net_asvspoof2019_rawboost",
              "MHFA": "mhfa_asvspoof2019_rawboost"}
SSL = {"wavlm-base": "WavLM Base", "hubert-base-ls960": "HuBERT Base", "wav2vec2-xls-r-300m": "XLS-R 300M",
       "wavlm-large": "WavLM Large", "hubert-large-ll60k": "HuBERT Large"}
LISTE = {"inthewild": "In-the-Wild", "asvspoof2019_eval": "2019 LA eval", "asvspoof2021la_eval": "2021 LA",
         "asvspoof2021df_eval": "2021 DF", "asvspoof5_eval_completo": "ASVspoof 5"}
N = 1000


def eer_veloce(label, punteggi, idx):
    """EER (%) su un ricampionamento (idx), senza copiare due volte gli array."""
    return 100 * metriche.eer(label[idx], punteggi[idx])


def bootstrap(label, sistemi, n=N, seed=0):
    """EER e intervallo al 95% di ogni sistema, e della differenza di ognuno rispetto al primo,
    con gli stessi ricampionamenti per tutti (bootstrap appaiato)."""
    rng = np.random.default_rng(seed)
    i_bona, i_spoof = np.flatnonzero(label == 0), np.flatnonzero(label == 1)
    valori = {nome: [] for nome in sistemi}
    for _ in range(n):
        idx = np.concatenate([rng.choice(i_bona, len(i_bona)), rng.choice(i_spoof, len(i_spoof))])
        for nome, p in sistemi.items():
            valori[nome].append(eer_veloce(label, p, idx))
    primo = np.array(valori[next(iter(sistemi))])
    righe = []
    for nome, p in sistemi.items():
        v = np.array(valori[nome])
        d = np.percentile(v - primo, [2.5, 97.5])
        righe.append({"sistema": nome, "EER (%)": 100 * metriche.eer(label, p),
                      "IC 2.5%": np.percentile(v, 2.5), "IC 97.5%": np.percentile(v, 97.5),
                      "diff. dal primo, IC": f"[{d[0]:+.2f}, {d[1]:+.2f}]",
                      "significativa": "-" if nome == next(iter(sistemi)) else ("sì" if d[0] > 0 or d[1] < 0 else "no")})
    return righe


def sistemi_lista(lista, migliori_media):
    """I punteggi di tutti i sistemi su una lista, nello stesso ordine di clip (quello del manifest)."""
    label, sistemi = None, {}
    # il migliore leggero su questa lista (per EER) va per primo: è quello con cui si confrontano gli altri
    candidati = []
    for ssl, nome_ssl in SSL.items():
        file = CARTELLA / ssl / "punteggi" / f"{lista}.npz"
        if not file.exists():
            continue
        d = np.load(file)
        if label is None:
            label = d["label"]
        assert (d["label"] == label).all(), f"etichette diverse: {ssl} {lista}"
        nomi = list(d["nomi"])
        eer = {n: metriche.eer(label, d["punteggi"][:, i]) for i, n in enumerate(nomi)}
        migliore = min(eer, key=eer.get)
        candidati.append((eer[migliore], f"{nome_ssl} · {migliore}", d["punteggi"][:, nomi.index(migliore)]))
        scelto = migliori_media[ssl]
        sistemi[f"{nome_ssl} · {scelto} (miglior media)"] = d["punteggi"][:, nomi.index(scelto)]
    candidati.sort(key=lambda c: c[0])
    sistemi = {f"{candidati[0][1]} (migliore su questa lista)": candidati[0][2], **sistemi}
    for nome, cartella in FINETUNATI.items():
        file = percorsi.PROGETTO / "artifacts" / "punteggi_finetunati" / cartella / f"punteggi_{lista}.csv"
        p = pd.read_csv(file, keep_default_na=False)
        assert ((p["label"] == "spoof").astype(int).to_numpy() == label).all(), f"ordine diverso: {nome} {lista}"
        sistemi[f"{nome} (fine-tuning + RawBoost)"] = p["punteggio"].to_numpy(dtype=float)
    return label, sistemi


if __name__ == "__main__":
    # per ogni SSL, il classificatore con l'EER medio più basso sui 5 test
    migliori_media = {}
    for ssl in SSL:
        df = pd.read_csv(CARTELLA / ssl / "risultati.csv")
        colonne = [f"{l} EER (%)" for l in LISTE]
        migliori_media[ssl] = df.loc[df[colonne].mean(axis=1).idxmin(), "classificatore"]
    print("miglior media per SSL:", migliori_media)

    def una_lista(lista):
        label, sistemi = sistemi_lista(lista, migliori_media)
        righe = bootstrap(label, sistemi)
        for r in righe:
            r["lista"] = LISTE[lista]
        print(f"{lista}: fatto")
        return righe

    risultati = Parallel(n_jobs=len(LISTE))(delayed(una_lista)(l) for l in LISTE)
    df = pd.DataFrame([r for righe in risultati for r in righe])
    df = df[["lista", "sistema", "EER (%)", "IC 2.5%", "IC 97.5%", "diff. dal primo, IC", "significativa"]]
    df.to_csv(CARTELLA / "intervalli.csv", index=False)
    testo = ["# Intervalli di confidenza al 95% dell'EER (bootstrap, 1000 ricampionamenti per classe)\n",
             "Training su 2019 LA. Il primo sistema di ogni lista è il miglior leggero su quella lista (scelto "
             "guardando il test, quindi ottimistico); la differenza è appaiata (stessi ricampionamenti).\n"]
    for lista, gruppo in df.groupby("lista", sort=False):
        testo.append(f"\n## {lista}\n\n| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |\n|---|---|---|---|---|")
        for _, r in gruppo.iterrows():
            testo.append(f"| {r['sistema']} | {r['EER (%)']:.2f} | [{r['IC 2.5%']:.2f}, {r['IC 97.5%']:.2f}] | "
                         f"{r['diff. dal primo, IC']} | {r['significativa']} |")
    (CARTELLA / "intervalli.md").write_text("\n".join(testo) + "\n", encoding="utf-8")
    print(df.round(2).to_string(index=False))
