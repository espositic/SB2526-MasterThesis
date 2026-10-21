# Tabelle dei risultati, a partire dai file punteggi_<lista>.csv degli esperimenti.

import json
import os
import shutil

import pandas as pd

from tesi import percorsi
from tesi.metriche import eer_per_gruppo, intervallo_eer, tutte


def leggi_esperimenti():
    """Tutti i seed in artifacts/esperimenti, con le informazioni di info.json e completato.json."""
    righe = []
    for file_info in sorted(percorsi.ESPERIMENTI.glob("*/seed*/info.json")):
        cartella = file_info.parent
        info = json.loads(file_info.read_text(encoding="utf-8"))
        riga = {"modello": info["modello"], "train": info["dataset"], "augmentation": info["augmentation"],
                "seed": info["seed"], "cartella": cartella}
        if (cartella / "completato.json").exists():
            riga.update(json.loads((cartella / "completato.json").read_text(encoding="utf-8")))
        righe.append(riga)
    return pd.DataFrame(righe)


def tabella_seed(con_intervallo=False):
    """Una riga per ogni (esperimento, seed, lista di test) con EER, minDCF, actDCF e Cllr."""
    righe = []
    for _, esp in leggi_esperimenti().iterrows():
        for file in sorted(esp["cartella"].glob("punteggi_*.csv")):
            if file.name.endswith(".parziale.csv"):
                continue
            p = pd.read_csv(file, keep_default_na=False)
            y = (p["label"] == "spoof").astype(int)
            riga = {"modello": esp["modello"], "train": esp["train"], "augmentation": esp["augmentation"],
                    "seed": esp["seed"], "test": file.stem.removeprefix("punteggi_"), "clip": len(p),
                    **tutte(y, p["punteggio"])}
            if con_intervallo:   # lento sulle liste grandi
                riga["EER IC 95%"] = intervallo_eer(y, p["punteggio"]).round(2).tolist()
            righe.append(riga)
    return pd.DataFrame(righe)


def tabella_media(tabella):
    """Media e deviazione standard sui seed (EER in %)."""
    gruppi = tabella.groupby(["modello", "train", "augmentation", "test"])
    media = gruppi[["EER (%)", "minDCF", "actDCF", "Cllr"]].mean().round(3)
    media["EER std"] = gruppi["EER (%)"].std().round(2)
    media["seed"] = gruppi["seed"].count()
    return media.reset_index()


NOMI_MODELLI = {"mambo": "XLSR-MamBo", "nes2net": "Nes2Net-X", "mhfa": "WavLM + MHFA", "tfparn": "TFPARN"}


def salva_excel(per_seed, file):
    """
    Tabella Excel dei risultati (serve openpyxl). Tre fogli:
    - Riepilogo: media ± deviazione standard sui seed, una riga per esperimento e lista di test
    - Per seed: tutte le metriche di ogni seed
    - Dev: EER sul dev dell'epoca migliore di ogni seed
    """
    gruppi = per_seed.groupby(["modello", "train", "augmentation", "test"])
    riepilogo = gruppi.agg(seed=("seed", "count"),
                           eer=("EER (%)", "mean"), eer_std=("EER (%)", "std"),
                           mindcf=("minDCF", "mean"), mindcf_std=("minDCF", "std"),
                           actdcf=("actDCF", "mean"), cllr=("Cllr", "mean")).reset_index()
    riepilogo["modello"] = riepilogo["modello"].map(NOMI_MODELLI).fillna(riepilogo["modello"])
    riepilogo = riepilogo.sort_values(["test", "train", "eer"]).round(3)
    riepilogo.columns = ["Modello", "Train", "Augmentation", "Test", "Seed", "EER (%)", "EER std",
                         "minDCF", "minDCF std", "actDCF", "Cllr"]

    esperimenti = leggi_esperimenti().drop(columns="cartella")
    with pd.ExcelWriter(file) as excel:
        for nome, tabella in [("Riepilogo", riepilogo), ("Per seed", per_seed.round(3)), ("Dev", esperimenti)]:
            tabella.to_excel(excel, sheet_name=nome, index=False)
            foglio = excel.sheets[nome]
            foglio.freeze_panes = "A2"
            for colonna in foglio.columns:   # larghezza delle colonne in base al testo
                foglio.column_dimensions[colonna[0].column_letter].width = max(len(str(c.value)) for c in colonna) + 2
    print("Salvato:", file)


def matrice(tabella, metrica="EER (%)"):
    """
    Tabella train x test: righe = modello, dataset di train e augmentation; colonne = liste di test.
    Si passa tabella_media(...) per avere la media sui seed.
    """
    return tabella.pivot_table(index=["modello", "train", "augmentation"], columns="test", values=metrica).round(2)


def eer_attacchi(cartella, nome_lista, colonna="attacco"):
    """EER (%) per attacco (o per codec, con colonna="codec") di un seed."""
    p = pd.read_csv(cartella / f"punteggi_{nome_lista}.csv", keep_default_na=False)
    return eer_per_gruppo((p["label"] == "spoof").astype(int), p["punteggio"], p[colonna])


# ---------------------------------------------------------------------------
# Esperimenti fatti con il codice vecchio (artifacts/runs/<nome>/seed<k>)
# ---------------------------------------------------------------------------

def importa_run_vecchia(nome_vecchio, modello, augmentation):
    """
    Porta un esperimento su ASVspoof 5 fatto con il codice vecchio nella struttura nuova:
    pesi del modello migliore, punteggi su dev ed eval (sottoinsiemi) e log.
    Esempio: importa_run_vecchia("mambo_rawboost_asvspoof5", "mambo", "rawboost")
    """
    cartelle = sorted((percorsi.PROGETTO / "artifacts" / "runs" / nome_vecchio).glob("seed*"))
    assert cartelle, f"nessun seed in artifacts/runs/{nome_vecchio}"
    for vecchia in cartelle:
        nuova = percorsi.ESPERIMENTI / f"{modello}_asvspoof5_{augmentation}" / vecchia.name
        nuova.mkdir(parents=True, exist_ok=True)
        seed = int(vecchia.name.removeprefix("seed"))
        info = {"modello": modello, "dataset": "asvspoof5", "augmentation": augmentation, "seed": seed,
                "importato_da": f"artifacts/runs/{nome_vecchio}/{vecchia.name}"}
        (nuova / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        if not (nuova / "migliore.pt").exists():
            try:
                os.link(vecchia / "model_best.pt", nuova / "migliore.pt")   # stesso file, nessuna copia
            except OSError:
                shutil.copy(vecchia / "model_best.pt", nuova / "migliore.pt")
        shutil.copy(vecchia / "log.csv", nuova / "log_vecchio.csv")
        metriche = json.loads((vecchia / "metrics.json").read_text(encoding="utf-8"))
        fine = {"epoca_migliore": metriche["epoca_migliore"], "eer_dev": 100 * metriche["dev"]["eer"]}
        (nuova / "completato.json").write_text(json.dumps(fine, indent=2), encoding="utf-8")
        for split in ("dev", "eval"):
            converti_punteggi(vecchia / f"scores_{split}.csv", nuova / f"punteggi_asvspoof5_{split}.csv")
        print("Importato:", nuova)


def converti_punteggi(vecchio, nuovo):
    """scores_<split>.csv del codice vecchio -> punteggi_<lista>.csv con le colonne nuove."""
    v = pd.read_csv(vecchio, keep_default_na=False)
    cartella_audio = v["file_name"].str[0].map({"T": "flac_T", "D": "flac_D", "E": "flac_E_eval"})
    pd.DataFrame({
        "percorso": "asvspoof5/" + cartella_audio + "/" + v["file_name"] + ".flac",
        "label": v["label"],
        "attacco": v["attack_label"].where(v["label"] == "spoof", "-"),
        "codec": v["codec"],
        "punteggio": v["score"],
    }).to_csv(nuovo, index=False)
