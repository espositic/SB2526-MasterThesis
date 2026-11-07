# Raccoglie i risultati dei classificatori leggeri dei tre modelli SSL in un unico Excel
# (artifacts/classificatori_leggeri/classificatori_leggeri.xlsx) e in un riepilogo Markdown.
#
#   python crea_excel.py
#
# Fogli, per ogni training (2019 LA e, se ci sono, ASVspoof 5 intero): Riepilogo EER (tutti i
# classificatori, tutti gli SSL, tutte le liste) e un foglio completo (EER e minDCF) per SSL;
# in comune Migliori per lista, Scorta (anomaly detection), Clustering, Layer dev (con la colonna "train").
# Nel riepilogo e nei fogli completi il valore migliore (il più basso) di ogni colonna è in verde.
import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from tesi import percorsi

CARTELLA = percorsi.PROGETTO / "artifacts" / "classificatori_leggeri"
SSL = {"wavlm-base": "WavLM Base", "hubert-base-ls960": "HuBERT Base", "wav2vec2-xls-r-300m": "XLS-R 300M",
       "wavlm-large": "WavLM Large", "hubert-large-ll60k": "HuBERT Large"}
# cartella dei risultati = <ssl><suffisso>; nome breve del training per i fogli
TRAINING = {"": ("2019 LA", ""), "_train-asvspoof5intero_train": ("ASVspoof 5 intero", " train ASV5")}
LISTE = {"EER dev (%)": "dev", "inthewild EER (%)": "In-the-Wild", "asvspoof2019_eval EER (%)": "2019 LA eval",
         "asvspoof2021la_eval EER (%)": "2021 LA", "asvspoof2021df_eval EER (%)": "2021 DF",
         "asvspoof5_eval_completo EER (%)": "ASVspoof 5"}
# classificatori da togliere dall'Excel e dal riepilogo (per ora nessuno)
# (restano in risultati.csv di ogni SSL)
ESCLUSI = []   # per esempio ["kNN (PCA 128)", "Naive Bayes gaussiano"]


def leggi(ssl, file):
    percorso = CARTELLA / ssl / file
    if not percorso.exists():
        return None
    df = pd.read_csv(percorso)
    return df[~df["classificatore"].isin(ESCLUSI)].reset_index(drop=True) if "classificatore" in df else df


def formatta(foglio, df, evidenzia, righe_titolo=1):
    """Intestazione in grassetto, larghezze, decimali; se evidenzia, minimo di ogni colonna EER/minDCF in verde."""
    foglio.freeze_panes = foglio.cell(row=righe_titolo + 1, column=2)
    for cella in foglio[righe_titolo]:
        cella.font = Font(bold=True)
    for j, colonna in enumerate(df.columns, start=1):
        lettera = get_column_letter(j)
        foglio.column_dimensions[lettera].width = max(12, min(42, len(str(colonna)) + 2))
        if not pd.api.types.is_float_dtype(df[colonna]):
            continue
        for i in range(len(df)):
            foglio.cell(row=righe_titolo + 1 + i, column=j).number_format = "0.000" if "minDCF" in str(colonna) else "0.00"
        if evidenzia and ("EER" in str(colonna) or "minDCF" in str(colonna)) and df[colonna].notna().any():
            i_min = int(df[colonna].fillna(float("inf")).to_numpy().argmin())
            cella = foglio.cell(row=righe_titolo + 1 + i_min, column=j)
            cella.fill, cella.font = PatternFill("solid", fgColor="C6EFCE"), Font(bold=True)
    foglio.column_dimensions["A"].width = 42


def tabelle(suffisso, train):
    """Riepilogo, fogli completi, migliori per lista e fogli di scorta per un training."""
    pezzi, migliori, completi, altri = [], [], {}, {"scorta": [], "clustering": [], "layer_dev": []}
    for ssl, nome in SSL.items():
        df = leggi(ssl + suffisso, "risultati.csv")
        if df is None:
            continue
        completi[nome] = df
        tab = df.set_index("classificatore")[[c for c in LISTE if c in df.columns]].rename(columns=LISTE)
        tab.columns = [f"{nome} (layer {df['layer'].iloc[0]}) | {c}" for c in tab.columns]
        pezzi.append(tab)
        for colonna, lista in LISTE.items():
            if colonna in df.columns and colonna != "EER dev (%)":
                r = df.loc[df[colonna].idxmin()]
                migliori.append({"train": train, "lista": lista, "SSL": nome, "classificatore": r["classificatore"],
                                 "famiglia": r["famiglia"], "EER (%)": r[colonna],
                                 "minDCF": r[colonna.replace("EER (%)", "minDCF")]})
        for chiave in altri:
            altro = leggi(ssl + suffisso, f"{'risultati_scorta' if chiave == 'scorta' else chiave}.csv")
            if altro is not None:
                altro.insert(0, "SSL", nome)
                altro.insert(0, "train", train)
                altri[chiave].append(altro)
    if not completi:
        return None
    piatto = pd.concat(pezzi, axis=1)
    famiglie = pd.concat(completi.values()).drop_duplicates("classificatore").set_index("classificatore")["famiglia"]
    piatto.insert(0, "famiglia", famiglie.reindex(piatto.index))
    return piatto.reset_index(), completi, migliori, altri


if __name__ == "__main__":
    fogli, migliori, altri, testo = {}, [], {"scorta": [], "clustering": [], "layer_dev": []}, [
        "# Classificatori leggeri: riepilogo EER (%)\n",
        "Scelte sul dev della lista di training, test sulle stesse liste, punteggio alto = bonafide.\n"]
    for suffisso, (train, breve) in TRAINING.items():
        risultato = tabelle(suffisso, train)
        if risultato is None:
            continue
        piatto, completi, m, a = risultato
        fogli[f"Riepilogo EER{breve}"] = piatto
        fogli.update({f"{nome} {'completo' if not breve else breve.strip()}": df for nome, df in completi.items()})
        migliori += m
        for chiave in altri:
            altri[chiave] += a[chiave]
        # Markdown: una tabella per SSL
        testo.append(f"\n# Training su {train}\n")
        for nome, df in completi.items():
            colonne = [c for c in LISTE if c in df.columns]
            t = df[["classificatore", "famiglia"] + colonne].rename(columns=LISTE)
            testo.append(f"\n## {nome} (layer {df['layer'].iloc[0]})\n")
            testo.append("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns))
            for _, r in t.iterrows():
                testo.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")

    fogli["Migliori per lista"] = pd.DataFrame(migliori).sort_values(["lista", "EER (%)"])
    fogli.update({"Scorta (anomaly)": pd.concat(altri["scorta"]), "Clustering": pd.concat(altri["clustering"]),
                  "Layer dev": pd.concat(altri["layer_dev"])})
    uscita = CARTELLA / "classificatori_leggeri.xlsx"
    with pd.ExcelWriter(uscita, engine="openpyxl") as excel:
        for titolo, df in fogli.items():
            df.to_excel(excel, sheet_name=titolo, index=False)
            formatta(excel.sheets[titolo], df.reset_index(drop=True), titolo.startswith("Riepilogo") or "completo" in titolo
                     or "train ASV5" in titolo)
    (CARTELLA / "riepilogo.md").write_text("\n".join(testo) + "\n", encoding="utf-8")
    print("Scritti", uscita, "e", CARTELLA / "riepilogo.md")
