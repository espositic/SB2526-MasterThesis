# Raccoglie i risultati dei classificatori leggeri dei tre modelli SSL in un unico Excel
# (artifacts/classificatori_leggeri/classificatori_leggeri.xlsx) e in un riepilogo Markdown.
#
#   python crea_excel.py
#
# Fogli: Riepilogo EER (tutti i classificatori, tutti gli SSL, tutte le liste), Migliori per lista,
# un foglio completo (EER e minDCF) per SSL, Scorta (anomaly detection), Clustering, Layer dev.
# Nel riepilogo e nei fogli completi il valore migliore (il più basso) di ogni colonna è in verde.
import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from tesi import percorsi

CARTELLA = percorsi.PROGETTO / "artifacts" / "classificatori_leggeri"
SSL = {"wavlm-base": "WavLM Base", "hubert-base-ls960": "HuBERT Base", "wav2vec2-xls-r-300m": "XLS-R 300M"}
LISTE = {"EER dev (%)": "dev 2019", "inthewild EER (%)": "In-the-Wild", "asvspoof2019_eval EER (%)": "2019 LA eval",
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


if __name__ == "__main__":
    # riepilogo: una riga per classificatore, colonne SSL x lista (solo EER)
    pezzi, migliori, completi, scorta, clustering, layer_dev = [], [], {}, [], [], []
    for ssl, nome in SSL.items():
        df = leggi(ssl, "risultati.csv")
        if df is None:
            continue
        completi[nome] = df
        tab = df.set_index("classificatore")[[c for c in LISTE if c in df.columns]].rename(columns=LISTE)
        tab.columns = pd.MultiIndex.from_product([[f"{nome} (layer {df['layer'].iloc[0]})"], tab.columns])
        pezzi.append(tab)
        for colonna, lista in LISTE.items():
            if colonna in df.columns and colonna != "EER dev (%)":
                r = df.loc[df[colonna].idxmin()]
                migliori.append({"lista": lista, "SSL": nome, "classificatore": r["classificatore"],
                                 "famiglia": r["famiglia"], "EER (%)": r[colonna],
                                 "minDCF": r[colonna.replace("EER (%)", "minDCF")]})
        for file, dove in [("risultati_scorta.csv", scorta), ("clustering.csv", clustering), ("layer_dev.csv", layer_dev)]:
            altro = leggi(ssl, file)
            if altro is not None:
                altro.insert(0, "SSL", nome)
                dove.append(altro)

    riepilogo = pd.concat(pezzi, axis=1)
    famiglie = pd.concat(completi.values()).drop_duplicates("classificatore").set_index("classificatore")["famiglia"]
    piatto = riepilogo.copy()
    piatto.columns = [f"{a} | {b}" for a, b in piatto.columns]
    piatto.insert(0, "famiglia", famiglie.reindex(piatto.index))
    piatto = piatto.reset_index()
    migliori = pd.DataFrame(migliori).sort_values(["lista", "EER (%)"])

    uscita = CARTELLA / "classificatori_leggeri.xlsx"
    with pd.ExcelWriter(uscita, engine="openpyxl") as excel:
        fogli = {"Riepilogo EER": piatto, "Migliori per lista": migliori}
        fogli.update({f"{nome} completo": df for nome, df in completi.items()})
        fogli.update({"Scorta (anomaly)": pd.concat(scorta), "Clustering": pd.concat(clustering),
                      "Layer dev": pd.concat(layer_dev)})
        for titolo, df in fogli.items():
            df.to_excel(excel, sheet_name=titolo, index=False)
            formatta(excel.sheets[titolo], df.reset_index(drop=True), titolo == "Riepilogo EER" or "completo" in titolo)

    # lo stesso riepilogo in Markdown (EER %, una tabella per SSL)
    testo = ["# Classificatori leggeri: riepilogo EER (%)\n",
             "Training su ASVspoof 2019 LA train, scelte su 2019 LA dev, punteggio alto = bonafide.\n"]
    for nome, df in completi.items():
        colonne = [c for c in LISTE if c in df.columns]
        t = df[["classificatore", "famiglia"] + colonne].rename(columns=LISTE)
        testo.append(f"\n## {nome} (layer {df['layer'].iloc[0]})\n")
        testo.append("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns))
        for _, r in t.iterrows():
            testo.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    (CARTELLA / "riepilogo.md").write_text("\n".join(testo) + "\n", encoding="utf-8")
    print("Scritti", uscita, "e", CARTELLA / "riepilogo.md")
