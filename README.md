# SB2526-MasterThesis

Tesi magistrale: rilevamento di audio contraffatto (anti-spoofing vocale, bonafide contro spoof)
con modelli auto-supervisionati, confronto di data augmentation e test su più dataset.

## Struttura

```
tesi/                   codice, importato dai notebook
  percorsi.py           cartelle del progetto (data/, manifests/, artifacts/)
  dati.py               download, lettura dei protocolli, liste di file, Dataset PyTorch
  augmentation.py       RawBoost, rumore + riverbero (MUSAN, RIR), codec (MP3, AAC, Opus)
  modelli/              XLSR-MamBo, Nes2Net-X, WavLM + MHFA, TFPARN
  training.py           training con early stopping sul dev, test
  metriche.py           EER, minDCF, actDCF, Cllr, EER per attacco
  risultati.py          tabelle dei risultati
  baseline_ssl.py       medie sul tempo dei layer di un SSL congelato (feature per i classificatori leggeri)
notebooks/
  01_preparazione_dati  download dei dataset e creazione delle liste
  02_training           training di un modello (scelta di modello, dataset, augmentation, seed)
  03_test               punteggi sulle liste di test
  04_risultati          tabelle e grafici
  05_colab              training su ASVspoof 5 intero su Google Colab
coda_tabella3.py        replica della Tabella 3 di XLSR-MamBo (parte "fisso" e parte "colab")
estrai_feature_ssl.py   estrae le feature SSL congelate (WavLM Base, HuBERT Base, XLS-R) sulla GPU
classificatori_leggeri.py  classificatori di scikit-learn sulle feature SSL, solo CPU
crea_excel.py           raccoglie i risultati dei classificatori leggeri in un Excel
risultati/              tabelle dei classificatori leggeri (copia di artifacts/classificatori_leggeri)
manifests/              liste di file (CSV), committate per poter rifare gli esperimenti
data/                   dataset                         (non in git)
artifacts/esperimenti/  pesi, log e punteggi di ogni esperimento   (non in git)
```

Tutto si lancia dai notebook, in ordine. Le etichette sono 0 = bonafide e 1 = spoof;
il punteggio di un modello è logit bonafide − logit spoof, quindi alto = bonafide.

## Installazione

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Esperimenti

| | |
|---|---|
| Modelli | XLSR-MamBo, Nes2Net-X, WavLM Base + MHFA (con SSL); TFPARN (su log-Mel, senza SSL) |
| Training | ASVspoof 5 (10.000 clip bilanciate) oppure ASVspoof 2019 LA (intero) |
| Test | ASVspoof 5, ASVspoof 2019 LA, ASVspoof 2021 LA e DF, In-the-Wild |
| Augmentation | nessuna, RawBoost (algoritmo 5), rumore + riverbero, codec; SpecAugment per TFPARN |

Ricetta comune: AdamW, lr 1e-5, warmup + coseno, focal loss, batch 8 × 4, clip di 4,175 s,
massimo 20 epoche, stop dopo 3 epoche senza miglioramenti dell'EER sul dev, 3 seed.
Le eccezioni (MHFA e TFPARN) sono in `tesi/modelli/__init__.py`.

Ogni esperimento finisce in `artifacts/esperimenti/<modello>_<dataset>_<augmentation>/seed<k>/`.

### Classificatori leggeri su feature SSL congelate

Stesso protocollo della Tabella 3 (training su 2019 LA train, scelte su 2019 LA dev, test su
2019 LA eval, 2021 LA e DF, In-the-Wild e ASVspoof 5 eval completo), ma senza fine-tuning:
`python estrai_feature_ssl.py` salva sulla GPU le medie sul tempo di ogni layer
(`artifacts/feature_ssl/<ssl>/`), poi `python classificatori_leggeri.py <ssl>` sceglie il layer sul dev
e confronta 24 classificatori (lineari, alberi, kernel, probabilistici, MLP, ensemble, somma pesata
dei layer, configurazione di Pascu et al. 2023) solo sulla CPU. Salva tabelle, modelli e punteggi
per clip in `artifacts/classificatori_leggeri/<ssl>/`; `python crea_excel.py` fa l'Excel di riepilogo.

### Replica della Tabella 3 di XLSR-MamBo

Training su ASVspoof 2019 LA intero con RawBoost algoritmo 5, test su 2019 LA, 2021 LA e DF,
In-the-Wild e ASVspoof 5. Il paper usa per il 2021 DF un secondo modello con RawBoost algoritmo 3:
noi teniamo un solo modello. In più si allena anche su ASVspoof 5 intero.
`python coda_tabella3.py fisso` fa TFPARN su 2019 LA e i test su ASVspoof 5;
`python coda_tabella3.py colab` (dal notebook 05) allena su ASVspoof 5 intero.

Il codice preso da altri repository è elencato in `THIRD_PARTY_NOTICES.md`.
