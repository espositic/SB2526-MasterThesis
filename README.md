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
notebooks/
  01_preparazione_dati  download dei dataset e creazione delle liste
  02_training           training di un modello (scelta di modello, dataset, augmentation, seed)
  03_test               punteggi sulle liste di test
  04_risultati          tabelle e grafici
  05_colab              la parte "colab" di coda_tabella3.py su Google Colab
coda_tabella3.py        replica della Tabella 3 di XLSR-MamBo (parte "fisso" e parte "colab")
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
| Augmentation | nessuna, RawBoost (algoritmo 5; algoritmo 3 per il modello DF), rumore + riverbero, codec; SpecAugment per TFPARN |

Ricetta comune: AdamW, lr 1e-5, warmup + coseno, focal loss, batch 8 × 4, clip di 4,175 s,
massimo 20 epoche, stop dopo 3 epoche senza miglioramenti dell'EER sul dev, 3 seed.
Le eccezioni (MHFA e TFPARN) sono in `tesi/modelli/__init__.py`.

Ogni esperimento finisce in `artifacts/esperimenti/<modello>_<dataset>_<augmentation>/seed<k>/`.

### Replica della Tabella 3 di XLSR-MamBo

Come nel paper (Sez. 4.2), su ASVspoof 2019 LA si allenano due modelli per ogni architettura:
uno con RawBoost algoritmo 5 (`rawboost`), testato su 2019 LA, 2021 LA, In-the-Wild e ASVspoof 5,
e uno con RawBoost algoritmo 3 (`rawboost_ssi`), testato su 2021 DF.
`python coda_tabella3.py fisso` fa i modelli `rawboost` e i test su ASVspoof 5;
`python coda_tabella3.py colab` (dal notebook 05) fa i modelli `rawboost_ssi` e il test su 2021 DF.

Il codice preso da altri repository è elencato in `THIRD_PARTY_NOTICES.md`.
