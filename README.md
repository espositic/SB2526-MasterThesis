# SB2526-MasterThesis

Esperimenti di anti-spoofing vocale (bonafide vs spoof) su più dataset e modelli.

## Struttura

```
src/thesis/            libreria riusabile, importata dai notebook
  config.py            percorsi (locale / Colab), seed
  datasets/            un modulo per dataset + interfaccia comune (base.py)
  features/            estrazione feature (Mel-spettrogrammi)
  models/              modelli (XLSR-MamBo con Hydra in PyTorch puro)
  waveform.py          dataset su forma d'onda grezza da manifest
  metrics.py           EER, minDCF, actDCF, Cllr, EER per gruppo, bootstrap
  train.py             entry point di training/valutazione
configs/               una config JSON per esperimento
notebooks/             un notebook per passo: NN_<dataset>_<cosa>.ipynb
manifests/             sottoinsiemi selezionati (committati per riproducibilità)
data/                  dataset grezzi        (ignorato da git)
artifacts/             feature, modelli, risultati (ignorato da git)
```

## Setup locale

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA, da adattare al driver
pip install -e ".[notebook]"
```

## Training

```bash
python -m thesis.train --config configs/mambo_asvspoof5.json --seed 1 2 3
```

Risultati in `artifacts/runs/<nome>/seed<k>/` (config, log per epoca, punteggi,
`metrics.json`); i seed già completati vengono saltati. Prima di un training
lungo conviene lanciare `configs/mambo_smoke.json` (pochi file, 2 epoche).

Su Colab la prima cella di ogni notebook clona il repo e installa tutto.
Le radici di dati e artefatti si cambiano con `THESIS_DATA_ROOT` e `THESIS_ARTIFACTS_ROOT`.

## Manifest

`manifests/<dataset>_<split>_n<N>_seed<S>_dur<min>-<max>.csv`: prima riga con i
parametri in JSON, poi `file_name,label,audio_path,duration` con percorsi relativi
alla radice del dataset.

## Aggiungere un dataset

Crea `src/thesis/datasets/<nome>.py` con una sottoclasse di `BaseDataset` che
implementa `prepare(split)` e `load_protocol(split)`; subset bilanciato, manifest
ed estrazione feature funzionano senza altre modifiche.
