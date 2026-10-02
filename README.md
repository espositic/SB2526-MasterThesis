# SB2526-MasterThesis

Esperimenti di anti-spoofing vocale (bonafide vs spoof) su più dataset e modelli.

## Struttura

```
src/thesis/            libreria riusabile, importata dai notebook
  config.py            percorsi (locale / Colab), seed
  datasets/            un modulo per dataset + interfaccia comune (base.py)
  features/            estrazione feature (Mel-spettrogrammi)
notebooks/             un notebook per passo: NN_<dataset>_<cosa>.ipynb
manifests/             sottoinsiemi selezionati (committati per riproducibilità)
data/                  dataset grezzi        (ignorato da git)
artifacts/             feature, modelli, risultati (ignorato da git)
```

## Setup locale

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[notebook]"
```

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
