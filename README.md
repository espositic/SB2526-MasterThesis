# SB2526-MasterThesis

Esperimenti di anti-spoofing vocale (bonafide vs spoof) su più dataset e modelli.

## Struttura

```
src/thesis/            libreria riusabile, importata dai notebook
  config.py            percorsi (locale / Colab), seed
  datasets/            un modulo per dataset + interfaccia comune (base.py)
  features/            estrazione feature (Mel-spettrogrammi)
  models/              modelli (XLSR-MamBo, WavLM + MHFA, MoLEx, XLSR-Conformer + TCM, TFPARN, LCNN)
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
lungo conviene lanciare lo smoke test del modello, `configs/<modello>_smoke.json`
(pochi file, 2 epoche). Il codice preso da altri repository è elencato in
`THIRD_PARTY_NOTICES.md`.

Configurazioni disponibili:

| config | modello | ingresso | hardware |
|---|---|---|---|
| `mambo_asvspoof5` | XLSR-MamBo (MamBo-3-Hydra-N3) | forma d'onda | GPU ≥ 8 GB |
| `mhfa_rawboost_asvspoof5` | WavLM Base fine-tuned + MHFA | forma d'onda | GPU ≥ 8 GB |
| `mhfa_frozen_rawboost_asvspoof5` | WavLM Base congelato + MHFA | forma d'onda | GPU |
| `molex_rawboost_asvspoof5` | MoLEx: WavLM-Large (12 layer) congelato + esperti LoRA | forma d'onda | GPU ≥ 8 GB |
| `molex_lora_rawboost_asvspoof5` | ablation: un solo LoRA, senza esperti | forma d'onda | GPU ≥ 8 GB |
| `tcm_rawboost_asvspoof5` | XLSR-Conformer + TCM | forma d'onda | GPU ≥ 8 GB |
| `tcm_kan_rawboost_asvspoof5` | variante XLSR-Kanformer (ChebyKAN) | forma d'onda | GPU ≥ 8 GB |
| `tfparn_rawboost_asvspoof5` | TFPARN: Transformer su log-Mel, focal + ranking a coppie | forma d'onda | GPU |
| `lcnn_asvspoof5` | LCNN, pooling medio, cross-entropy | Mel | CPU |
| `lcnn_lstm_asvspoof5` | + BiLSTM e attention pooling | Mel | CPU |
| `lcnn_lstm_specaug_asvspoof5` | + SpecAugment in training | Mel | CPU |
| `lcnn_lstm_specaug_oc_asvspoof5` | + OC-Softmax | Mel | CPU |

Le LCNN non leggono l'audio: servono solo le feature Mel del notebook 01
(`artifacts/features/asvspoof5/<tag>/`) e i TSV di protocollo in `data/asvspoof5/`.

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
