# Classificatori leggeri: riepilogo EER (%)

Scelte sul dev della lista di training, test sulle stesse liste, punteggio alto = bonafide.


# Training su 2019 LA


## WavLM Base (layer 3)

| classificatore | famiglia | dev | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
|---|---|---|---|---|---|---|---|
| Regressione logistica | lineare | 0.82 | 22.66 | 2.98 | 20.51 | 14.51 | 27.26 |
| SVM lineare | lineare | 0.63 | 18.62 | 2.58 | 20.11 | 13.86 | 27.85 |
| LDA | lineare | 0.55 | 29.08 | 2.27 | 14.80 | 14.53 | 27.36 |
| Albero di decisione | alberi | 14.84 | 53.17 | 14.29 | 32.55 | 32.44 | 45.98 |
| Random forest | alberi | 1.41 | 18.99 | 3.65 | 11.72 | 12.51 | 31.27 |
| Extra trees | alberi | 1.34 | 22.55 | 3.89 | 10.62 | 12.21 | 28.54 |
| Gradient boosting | alberi | 0.75 | 20.44 | 2.58 | 14.94 | 12.30 | 26.93 |
| XGBoost | alberi | 0.67 | 18.06 | 2.35 | 19.24 | 12.06 | 27.31 |
| AdaBoost | alberi | 1.14 | 26.12 | 2.83 | 20.66 | 11.65 | 26.93 |
| kNN (PCA 128) | vicini | 22.37 | 35.16 | 7.02 | 16.31 | 22.99 | 40.77 |
| Naive Bayes gaussiano | probabilistico | 14.28 | 50.04 | 9.81 | 33.00 | 31.47 | 49.67 |
| QDA (PCA 128) | probabilistico | 3.17 | 26.40 | 3.56 | 25.10 | 12.99 | 25.18 |
| SVM RBF (Nystroem 2000) | kernel | 0.46 | 21.72 | 1.65 | 11.45 | 10.88 | 31.78 |
| SVM RBF (esatta) | kernel | 0.16 | 18.26 | 1.46 | 11.32 | 10.77 | 30.55 |
| MLP (256) | rete | 0.31 | 17.31 | 1.32 | 13.44 | 11.03 | 24.20 |
| Due GMM (PCA 128, LLR) | probabilistico | 9.38 | 37.11 | 10.05 | 29.38 | 23.48 | 31.99 |
| Regressione logistica (C sul dev) | lineare | 0.43 | 16.88 | 1.41 | 18.93 | 9.98 | 24.79 |
| Voting (RF + GB + SVM lineare) | ensemble | 0.43 | 15.90 | 1.65 | 15.44 | 10.59 | 24.94 |
| SelectKBest + regressione logistica | selezione feature | 1.13 | 21.60 | 3.60 | 26.34 | 11.40 | 27.71 |
| Regressione logistica (sottocampionata) | campionamento | 0.52 | 19.39 | 1.85 | 23.37 | 10.54 | 25.09 |
| Random forest (sottocampionata) | campionamento | 2.51 | 21.16 | 4.22 | 11.98 | 11.77 | 28.58 |
| Regressione logistica (3 layer) | multi-layer | 0.67 | 27.74 | 2.12 | 21.85 | 14.94 | 31.00 |
| Somma pesata dei layer + MLP | multi-layer | 0.08 | 18.52 | 1.32 | 9.27 | 14.37 | 36.50 |
| Regressione logistica (Pascu: ultimo layer, C=1e6) | lineare | 4.67 | 29.35 | 6.73 | 13.14 | 16.73 | 21.73 |

## HuBERT Base (layer 2)

| classificatore | famiglia | dev | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
|---|---|---|---|---|---|---|---|
| Regressione logistica | lineare | 0.51 | 26.20 | 2.08 | 9.56 | 18.39 | 27.20 |
| SVM lineare | lineare | 0.31 | 28.42 | 1.76 | 9.54 | 16.77 | 26.73 |
| LDA | lineare | 0.31 | 32.55 | 2.58 | 9.40 | 18.48 | 33.11 |
| Albero di decisione | alberi | 17.27 | 44.79 | 12.85 | 28.96 | 34.06 | 49.43 |
| Random forest | alberi | 1.45 | 30.77 | 4.07 | 8.34 | 19.85 | 36.31 |
| Extra trees | alberi | 1.34 | 32.08 | 4.24 | 8.12 | 19.10 | 34.21 |
| Gradient boosting | alberi | 1.09 | 32.83 | 3.35 | 11.36 | 19.10 | 32.07 |
| XGBoost | alberi | 0.90 | 36.01 | 2.88 | 11.60 | 18.37 | 30.59 |
| AdaBoost | alberi | 1.61 | 40.20 | 2.88 | 11.22 | 17.79 | 30.30 |
| kNN (PCA 128) | vicini | 17.31 | 45.08 | 8.51 | 18.08 | 25.40 | 42.83 |
| Naive Bayes gaussiano | probabilistico | 13.54 | 50.03 | 10.23 | 22.09 | 31.84 | 49.76 |
| QDA (PCA 128) | probabilistico | 4.75 | 34.79 | 4.80 | 13.81 | 18.54 | 28.77 |
| SVM RBF (Nystroem 2000) | kernel | 0.68 | 33.64 | 2.46 | 8.30 | 17.41 | 34.85 |
| SVM RBF (esatta) | kernel | 0.16 | 33.78 | 1.92 | 7.20 | 17.60 | 32.62 |
| MLP (256) | rete | 0.47 | 23.72 | 1.58 | 9.09 | 17.14 | 27.97 |
| Due GMM (PCA 128, LLR) | probabilistico | 10.40 | 35.84 | 10.13 | 19.32 | 24.88 | 35.56 |
| Regressione logistica (C sul dev) | lineare | 0.35 | 25.98 | 1.63 | 8.30 | 16.08 | 26.18 |
| Voting (RF + GB + SVM lineare) | ensemble | 0.35 | 26.79 | 1.91 | 7.91 | 16.57 | 26.46 |
| SelectKBest + regressione logistica | selezione feature | 1.14 | 29.43 | 2.72 | 9.15 | 15.91 | 27.42 |
| Regressione logistica (sottocampionata) | campionamento | 0.63 | 29.13 | 2.32 | 9.76 | 15.91 | 26.32 |
| Random forest (sottocampionata) | campionamento | 2.08 | 28.61 | 4.49 | 8.46 | 18.96 | 32.88 |
| Regressione logistica (3 layer) | multi-layer | 0.43 | 22.55 | 1.75 | 12.95 | 15.46 | 28.68 |
| Somma pesata dei layer + MLP | multi-layer | 0.13 | 24.63 | 2.03 | 7.03 | 13.68 | 29.22 |
| Regressione logistica (Pascu: ultimo layer, C=1e6) | lineare | 2.71 | 26.80 | 7.72 | 16.27 | 13.96 | 21.98 |

## XLS-R 300M (layer 4)

| classificatore | famiglia | dev | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
|---|---|---|---|---|---|---|---|
| Regressione logistica | lineare | 0.00 | 49.29 | 0.79 | 24.87 | 20.55 | 31.13 |
| SVM lineare | lineare | 0.00 | 52.61 | 0.73 | 25.25 | 19.12 | 29.05 |
| LDA | lineare | 0.00 | 59.77 | 0.57 | 24.62 | 19.84 | 34.00 |
| Albero di decisione | alberi | 12.20 | 44.55 | 9.42 | 32.60 | 28.29 | 46.27 |
| Random forest | alberi | 0.35 | 19.32 | 1.58 | 10.09 | 10.29 | 29.56 |
| Extra trees | alberi | 0.31 | 15.67 | 1.54 | 9.44 | 10.17 | 26.09 |
| Gradient boosting | alberi | 0.27 | 27.87 | 1.20 | 14.21 | 12.13 | 30.52 |
| XGBoost | alberi | 0.16 | 26.87 | 0.97 | 15.58 | 11.91 | 29.60 |
| AdaBoost | alberi | 0.08 | 28.70 | 0.65 | 16.39 | 13.52 | 28.31 |
| kNN (PCA 128) | vicini | 20.92 | 41.42 | 2.97 | 11.42 | 23.93 | 42.01 |
| Naive Bayes gaussiano | probabilistico | 12.56 | 50.04 | 7.03 | 33.18 | 32.89 | 49.83 |
| QDA (PCA 128) | probabilistico | 2.91 | 26.36 | 0.81 | 15.58 | 12.74 | 21.28 |
| SVM RBF (Nystroem 2000) | kernel | 0.04 | 25.59 | 0.76 | 10.73 | 12.72 | 36.54 |
| SVM RBF (esatta) | kernel | 0.00 | 29.00 | 0.57 | 12.65 | 14.24 | 36.94 |
| MLP (256) | rete | 0.04 | 19.64 | 0.34 | 9.07 | 14.35 | 23.66 |
| Due GMM (PCA 128, LLR) | probabilistico | 7.80 | 46.04 | 7.60 | 17.18 | 21.06 | 26.21 |
| Regressione logistica (C sul dev) | lineare | 0.00 | 25.03 | 0.30 | 17.15 | 13.52 | 23.03 |
| Voting (RF + GB + SVM lineare) | ensemble | 0.04 | 42.02 | 0.45 | 16.15 | 14.65 | 26.55 |
| SelectKBest + regressione logistica | selezione feature | 0.16 | 29.47 | 1.71 | 15.77 | 13.46 | 26.52 |
| Regressione logistica (sottocampionata) | campionamento | 0.00 | 24.40 | 0.35 | 17.89 | 14.56 | 23.31 |
| Random forest (sottocampionata) | campionamento | 0.59 | 16.57 | 1.58 | 10.15 | 9.66 | 27.15 |
| Regressione logistica (3 layer) | multi-layer | 0.00 | 47.77 | 1.24 | 25.77 | 22.81 | 35.73 |
| Somma pesata dei layer + MLP | multi-layer | 0.00 | 22.13 | 0.27 | 8.69 | 8.94 | 16.44 |
| Regressione logistica (Pascu: ultimo layer, C=1e6) | lineare | 0.16 | 21.58 | 1.05 | 9.90 | 8.41 | 14.59 |

## WavLM Large (layer 1)

| classificatore | famiglia | dev | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
|---|---|---|---|---|---|---|---|
| Regressione logistica | lineare | 0.11 | 44.52 | 4.78 | 33.61 | 29.32 | 38.83 |
| SVM lineare | lineare | 0.03 | 51.90 | 4.42 | 33.63 | 29.46 | 38.03 |
| LDA | lineare | 0.00 | 51.13 | 2.84 | 19.17 | 24.69 | 38.89 |
| Albero di decisione | alberi | 15.97 | 49.48 | 12.48 | 42.18 | 37.39 | 49.99 |
| Random forest | alberi | 1.22 | 27.70 | 5.64 | 15.42 | 23.01 | 42.86 |
| Extra trees | alberi | 1.34 | 28.93 | 5.90 | 15.47 | 22.40 | 42.02 |
| Gradient boosting | alberi | 0.66 | 31.98 | 4.57 | 22.37 | 23.41 | 39.73 |
| XGBoost | alberi | 0.59 | 32.23 | 4.95 | 25.30 | 25.20 | 40.62 |
| AdaBoost | alberi | 0.51 | 35.11 | 5.36 | 28.22 | 28.10 | 40.76 |
| kNN (PCA 128) | vicini | 26.53 | 46.75 | 11.81 | 19.46 | 26.47 | 44.59 |
| Naive Bayes gaussiano | probabilistico | 14.72 | 50.04 | 12.60 | 40.54 | 36.38 | 49.90 |
| QDA (PCA 128) | probabilistico | 4.75 | 29.04 | 3.30 | 28.70 | 26.57 | 39.88 |
| SVM RBF (Nystroem 2000) | kernel | 0.27 | 42.47 | 3.48 | 25.94 | 27.58 | 42.92 |
| SVM RBF (esatta) | kernel | 0.04 | 44.73 | 3.70 | 29.28 | 29.27 | 44.09 |
| MLP (256) | rete | 0.08 | 33.00 | 5.14 | 27.87 | 31.00 | 39.61 |
| Due GMM (PCA 128, LLR) | probabilistico | 14.95 | 31.51 | 18.19 | 29.16 | 31.44 | 36.38 |
| Regressione logistica (C sul dev) | lineare | 0.04 | 38.60 | 4.57 | 31.92 | 30.61 | 37.49 |
| Voting (RF + GB + SVM lineare) | ensemble | 0.03 | 43.57 | 3.21 | 28.22 | 26.82 | 37.33 |
| SelectKBest + regressione logistica | selezione feature | 0.19 | 35.41 | 5.49 | 29.74 | 29.79 | 37.90 |
| Regressione logistica (sottocampionata) | campionamento | 0.12 | 33.14 | 5.02 | 32.82 | 31.72 | 38.09 |
| Random forest (sottocampionata) | campionamento | 1.70 | 28.32 | 5.93 | 15.50 | 21.99 | 41.14 |
| Regressione logistica (3 layer) | multi-layer | 0.08 | 34.37 | 1.90 | 24.07 | 21.60 | 33.43 |
| Somma pesata dei layer + MLP | multi-layer | 0.04 | 21.39 | 0.34 | 7.30 | 9.23 | 17.00 |
| Regressione logistica (Pascu: ultimo layer, C=1e6) | lineare | 0.98 | 19.62 | 2.28 | 10.50 | 7.58 | 18.94 |

## HuBERT Large (layer 11)

| classificatore | famiglia | dev | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
|---|---|---|---|---|---|---|---|
| Regressione logistica | lineare | 0.12 | 24.35 | 0.84 | 10.37 | 9.50 | 22.54 |
| SVM lineare | lineare | 0.16 | 23.99 | 0.65 | 10.83 | 8.85 | 22.96 |
| LDA | lineare | 0.12 | 23.85 | 0.55 | 8.38 | 8.20 | 22.75 |
| Albero di decisione | alberi | 16.29 | 45.40 | 9.90 | 27.51 | 31.43 | 46.64 |
| Random forest | alberi | 1.54 | 17.77 | 3.10 | 6.94 | 8.08 | 15.37 |
| Extra trees | alberi | 1.30 | 19.07 | 3.17 | 6.46 | 8.33 | 15.96 |
| Gradient boosting | alberi | 0.47 | 18.87 | 1.62 | 7.91 | 7.73 | 16.59 |
| XGBoost | alberi | 0.83 | 18.36 | 1.69 | 9.36 | 7.02 | 16.89 |
| AdaBoost | alberi | 0.67 | 21.39 | 1.47 | 6.28 | 9.48 | 20.25 |
| kNN (PCA 128) | vicini | 25.51 | 36.15 | 4.54 | 10.40 | 17.18 | 36.08 |
| Naive Bayes gaussiano | probabilistico | 16.48 | 49.87 | 10.81 | 15.23 | 32.82 | 49.63 |
| QDA (PCA 128) | probabilistico | 2.35 | 17.58 | 2.04 | 10.14 | 8.27 | 16.73 |
| SVM RBF (Nystroem 2000) | kernel | 0.27 | 16.33 | 0.68 | 8.17 | 5.56 | 21.15 |
| SVM RBF (esatta) | kernel | 0.04 | 15.00 | 0.51 | 7.28 | 6.58 | 21.92 |
| MLP (256) | rete | 0.12 | 16.70 | 0.66 | 4.71 | 11.03 | 20.27 |
| Due GMM (PCA 128, LLR) | probabilistico | 9.26 | 44.24 | 8.33 | 13.51 | 19.95 | 32.45 |
| Regressione logistica (C sul dev) | lineare | 0.12 | 24.35 | 0.84 | 10.37 | 9.50 | 22.54 |
| Voting (RF + GB + SVM lineare) | ensemble | 0.15 | 17.08 | 0.57 | 6.89 | 5.98 | 18.17 |
| SelectKBest + regressione logistica | selezione feature | 0.71 | 19.57 | 1.18 | 13.88 | 8.91 | 20.61 |
| Regressione logistica (sottocampionata) | campionamento | 0.27 | 17.51 | 0.91 | 10.19 | 8.02 | 19.82 |
| Random forest (sottocampionata) | campionamento | 3.22 | 18.07 | 3.51 | 7.41 | 7.84 | 14.67 |
| Regressione logistica (3 layer) | multi-layer | 0.16 | 32.36 | 0.71 | 15.57 | 11.18 | 31.84 |
| Somma pesata dei layer + MLP | multi-layer | 0.04 | 17.03 | 1.09 | 9.44 | 11.45 | 24.37 |
| Regressione logistica (Pascu: ultimo layer, C=1e6) | lineare | 0.67 | 18.60 | 2.96 | 12.07 | 9.03 | 21.42 |

# Training su ASVspoof 5 intero


## WavLM Base (layer 8)

| classificatore | famiglia | dev | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
|---|---|---|---|---|---|---|---|
| Regressione logistica | lineare | 9.24 | 22.28 | 16.69 | 20.14 | 17.93 | 12.09 |
| SVM lineare | lineare | 9.84 | 21.86 | 16.25 | 18.88 | 16.52 | 12.81 |
| LDA | lineare | 18.24 | 23.92 | 18.38 | 23.68 | 21.19 | 20.95 |
| Albero di decisione | alberi | 23.48 | 39.69 | 25.72 | 26.88 | 27.90 | 31.84 |
| Random forest | alberi | 11.76 | 29.32 | 14.06 | 14.55 | 17.40 | 20.36 |
| Extra trees | alberi | 11.84 | 29.13 | 12.94 | 14.40 | 18.25 | 20.33 |
| Gradient boosting | alberi | 9.96 | 26.52 | 15.02 | 17.64 | 16.88 | 15.38 |
| XGBoost | alberi | 9.72 | 27.07 | 14.89 | 17.30 | 16.81 | 15.43 |
| AdaBoost | alberi | 11.76 | 27.85 | 18.35 | 22.94 | 19.62 | 16.26 |
| kNN (PCA 128) | vicini | 15.12 | 30.97 | 13.88 | 16.52 | 19.36 | 21.50 |
| Naive Bayes gaussiano | probabilistico | 23.08 | 39.41 | 44.55 | 48.36 | 42.03 | 26.99 |
| QDA (PCA 128) | probabilistico | 14.64 | 35.66 | 21.82 | 23.45 | 29.28 | 22.32 |
| SVM RBF (Nystroem 2000) | kernel | 8.76 | 24.46 | 15.76 | 20.67 | 17.08 | 17.22 |
| SVM RBF (esatta) | kernel | 8.56 | 24.62 | 15.44 | 19.57 | 16.61 | 17.11 |
| MLP (256) | rete | 7.84 | 28.77 | 22.51 | 22.89 | 21.12 | 11.48 |
| Due GMM (PCA 128, LLR) | probabilistico | 13.88 | 37.71 | 33.83 | 44.80 | 33.23 | 20.17 |
| Regressione logistica (C sul dev) | lineare | 8.44 | 22.98 | 16.67 | 20.46 | 18.89 | 12.14 |
| Voting (RF + GB + SVM lineare) | ensemble | 8.56 | 22.48 | 14.04 | 16.25 | 14.81 | 13.46 |
| SelectKBest + regressione logistica | selezione feature | 12.44 | 26.65 | 18.04 | 22.52 | 20.30 | 18.26 |
| Regressione logistica (sottocampionata) | campionamento | 9.16 | 24.15 | 19.18 | 23.76 | 20.91 | 12.92 |
| Random forest (sottocampionata) | campionamento | 12.00 | 29.39 | 14.58 | 14.32 | 17.30 | 20.57 |
| Regressione logistica (3 layer) | multi-layer | 10.92 | 20.34 | 15.81 | 16.60 | 16.52 | 12.62 |
| Somma pesata dei layer + MLP | multi-layer | 5.48 | 12.91 | 18.50 | 20.81 | 17.15 | 11.56 |
| Regressione logistica (Pascu: ultimo layer, C=1e6) | lineare | 10.68 | 23.80 | 14.93 | 16.35 | 17.59 | 13.93 |

## HuBERT Base (layer 10)

| classificatore | famiglia | dev | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
|---|---|---|---|---|---|---|---|
| Regressione logistica | lineare | 11.68 | 18.01 | 16.62 | 18.47 | 17.45 | 13.81 |
| SVM lineare | lineare | 11.68 | 17.96 | 16.65 | 18.39 | 17.86 | 13.55 |
| LDA | lineare | 18.60 | 19.84 | 17.02 | 18.14 | 20.35 | 22.62 |
| Albero di decisione | alberi | 26.32 | 37.93 | 26.19 | 26.95 | 30.24 | 36.52 |
| Random forest | alberi | 15.68 | 27.95 | 12.83 | 13.69 | 18.14 | 21.65 |
| Extra trees | alberi | 15.32 | 26.97 | 12.54 | 13.31 | 17.29 | 21.07 |
| Gradient boosting | alberi | 13.76 | 21.99 | 13.99 | 16.72 | 18.60 | 18.95 |
| XGBoost | alberi | 13.56 | 22.51 | 13.70 | 16.55 | 18.37 | 18.60 |
| AdaBoost | alberi | 17.00 | 24.16 | 16.85 | 19.96 | 21.87 | 22.59 |
| kNN (PCA 128) | vicini | 20.92 | 30.26 | 13.93 | 16.41 | 19.80 | 25.50 |
| Naive Bayes gaussiano | probabilistico | 14.60 | 33.46 | 35.31 | 35.59 | 30.18 | 21.68 |
| QDA (PCA 128) | probabilistico | 16.76 | 27.35 | 17.09 | 17.08 | 21.70 | 23.45 |
| SVM RBF (Nystroem 2000) | kernel | 13.32 | 20.04 | 14.64 | 16.51 | 16.81 | 18.30 |
| SVM RBF (esatta) | kernel | 12.84 | 18.92 | 14.58 | 16.16 | 16.98 | 18.40 |
| MLP (256) | rete | 12.64 | 27.73 | 24.02 | 27.40 | 24.04 | 15.76 |
| Due GMM (PCA 128, LLR) | probabilistico | 19.04 | 30.84 | 19.47 | 23.23 | 21.29 | 23.83 |
| Regressione logistica (C sul dev) | lineare | 11.40 | 17.54 | 16.41 | 18.18 | 17.27 | 13.53 |
| Voting (RF + GB + SVM lineare) | ensemble | 11.32 | 18.60 | 14.02 | 15.07 | 16.26 | 15.10 |
| SelectKBest + regressione logistica | selezione feature | 15.40 | 20.63 | 15.95 | 17.66 | 19.38 | 17.13 |
| Regressione logistica (sottocampionata) | campionamento | 11.24 | 18.90 | 16.00 | 17.80 | 17.14 | 14.32 |
| Random forest (sottocampionata) | campionamento | 15.84 | 28.71 | 12.62 | 13.54 | 17.70 | 21.56 |
| Regressione logistica (3 layer) | multi-layer | 13.48 | 17.36 | 16.95 | 17.38 | 16.75 | 13.21 |
| Somma pesata dei layer + MLP | multi-layer | 10.08 | 21.31 | 17.72 | 17.41 | 17.14 | 17.42 |
| Regressione logistica (Pascu: ultimo layer, C=1e6) | lineare | 17.32 | 22.25 | 18.48 | 19.36 | 19.49 | 20.70 |

## XLS-R 300M (layer 23)

| classificatore | famiglia | dev | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
|---|---|---|---|---|---|---|---|
| Regressione logistica | lineare | 4.80 | 19.88 | 32.36 | 35.14 | 26.52 | 13.81 |
| SVM lineare | lineare | 4.24 | 15.52 | 22.96 | 26.91 | 19.21 | 9.60 |
| LDA | lineare | 7.48 | 16.59 | 22.65 | 27.30 | 19.99 | 11.18 |
| Albero di decisione | alberi | 13.60 | 32.93 | 26.70 | 30.70 | 27.40 | 24.97 |
| Random forest | alberi | 5.56 | 26.02 | 12.69 | 17.00 | 16.79 | 16.78 |
| Extra trees | alberi | 6.52 | 26.56 | 12.40 | 16.44 | 17.09 | 17.94 |
| Gradient boosting | alberi | 2.40 | 19.13 | 13.34 | 19.90 | 17.14 | 12.56 |
| XGBoost | alberi | 2.56 | 19.44 | 13.03 | 19.28 | 16.56 | 12.73 |
| AdaBoost | alberi | 2.76 | 20.67 | 18.59 | 26.82 | 19.29 | 13.29 |
| kNN (PCA 128) | vicini | 10.92 | 33.76 | 17.57 | 23.26 | 23.91 | 24.38 |
| Naive Bayes gaussiano | probabilistico | 31.12 | 45.46 | 35.13 | 38.18 | 39.17 | 34.46 |
| QDA (PCA 128) | probabilistico | 4.92 | 22.45 | 23.48 | 23.66 | 24.55 | 24.13 |
| SVM RBF (Nystroem 2000) | kernel | 1.56 | 18.65 | 15.43 | 23.41 | 15.39 | 12.28 |
| SVM RBF (esatta) | kernel | 1.24 | 17.84 | 16.23 | 23.32 | 16.33 | 11.44 |
| MLP (256) | rete | 1.64 | 27.73 | 38.65 | 45.30 | 27.91 | 10.15 |
| Due GMM (PCA 128, LLR) | probabilistico | 19.64 | 27.40 | 30.92 | 32.40 | 32.73 | 42.12 |
| Regressione logistica (C sul dev) | lineare | 0.88 | 17.46 | 20.72 | 25.99 | 17.32 | 9.67 |
| Voting (RF + GB + SVM lineare) | ensemble | 1.64 | 15.69 | 12.97 | 17.88 | 15.13 | 10.33 |
| SelectKBest + regressione logistica | selezione feature | 2.28 | 18.14 | 30.99 | 32.49 | 20.77 | 13.10 |
| Regressione logistica (sottocampionata) | campionamento | 1.24 | 18.97 | 27.30 | 31.40 | 20.57 | 11.42 |
| Random forest (sottocampionata) | campionamento | 5.68 | 26.51 | 13.15 | 18.55 | 17.11 | 17.18 |
| Regressione logistica (3 layer) | multi-layer | 7.76 | 13.36 | 16.22 | 15.92 | 15.30 | 9.97 |
| Somma pesata dei layer + MLP | multi-layer | 2.24 | 14.41 | 14.70 | 14.23 | 13.54 | 6.33 |
| Regressione logistica (Pascu: ultimo layer, C=1e6) | lineare | 0.88 | 16.43 | 21.16 | 26.90 | 16.37 | 9.88 |

## WavLM Large (layer 12)

| classificatore | famiglia | dev | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
|---|---|---|---|---|---|---|---|
| Regressione logistica | lineare | 6.40 | 12.10 | 14.78 | 17.33 | 11.23 | 6.03 |
| SVM lineare | lineare | 4.24 | 10.20 | 14.32 | 13.53 | 10.19 | 6.34 |
| LDA | lineare | 1.76 | 10.63 | 11.83 | 14.13 | 10.14 | 5.70 |
| Albero di decisione | alberi | 12.72 | 38.84 | 32.21 | 36.92 | 25.89 | 19.55 |
| Random forest | alberi | 4.36 | 19.89 | 9.26 | 11.16 | 10.53 | 9.52 |
| Extra trees | alberi | 4.80 | 18.33 | 7.89 | 9.52 | 10.65 | 10.36 |
| Gradient boosting | alberi | 2.64 | 17.32 | 14.10 | 16.37 | 11.74 | 7.31 |
| XGBoost | alberi | 2.60 | 18.06 | 14.85 | 17.41 | 12.24 | 7.56 |
| AdaBoost | alberi | 2.60 | 16.12 | 19.18 | 22.22 | 13.48 | 6.90 |
| kNN (PCA 128) | vicini | 5.72 | 19.54 | 16.40 | 21.01 | 16.51 | 14.03 |
| Naive Bayes gaussiano | probabilistico | 31.88 | 47.98 | 47.21 | 48.56 | 41.17 | 29.49 |
| QDA (PCA 128) | probabilistico | 8.72 | 18.84 | 16.79 | 17.29 | 18.84 | 14.76 |
| SVM RBF (Nystroem 2000) | kernel | 4.28 | 16.59 | 25.00 | 28.16 | 14.83 | 10.58 |
| SVM RBF (esatta) | kernel | 3.76 | 15.48 | 18.18 | 22.69 | 11.80 | 10.39 |
| MLP (256) | rete | 7.16 | 38.57 | 34.63 | 38.15 | 24.51 | 6.74 |
| Due GMM (PCA 128, LLR) | probabilistico | 20.56 | 31.86 | 44.23 | 56.77 | 37.08 | 25.69 |
| Regressione logistica (C sul dev) | lineare | 3.72 | 10.67 | 15.76 | 17.55 | 12.21 | 5.93 |
| Voting (RF + GB + SVM lineare) | ensemble | 1.72 | 10.63 | 11.86 | 11.71 | 9.25 | 5.69 |
| SelectKBest + regressione logistica | selezione feature | 7.28 | 15.73 | 17.87 | 19.96 | 12.97 | 10.37 |
| Regressione logistica (sottocampionata) | campionamento | 3.32 | 12.93 | 18.33 | 20.94 | 13.89 | 6.20 |
| Random forest (sottocampionata) | campionamento | 4.44 | 19.72 | 9.29 | 10.86 | 10.38 | 9.75 |
| Regressione logistica (3 layer) | multi-layer | 4.12 | 12.07 | 14.22 | 13.97 | 10.24 | 5.59 |
| Somma pesata dei layer + MLP | multi-layer | 3.12 | 13.61 | 15.13 | 16.56 | 15.31 | 6.65 |
| Regressione logistica (Pascu: ultimo layer, C=1e6) | lineare | 5.20 | 16.27 | 16.64 | 21.59 | 18.38 | 8.23 |

## HuBERT Large (layer 15)

| classificatore | famiglia | dev | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
|---|---|---|---|---|---|---|---|
| Regressione logistica | lineare | 5.48 | 13.81 | 14.46 | 16.25 | 13.37 | 8.16 |
| SVM lineare | lineare | 5.96 | 13.88 | 14.68 | 15.70 | 12.55 | 8.40 |
| LDA | lineare | 4.76 | 15.73 | 13.46 | 14.30 | 12.38 | 7.87 |
| Albero di decisione | alberi | 13.16 | 31.74 | 21.31 | 20.04 | 20.51 | 20.67 |
| Random forest | alberi | 7.88 | 23.81 | 8.88 | 9.10 | 11.41 | 13.51 |
| Extra trees | alberi | 8.08 | 23.55 | 8.26 | 9.43 | 11.90 | 14.09 |
| Gradient boosting | alberi | 4.96 | 21.17 | 12.68 | 14.88 | 12.11 | 10.62 |
| XGBoost | alberi | 4.96 | 21.66 | 13.57 | 14.50 | 12.25 | 10.81 |
| AdaBoost | alberi | 4.76 | 20.88 | 17.89 | 21.02 | 14.82 | 10.64 |
| kNN (PCA 128) | vicini | 8.68 | 26.64 | 12.97 | 15.19 | 14.41 | 16.97 |
| Naive Bayes gaussiano | probabilistico | 14.72 | 36.70 | 36.18 | 38.66 | 28.41 | 17.18 |
| QDA (PCA 128) | probabilistico | 8.68 | 21.76 | 16.64 | 16.68 | 17.04 | 13.47 |
| SVM RBF (Nystroem 2000) | kernel | 5.00 | 19.84 | 16.31 | 20.91 | 13.14 | 11.21 |
| SVM RBF (esatta) | kernel | 4.56 | 20.26 | 13.59 | 19.09 | 12.49 | 11.31 |
| MLP (256) | rete | 5.28 | 29.88 | 22.71 | 28.19 | 16.91 | 11.15 |
| Due GMM (PCA 128, LLR) | probabilistico | 11.76 | 30.99 | 25.33 | 32.87 | 22.70 | 17.12 |
| Regressione logistica (C sul dev) | lineare | 3.20 | 14.40 | 12.79 | 13.92 | 10.48 | 7.77 |
| Voting (RF + GB + SVM lineare) | ensemble | 3.64 | 15.29 | 11.11 | 12.05 | 10.46 | 8.86 |
| SelectKBest + regressione logistica | selezione feature | 5.12 | 19.08 | 14.10 | 14.38 | 12.58 | 11.43 |
| Regressione logistica (sottocampionata) | campionamento | 3.92 | 14.18 | 13.76 | 15.67 | 12.15 | 9.00 |
| Random forest (sottocampionata) | campionamento | 8.08 | 24.08 | 9.61 | 9.46 | 11.52 | 13.93 |
| Regressione logistica (3 layer) | multi-layer | 5.32 | 14.54 | 15.73 | 17.38 | 14.14 | 7.63 |
| Somma pesata dei layer + MLP | multi-layer | 3.72 | 15.49 | 10.86 | 13.44 | 10.71 | 10.06 |
| Regressione logistica (Pascu: ultimo layer, C=1e6) | lineare | 6.36 | 14.40 | 16.53 | 20.05 | 15.70 | 11.10 |
