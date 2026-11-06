# Classificatori leggeri: riepilogo EER (%)

Training su ASVspoof 2019 LA train, scelte su 2019 LA dev, punteggio alto = bonafide.


## WavLM Base (layer 3)

| classificatore | famiglia | dev 2019 | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
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

| classificatore | famiglia | dev 2019 | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
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

| classificatore | famiglia | dev 2019 | In-the-Wild | 2019 LA eval | 2021 LA | 2021 DF | ASVspoof 5 |
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
