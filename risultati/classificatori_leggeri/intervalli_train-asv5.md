# Intervalli di confidenza al 95% dell'EER (bootstrap, 1000 ricampionamenti per classe)

Training su ASVspoof 5 intero. Il primo sistema di ogni lista è il miglior leggero su quella lista (scelto guardando il test, quindi ottimistico); la differenza è appaiata (stessi ricampionamenti).


## In-the-Wild

| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |
|---|---|---|---|---|
| WavLM Large · SVM lineare (migliore su questa lista) | 10.20 | [9.81, 10.55] | [+0.00, +0.00] | - |
| WavLM Base · Somma pesata dei layer + MLP (miglior media) | 12.91 | [12.48, 13.32] | [+2.28, +3.12] | sì |
| HuBERT Base · Voting (RF + GB + SVM lineare) (miglior media) | 18.60 | [18.15, 19.09] | [+7.95, +8.92] | sì |
| XLS-R 300M · Somma pesata dei layer + MLP (miglior media) | 14.41 | [13.97, 14.87] | [+3.82, +4.67] | sì |
| WavLM Large · Voting (RF + GB + SVM lineare) (miglior media) | 10.63 | [10.31, 11.03] | [+0.19, +0.74] | sì |
| HuBERT Large · Voting (RF + GB + SVM lineare) (miglior media) | 15.29 | [14.89, 15.73] | [+4.67, +5.54] | sì |
| MamBo (fine-tuning + RawBoost) | 18.87 | [18.53, 19.47] | [+8.24, +9.34] | sì |
| MHFA (fine-tuning + RawBoost) | 19.76 | [19.31, 20.27] | [+9.08, +10.15] | sì |

## 2019 LA eval

| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |
|---|---|---|---|---|
| WavLM Large · Extra trees (migliore su questa lista) | 7.89 | [7.63, 8.18] | [+0.00, +0.00] | - |
| WavLM Base · Somma pesata dei layer + MLP (miglior media) | 18.50 | [18.16, 18.79] | [+10.21, +10.94] | sì |
| HuBERT Base · Voting (RF + GB + SVM lineare) (miglior media) | 14.02 | [13.75, 14.28] | [+5.78, +6.43] | sì |
| XLS-R 300M · Somma pesata dei layer + MLP (miglior media) | 14.70 | [14.38, 14.98] | [+6.42, +7.12] | sì |
| WavLM Large · Voting (RF + GB + SVM lineare) (miglior media) | 11.86 | [11.60, 12.17] | [+3.67, +4.27] | sì |
| HuBERT Large · Voting (RF + GB + SVM lineare) (miglior media) | 11.11 | [10.84, 11.39] | [+2.88, +3.51] | sì |
| MamBo (fine-tuning + RawBoost) | 23.91 | [23.63, 24.57] | [+15.68, +16.73] | sì |
| MHFA (fine-tuning + RawBoost) | 22.05 | [21.71, 22.50] | [+13.76, +14.67] | sì |

## 2021 LA

| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |
|---|---|---|---|---|
| HuBERT Large · Random forest (migliore su questa lista) | 9.10 | [8.90, 9.31] | [+0.00, +0.00] | - |
| WavLM Base · Somma pesata dei layer + MLP (miglior media) | 20.81 | [20.52, 21.06] | [+11.40, +11.98] | sì |
| HuBERT Base · Voting (RF + GB + SVM lineare) (miglior media) | 15.07 | [14.85, 15.31] | [+5.72, +6.24] | sì |
| XLS-R 300M · Somma pesata dei layer + MLP (miglior media) | 14.23 | [14.01, 14.45] | [+4.87, +5.41] | sì |
| WavLM Large · Voting (RF + GB + SVM lineare) (miglior media) | 11.71 | [11.49, 11.93] | [+2.34, +2.87] | sì |
| HuBERT Large · Voting (RF + GB + SVM lineare) (miglior media) | 12.05 | [11.81, 12.27] | [+2.71, +3.17] | sì |
| MamBo (fine-tuning + RawBoost) | 26.28 | [26.08, 26.77] | [+16.93, +17.68] | sì |
| MHFA (fine-tuning + RawBoost) | 28.60 | [28.31, 28.95] | [+19.16, +19.88] | sì |

## 2021 DF

| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |
|---|---|---|---|---|
| WavLM Large · Voting (RF + GB + SVM lineare) (migliore su questa lista) | 9.25 | [9.04, 9.46] | [+0.00, +0.00] | - |
| WavLM Base · Somma pesata dei layer + MLP (miglior media) | 17.15 | [16.92, 17.43] | [+7.67, +8.20] | sì |
| HuBERT Base · Voting (RF + GB + SVM lineare) (miglior media) | 16.26 | [15.99, 16.52] | [+6.74, +7.30] | sì |
| XLS-R 300M · Somma pesata dei layer + MLP (miglior media) | 13.54 | [13.28, 13.75] | [+4.01, +4.54] | sì |
| WavLM Large · Voting (RF + GB + SVM lineare) (miglior media) | 9.25 | [9.04, 9.46] | [+0.00, +0.00] | no |
| HuBERT Large · Voting (RF + GB + SVM lineare) (miglior media) | 10.46 | [10.26, 10.69] | [+1.00, +1.44] | sì |
| MamBo (fine-tuning + RawBoost) | 15.50 | [15.33, 15.78] | [+6.05, +6.62] | sì |
| MHFA (fine-tuning + RawBoost) | 16.88 | [16.68, 17.12] | [+7.36, +7.91] | sì |

## ASVspoof 5

| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |
|---|---|---|---|---|
| WavLM Large · Regressione logistica (3 layer) (migliore su questa lista) | 5.59 | [5.51, 5.67] | [+0.00, +0.00] | - |
| WavLM Base · Somma pesata dei layer + MLP (miglior media) | 11.56 | [11.46, 11.65] | [+5.87, +6.07] | sì |
| HuBERT Base · Voting (RF + GB + SVM lineare) (miglior media) | 15.10 | [15.00, 15.20] | [+9.40, +9.62] | sì |
| XLS-R 300M · Somma pesata dei layer + MLP (miglior media) | 6.33 | [6.25, 6.41] | [+0.67, +0.82] | sì |
| WavLM Large · Voting (RF + GB + SVM lineare) (miglior media) | 5.69 | [5.62, 5.76] | [+0.04, +0.17] | sì |
| HuBERT Large · Voting (RF + GB + SVM lineare) (miglior media) | 8.86 | [8.78, 8.94] | [+3.19, +3.36] | sì |
| MamBo (fine-tuning + RawBoost) | 4.67 | [4.61, 4.75] | [-0.98, -0.83] | sì |
| MHFA (fine-tuning + RawBoost) | 6.58 | [6.50, 6.65] | [+0.90, +1.07] | sì |
