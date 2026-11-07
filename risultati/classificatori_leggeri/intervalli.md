# Intervalli di confidenza al 95% dell'EER (bootstrap, 1000 ricampionamenti per classe)

Training su 2019 LA. Il primo sistema di ogni lista è il miglior leggero su quella lista (scelto guardando il test, quindi ottimistico); la differenza è appaiata (stessi ricampionamenti).


## In-the-Wild

| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |
|---|---|---|---|---|
| HuBERT Large · SVM RBF (esatta) (migliore su questa lista) | 15.00 | [14.61, 15.40] | [+0.00, +0.00] | - |
| WavLM Base · MLP (256) (miglior media) | 17.31 | [16.89, 17.74] | [+1.78, +2.85] | sì |
| HuBERT Base · Somma pesata dei layer + MLP (miglior media) | 24.63 | [24.18, 25.12] | [+9.06, +10.21] | sì |
| XLS-R 300M · Regressione logistica (Pascu: ultimo layer, C=1e6) (miglior media) | 21.58 | [21.13, 22.01] | [+6.05, +7.09] | sì |
| WavLM Large · Somma pesata dei layer + MLP (miglior media) | 21.39 | [20.93, 21.87] | [+5.89, +6.93] | sì |
| HuBERT Large · Voting (RF + GB + SVM lineare) (miglior media) | 17.08 | [16.68, 17.48] | [+1.68, +2.52] | sì |
| MamBo (fine-tuning + RawBoost) | 10.32 | [9.98, 10.66] | [-5.17, -4.19] | sì |
| Nes2Net-X (fine-tuning + RawBoost) | 17.65 | [17.28, 18.10] | [+2.18, +3.17] | sì |
| MHFA (fine-tuning + RawBoost) | 16.11 | [15.73, 16.53] | [+0.63, +1.63] | sì |

## 2019 LA eval

| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |
|---|---|---|---|---|
| XLS-R 300M · Somma pesata dei layer + MLP (migliore su questa lista) | 0.27 | [0.22, 0.32] | [+0.00, +0.00] | - |
| WavLM Base · MLP (256) (miglior media) | 1.32 | [1.20, 1.45] | [+0.94, +1.19] | sì |
| HuBERT Base · Somma pesata dei layer + MLP (miglior media) | 2.03 | [1.86, 2.22] | [+1.61, +1.95] | sì |
| XLS-R 300M · Regressione logistica (Pascu: ultimo layer, C=1e6) (miglior media) | 1.05 | [0.95, 1.16] | [+0.68, +0.90] | sì |
| WavLM Large · Somma pesata dei layer + MLP (miglior media) | 0.34 | [0.28, 0.42] | [+0.01, +0.15] | sì |
| HuBERT Large · Voting (RF + GB + SVM lineare) (miglior media) | 0.57 | [0.46, 0.67] | [+0.20, +0.41] | sì |
| MamBo (fine-tuning + RawBoost) | 0.72 | [0.60, 0.87] | [+0.34, +0.60] | sì |
| Nes2Net-X (fine-tuning + RawBoost) | 1.10 | [1.00, 1.21] | [+0.72, +0.95] | sì |
| MHFA (fine-tuning + RawBoost) | 2.16 | [2.04, 2.30] | [+1.78, +2.04] | sì |

## 2021 LA

| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |
|---|---|---|---|---|
| HuBERT Large · MLP (256) (migliore su questa lista) | 4.71 | [4.54, 4.90] | [+0.00, +0.00] | - |
| WavLM Base · MLP (256) (miglior media) | 13.44 | [13.14, 13.73] | [+8.43, +9.00] | sì |
| HuBERT Base · Somma pesata dei layer + MLP (miglior media) | 7.03 | [6.82, 7.22] | [+2.13, +2.50] | sì |
| XLS-R 300M · Regressione logistica (Pascu: ultimo layer, C=1e6) (miglior media) | 9.90 | [9.70, 10.12] | [+4.97, +5.43] | sì |
| WavLM Large · Somma pesata dei layer + MLP (miglior media) | 7.30 | [7.06, 7.51] | [+2.34, +2.81] | sì |
| HuBERT Large · Voting (RF + GB + SVM lineare) (miglior media) | 6.89 | [6.70, 7.09] | [+1.99, +2.37] | sì |
| MamBo (fine-tuning + RawBoost) | 1.76 | [1.64, 1.90] | [-3.15, -2.74] | sì |
| Nes2Net-X (fine-tuning + RawBoost) | 2.71 | [2.60, 2.83] | [-2.21, -1.83] | sì |
| MHFA (fine-tuning + RawBoost) | 4.42 | [4.29, 4.56] | [-0.47, -0.10] | sì |

## 2021 DF

| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |
|---|---|---|---|---|
| HuBERT Large · SVM RBF (Nystroem 2000) (migliore su questa lista) | 5.56 | [5.39, 5.72] | [+0.00, +0.00] | - |
| WavLM Base · MLP (256) (miglior media) | 11.03 | [10.79, 11.29] | [+5.22, +5.70] | sì |
| HuBERT Base · Somma pesata dei layer + MLP (miglior media) | 13.68 | [13.39, 13.97] | [+7.84, +8.37] | sì |
| XLS-R 300M · Regressione logistica (Pascu: ultimo layer, C=1e6) (miglior media) | 8.41 | [8.17, 8.68] | [+2.62, +3.11] | sì |
| WavLM Large · Somma pesata dei layer + MLP (miglior media) | 9.23 | [8.92, 9.48] | [+3.38, +3.90] | sì |
| HuBERT Large · Voting (RF + GB + SVM lineare) (miglior media) | 5.98 | [5.79, 6.19] | [+0.24, +0.58] | sì |
| MamBo (fine-tuning + RawBoost) | 5.86 | [5.68, 6.05] | [+0.09, +0.50] | sì |
| Nes2Net-X (fine-tuning + RawBoost) | 7.92 | [7.74, 8.13] | [+2.14, +2.57] | sì |
| MHFA (fine-tuning + RawBoost) | 10.80 | [10.55, 11.08] | [+4.96, +5.51] | sì |

## ASVspoof 5

| sistema | EER (%) | IC 95% | differenza dal primo (IC 95%) | significativa |
|---|---|---|---|---|
| XLS-R 300M · Regressione logistica (Pascu: ultimo layer, C=1e6) (migliore su questa lista) | 14.59 | [14.47, 14.70] | [+0.00, +0.00] | - |
| WavLM Base · MLP (256) (miglior media) | 24.20 | [24.10, 24.36] | [+9.53, +9.74] | sì |
| HuBERT Base · Somma pesata dei layer + MLP (miglior media) | 29.22 | [29.08, 29.36] | [+14.48, +14.78] | sì |
| XLS-R 300M · Regressione logistica (Pascu: ultimo layer, C=1e6) (miglior media) | 14.59 | [14.47, 14.70] | [+0.00, +0.00] | no |
| WavLM Large · Somma pesata dei layer + MLP (miglior media) | 17.00 | [16.89, 17.11] | [+2.32, +2.50] | sì |
| HuBERT Large · Voting (RF + GB + SVM lineare) (miglior media) | 18.17 | [18.05, 18.29] | [+3.49, +3.67] | sì |
| MamBo (fine-tuning + RawBoost) | 15.48 | [15.35, 15.67] | [+0.82, +1.04] | sì |
| Nes2Net-X (fine-tuning + RawBoost) | 12.36 | [12.25, 12.47] | [-2.31, -2.14] | sì |
| MHFA (fine-tuning + RawBoost) | 16.88 | [16.79, 17.02] | [+2.22, +2.40] | sì |
