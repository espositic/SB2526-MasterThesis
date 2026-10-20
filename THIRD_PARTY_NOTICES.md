# Codice di terze parti

Alcune parti di `tesi/` sono port di codice pubblicato con licenza MIT o Apache 2.0.
La licenza chiede di riportare la nota di copyright e il testo della licenza,
che sono qui sotto. Gli altri modelli sono re-implementati dai paper, senza
copiare codice.

## Port di codice con licenza MIT

| File | Origine | Copyright |
|---|---|---|
| `tesi/modelli/mambo.py`, `tesi/modelli/hydra.py` | [saki-ciallo/MamBo-for-ADD](https://github.com/saki-ciallo/MamBo-for-ADD) | Copyright (c) 2026 ayase |
| `tesi/augmentation.py` (RawBoost) | [TakHemlata/RawBoost-antispoofing](https://github.com/TakHemlata/RawBoost-antispoofing), nella versione di MamBo-for-ADD | Copyright (c) 2021 Hemlata; Copyright (c) 2026 ayase |
| `tesi/modelli/tfparn.py` | [SomeB1oody/TFPARN](https://github.com/SomeB1oody/TFPARN) | Copyright (c) 2025 SomeB1oody |

Testo della licenza, uguale per tutti:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Port di codice con licenza Apache 2.0

| File | Origine |
|---|---|
| `tesi/modelli/nes2net.py` (back-end Nested Res2Net TDNN) | [Liu-Tianchi/Nes2Net_ASVspoof_ITW](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW), branch `asvspoof5`, `models/wavlm_nes2net.py` |
| `tesi/augmentation.py` (classe `RumoreRiverbero`) | stesso repository, `musan.py` e `rir.py` |

Il codice è distribuito con la [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
Modifiche: nomi e commenti in italiano, WavLM caricato da Hugging Face, solo la configurazione
usata su ASVspoof 5 (pooling medio, niente ASTP), generatore casuale passato come argomento.

## Re-implementazioni dai paper

Per questo modello il codice degli autori non ha licenza, quindi non se ne
può copiare nulla: è scritto da zero seguendo il paper.

| File | Paper | Codice degli autori (senza licenza) |
|---|---|---|
| `tesi/modelli/mhfa.py` | Peng et al., SLT 2022; Stourbe et al., arXiv 2409.05032 | JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling |
