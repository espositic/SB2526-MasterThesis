# Backend per interrogare un modello già allenato: si manda un file audio, risponde con il punteggio.
#
# Avvio (dalla cartella del repo, con il .venv attivo):
#   uvicorn backend:app
# poi aprire http://127.0.0.1:8000/docs, "Try it out" su /predict, scegliere un file e premere Execute.
#
# L'audio è preparato come nel test (16 kHz mono, primi 66.800 campioni) e il punteggio è
# logit bonafide - logit spoof, come in training.calcola_punteggi: alto = bonafide.

import json
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile

from tesi import training
from tesi.dati import FREQ, N_CAMPIONI, leggi_audio, taglia
from tesi.modelli import crea_modello

# esperimento da usare: basta cambiare queste quattro righe
MODELLO = "mambo"
DATASET = "asvspoof2019"
AUGMENTATION = "rawboost"
SEED = 1
# sopra la soglia la clip è bonafide. 0 vuol dire "il logit bonafide supera quello spoof";
# per la tesi è più corretta la soglia dell'EER sul dev
SOGLIA = 0.0

cartella = training.cartella_esperimento(MODELLO, DATASET, AUGMENTATION, SEED)
info = json.loads((cartella / "info.json").read_text(encoding="utf-8"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modello = crea_modello(info["modello"])
training.carica_pesi(modello, cartella / "migliore.pt")
modello.to(device).eval()

app = FastAPI(title="Anti-spoofing vocale")


@app.get("/")
def stato():
    return {"esperimento": f"{cartella.parent.name}/{cartella.name}", "device": str(device), "soglia": SOGLIA}


@app.post("/predict")
async def predict(file: UploadFile):
    # il file va scritto su disco per poterlo leggere come nel test (anche con ffmpeg, per mp3 e simili)
    with tempfile.TemporaryDirectory() as tmp:
        percorso = Path(tmp) / Path(file.filename or "audio").name
        percorso.write_bytes(await file.read())
        wave = leggi_audio(percorso)
    durata = len(wave) / FREQ
    x = torch.from_numpy(taglia(wave, N_CAMPIONI).copy()).unsqueeze(0).to(device)

    with torch.no_grad(), training.autocast(device):
        logit = modello(x)
    logit = logit.float()[0]
    punteggio = float(logit[0] - logit[1])   # alto = bonafide
    return {
        "file": file.filename,
        "durata_s": round(durata, 2),
        "punteggio": round(punteggio, 4),
        "verdetto": "bonafide" if punteggio > SOGLIA else "spoof",
    }
