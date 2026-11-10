# Baseline semplice: modello SSL congelato + media sul tempo + classificatore classico.
#
# 1. Ogni clip passa in WavLM o XLS-R senza allenarlo (solo inferenza).
# 2. Per ogni layer si fa la media sul tempo: una clip diventa una matrice (n_layer, dim).
# 3. Sopra si allena una regressione logistica (scikit-learn), un layer alla volta.
#
# Le medie si salvano in artifacts/feature_ssl/<ssl>/<lista>.npy (float16), così
# la parte di statistica gira solo sulla CPU e si può ripetere quante volte si vuole.

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tesi import percorsi
from tesi.dati import AudioDataset, carica_lista
from tesi.modelli.ssl import WAVLM_BASE, carica_ssl, normalizza_wave

CARTELLA = percorsi.PROGETTO / "artifacts" / "feature_ssl"


def nome_cartella(nome_ssl):
    """'microsoft/wavlm-base' -> 'wavlm-base'"""
    return nome_ssl.split("/")[-1]


@torch.no_grad()
def estrai_medie(nome_lista, nome_ssl=WAVLM_BASE, batch=16, num_workers=4, blocco=50_000):
    """
    Media sul tempo di ogni layer per tutte le clip della lista. Se il file c'è già lo rilegge.
    Le medie si scrivono direttamente sul disco (memmap), un blocco alla volta: le liste intere
    con XLS-R non starebbero nella RAM, e se si interrompe si riparte dall'ultimo blocco finito.
    """
    file = CARTELLA / nome_cartella(nome_ssl) / f"{nome_lista}.npy"
    if file.exists():
        return np.load(file, mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl, normalizza = carica_ssl(nome_ssl, congelato=True, checkpointing=False)
    ssl.eval().to(device)

    lista = carica_lista(nome_lista)
    file.parent.mkdir(parents=True, exist_ok=True)
    parziale = file.with_name(f"{nome_lista}.parziale.npy")
    contatore = file.with_name(f"{nome_lista}.parziale.txt")   # quante clip sono già scritte
    if parziale.exists() and contatore.exists():
        medie = np.lib.format.open_memmap(parziale, mode="r+")
        fatti = int(contatore.read_text())
    else:
        forma = (len(lista), ssl.config.num_hidden_layers + 1, ssl.config.hidden_size)
        medie = np.lib.format.open_memmap(parziale, mode="w+", dtype=np.float16, shape=forma)
        fatti = 0

    for inizio in range(fatti, len(lista), blocco):
        pezzo = lista.iloc[inizio:inizio + blocco]
        loader = DataLoader(AudioDataset(pezzo), batch_size=batch, num_workers=num_workers)
        print(f"{nome_lista}: clip {inizio}-{inizio + len(pezzo)} di {len(lista)}")
        pos = inizio
        for wave, _ in tqdm(loader, desc=nome_lista):
            wave = wave.to(device)
            if normalizza:
                wave = normalizza_wave(wave)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                layer = ssl(wave, output_hidden_states=True).hidden_states   # n_layer x (batch, frame, dim)
            m = torch.stack([h.float().mean(dim=1) for h in layer], dim=1).cpu().numpy().astype(np.float16)
            medie[pos:pos + len(m)] = m
            pos += len(m)
        medie.flush()
        contatore.write_text(str(pos))

    del medie   # chiude il file, altrimenti su Windows non si può rinominare
    # prima le etichette e poi le medie: se c'è il file delle medie, la lista è completa
    np.save(file.with_name(f"{nome_lista}_label.npy"), etichette(nome_lista))
    parziale.rename(file)
    contatore.unlink()
    return np.load(file, mmap_mode="r")


def etichette(nome_lista):
    """0 = bonafide, 1 = spoof, come nel resto della tesi."""
    return (carica_lista(nome_lista)["label"] == "spoof").astype(int).to_numpy()


def allena_logistica(x, y, c=1.0):
    """Standardizzazione + regressione logistica. x: (n_clip, dim)."""
    modello = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    return modello.fit(x.astype(np.float32), y)


def punteggi(modello, x):
    """Punteggio alto = bonafide: è il log-odds di bonafide contro spoof."""
    return -modello.decision_function(x.astype(np.float32))
