# Training e test dei modelli. Le funzioni si chiamano dai notebook 02 (training) e 03 (test).
#
# Ogni esperimento ha una cartella artifacts/esperimenti/<modello>_<dataset>_<augmentation>/seed<k>/:
#   info.json                   modello, dataset, augmentation, seed, iperparametri
#   log.csv                     una riga per epoca (loss di train, EER sul dev, tempo)
#   migliore.pt                 pesi dell'epoca con l'EER sul dev più basso
#   ultimo.pt                   ultima epoca finita, per riprendere se il training si interrompe
#   completato.json             scritto alla fine: se c'è, il training di quel seed è finito
#   punteggi_<lista>.csv        punteggi del modello migliore su una lista di test
# L'epoca migliore si sceglie solo sul dev; i test si fanno dopo, con il modello migliore.

import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tesi import percorsi
from tesi.augmentation import crea_augmentation
from tesi.dati import AudioDataset, carica_lista, leggi_audio
from tesi.metriche import eer
from tesi.modelli import IPERPARAMETRI, conta_parametri, crea_modello


def cartella_esperimento(modello, dataset, augmentation, seed):
    return percorsi.ESPERIMENTI / f"{modello}_{dataset}_{augmentation}" / f"seed{seed}"


def fissa_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def focal_loss(logit, label, alpha, gamma=2.0):
    """Focal loss (Lin et al., 2017): pesa di più le clip classificate male. alpha = peso di ogni classe."""
    logp = F.log_softmax(logit.float(), dim=-1).gather(1, label[:, None]).squeeze(1)
    peso = torch.tensor(alpha, device=logit.device)[label]
    return (-peso * (1 - logp.exp()) ** gamma * logp).mean()


def ranking_loss(logit, label, margine=1.0):
    """
    Ranking loss a coppie di TFPARN: ogni bonafide del batch deve avere un punteggio
    più alto di ogni spoof di almeno `margine`.
    """
    punteggio = logit[:, 0].float() - logit[:, 1].float()
    bona, spoof = punteggio[label == 0], punteggio[label == 1]
    if len(bona) == 0 or len(spoof) == 0:
        return punteggio.sum() * 0
    return F.relu(margine - (bona[:, None] - spoof[None, :])).mean()


def fattore_lr(passo, totale, warmup):
    """Warmup lineare, poi coseno fino al 10% del learning rate iniziale."""
    if passo < warmup:
        return passo / max(1, warmup)
    avanzamento = (passo - warmup) / max(1, totale - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * avanzamento))


def autocast(device):
    """bfloat16 sulla GPU per risparmiare memoria; sulla CPU si resta in float32."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def calcola_punteggi(modello, loader, device):
    """Punteggio di ogni clip: logit bonafide - logit spoof (alto = bonafide)."""
    modello.eval()
    punteggi = []
    for wave, _ in tqdm(loader, desc="punteggi", leave=False):
        with autocast(device):
            logit = modello(wave.to(device))
        logit = logit.float()
        punteggi.append((logit[:, 0] - logit[:, 1]).cpu().numpy())
    return np.concatenate(punteggi)


def carica_pesi(modello, file):
    """
    Carica i pesi salvati. L'unico parametro che può mancare o avanzare è masked_spec_embed
    del modello SSL, che il training non usa (il masking è spento).
    """
    pesi = torch.load(file, map_location="cpu", weights_only=True)
    risultato = modello.load_state_dict(pesi, strict=False)
    diversi = [k for k in risultato.missing_keys + risultato.unexpected_keys if "masked_spec_embed" not in k]
    if diversi:
        raise RuntimeError(f"i pesi di {file} non corrispondono al modello, per esempio: {diversi[:5]}")


def addestra(modello_nome, dataset, augmentation="nessuna", seed=1, num_workers=4, **modifiche):
    """
    Allena un modello su <dataset>_train e sceglie l'epoca migliore su <dataset>_dev.
    modifiche: per cambiare un iperparametro al volo, per esempio addestra(..., lr_backend=1e-3).
    """
    cartella = cartella_esperimento(modello_nome, dataset, augmentation, seed)
    if (cartella / "completato.json").exists():
        print("Già fatto:", cartella)
        return cartella
    cartella.mkdir(parents=True, exist_ok=True)
    ip = {**IPERPARAMETRI[modello_nome], **modifiche}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- dati -----------------------------------------------------------------
    train, dev = carica_lista(f"{dataset}_train"), carica_lista(f"{dataset}_dev")
    aug = crea_augmentation(augmentation)
    if aug is not None:
        # controllo che l'augmentation cambi davvero la clip
        wave = leggi_audio(train["percorso"][0])
        assert not np.allclose(aug.trasforma(wave, np.random.default_rng(0)), wave), "l'augmentation non cambia la clip"
    # focal loss: peso di ogni classe inversamente proporzionale al numero di clip
    # (0,5 e 0,5 se il train è bilanciato; su ASVspoof 2019 LA circa 0,9 per i bonafide)
    n_bona, n_spoof = (train["label"] == "bonafide").sum(), (train["label"] == "spoof").sum()
    alpha = (n_spoof / len(train), n_bona / len(train))

    fissa_seed(seed)
    loader_train = DataLoader(AudioDataset(train, augmentation=aug), batch_size=ip["batch"], shuffle=True,
                              drop_last=True, num_workers=num_workers, persistent_workers=num_workers > 0,
                              pin_memory=device.type == "cuda")
    loader_dev = DataLoader(AudioDataset(dev), batch_size=2 * ip["batch"], num_workers=num_workers,
                            persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")
    y_dev = (dev["label"] == "spoof").astype(int).to_numpy()

    # --- modello e ottimizzatore -----------------------------------------------
    modello = crea_modello(modello_nome, specaugment=augmentation == "specaugment").to(device)
    parametri = [(n, p) for n, p in modello.named_parameters() if p.requires_grad]
    if ip["lr_backend"] is None:
        gruppi = [{"params": [p for _, p in parametri], "lr": ip["lr"]}]
    else:
        # due learning rate: uno per il modello SSL (parametri "ssl.*"), uno per il resto
        gruppi = [{"params": [p for n, p in parametri if n.startswith("ssl.")], "lr": ip["lr"]},
                  {"params": [p for n, p in parametri if not n.startswith("ssl.")], "lr": ip["lr_backend"]}]
    ottimizzatore = torch.optim.AdamW(gruppi, betas=(0.9, 0.95), weight_decay=ip["weight_decay"])
    passi_totali = len(loader_train) // ip["accumulo"] * ip["epoche"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        ottimizzatore, lambda passo: fattore_lr(passo, passi_totali, int(ip["warmup"] * passi_totali)))

    info = {"modello": modello_nome, "dataset": dataset, "augmentation": augmentation, "seed": seed,
            "iperparametri": ip, "alpha_focal": alpha, "clip_train": len(train), "clip_dev": len(dev),
            "parametri_milioni": conta_parametri(modello)}
    (cartella / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{cartella.parent.name} seed {seed} su {device}: {len(train)} clip di train, {len(dev)} di dev, "
          f"parametri (totali, allenabili) = {info['parametri_milioni']} milioni")

    # --- ripresa dopo un'interruzione --------------------------------------------
    prima_epoca, migliore_eer, epoca_migliore, attesa = 1, float("inf"), 0, 0
    if (cartella / "ultimo.pt").exists():
        stato = torch.load(cartella / "ultimo.pt", map_location="cpu", weights_only=False)
        modello.load_state_dict(stato["modello"])
        ottimizzatore.load_state_dict(stato["ottimizzatore"])
        scheduler.load_state_dict(stato["scheduler"])
        prima_epoca = stato["epoca"] + 1
        migliore_eer, epoca_migliore, attesa = stato["migliore_eer"], stato["epoca_migliore"], stato["attesa"]
        torch.set_rng_state(stato["rng"])
        del stato
        print(f"Riprendo dall'epoca {prima_epoca}")
    file_log = cartella / "log.csv"
    if file_log.exists():
        log = pd.read_csv(file_log)
        log[log["epoca"] < prima_epoca].to_csv(file_log, index=False)   # toglie l'epoca interrotta

    # --- training ------------------------------------------------------------------
    for epoca in range(prima_epoca, ip["epoche"] + 1):
        if attesa >= ip["pazienza"]:
            break
        modello.train()
        inizio, somma_loss = time.time(), 0.0
        barra = tqdm(loader_train, desc=f"epoca {epoca}", leave=False)
        for i, (wave, label) in enumerate(barra, 1):
            wave, label = wave.to(device), label.to(device)
            with autocast(device):
                logit = modello(wave)
            loss = focal_loss(logit, label, alpha)
            if ip["peso_ranking"] > 0:
                loss = loss + ip["peso_ranking"] * ranking_loss(logit, label)
            assert torch.isfinite(loss), f"loss non valida all'epoca {epoca}: prova un learning rate più basso"
            (loss / ip["accumulo"]).backward()
            if i % ip["accumulo"] == 0:
                torch.nn.utils.clip_grad_norm_(modello.parameters(), 1.0)
                ottimizzatore.step()
                scheduler.step()
                ottimizzatore.zero_grad()
            somma_loss += loss.item()
            barra.set_postfix(loss=f"{somma_loss / i:.4f}")

        eer_dev = 100 * eer(y_dev, calcola_punteggi(modello, loader_dev, device))
        riga = {"epoca": epoca, "loss_train": somma_loss / len(loader_train), "eer_dev": eer_dev,
                "lr": scheduler.get_last_lr()[0], "minuti": round((time.time() - inizio) / 60, 1)}
        pd.DataFrame([riga]).to_csv(file_log, mode="a", header=not file_log.exists(), index=False)
        print(f"epoca {epoca}: loss {riga['loss_train']:.4f}, EER dev {eer_dev:.2f}%, {riga['minuti']} min")

        if eer_dev < migliore_eer:
            migliore_eer, epoca_migliore, attesa = eer_dev, epoca, 0
            torch.save(modello.state_dict(), cartella / "migliore.pt")
        else:
            attesa += 1
        torch.save({"modello": modello.state_dict(), "ottimizzatore": ottimizzatore.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoca": epoca, "migliore_eer": migliore_eer,
                    "epoca_migliore": epoca_migliore, "attesa": attesa, "rng": torch.get_rng_state()},
                   cartella / "ultimo.pt")

    fine = {"epoca_migliore": epoca_migliore, "eer_dev": migliore_eer}
    (cartella / "completato.json").write_text(json.dumps(fine, indent=2), encoding="utf-8")
    (cartella / "ultimo.pt").unlink()   # serve solo per riprendere e pesa qualche GB
    print(f"Finito: epoca migliore {epoca_migliore}, EER dev {migliore_eer:.2f}%")
    return cartella


def testa(cartella, nome_lista, batch=16, num_workers=4, blocco=50_000):
    """
    Punteggi del modello migliore su una lista di test, salvati in punteggi_<nome_lista>.csv.
    Le liste lunghe (eval completi) si fanno a blocchi: se si interrompe, riparte dal blocco
    dove era arrivato.
    """
    file = cartella / f"punteggi_{nome_lista}.csv"
    if file.exists():
        print("Già fatto:", file.name)
        return pd.read_csv(file, keep_default_na=False)
    info = json.loads((cartella / "info.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modello = crea_modello(info["modello"])
    carica_pesi(modello, cartella / "migliore.pt")
    modello.to(device)

    lista = carica_lista(nome_lista)
    parziale = cartella / f"punteggi_{nome_lista}.parziale.csv"
    fatti = len(pd.read_csv(parziale)) if parziale.exists() else 0
    for inizio in range(fatti, len(lista), blocco):
        pezzo = lista.iloc[inizio:inizio + blocco].copy()
        loader = DataLoader(AudioDataset(pezzo), batch_size=batch, num_workers=num_workers)
        print(f"{nome_lista}: clip {inizio}-{inizio + len(pezzo)} di {len(lista)}")
        pezzo["punteggio"] = calcola_punteggi(modello, loader, device)
        pezzo.to_csv(parziale, mode="a", header=inizio == 0, index=False)
    parziale.rename(file)
    return pd.read_csv(file, keep_default_na=False)
