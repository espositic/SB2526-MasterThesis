# Dataset: download, lettura dei protocolli, liste di file e Dataset PyTorch.
#
# Ogni esperimento usa delle "liste": CSV salvati in manifests/ con le colonne
#   percorso  file audio, relativo alla cartella data/
#   label     "bonafide" oppure "spoof"
#   attacco   sistema che ha generato lo spoof ("-" per i bonafide o se non si sa)
#   codec     compressione applicata alla clip ("-" se nessuna o se non si sa)
# Nei tensori la label diventa un numero: 0 = bonafide, 1 = spoof.

import shutil
import urllib.request
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from tesi import percorsi

FREQ = 16000          # tutti i modelli lavorano a 16 kHz
N_CAMPIONI = 66800    # 4,175 secondi per clip, come in XLSR-MamBo
COLONNE = ["percorso", "label", "attacco", "codec"]


# ---------------------------------------------------------------------------
# Download ed estrazione
# ---------------------------------------------------------------------------

def scarica(url, file):
    """Scarica un file con la barra di avanzamento. Se c'è già non fa niente."""
    file = Path(file)
    if file.exists():
        print("Già scaricato:", file.name)
        return file
    file.parent.mkdir(parents=True, exist_ok=True)
    parziale = file.with_name(file.name + ".parziale")   # rinominato solo a download finito
    with tqdm(unit="B", unit_scale=True, desc=file.name) as barra:
        def avanzamento(n_blocchi, dim_blocco, totale):
            barra.total = totale
            barra.update(n_blocchi * dim_blocco - barra.n)
        urllib.request.urlretrieve(url, parziale, avanzamento)
    parziale.rename(file)
    return file


def estrai(archivio, cartella):
    """Estrae zip, tar o tar.gz. Un file nascosto ricorda che l'estrazione è finita."""
    archivio, cartella = Path(archivio), Path(cartella)
    fatto = cartella / f".{archivio.name}.estratto"
    if fatto.exists() or (archivio.parent / f".{archivio.name}.extracted").exists():   # anche il segno del codice vecchio
        print("Già estratto:", archivio.name)
        return
    print("Estrazione di", archivio.name, "...")
    cartella.mkdir(parents=True, exist_ok=True)
    if archivio.suffix == ".zip":
        shutil.unpack_archive(archivio, cartella)
    else:
        shutil.unpack_archive(archivio, cartella, filter="data")
    fatto.touch()


def scarica_asvspoof5(split, cancella_tar=False):
    """ASVspoof 5 da Hugging Face (jungjee/asvspoof5): protocolli e TAR audio dello split, uno alla volta."""
    from huggingface_hub import HfApi, hf_hub_download

    if (percorsi.ASVSPOOF5 / f".{split}.complete").exists():
        print(f"ASVspoof 5 {split}: già pronto")
        return
    schema = {"train": "flac_T_*.tar", "dev": "flac_D_*.tar", "eval": "flac_E_*.tar"}[split]
    tutti = HfApi().list_repo_files("jungjee/asvspoof5", repo_type="dataset")
    da_scaricare = sorted(f for f in tutti if f == "ASVspoof5_protocols.tar" or fnmatch(f, schema))
    for nome in da_scaricare:
        tar = percorsi.ASVSPOOF5 / nome
        if (percorsi.ASVSPOOF5 / f".{nome}.estratto").exists() or (percorsi.ASVSPOOF5 / f".{nome}.extracted").exists():
            continue
        hf_hub_download("jungjee/asvspoof5", nome, repo_type="dataset", local_dir=percorsi.ASVSPOOF5)
        estrai(tar, percorsi.ASVSPOOF5)
        if cancella_tar:
            tar.unlink()
    (percorsi.ASVSPOOF5 / f".{split}.complete").touch()


# ---------------------------------------------------------------------------
# Lettura dei protocolli: ogni funzione restituisce una lista con COLONNE
# ---------------------------------------------------------------------------

def trova(cartella, nome):
    """Cerca un file (o una cartella) dentro `cartella`, fino a tre sottocartelle più in basso."""
    for schema in (nome, f"*/{nome}", f"*/*/{nome}", f"*/*/*/{nome}"):
        trovato = next(Path(cartella).glob(schema), None)
        if trovato is not None:
            return trovato
    raise FileNotFoundError(f"'{nome}' non trovato in {cartella}: hai scaricato ed estratto il dataset?")


def relativo(cartella):
    """Percorso di una cartella rispetto a data/, con le barre "/" anche su Windows."""
    return Path(cartella).relative_to(percorsi.DATI).as_posix()


def lista_asvspoof5(split):
    """Tutto il protocollo di uno split di ASVspoof 5 (train, dev o eval, Track 1)."""
    tsv = {"train": "ASVspoof5.train.tsv", "dev": "ASVspoof5.dev.track_1.tsv", "eval": "ASVspoof5.eval.track_1.tsv"}[split]
    audio = {"train": "flac_T", "dev": "flac_D", "eval": "flac_E_eval"}[split]
    # colonne: speaker, file, genere, codec, qualità codec, seed codec, tag attacco, attacco, label, -
    p = pd.read_csv(trova(percorsi.ASVSPOOF5, tsv), sep=r"\s+", header=None)
    return pd.DataFrame({
        "percorso": f"asvspoof5/{audio}/" + p[1] + ".flac",
        "label": p[8],
        "attacco": np.where(p[8] == "bonafide", "-", p[7]),
        "codec": p[3],
    })


def lista_da_manifest(nome_manifest, split):
    """
    Le stesse clip dei sottoinsiemi di ASVspoof 5 scelti all'inizio della tesi
    (manifests/asvspoof5_<split>_n..._seed42_dur1-15.csv), con attacco e codec presi dal protocollo.
    """
    vecchio = pd.read_csv(percorsi.LISTE / nome_manifest, comment="#")
    tutto = lista_asvspoof5(split)
    tutto["file_name"] = tutto["percorso"].str.split("/").str[-1].str.removesuffix(".flac")
    scelti = vecchio[["file_name"]].merge(tutto, on="file_name", how="left")
    assert scelti["percorso"].notna().all(), "alcuni file del manifest non sono nel protocollo"
    return scelti[COLONNE]


def lista_asvspoof2019(split):
    """ASVspoof 2019 LA intero: train (25.380 clip), dev (24.844) o eval (71.237)."""
    nome = {"train": "train.trn", "dev": "dev.trl", "eval": "eval.trl"}[split]
    protocollo = trova(percorsi.ASVSPOOF2019, f"ASVspoof2019.LA.cm.{nome}.txt")
    # colonne: speaker, file, -, attacco ("-" per i bonafide), label
    p = pd.read_csv(protocollo, sep=r"\s+", header=None)
    audio = protocollo.parent.parent / f"ASVspoof2019_LA_{split}" / "flac"
    return pd.DataFrame({
        "percorso": relativo(audio) + "/" + p[1] + ".flac",
        "label": p[4],
        "attacco": p[3],
        "codec": "-",
    })


def lista_asvspoof2021(traccia):
    """ASVspoof 2021 LA o DF (traccia = "LA" o "DF"): solo la parte "eval" del protocollo ufficiale."""
    chiavi = trova(percorsi.ASVSPOOF2021, f"keys/{traccia}/CM/trial_metadata.txt")
    # colonne: speaker, file, codec, trasmissione/sorgente, attacco, label, trim, fase (progress/eval/hidden_track), ...
    p = pd.read_csv(chiavi, sep=r"\s+", header=None, usecols=range(8))
    p = p[p[7] == "eval"]
    audio = trova(percorsi.ASVSPOOF2021, f"ASVspoof2021_{traccia}_eval") / "flac"
    return pd.DataFrame({
        "percorso": relativo(audio) + "/" + p[1] + ".flac",
        "label": p[5],
        "attacco": np.where(p[5] == "bonafide", "-", p[4]),
        "codec": p[2],
    }).reset_index(drop=True)


def lista_in_the_wild():
    """In-the-Wild (Müller et al., 2022): 31.779 clip di personaggi pubblici, solo per il test."""
    meta = trova(percorsi.IN_THE_WILD, "meta.csv")
    m = pd.read_csv(meta)   # colonne: file, speaker, label ("bona-fide" o "spoof")
    return pd.DataFrame({
        "percorso": relativo(meta.parent) + "/" + m["file"],
        "label": m["label"].replace({"bona-fide": "bonafide"}),
        "attacco": "-",
        "codec": "-",
    })


def sottoinsieme(lista, n_per_classe, seed=42):
    """n_per_classe clip scelte a caso per ogni classe (o tutte, se sono di meno)."""
    parti = []
    for label in ("bonafide", "spoof"):
        classe = lista[lista["label"] == label]
        parti.append(classe.sample(n=min(n_per_classe, len(classe)), random_state=seed))
    return pd.concat(parti).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Salvataggio e caricamento delle liste
# ---------------------------------------------------------------------------

def salva_lista(lista, nome):
    assert lista["label"].isin(["bonafide", "spoof"]).all(), "label diverse da bonafide/spoof"
    percorsi.LISTE.mkdir(exist_ok=True)
    lista[COLONNE].to_csv(percorsi.LISTE / f"{nome}.csv", index=False)
    print(f"{nome}: {len(lista)} clip", lista["label"].value_counts().to_dict())


def carica_lista(nome):
    return pd.read_csv(percorsi.LISTE / f"{nome}.csv", keep_default_na=False)


def file_mancanti(lista):
    """Quanti file della lista non ci sono su disco (deve essere 0 prima di usarla)."""
    return sum(not (percorsi.DATI / p).exists() for p in tqdm(lista["percorso"], desc="controllo file", leave=False))


# ---------------------------------------------------------------------------
# Audio e Dataset PyTorch
# ---------------------------------------------------------------------------

def leggi_audio(percorso):
    """Legge una clip come array float32 mono a 16 kHz."""
    wave, freq = sf.read(percorsi.DATI / percorso, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if freq != FREQ:
        wave = resample_poly(wave, FREQ, freq).astype(np.float32)
    assert len(wave) > 0, f"clip vuota: {percorso}"
    return wave


def taglia(wave, n):
    """Porta la clip a n campioni: se è più lunga tiene l'inizio, se è più corta la ripete."""
    if len(wave) >= n:
        return wave[:n]
    return np.tile(wave, n // len(wave) + 1)[:n]


class AudioDataset(Dataset):
    """
    Restituisce (forma d'onda di n_campioni, label). L'augmentation (solo per il train)
    si applica alla clip intera, prima del taglio, come nei primi esperimenti.
    """

    def __init__(self, lista, n_campioni=N_CAMPIONI, augmentation=None):
        assert lista["label"].isin(["bonafide", "spoof"]).all(), "label diverse da bonafide/spoof"
        self.percorsi = list(lista["percorso"])
        self.label = list((lista["label"] == "spoof").astype(int))
        self.n_campioni = n_campioni
        self.augmentation = augmentation

    def __len__(self):
        return len(self.percorsi)

    def __getitem__(self, i):
        wave = leggi_audio(self.percorsi[i])
        if self.augmentation is not None:
            # generatore casuale preso da torch: diverso in ogni worker e riproducibile con il seed
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            wave = self.augmentation(wave, rng)
        wave = taglia(wave, self.n_campioni).astype(np.float32)
        return torch.from_numpy(wave.copy()), self.label[i]
