# Cartelle del progetto. Tutto è relativo alla cartella del repository,
# così funziona su qualunque PC senza cambiare nulla.

from pathlib import Path

PROGETTO = Path(__file__).resolve().parent.parent

DATI = PROGETTO / "data"                                 # dataset scaricati (non va su git)
LISTE = PROGETTO / "manifests"                           # liste dei file usati negli esperimenti (vanno su git)
ESPERIMENTI = PROGETTO / "artifacts" / "esperimenti"     # modelli, log e punteggi (non va su git)

# Cartelle dei singoli dataset dentro DATI
ASVSPOOF5 = DATI / "asvspoof5"
ASVSPOOF2019 = DATI / "asvspoof2019"
ASVSPOOF2021 = DATI / "asvspoof2021"
IN_THE_WILD = DATI / "in_the_wild"
MUSAN = DATI / "musan"
RIR = DATI / "RIRS_NOISES"
