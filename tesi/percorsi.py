# Cartelle del progetto. Tutto è relativo alla cartella del repository,
# così funziona su qualunque PC senza cambiare nulla.

import os
from pathlib import Path

PROGETTO = Path(__file__).resolve().parent.parent

# Su Colab i dati e i risultati stanno in altre cartelle: si cambiano con due variabili
# d'ambiente (vedi notebooks/05_colab.ipynb). Sul fisso non servono.
DATI = Path(os.environ.get("THESIS_DATA_ROOT", PROGETTO / "data"))     # dataset scaricati (non va su git)
LISTE = PROGETTO / "manifests"                                          # liste dei file usati negli esperimenti (vanno su git)
ESPERIMENTI = Path(os.environ.get("THESIS_ARTIFACTS_ROOT", PROGETTO / "artifacts")) / "esperimenti"   # modelli, log e punteggi

# Cartelle dei singoli dataset dentro DATI
ASVSPOOF5 = DATI / "asvspoof5"
ASVSPOOF2019 = DATI / "asvspoof2019"
ASVSPOOF2021 = DATI / "asvspoof2021"
IN_THE_WILD = DATI / "in_the_wild"
MUSAN = DATI / "musan"
RIR = DATI / "RIRS_NOISES"
