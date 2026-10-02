import os

# Download HF via HTTP classico invece del backend Xet: dentro Jupyter su Windows
# Xet si blocca a fine file (download completo ma mai finalizzato). La velocità
# è la stessa (limita la connessione) e l'HTTP riprende i download interrotti.
# Va impostato prima che huggingface_hub venga importato.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
