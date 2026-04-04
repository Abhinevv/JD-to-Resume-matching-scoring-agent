#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Creating virtual environment"
python3 -m venv venv
source venv/bin/activate

echo "[INFO] Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "[INFO] Downloading NLP resources"
python3 -m spacy download en_core_web_sm || true
python3 -c "import nltk; [nltk.download(r, quiet=True) for r in ['punkt','stopwords','averaged_perceptron_tagger','punkt_tab']]"

echo "[INFO] Running smoke test"
python3 main.py --test || true

echo "[OK] Setup complete"
echo "Start app: python main.py"
