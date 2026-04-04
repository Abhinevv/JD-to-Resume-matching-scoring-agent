#!/usr/bin/env bash
# =============================================================================
#  setup.sh  —  One-click environment setup for Resume Matcher
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh           # full setup + run smoke test
#    ./setup.sh --no-test # skip smoke test
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   🎯 Resume–JD Matching System — Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# ── Python version check ──────────────────────────────────────────────────────
info "Checking Python version…"
python_ver=$(python3 --version 2>&1 | awk '{print $2}')
required="3.9"
if python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)"; then
    success "Python $python_ver ✓"
else
    error "Python ≥ 3.9 required (found $python_ver)"
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    info "Creating virtual environment…"
    python3 -m venv venv
    success "venv created"
else
    success "venv already exists"
fi

source venv/bin/activate

# ── pip upgrade ───────────────────────────────────────────────────────────────
info "Upgrading pip…"
pip install --upgrade pip --quiet
success "pip upgraded"

# ── Install dependencies ──────────────────────────────────────────────────────
info "Installing Python dependencies (this may take a few minutes)…"
pip install -r requirements.txt --quiet
success "Python packages installed"

# ── spaCy model ───────────────────────────────────────────────────────────────
info "Downloading spaCy en_core_web_sm…"
if python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    success "spaCy model already installed"
else
    python3 -m spacy download en_core_web_sm --quiet
    success "spaCy model downloaded"
fi

# ── NLTK resources ────────────────────────────────────────────────────────────
info "Downloading NLTK resources…"
python3 -c "
import nltk
for r in ['punkt','stopwords','averaged_perceptron_tagger','punkt_tab']:
    nltk.download(r, quiet=True)
print('NLTK OK')
"
success "NLTK resources ready"

# ── Tesseract check ───────────────────────────────────────────────────────────
info "Checking Tesseract OCR…"
if command -v tesseract &>/dev/null; then
    tess_ver=$(tesseract --version 2>&1 | head -1)
    success "Tesseract: $tess_ver"
else
    warn "Tesseract not found. Image OCR will be disabled."
    warn "Install with:  sudo apt install tesseract-ocr  (Ubuntu)"
    warn "            or brew install tesseract            (macOS)"
fi

# ── Project directories ───────────────────────────────────────────────────────
info "Creating project directories…"
mkdir -p data/raw_resumes models database
success "Directories ready"

# ── Smoke test ────────────────────────────────────────────────────────────────
if [[ "${1:-}" != "--no-test" ]]; then
    info "Running smoke test…"
    python3 main.py --test && success "Smoke test passed ✓" || warn "Smoke test had issues (check logs above)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   ✅  Setup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BLUE}Start Streamlit UI:${NC}  streamlit run app.py"
echo -e "  ${BLUE}Start FastAPI:${NC}        uvicorn api.main_api:app --reload --port 8000"
echo -e "  ${BLUE}Run tests:${NC}            python -m pytest tests/ -v"
echo ""
