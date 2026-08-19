#!/usr/bin/env bash
# One-shot environment setup for Linux/macOS.
#
# Usage:
#   ./setup.sh          # base environment (API + Streamlit + training)
#   ./setup.sh --dev     # also installs Playwright + a Chromium build (UI smoke tests)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}

if [ ! -d .venv ]; then
  echo "Creating virtualenv in .venv ..."
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip -q

if [ "${1:-}" = "--dev" ]; then
  echo "Installing requirements-dev.txt (adds Playwright)..."
  pip install -r requirements-dev.txt -q
  echo "Downloading the Chromium browser for Playwright..."
  python -m playwright install chromium
else
  echo "Installing requirements.txt..."
  pip install -r requirements.txt -q
fi

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo
echo "Next steps:"
echo "  python scripts/prepare_data.py                 # build data/processed from 10573036.zip"
echo "  python scripts/train.py --model resnet50 --epochs 5 --finetune-epochs 5"
echo "  python scripts/evaluate.py --model resnet50"
echo "  streamlit run streamlit_app.py --server.headless true   # http://localhost:8501"
echo "  uvicorn api.main:app --reload                            # http://127.0.0.1:8000/docs"
