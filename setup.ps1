# One-shot environment setup for Windows (PowerShell).
#
# Usage:
#   .\setup.ps1          # base environment (API + Streamlit + training)
#   .\setup.ps1 -Dev      # also installs Playwright + a Chromium build (UI smoke tests)
param(
    [switch]$Dev
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    Write-Host "Creating virtualenv in .venv ..."
    python -m venv .venv
}

$venvPython = ".\.venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip -q

if ($Dev) {
    Write-Host "Installing requirements-dev.txt (adds Playwright)..."
    & $venvPython -m pip install -r requirements-dev.txt -q
    Write-Host "Downloading the Chromium browser for Playwright..."
    & $venvPython -m playwright install chromium
} else {
    Write-Host "Installing requirements.txt..."
    & $venvPython -m pip install -r requirements.txt -q
}

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  python scripts/prepare_data.py                 # build data/processed from 10573036.zip"
Write-Host "  python scripts/train.py --model resnet50 --epochs 5 --finetune-epochs 5"
Write-Host "  python scripts/evaluate.py --model resnet50"
Write-Host "  streamlit run streamlit_app.py --server.headless true   # http://localhost:8501"
Write-Host "  uvicorn api.main:app --reload                            # http://127.0.0.1:8000/docs"
