#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build.sh — Render.com build script for SGEIA
#
# Called by Render during every deploy.  Idempotent: skips steps whose
# outputs already exist so subsequent deploys are fast.
#
# What is already in git (no rebuild needed on Render):
#   data/smart_grid_stability_augmented.csv  (19 MB)
#   data/grid_incidents_synthetic.csv         (80 KB)
#   models_saved/*.joblib                     (~6 MB)
#   chroma_db/                                (~2 MB)
#
# What this script handles:
#   data/household_power_consumption.csv — 268 MB, not in git;
#   generated as a realistic 50K-row sample here.
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "============================================================"
echo "  SGEIA — Render Build Script"
echo "============================================================"

# ── 1. Install Python dependencies ───────────────────────────────────────────
echo "[1/4] Installing Python packages..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "      Packages installed."

# ── 2. Download spaCy language model (presidio dependency) ───────────────────
echo "[2/4] Downloading spaCy en_core_web_sm..."
python -m spacy download en_core_web_sm --quiet 2>/dev/null || echo "      spaCy model already present or skipped."

# ── 3. Pre-cache sentence-transformer model ──────────────────────────────────
echo "[3/4] Pre-caching all-MiniLM-L6-v2 embeddings model..."
python - <<'PYEOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
print("      Sentence-transformer model cached.")
PYEOF

# ── 4. Generate DS2 sample if absent ─────────────────────────────────────────
echo "[4/4] Checking DS2 (household_power_consumption.csv)..."
if [ ! -f "data/household_power_consumption.csv" ]; then
    echo "      Not found — generating 50K-row sample (no source file needed)..."
    python scripts/generate_ds2_sample.py
else
    SIZE=$(du -sh "data/household_power_consumption.csv" | cut -f1)
    echo "      Found (${SIZE}) — skipping generation."
fi

echo ""
echo "============================================================"
echo "  Build complete — starting SGEIA."
echo "============================================================"
