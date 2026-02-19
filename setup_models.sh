#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_models.sh — Voice Timbre Transfer · Model Setup Helper
# ─────────────────────────────────────────────────────────────────────────────
# Creates the required directory structure and verifies voice models are in
# place. Run this once after cloning or setting up the project.
#
# Usage:
#   chmod +x setup_models.sh
#   ./setup_models.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICES_DIR="$SCRIPT_DIR/voices"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  🎙️  Voice Timbre Transfer — Setup"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Create directories ──
echo "📁 Creating directories..."
mkdir -p "$VOICES_DIR"
mkdir -p "$OUTPUT_DIR"
echo "   ✓ voices/  → $VOICES_DIR"
echo "   ✓ output/  → $OUTPUT_DIR"
echo ""

# ── Check for voice models ──
echo "🔍 Scanning for voice models (.pth)..."
MODEL_COUNT=$(find "$VOICES_DIR" -maxdepth 1 -name "*.pth" | wc -l | tr -d ' ')

if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "   ✓ Found $MODEL_COUNT model(s):"
    for f in "$VOICES_DIR"/*.pth; do
        SIZE=$(du -h "$f" | cut -f1)
        echo "     • $(basename "$f" .pth)  ($SIZE)"
    done
else
    echo "   ⚠️  No voice models found!"
    echo ""
    echo "   To add a voice model:"
    echo "     1. Download or train an RVC .pth model"
    echo "     2. Place it in: $VOICES_DIR"
    echo "     3. The filename becomes the voice name"
    echo "        e.g. MarinaAI.pth → 'MarinaAI'"
fi

echo ""

# ── Check Python deps ──
echo "📦 Checking Python dependencies..."
if command -v pip &> /dev/null; then
    if pip show rvc-python &> /dev/null 2>&1; then
        echo "   ✓ rvc-python installed"
    else
        echo "   ⚠️  rvc-python not installed"
        echo "   Run: pip install -r requirements.txt"
    fi
    if pip show streamlit &> /dev/null 2>&1; then
        echo "   ✓ streamlit installed"
    else
        echo "   ⚠️  streamlit not installed"
        echo "   Run: pip install -r requirements.txt"
    fi
else
    echo "   ⚠️  pip not found — install Python dependencies manually"
fi

echo ""

# ── Backbone weights note ──
echo "📝 Note: hubert_base.pt and rmvpe.pt will be"
echo "   auto-downloaded on first app launch if missing."
echo ""
echo "══════════════════════════════════════════════════════"
echo "  ✅ Setup complete! Run the app with:"
echo "     streamlit run app.py"
echo "══════════════════════════════════════════════════════"
echo ""
