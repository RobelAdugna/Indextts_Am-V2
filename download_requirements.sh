#!/bin/bash
# Quick setup script for IndexTTS2 - Downloads all requirements

set -e

echo "========================================"
echo "IndexTTS2 Setup - Downloading Requirements"
echo "========================================"
echo ""

echo "[1/2] Installing Python dependencies..."
echo ""
pip install huggingface-hub

echo ""
echo "[2/2] Downloading pretrained model checkpoints..."
echo "This may take 10-30 minutes depending on your connection."
echo ""
python tools/download_checkpoints.py --output-dir checkpoints

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "All requirements downloaded successfully."
echo "You can now start training with:"
echo "  python webui_amharic.py"
echo ""
echo "Or run the end-to-end pipeline:"
echo "  bash scripts/amharic/end_to_end.sh"
echo ""
