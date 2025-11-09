#!/bin/bash

# End-to-End Amharic Training Pipeline for IndexTTS2
# This script automates the entire process from data collection to model training

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/amharic_data"
OUTPUT_DIR="${PROJECT_ROOT}/amharic_output"
CHECKPOINTS_DIR="${PROJECT_ROOT}/checkpoints"

# Create directories
mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "IndexTTS2 Amharic Training Pipeline"
echo "========================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Step 0: Check and download required checkpoints
echo "[Step 0/7] Checking for required model checkpoints..."
MISSING_CHECKPOINTS=false

for checkpoint in "gpt.pth" "bpe.model" "s2mel.pth" "wav2vec2bert_stats.pt" "feat1.pt" "feat2.pt"; do
    if [ ! -f "${CHECKPOINTS_DIR}/${checkpoint}" ]; then
        echo "  ✗ Missing: ${checkpoint}"
        MISSING_CHECKPOINTS=true
    fi
done

if [ "$MISSING_CHECKPOINTS" = true ]; then
    echo "  ⬇ Downloading missing checkpoints from HuggingFace..."
    uv run python tools/download_checkpoints.py --output-dir "${CHECKPOINTS_DIR}"
    
    if [ $? -ne 0 ]; then
        echo "  ✗ Failed to download checkpoints. Exiting."
        exit 1
    fi
    echo "  ✓ Checkpoints downloaded successfully"
else
    echo "  ✓ All required checkpoints present"
fi
echo ""

# Step 1: Download content (YouTube OR direct URLs)
echo "[Step 1/8] Downloading Amharic content..."

# Check for media manifest (direct URLs)
if [ -f "${PROJECT_ROOT}/examples/media_manifest.jsonl" ]; then
    echo "  Using media manifest for direct URL downloads"
    uv run python tools/universal_media_downloader.py \
        --manifest "${PROJECT_ROOT}/examples/media_manifest.jsonl" \
        --output-dir "${DATA_DIR}/downloads"
    echo "✓ Step 1 complete (direct URLs)"

# Fall back to YouTube
elif [ -f "${PROJECT_ROOT}/examples/amharic_youtube_urls.txt" ]; then
    echo "  Using YouTube downloader"
    uv run python tools/youtube_amharic_downloader.py \
        --url-file "${PROJECT_ROOT}/examples/amharic_youtube_urls.txt" \
        --output-dir "${DATA_DIR}/downloads" \
        --subtitle-langs am en amh \
        --audio-format wav
    echo "✓ Step 1 complete (YouTube)"

else
    echo "⚠ Skipping: No source file found"
    echo "  Create either:"
    echo "    - examples/media_manifest.jsonl (for direct URLs/local files)"
    echo "    - examples/amharic_youtube_urls.txt (for YouTube)"
fi
echo ""

# Step 2: Create dataset from media files
echo "[Step 2/8] Creating dataset from audio and subtitle files..."
if [ -d "${DATA_DIR}/downloads" ]; then
    uv run python tools/create_amharic_dataset.py \
        --input-dir "${DATA_DIR}/downloads" \
        --output-dir "${DATA_DIR}/raw_dataset" \
        --manifest "${DATA_DIR}/raw_dataset/manifest.jsonl" \
        --min-duration 1.0 \
        --max-duration 30.0
    echo "✓ Step 2 complete"
else
    echo "⚠ Skipping: No downloads directory found"
fi
echo ""

# Step 3: Collect and clean corpus
echo "[Step 3/8] Collecting and cleaning Amharic corpus..."
if [ -f "${DATA_DIR}/raw_dataset/manifest.jsonl" ]; then
    uv run python tools/collect_amharic_corpus.py \
        --input "${DATA_DIR}/raw_dataset/manifest.jsonl" \
        --output "${DATA_DIR}/amharic_corpus.txt" \
        --text-field text \
        --stats
    echo "✓ Step 3 complete"
else
    echo "⚠ Skipping: No manifest file found"
fi
echo ""

# Step 4: Train/extend BPE tokenizer
echo "[Step 4/8] Training multilingual BPE tokenizer..."
if [ -f "${DATA_DIR}/amharic_corpus.txt" ]; then
    # Check if base tokenizer exists
    if [ -f "${CHECKPOINTS_DIR}/bpe.model" ]; then
        echo "  Extending existing tokenizer with Amharic data..."
        # Train new tokenizer with combined corpus
        uv run python tools/train_multilingual_bpe.py \
            --corpus "${DATA_DIR}/amharic_corpus.txt" \
            --model-prefix "${OUTPUT_DIR}/amharic_bpe" \
            --vocab-size 40000 \
            --character-coverage 0.9999 \
            --test-files "${DATA_DIR}/amharic_corpus.txt"
    else
        echo "  Training new tokenizer..."
        uv run python tools/train_multilingual_bpe.py \
            --corpus "${DATA_DIR}/amharic_corpus.txt" \
            --model-prefix "${OUTPUT_DIR}/amharic_bpe" \
            --vocab-size 32000 \
            --character-coverage 0.9999 \
            --test-files "${DATA_DIR}/amharic_corpus.txt"
    fi
    echo "✓ Step 4 complete"
else
    echo "⚠ Skipping: No corpus file found"
fi
echo ""

# Step 5: Preprocess data
echo "[Step 5/8] Preprocessing Amharic dataset..."
if [ -f "${DATA_DIR}/raw_dataset/manifest.jsonl" ] && [ -f "${OUTPUT_DIR}/amharic_bpe.model" ]; then
    uv run python tools/preprocess_data.py \
        --manifest "${DATA_DIR}/raw_dataset/manifest.jsonl" \
        --output-dir "${DATA_DIR}/processed" \
        --tokenizer "${OUTPUT_DIR}/amharic_bpe.model" \
        --language am \
        --val-ratio 0.01 \
        --batch-size 4
    echo "✓ Step 5 complete"
else
    echo "⚠ Skipping: Missing manifest or tokenizer"
fi
echo ""

# Step 6: Generate prompt-target pairs
echo "[Step 6/8] Generating GPT prompt-target pairs..."
if [ -f "${DATA_DIR}/processed/train_manifest.jsonl" ]; then
    # Check if pair generation script exists
    if [ -f "${PROJECT_ROOT}/tools/build_gpt_prompt_pairs.py" ]; then
        uv run python tools/build_gpt_prompt_pairs.py \
            --manifest "${DATA_DIR}/processed/train_manifest.jsonl" \
            --output "${DATA_DIR}/processed/train_pairs.jsonl"
        
        uv run python tools/build_gpt_prompt_pairs.py \
            --manifest "${DATA_DIR}/processed/val_manifest.jsonl" \
            --output "${DATA_DIR}/processed/val_pairs.jsonl"
        echo "✓ Step 6 complete"
    else
        echo "⚠ Warning: build_gpt_prompt_pairs.py not found"
        echo "  Continuing with single-sample manifests..."
        # Copy manifests as fallback
        cp "${DATA_DIR}/processed/train_manifest.jsonl" "${DATA_DIR}/processed/train_pairs.jsonl"
        cp "${DATA_DIR}/processed/val_manifest.jsonl" "${DATA_DIR}/processed/val_pairs.jsonl"
    fi
else
    echo "⚠ Skipping: No processed manifests found"
fi
echo ""

# Step 7: Train GPT model
echo "[Step 7/8] Training GPT model (this will take a long time)..."
if [ -f "${DATA_DIR}/processed/train_pairs.jsonl" ]; then
    echo "  Starting training..."
    echo "  You can monitor progress in: ${OUTPUT_DIR}/trained_ckpts/logs/"
    echo ""
    
    uv run python trainers/train_gpt_v2.py \
        --train-manifest "${DATA_DIR}/processed/train_pairs.jsonl" \
        --val-manifest "${DATA_DIR}/processed/val_pairs.jsonl" \
        --tokenizer "${OUTPUT_DIR}/amharic_bpe.model" \
        --config "${CHECKPOINTS_DIR}/config.yaml" \
        --base-checkpoint "${CHECKPOINTS_DIR}/gpt.pth" \
        --output-dir "${OUTPUT_DIR}/trained_ckpts" \
        --batch-size 4 \
        --grad-accumulation 4 \
        --epochs 10 \
        --learning-rate 2e-5 \
        --warmup-steps 1000 \
        --log-interval 100 \
        --val-interval 500 \
        --amp
    
    echo "✓ Step 7 complete"
else
    echo "⚠ Skipping: No paired manifests found"
fi
echo ""

echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Trained checkpoints: ${OUTPUT_DIR}/trained_ckpts/"
echo ""
echo "To use the trained model, update your config to point to:"
echo "  Tokenizer: ${OUTPUT_DIR}/amharic_bpe.model"
echo "  GPT: ${OUTPUT_DIR}/trained_ckpts/model_stepXXXX.pth"
echo ""
