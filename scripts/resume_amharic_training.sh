#!/bin/bash
# Production-Ready Amharic Training Script (Resume)
# This script resumes training from the latest checkpoint

set -e  # Exit on error

echo "========================================"
echo "Amharic Training - Resume"
echo "========================================"
echo ""

# Configuration
OUTPUT_DIR="trained_ckpts_fixed_fresh"
TRAIN_MANIFEST="preprocessed_amharic/train_pairs.jsonl"
VAL_MANIFEST="preprocessed_amharic/val_pairs.jsonl"
TOKENIZER="tokenizers/amharic_extended_bpe.model"
CONFIG="checkpoints/config.yaml"
BASE_CHECKPOINT="checkpoints/gpt.pth"

# Training hyperparameters (MUST match original training!)
LEARNING_RATE="5e-6"
TEXT_LOSS_WEIGHT="0.4"
MEL_LOSS_WEIGHT="0.6"
WARMUP_STEPS="2000"
EPOCHS="10"
LOG_INTERVAL="100"
VAL_INTERVAL="500"
SAVE_INTERVAL="1000"
KEEP_CHECKPOINTS="3"

# Check if checkpoint exists
echo "[1/4] Checking for checkpoint..."
LATEST_CHECKPOINT="$OUTPUT_DIR/latest.pth"

if [ ! -f "$LATEST_CHECKPOINT" ]; then
    echo "❌ Error: No checkpoint found at $LATEST_CHECKPOINT"
    echo ""
    echo "To start fresh training instead, run:"
    echo "  bash scripts/start_fresh_amharic_training.sh"
    echo ""
    exit 1
fi

echo "✅ Found checkpoint: $LATEST_CHECKPOINT"
echo ""

# Verify required files exist
echo "[2/4] Verifying files..."
if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "❌ Error: Train manifest not found: $TRAIN_MANIFEST"
    exit 1
fi

if [ ! -f "$VAL_MANIFEST" ]; then
    echo "❌ Error: Validation manifest not found: $VAL_MANIFEST"
    exit 1
fi

if [ ! -f "$TOKENIZER" ]; then
    echo "❌ Error: Tokenizer not found: $TOKENIZER"
    exit 1
fi

echo "✅ All required files found"
echo ""

# Show configuration
echo "[3/4] Training configuration:"
echo "  Resume from: $LATEST_CHECKPOINT"
echo "  Train manifest: $TRAIN_MANIFEST"
echo "  Val manifest: $VAL_MANIFEST"
echo "  Tokenizer: $TOKENIZER"
echo "  Output: $OUTPUT_DIR"
echo "  Learning rate: $LEARNING_RATE"
echo "  Text loss weight: $TEXT_LOSS_WEIGHT"
echo "  Mel loss weight: $MEL_LOSS_WEIGHT"
echo ""

# Resume training
echo "[4/4] Resuming training..."
echo "========================================"
echo ""

python trainers/train_gpt_v2.py \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --tokenizer "$TOKENIZER" \
  --config "$CONFIG" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --resume auto \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --text-loss-weight "$TEXT_LOSS_WEIGHT" \
  --mel-loss-weight "$MEL_LOSS_WEIGHT" \
  --warmup-steps "$WARMUP_STEPS" \
  --log-interval "$LOG_INTERVAL" \
  --val-interval "$VAL_INTERVAL" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-checkpoints "$KEEP_CHECKPOINTS" \
  --amp

echo ""
echo "========================================"
echo "Training completed!"
echo "Checkpoints saved in: $OUTPUT_DIR"
echo "========================================"
