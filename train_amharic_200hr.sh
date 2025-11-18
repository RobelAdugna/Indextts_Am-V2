#!/bin/bash
# Optimized Training Script for 200-Hour Amharic Dataset
# Features: Overfitting protection, early stopping, quality monitoring

set -e  # Exit on error

echo "🚀 Starting Optimized Amharic Training (200hr dataset)"
echo "================================================="

# Configuration
TRAIN_MANIFEST="processed/GPT_pairs_train.jsonl"
VAL_MANIFEST="processed/GPT_pairs_val.jsonl"
TOKENIZER="tokenizers/amharic_extended_bpe.model"
OUTPUT_DIR="trained_ckpts"

# Verify files exist
if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "❌ Error: Training manifest not found: $TRAIN_MANIFEST"
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

# Count samples
TRAIN_SAMPLES=$(wc -l < "$TRAIN_MANIFEST")
VAL_SAMPLES=$(wc -l < "$VAL_MANIFEST")

echo "📊 Dataset Statistics:"
echo "   Training samples: $TRAIN_SAMPLES"
echo "   Validation samples: $VAL_SAMPLES"
echo "   Val ratio: $(echo "scale=2; $VAL_SAMPLES*100/($TRAIN_SAMPLES+$VAL_SAMPLES)" | bc)%"
echo ""

# Validate split ratio
VAL_RATIO=$(echo "scale=3; $VAL_SAMPLES/($TRAIN_SAMPLES+$VAL_SAMPLES)" | bc)
if (( $(echo "$VAL_RATIO < 0.02" | bc -l) )); then
    echo "⚠️  Warning: Validation set <2% of data (recommended 3%)"
elif (( $(echo "$VAL_RATIO > 0.05" | bc -l) )); then
    echo "⚠️  Warning: Validation set >5% of data (recommended 3%)"
else
    echo "✅ Validation split is optimal (2-5%)"
fi
echo ""

# Ask for confirmation
echo "🎯 Training Configuration:"
echo "   Epochs: 3"
echo "   Learning rate: 5e-5 (conservative for extended vocab)"
echo "   Weight decay: 1e-5 (L2 regularization)"
echo "   Warmup: 4000 steps"
echo "   Validation: Every 500 steps"
echo "   Checkpoints: Save every 1000 steps, keep best 5"
echo "   Loss weights: Text=0.3, Mel=0.7"
echo ""

read -p "▶️  Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 0
fi

echo ""
echo "🏋️  Starting training..."
echo "💡 Monitor with: uv run tensorboard --logdir $OUTPUT_DIR"
echo "🛑 Stop early if validation loss plateaus for 10k steps!"
echo ""

# Start training with optimal settings
python trainers/train_gpt_v2.py \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --tokenizer "$TOKENIZER" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 3 \
  --learning-rate 5e-5 \
  --weight-decay 1e-5 \
  --warmup-steps 4000 \
  --val-interval 500 \
  --save-interval 1000 \
  --keep-checkpoints 5 \
  --text-loss-weight 0.3 \
  --mel-loss-weight 0.7 \
  --grad-clip 1.0 \
  --resume auto \
  --amp

echo ""
echo "✅ Training complete!"
echo ""
echo "📦 Next steps:"
echo "   1. Check TensorBoard for validation metrics"
echo "   2. Select best checkpoint (lowest val_loss, gap <0.3)"
echo "   3. Test with inference on validation samples"
echo "   4. If overfitting (gap >0.5), use earlier checkpoint"
