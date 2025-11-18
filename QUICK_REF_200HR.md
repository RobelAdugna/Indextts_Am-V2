# Quick Reference: 200-Hour Dataset Training

## One-Command Start

**Linux/Mac:**
```bash
bash train_amharic_200hr.sh
```

**Windows:**
```bash
train_amharic_200hr.bat
```

## Key Numbers (Memorize These!)

| Setting | Value | Why |
|---------|-------|-----|
| Epochs | **2-3** | Not 10! Prevents overfitting |
| Learning Rate | **5e-5** | Conservative for extended vocab |
| Weight Decay | **1e-5** | L2 regularization active |
| Warmup Steps | **4000** | Smooth start |
| Val Interval | **500** | Catch overfitting early |
| Save Interval | **1000** | Regular checkpoints |
| Keep Checkpoints | **5** | Best model selection |
| Text Weight | **0.3** | Amharic text accuracy |
| Mel Weight | **0.7** | Speech quality |
| Expected Steps | **60k-70k** | Stop here if val plateaus |

## Validation Split

```bash
python tools/preprocess_data.py \
  --val-ratio 0.03 \
  # 3% = ~6 hours = ~500-800 utterances
```

## TensorBoard

```bash
uv run tensorboard --logdir trained_ckpts
```

**Watch for:**
- ✅ Gap <0.3 between train/val loss (healthy)
- ⚠️ Gap 0.3-0.5 (monitor closely)
- ❌ Gap >0.5 (overfitting! use earlier checkpoint)

## Stop Training When:

1. Val loss flat for 10k+ steps
2. Train loss << Val loss (gap >0.5)
3. Mel_top1 decreasing on validation
4. Audio quality degrades

## Expected Timeline

| GPU | Time/Epoch | Total (3 epochs) |
|-----|------------|------------------|
| L4 24GB | 2-3 days | **6-9 days** |
| A100 80GB | 18-24 hours | **2-3 days** |
| V100 16GB | 3-4 days | **9-12 days** |

## Healthy Metrics (60k steps)

```
train_loss:  0.8-1.0
val_loss:    1.0-1.2
gap:         0.2-0.3 ✅
mel_top1:    0.75-0.80
```

## Resume After Interruption

```bash
# Script automatically uses --resume auto
bash train_amharic_200hr.sh
```

## Select Best Checkpoint

```bash
# In trained_ckpts/, find:
# - Lowest val_loss
# - Gap <0.3
# - Around step 60k-70k
```

## Common Mistakes

❌ Training 10 epochs → Use 2-3
❌ LR 2e-5 → Use 5e-5
❌ Val interval 1000 → Use 500
❌ Ignoring val loss → Monitor closely!
❌ Training to 120k steps → Stop at 60k-70k

## Files You Need

```
processed/
  ├── GPT_pairs_train.jsonl  (97% of data)
  ├── GPT_pairs_val.jsonl    (3% of data)
  └── ...
tokenizers/
  └── amharic_extended_bpe.model
```

## Full Details

See `TRAINING_200HR_OPTIMIZATIONS.md`
