# Training Optimizations for 200-Hour Amharic Dataset

## Dataset Size: Medium-Scale (~200 hours)

**Key Challenge:** Sweet spot where overfitting is a real concern, but you have enough data for quality training.

## Recommended Training Configuration

### 1. Core Hyperparameters

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed/GPT_pairs_train.jsonl \
  --val-manifest processed/GPT_pairs_val.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
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
  --amp
```

### 2. Overfitting Protection (Built-In)

**Already Implemented:**
- ✅ Weight Decay (L2 regularization)
- ✅ Gradient Clipping
- ✅ Learning Rate Warmup
- ✅ Cosine Annealing Scheduler
- ✅ Early Stopping (via validation monitoring)

**What You Get:**
- Regularization through `--weight-decay` (prevents large weights)
- Gradient stability through `--grad-clip` (prevents explosion)
- Smooth training through warmup + cosine schedule
- Best checkpoint selection through validation

### 3. Dataset Split (Critical!)

**Recommended Split for 200 hours:**
- Training: 97% (~194 hours)
- Validation: 3% (~6 hours)

**Why?**
- 6 hours validation = ~500-800 utterances (enough for reliable metrics)
- Maximizes training data utilization
- Still prevents overfitting through monitoring

**Implementation:**
```bash
python tools/preprocess_data.py \
  --manifest dataset/manifest.jsonl \
  --val-ratio 0.03 \
  # ... other args
```

### 4. Training Duration

**Recommended:** 2-3 epochs (NOT 10!)

**Why?**
- 200 hours is medium-sized (not big data)
- 3 epochs ≈ 60k-90k steps (optimal for convergence)
- Extended vocab needs focused training (not memorization)

**Expected Timeline:**
- L4 24GB: 6-9 days (3 epochs × 2-3 days/epoch)
- A100 80GB: 2-3 days (3 epochs × 18-24 hours/epoch)

**Stop Early If:**
- Validation loss stops improving for 10k steps
- Training loss << Validation loss (overfitting sign)
- Mel_top1 plateaus on validation set

### 5. Learning Rate Strategy

**Recommended:** Conservative + Gradual

```
Phase 1 (0-4k steps): Warmup 0 → 5e-5
Phase 2 (4k-60k steps): Cosine decay 5e-5 → 5e-6
Phase 3 (60k+ steps): Fine-tune at 5e-6
```

**Why Lower LR (5e-5 not 2e-5)?**
- Extended vocab (12k new tokens) needs careful tuning
- Prevents catastrophic forgetting of base tokens
- Gradient hooks freeze base embeddings (stability)

### 6. Validation Monitoring

**Check Every 500 Steps (Not 1000!)**

```bash
--val-interval 500
```

**Why?**
- 200 hours = smaller dataset → faster overfitting possible
- Frequent validation catches issues early
- Minimal time overhead (~2-3 min per validation)

**What to Watch:**
```
Good Training:
  train_loss: 2.5 → 1.2 → 0.9
  val_loss:   2.6 → 1.4 → 1.1
  gap:        0.1   0.2   0.2  ✅

Overfitting:
  train_loss: 2.5 → 1.2 → 0.5
  val_loss:   2.6 → 1.4 → 1.8  ❌
  gap:        0.1   0.2   1.3  (STOP!)
```

### 7. Checkpoint Strategy

**Keep Top 5 Checkpoints:**

```bash
--keep-checkpoints 5
```

**Selection Criteria:**
1. Lowest validation loss (primary)
2. Highest mel_top1 on validation
3. Best training/val gap (<0.3)

**Why 5?**
- Allows ensemble averaging (average top 3)
- Safe fallback if best checkpoint has issues
- Covers ~5k step range for fine selection

### 8. Loss Weight Tuning

**Recommended for Amharic:**

```bash
--text-loss-weight 0.3 \
--mel-loss-weight 0.7
```

**Why?**
- Amharic has complex syllabary (231 base characters)
- Text accuracy critical (each fidel = syllable)
- Mel quality ensures prosody/naturalness
- 30/70 split balances both needs

**Adjust If:**
- Text accuracy low → increase text weight to 0.4
- Speech quality poor → increase mel weight to 0.8

### 9. Batch Size Optimization

**Auto-Detected (Trust It!):**
- L4 24GB: batch=8, grad_accum=4 → effective=32
- A100 80GB: batch=64, grad_accum=1 → effective=64

**Don't Override Unless:**
- OOM errors (reduce batch, increase grad_accum)
- Unstable training (reduce batch to 4-6)

**Sweet Spot:** effective_batch = 24-48 for 200hr dataset

## Advanced Optimizations

### 10. Data Quality Filtering

**Pre-Training Cleanup:**

```bash
# Remove poor quality segments
python tools/create_amharic_dataset.py \
  --min-snr 20.0 \
  --max-silence-ratio 0.25 \
  --min-words 4 \
  --quality-report quality.json
```

**Why?**
- 200 hours → ~15k-20k segments
- Even 5% poor quality = 1k bad samples
- Bad data amplifies overfitting
- Quality > Quantity at this scale

### 11. Gradient Accumulation Strategy

**For Stable Training:**

```
Small GPU (12-16GB):
  batch=4, grad_accum=8 → effective=32
  
Medium GPU (24GB):
  batch=8, grad_accum=4 → effective=32
  
Large GPU (40-80GB):
  batch=16-32, grad_accum=2 → effective=32-64
```

**Trade-off:**
- Larger batches = smoother gradients (less noise)
- Smaller batches = more updates (implicit regularization)
- Gradient accumulation = best of both worlds

### 12. Early Stopping Implementation

**Manual Monitoring (TensorBoard):**

```bash
uv run tensorboard --logdir trained_ckpts
```

**Stop Training When:**

1. **Validation Loss Plateau** (10k+ steps flat)
2. **Overfitting Gap** (train_loss - val_loss > 0.5)
3. **Mel_top1 Degradation** (validation metric drops)
4. **Audio Quality Issues** (listen to validation samples)

**Expected Convergence:**
- 30k steps: losses converging
- 60k steps: near-optimal performance
- 90k steps: diminishing returns
- 120k+ steps: likely overfitting

### 13. Progressive Training (Optional)

**Phase 1: Foundation (0-30k steps)**
```bash
--learning-rate 5e-5 \
--text-loss-weight 0.4 \
--mel-loss-weight 0.6
```
Focus: Text alignment and basic phonetics

**Phase 2: Refinement (30k-60k steps)**
```bash
--learning-rate 2e-5 \
--text-loss-weight 0.3 \
--mel-loss-weight 0.7
```
Focus: Mel quality and prosody

**Phase 3: Fine-Tuning (60k-90k steps)**
```bash
--learning-rate 1e-5 \
--text-loss-weight 0.3 \
--mel-loss-weight 0.7
```
Focus: Polish and generalization

## Overfitting Prevention Checklist

- [x] Weight decay enabled (`--weight-decay 1e-5`)
- [x] Gradient clipping (`--grad-clip 1.0`)
- [x] Learning rate warmup (4k steps)
- [x] Cosine annealing scheduler (built-in)
- [x] Frequent validation checks (500 steps)
- [x] Multiple checkpoint retention (top 5)
- [x] Quality filtering before training
- [x] Conservative learning rate (5e-5)
- [x] Limited epochs (2-3 not 10)
- [x] Validation split (3% = 6 hours)

## Expected Metrics Timeline

**Healthy Training Progression:**

```
Step     Train Loss  Val Loss   Mel_top1  Gap
-----    ----------  --------   --------  ---
0        4.8         4.9        0.02      0.1
5k       3.2         3.3        0.15      0.1
10k      2.4         2.6        0.28      0.2
20k      1.8         2.0        0.42      0.2
30k      1.4         1.6        0.56      0.2
40k      1.1         1.3        0.65      0.2
50k      0.9         1.1        0.72      0.2
60k      0.8         1.0        0.77      0.2  ← Sweet spot
70k      0.7         1.0        0.78      0.3  ← Watch carefully
80k      0.6         1.1        0.77      0.5  ❌ OVERFITTING!
```

**Stop at:** 60k-70k steps (best val loss + gap <0.3)

## Common Mistakes to Avoid

❌ **Don't:** Train for 10 epochs (too much!)
✅ **Do:** Train for 2-3 epochs

❌ **Don't:** Use learning rate 2e-5 (too high for extended vocab)
✅ **Do:** Use 5e-5 or lower

❌ **Don't:** Validate every 1000 steps (too infrequent)
✅ **Do:** Validate every 500 steps

❌ **Don't:** Keep only 2-3 checkpoints
✅ **Do:** Keep 5 checkpoints for selection

❌ **Don't:** Ignore validation loss increases
✅ **Do:** Stop when val loss stops improving

❌ **Don't:** Use all data for training
✅ **Do:** Hold out 3% for validation

## Quick Start Command

**Optimized for 200hr Amharic:**

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed/GPT_pairs_train.jsonl \
  --val-manifest processed/GPT_pairs_val.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
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
```

**Monitor with:**
```bash
uv run tensorboard --logdir trained_ckpts
```

**Expected Duration:** 2-3 days (L4) or 18-24 hours (A100 80GB)

## Summary

200 hours is the **sweet spot** for TTS training:
- ✅ Enough data for quality
- ⚠️ Not enough to ignore overfitting
- 🎯 Perfect for careful regularization

**Your Strategy:**
1. Quality filtering before training
2. Conservative hyperparameters
3. Frequent validation monitoring
4. Early stopping at 60k-70k steps
5. Ensemble top 3 checkpoints if needed

**Result:** High-quality Amharic TTS without overfitting! 🎉
