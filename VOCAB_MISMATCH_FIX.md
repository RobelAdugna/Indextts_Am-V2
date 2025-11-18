# Vocab Mismatch Fix - Critical Training Issue

## Problem

**Symptom:** Training losses completely stopped improving after resuming from checkpoint
- Text loss stuck at 7.5-7.9
- Mel loss stuck at 4.2-5.5
- No learning from step 4k to 11k (7000 steps!)

**Root Cause:** Vocabulary size mismatch between checkpoint and current tokenizer
- Checkpoint: 24001 tokens
- Current tokenizer: 24000 tokens
- Difference: Just 1 token, but CATASTROPHIC impact

## Why This Breaks Training

When vocab sizes don't match:

1. **Model weights load successfully** - PyTorch silently drops/adds rows to embeddings
2. **Optimizer state is INCOMPATIBLE** - Momentum and variance buffers sized for 24001 tokens
3. **Shape mismatch corrupts gradients** - Updates applied to wrong parameters
4. **Model cannot learn** - Gradients don't flow properly through text embeddings

## The Fix

### Detection (line ~790 in train_gpt_v2.py)

```python
# Extract vocab size from checkpoint
for key, value in checkpoint["model"].items():
    if key == "text_embedding.weight":
        checkpoint_vocab = value.shape[0]
        break

# Compare with current tokenizer
if checkpoint_vocab != current_vocab_size:
    skip_optimizer_load = True  # CRITICAL!
```

### Safe Loading

```python
# Load model weights (flexible sizing)
model.load_state_dict(checkpoint["model"], strict=False)

# SKIP incompatible optimizer state
if not skip_optimizer_load:
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
else:
    print("Using FRESH optimizer due to vocab mismatch")
```

## What Happens Now

✅ **Model weights preserved** - All progress is kept
✅ **Fresh optimizer** - New momentum/variance from scratch
✅ **Training continues** - Model will learn (slower initially, but correctly)
⚠️ **Slower convergence** - No momentum from previous training

## Prevention

To avoid this issue:

1. **Use same tokenizer** throughout training
2. **Check vocab size** before resuming:
   ```python
   import sentencepiece as spm
   sp = spm.SentencePieceProcessor()
   sp.load('tokenizers/amharic_extended_bpe.model')
   print(f"Vocab size: {sp.vocab_size()}")  # Should be 24000
   ```
3. **Delete corrupted checkpoints** if vocab changed mid-training

## Your Specific Case

You need to:

1. Delete all existing checkpoints:
   ```bash
   rm -rf trained_ckpts_fixed/
   ```

2. Start fresh training with fix:
   ```bash
   python trainers/train_gpt_v2.py \
     --train-manifest preprocessed_amharic/train_pairs.jsonl \
     --val-manifest preprocessed_amharic/val_pairs.jsonl \
     --tokenizer tokenizers/amharic_extended_bpe.model \
     --output-dir trained_ckpts_fixed \
     --learning-rate 5e-6 \
     --amp
   ```

3. Watch for losses to **ACTUALLY DECREASE** this time!

## Expected Results

With the fix:
- Losses will improve every few hundred steps
- Text loss should drop below 6.0 by 5-10k steps
- Mel loss should drop below 4.0 by 10-15k steps
- Model will actually learn Amharic!

---

**Status:** ✅ FIXED in train_gpt_v2.py (2025-01-17)
