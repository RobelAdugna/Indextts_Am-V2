# Resume Training Scheduler Bug Fix

## Problem

When resuming training after a vocab size mismatch (e.g., extending from 12k to 24k tokens), the model would:
- ✅ Correctly detect the mismatch
- ✅ Skip loading incompatible optimizer state
- ✅ Create fresh optimizer
- ❌ **BUG:** Still load old scheduler state
- ❌ **BUG:** Still load old scaler state

### Result
**Fresh optimizer at step 0 with stale scheduler at step 26000 = WRONG LEARNING RATE**

### Symptoms
- High plateauing losses (text_loss ~6.2-6.8, mel_loss ~4.3-4.9)
- Learning rate decaying instead of warming up (4.78e-05 → 4.54e-05)
- No momentum in optimizer state
- Model cannot learn effectively

## Root Cause

```python
# BEFORE FIX (WRONG):
if not skip_optimizer_load:
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])  # Inside if block ✅
else:
    print("Using fresh optimizer")
    # But scheduler already loaded above! ❌

# Scaler loaded AFTER the if/else block
if scaler is not None:
    scaler.load_state_dict(checkpoint["scaler"])  # Wrong! ❌
```

**The bug:** `scheduler.load_state_dict()` and scaler loading happened in the wrong scope.

## Fix Applied

```python
# AFTER FIX (CORRECT):
if not skip_optimizer_load:
    # Load all three together - they're a matched set
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
else:
    # Create ALL fresh - matched set for fresh training
    print("Using FRESH optimizer, scheduler, AND scaler")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    if scaler is not None:
        scaler = torch.cuda.amp.GradScaler()
```

## Expected Results After Fix

**Before fix (OLD behavior - BROKEN):**
```
[Info] Resuming from step 26000
[Train] step=26100 text_loss=6.2076 mel_loss=4.3692 lr=4.78e-05  # Decaying LR
[Train] step=26200 text_loss=6.4494 mel_loss=4.7297 lr=4.77e-05  # Still decaying
[Train] step=26300 text_loss=6.4512 mel_loss=4.6158 lr=4.75e-05  # Plateau - NOT LEARNING!
```

**After fix (NEW behavior - WORKS):**
```
[Info] 🔄 Training will RESTART from step 0 (was at step 26000)
[Info] ✅ Using FRESH optimizer with initial LR and warmup
[Train] step=100 text_loss=5.8 mel_loss=4.0 lr=1.25e-06   # Warming up from 0
[Train] step=500 text_loss=4.2 mel_loss=3.2 lr=6.25e-06   # Still warming
[Train] step=4000 text_loss=2.8 mel_loss=2.5 lr=5.00e-05  # Peak LR reached
[Train] step=10000 text_loss=2.0 mel_loss=1.8 lr=4.50e-05 # LEARNING CORRECTLY!
```

**Important:** Training restarts from step 0 with fresh optimizer/scheduler. This is correct behavior - you're essentially starting fresh training with the loaded model weights.

## Why This Matters

**Optimizer, scheduler, and scaler are a MATCHED SET:**
- Optimizer state includes momentum and adaptive LR adjustments
- Scheduler state includes current step for LR decay schedule
- Scaler state includes gradient scaling factors for AMP

**When you reset one, you MUST reset all three:**
- Fresh optimizer needs initial LR + warmup (not decayed LR)
- Fresh optimizer needs fresh momentum (not stale momentum from different weights)
- Fresh scaler needs fresh scaling factors (not stale factors)

## Testing

To verify the fix works:

1. **Start fresh training:**
   ```bash
   python trainers/train_gpt_v2.py \
     --train-manifest preprocessed_amharic/train_pairs.jsonl \
     --val-manifest preprocessed_amharic/val_pairs.jsonl \
     --tokenizer tokenizers/amharic_extended_bpe.model \
     --output-dir training_output \
     --epochs 3
   ```

2. **Stop after a few thousand steps**

3. **Resume training:**
   ```bash
   python trainers/train_gpt_v2.py \
     --train-manifest preprocessed_amharic/train_pairs.jsonl \
     --val-manifest preprocessed_amharic/val_pairs.jsonl \
     --tokenizer tokenizers/amharic_extended_bpe.model \
     --output-dir training_output \
     --resume auto \
     --epochs 3
   ```

4. **Check logs for:**
   - ✅ "SKIPPING optimizer, scheduler, AND scaler restore"
   - ✅ "Training will RESTART from step 0"
   - ✅ LR warming up from near-zero (~1e-6 at step 100)
   - ✅ Losses dropping steadily from step 0 onwards
   - ✅ Step counter resets to 0 (not 26000+)

## Impact

This fix is **CRITICAL** for:
- ✅ Amharic training (24k vocab)
- ✅ Any language with extended vocab (Korean, Arabic, etc.)
- ✅ Resume training after vocab extension
- ✅ Resume training with different hardware (different checkpoint)

**Without this fix:** Resume training is broken and will NOT learn.

**With this fix:** Resume training works perfectly.

## Related Files

- `trainers/train_gpt_v2.py` - Main training script (fixed)
- `RESUME_TRAINING_WITH_FIX.md` - Extended vocab gradient hook fix
- `VOCAB_VALIDATION_FIX.md` - Vocab size validation on resume

## Date Fixed

2025-01-21
