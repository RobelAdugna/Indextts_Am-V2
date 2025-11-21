# Vocab Validation Fix (2025-01)

## Problem

When resuming training with extended vocabularies (e.g., Amharic 24k tokens), users were seeing a false "vocab mismatch" warning:

```
🚨 CRITICAL: Vocab size mismatch detected!
   Checkpoint vocab: 24001
   Current tokenizer: 24000
   Optimizer state is INCOMPATIBLE with current model.
```

This caused confusion because:
1. Training appeared to be working fine
2. The warning suggested the optimizer would be reset (even though it wasn't)
3. Users didn't understand why there was a 1-token difference

## Root Cause

The resume validation code was comparing:
- Checkpoint embeddings: `text_embedding.weight.shape[0]` = 24,001
- Tokenizer vocab: `tokenizer.vocab_size` = 24,000

This **1-token difference is expected** because:
- The model adds STOP_TEXT_TOKEN at position `vocab_size`
- So embeddings = vocab_size + 1
- Example: 24,000 tokens → 24,001 embeddings

## Fix

**File:** `trainers/train_gpt_v2.py` (lines 900-907)

**Before:**
```python
if checkpoint_vocab is not None and checkpoint_vocab != current_vocab_size:
    print("🚨 CRITICAL: Vocab size mismatch detected!")
    # ... warning message ...
    skip_optimizer_load = True
```

**After:**
```python
# Subtract 1 to account for STOP_TEXT_TOKEN
checkpoint_actual_vocab = checkpoint_vocab - 1 if checkpoint_vocab is not None else None

if checkpoint_actual_vocab is not None and checkpoint_actual_vocab != current_vocab_size:
    print("🚨 CRITICAL: Vocab size mismatch detected!")
    print(f"   Checkpoint vocab: {checkpoint_vocab} embeddings = {checkpoint_actual_vocab} tokens + STOP")
    print(f"   Difference: {abs(checkpoint_actual_vocab - current_vocab_size)} tokens")
    # ... warning message ...
    skip_optimizer_load = True
elif checkpoint_vocab is not None:
    # Success message
    print(f"✅ Vocab size validated: {checkpoint_vocab} embeddings ({checkpoint_actual_vocab} tokens + STOP)")
```

## Result

**Before fix:** False positive warnings for all extended vocab models
- 24k tokenizer + 24,001 checkpoint = ❌ Warning
- 12k tokenizer + 12,001 checkpoint = ❌ Warning

**After fix:** Only true mismatches trigger warnings
- 24k tokenizer + 24,001 checkpoint = ✅ No warning
- 24k tokenizer + 12,001 checkpoint = ❌ Warning (correct!)
- 12k tokenizer + 24,001 checkpoint = ❌ Warning (correct!)

## User Impact

✅ **Resume training works correctly without confusing warnings**

Users can now resume training with extended vocabularies and will only see warnings for **actual** vocab mismatches (e.g., wrong tokenizer file).

## Related Documentation

- `VOCAB_SIZE_EXPLAINED.md` - Detailed explanation of why vocab_size ≠ embedding_count
- `knowledge.md` - Updated troubleshooting section
- `AMHARIC_TRAINING_FIX.md` - Extended vocab gradient masking fix

## Testing

To verify the fix works:

```bash
# 1. Train with extended vocab (creates 24,001 embedding checkpoint)
python trainers/train_gpt_v2.py \
  --train-manifest train_pairs.jsonl \
  --val-manifest val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model

# 2. Resume training (should show ✅ validation success)
python trainers/train_gpt_v2.py \
  --train-manifest train_pairs.jsonl \
  --val-manifest val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --resume auto

# Expected output:
# [Info] ✅ Vocab size validated: 24001 embeddings (24000 tokens + STOP)
```

## Technical Details

**Why does the model have vocab_size + 1 embeddings?**

See `trainers/train_gpt_v2.py` line 454:
```python
text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
```

The model appends STOP_TEXT_TOKEN (ID = vocab_size) to mark the end of sequences during training. This token needs its own embedding vector, hence vocab_size + 1 total embeddings.

**Design consistency:**
- Base model: 12,000 tokens → 12,001 embeddings
- Amharic extended: 24,000 tokens → 24,001 embeddings
- Always: N tokens → N+1 embeddings
