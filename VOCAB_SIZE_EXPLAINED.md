# Vocab Size Difference: Expected Behavior

## Summary

✅ **The 1-token difference between tokenizer (24000) and model (24001) is EXPECTED and CORRECT!**

This is **not a bug** - it's the standard design of IndexTTS2.

## Your Current Setup

```
Tokenizer vocab: 24,000 tokens (IDs 0-23,999)
Model embeddings: 24,001 embeddings (24,000 + 1 STOP token)
Amharic tokens: IDs 12,000-23,999 encode correctly ✅
```

## Why The Difference?

### Tokenizer (24,000 tokens)
- **Base tokens:** IDs 0-11,999 (English/Chinese from pretrained model)
- **Amharic tokens:** IDs 12,000-23,999 (added via extension)
- **Total:** 24,000 real text tokens

### Model (24,001 embeddings)
- **Text tokens:** IDs 0-23,999 (same 24,000 as tokenizer)
- **STOP_TEXT_TOKEN:** ID 24,000 (special token added by model)
- **Total:** 24,001 embedding vectors

## How STOP_TEXT_TOKEN Works

```python
# In trainers/train_gpt_v2.py, line 454:
text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
```

The model:
1. Takes tokenized text (IDs 0-23,999)
2. Appends STOP_TEXT_TOKEN (ID 24,000) to mark end of sequence
3. Uses this for training targets

This is why `text_embedding.weight` has shape `[24001, hidden_dim]` - it needs an embedding for the STOP token!

## Base Model Design

The base checkpoint also follows this pattern:
- Base tokenizer: 12,000 tokens
- Base model embeddings: 12,001 (12,000 + STOP)

When you extend to Amharic:
- Extended tokenizer: 24,000 tokens
- Extended model embeddings: 24,001 (24,000 + STOP)

## Training Code Validation

The training script **correctly handles** this:

```python
# Line 275-277 in trainers/train_gpt_v2.py:
vocab_size = tokenizer.vocab_size  # 24,000
if cfg.gpt.number_text_tokens != vocab_size:
    cfg.gpt.number_text_tokens = vocab_size  # Sets to 24,000

# Line 279 (UnifiedVoice model):
model = UnifiedVoice(**cfg.gpt)  # Creates 24,001 embeddings internally
```

The `UnifiedVoice` model automatically adds +1 for STOP_TEXT_TOKEN.

## Resume Training Validation

The vocab validation in resume code now correctly handles the STOP_TEXT_TOKEN:
- Checkpoint `text_embedding.weight.shape[0]` (24,001)
- Current tokenizer `vocab_size` (24,000)

**✅ Fixed!** The code now correctly compares:
```python
checkpoint_actual_vocab = checkpoint_vocab - 1  # Subtract STOP_TEXT_TOKEN
current_vocab = tokenizer.vocab_size
# Both should be 24,000
```

The validation now accounts for the +1 STOP token, so you won't get false "mismatch" warnings when resuming training with extended vocabularies.

## Is This Causing Training Issues?

**No!** If training is stuck, it's NOT because of this vocab difference. The real issues were:

1. ✅ **Extended vocab gradient masking** (FIXED in train_gpt_v2.py)
   - Base embeddings (0-11,999) now frozen during training
   - New Amharic embeddings (12,000-23,999) train properly

2. ✅ **Inference vocab bug** (FIXED in infer_v2_modded.py)  
   - Was adding +1 twice (24,001 → 24,002)
   - Now correctly uses 24,001

## Verification

Your terminal output shows everything working correctly:

```bash
# Tokenizer correctly has 24k tokens
>>> sp.vocab_size()
24000

# Checkpoint correctly has 24k+1 embeddings
>>> ckpt['model']['text_embedding.weight'].shape[0]
24001

# Amharic tokens encode within valid range
>>> tokens = sp.encode('ሰላም ልጆች')
>>> tokens
[12040, 12407, 12535]  # All < 24000 ✅
```

## Conclusion

**No action needed!** The 1-token difference is by design and working correctly.

If you're experiencing training issues, they're caused by:
- Frozen embeddings not training (check gradient hooks are applied)
- Wrong learning rate or loss weights
- Insufficient data
- Hardware limitations

But **NOT** the vocab size difference - that's normal!
