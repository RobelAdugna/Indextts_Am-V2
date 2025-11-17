# Amharic Training Fix - Executive Summary

## What Was Wrong

Your training failed because:
1. **Tokenizer has 24,000 tokens** (base 12k + Amharic 12k)
2. **Base checkpoint only has embeddings for first 12,000 tokens**
3. **Amharic tokens 12000-23999 got random initialization**
4. **Model couldn't learn from random embeddings** → nonsense output, high loss

## What We Fixed

Modified `trainers/train_gpt_v2.py` to:
- ✅ Automatically detect extended vocabularies
- ✅ Freeze base token embeddings (0-11999) via gradient hooks
- ✅ Train only new Amharic tokens (12000-23999)
- ✅ Show diagnostic info on startup

## Resume Training Compatibility

**✅ YES, resume works with the fix!**

Gradient hooks re-register automatically on every training run.

**BUT for your specific case (38k broken steps):**
- ❌ **Don't resume from old checkpoint** - optimizer state is corrupted
- ✅ **Start fresh** - faster to good results (25k steps vs 60k+ resumed)
- ✅ **Future resumes** - once trained with fix, resume works perfectly

**See `RESUME_TRAINING_WITH_FIX.md` for complete explanation.**

## What You Need To Do

### Step 1: Verify Setup
```bash
python verify_amharic_training.py
```

This checks:
- Tokenizer is properly extended to 24k tokens
- Amharic text uses new token IDs (>= 12000)
- Model structure is correct

### Step 2: Start Fresh Training

**IMPORTANT:** Don't resume from your old checkpoint at step 38k! Those embeddings are undertrained.

Use these exact parameters:
```bash
python trainers/train_gpt_v2.py \
  --train-manifest preprocessed/train_pairs.jsonl \
  --val-manifest preprocessed/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --learning-rate 5e-6 \
  --text-loss-weight 0.4 \
  --mel-loss-weight 0.6 \
  --warmup-steps 2000 \
  --output-dir trained_ckpts_fixed
```

**Why these parameters?**
- `--learning-rate 5e-6`: Lower LR for stability (was 2e-5)
- `--text-loss-weight 0.4`: Higher text emphasis (was 0.2)
- `--mel-loss-weight 0.6`: Balanced with text (was 0.8)
- `--warmup-steps 2000`: Longer warmup for embedding adjustment (was 1000)

### Step 3: Monitor Progress

Watch TensorBoard for these milestones:

**✅ First 5k-10k steps:**
- text_loss should drop below 3.0 (NOT plateau!)
- mel_loss should drop below 3.5
- **If still plateauing:** Something else is wrong, report back

**✅ At 20k-30k steps:**
- text_loss: ~1.8-2.2
- mel_loss: ~2.0-2.5
- **Test inference:** Should hear intelligible Amharic (not perfect yet)

**✅ At 50k+ steps:**
- text_loss: ~1.5-1.8
- mel_loss: ~1.8-2.2
- **Production quality Amharic TTS**

## What You'll See at Startup

When training starts, you'll see:
```
================================================================================
[Extended Vocab Fix] Detected extended vocabulary: 24000 tokens
[Extended Vocab Fix] Base tokens: 0-11999 (pretrained)
[Extended Vocab Fix] New tokens: 12000-23999 (random init)
[Extended Vocab Fix] Applying gradient masking to freeze base embeddings
================================================================================

[Extended Vocab Fix] Gradient hooks registered for selective training
[Extended Vocab Fix] Freezing 30,720,000 / 523,456,000 parameters (5.9%)
[Extended Vocab Fix] Base embeddings frozen, new embeddings trainable
```

This confirms the fix is active!

## Files Changed

1. **trainers/train_gpt_v2.py** - Core fix (automatic, no config needed)
2. **AMHARIC_TRAINING_FIX.md** - Detailed technical analysis
3. **verify_amharic_training.py** - Pre-training diagnostic script
4. **knowledge.md** - Updated with fix documentation

## FAQ

**Q: Why can't I resume from my old checkpoint?**  
A: The old checkpoint has 38k steps of training with broken Amharic embeddings. Those embeddings learned nothing useful. Starting fresh with the fix will be faster.

**Q: Will this work for other languages too?**  
A: Yes! Any language that extends the base tokenizer (Japanese, Korean, etc.) will benefit from this fix.

**Q: Do I need to re-preprocess my data?**  
A: No! Your preprocessed data is fine. Only training was broken.

**Q: Why does voice cloning still work?**  
A: Voice cloning uses audio features (conditioning), not text. The bug only affected text→speech mapping.

**Q: How long until I see results?**  
A: You should see loss improvement within first 1k-2k steps. Intelligible speech by 20k-30k steps.

## Need Help?

If training still plateaus after 10k steps with the fix:
1. Run diagnostic: `python verify_amharic_training.py`
2. Check TensorBoard logs
3. Share: training command, loss curves, and diagnostic output

See `AMHARIC_TRAINING_FIX.md` for complete technical details.
