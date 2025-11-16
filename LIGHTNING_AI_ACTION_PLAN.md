# Lightning.ai Action Plan - Fix Amharic Training

## Quick Summary

**Problem:** Training plateaus at 10k steps with high loss (~4.5-4.8) producing nonsense Amharic.
**Root Cause:** Extended tokenizer (24k vocab) has randomly initialized embeddings for Amharic tokens (12k-24k).
**Solution:** Applied fix to freeze base embeddings and train only new Amharic tokens.

## Steps to Execute on Lightning.ai

### Step 1: Sync Fixed Code (2 minutes)

**Option A - Git (Recommended):**
```bash
cd /path/to/your/project
git pull origin main
```

**Option B - Manual Upload:**
Upload these files from local PC to Lightning.ai:
- `trainers/train_gpt_v2.py`
- `verify_amharic_training.py`
- `check_amharic_data.py`

### Step 2: Verify Your Data Setup (5 minutes)

```bash
# Check data quality
python check_amharic_data.py \
  --train-manifest preprocessed_amharic/train_manifest.jsonl \
  --train-pairs preprocessed_amharic/train_pairs.jsonl
```

**Expected output:**
```
Total samples: 150,000+
Amharic samples: 145,000+ (>95%)
Unique speakers: 50-200
✅ Dataset is primarily Amharic
✅ Good speaker diversity
```

**If you see warnings:** Report back the specific warnings.

### Step 3: Verify Tokenizer (2 minutes)

```bash
python verify_amharic_training.py \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --config checkpoints/config.yaml
```

**Expected output:**
```
✓ Tokenizer loaded: tokenizers/amharic_extended_bpe.model
✓ Vocabulary size: 24000
✓ Uses extended vocabulary (ID >= 12000)
✅ All checks passed! Ready for training.
```

### Step 4: Start Fresh Training (CRITICAL)

**DO NOT resume from your old checkpoint!** Start fresh:

```bash
# Stop any existing training first
# Then run:

python trainers/train_gpt_v2.py \
  --train-manifest preprocessed_amharic/train_pairs.jsonl \
  --val-manifest preprocessed_amharic/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --config checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt.pth \
  --output-dir trained_ckpts_fixed \
  --epochs 10 \
  --learning-rate 5e-6 \
  --text-loss-weight 0.4 \
  --mel-loss-weight 0.6 \
  --warmup-steps 2000 \
  --log-interval 100 \
  --val-interval 500 \
  --amp
```

**CRITICAL:** You MUST see this at startup:
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

**If you DON'T see these messages:**
1. Check that `trainers/train_gpt_v2.py` was updated (run `git log -1` to verify latest commit)
2. Check tokenizer path is correct
3. Report back the actual startup messages

### Step 5: Monitor First 5k Steps (30-60 minutes)

Watch TensorBoard or training logs for:

**✅ GOOD (Fix Working):**
```
Step 1000: text_loss=3.8, mel_loss=4.2  # Dropping!
Step 2000: text_loss=3.2, mel_loss=3.8  # Still dropping!
Step 5000: text_loss=2.7, mel_loss=3.2  # Clear improvement!
```

**❌ BAD (Still Broken):**
```
Step 1000: text_loss=4.5, mel_loss=4.8  # Not moving
Step 2000: text_loss=4.5, mel_loss=4.8  # Stuck
Step 5000: text_loss=4.5, mel_loss=4.8  # Plateau
```

### Step 6: Report Back

Share with me:

1. **Startup messages:** Did you see "[Extended Vocab Fix]"?
2. **Loss values:** At steps 1k, 2k, 5k
3. **Data verification output:** From `check_amharic_data.py`
4. **Any errors or warnings**

Based on your report, I'll provide next steps!

## Troubleshooting

### Issue: Still plateauing after fix

**Possible causes:**
1. Training script not updated → verify git pull worked
2. Prompt-target language mismatch → check data verification output
3. Tokenizer not properly extended → verify it shows 24k tokens
4. Data quality issues → check speaker diversity and Amharic percentage

**Debug steps:**
```bash
# Check training script has fix
grep -n "Extended Vocab Fix" trainers/train_gpt_v2.py
# Should show multiple matches around line 450-480

# Check tokenizer size
python -c "from sentencepiece import SentencePieceProcessor; sp = SentencePieceProcessor(); sp.load('tokenizers/amharic_extended_bpe.model'); print(f'Vocab: {sp.get_piece_size()}')"
# Should output: Vocab: 24000

# Check sample tokenization
python -c "from indextts.utils.front import TextTokenizer, TextNormalizer; t = TextTokenizer('tokenizers/amharic_extended_bpe.model', TextNormalizer('am')); ids = t.encode('ሰላም ዓለም'); print(f'IDs: {ids}'); print(f'Max: {max(ids)}')"
# Should output: Max: >12000 (using Amharic tokens)
```

### Issue: Can't find files on Lightning.ai

Paths might be different. Find your files:
```bash
find . -name "train_pairs.jsonl" -type f
find . -name "*_bpe.model" -type f
```

Then adjust paths in commands accordingly.

### Issue: Git pull fails

Manual fix:
```bash
# Download train_gpt_v2.py from GitHub or local machine
# Replace the old file on Lightning.ai
cp /path/to/new/train_gpt_v2.py trainers/train_gpt_v2.py
```

## Expected Timeline

**With fix applied:**
- **First 5k steps:** 30-60 minutes, should see loss dropping
- **20k steps:** 2-4 hours, should have intelligible Amharic
- **50k steps:** 8-12 hours, production quality

**Total training time (200hrs dataset on L4/A100):**
- L4 GPU: ~1-2 days
- A100 GPU: ~12-18 hours

## Success Criteria

You'll know it's working when:

1. ✅ See "[Extended Vocab Fix]" messages at startup
2. ✅ text_loss < 3.0 by step 5000 (not stuck at 4.5)
3. ✅ Steady decrease in both losses (not plateau)
4. ✅ At step 20k-30k, inference produces recognizable Amharic syllables

## What to Do After Success

Once training shows improvement:

1. **Let it run to 50k-100k steps** for best quality
2. **Test at checkpoints:** 20k, 40k, 60k, 80k, 100k
3. **Compare outputs:** Listen for pronunciation clarity, emotion transfer
4. **Share your success!** This will be valuable for other Amharic/extended-vocab users

## Need More Help?

If issues persist after following this plan, gather:
- Complete startup log (first 50 lines)
- Loss curves (screenshot or values at 1k, 2k, 5k, 10k)
- Output of verification scripts
- Any error messages

Then we'll debug further!
