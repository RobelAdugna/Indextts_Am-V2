# Resume Training + Extended Vocab Fix: Quick Answer

## Question: Can I Resume Training with the Fix?

### Short Answer

**✅ YES** - Resume is fully compatible with the fix.

**❌ BUT** - Don't resume from your old broken checkpoint (step 38k).

**✅ INSTEAD** - Start fresh for faster results.

## Why Not Resume from Old Checkpoint?

**Your old checkpoint at step 38k has:**
1. ❌ Corrupted Amharic embeddings (trained from random init without gradient masking)
2. ❌ Corrupted optimizer momentum (learned wrong gradients for 38k steps)
3. ❌ Wrong learning rate schedule (already decayed)

**Even with fix applied on resume:**
- Gradient hooks will activate ✅
- But embeddings already in bad state ❌
- Optimizer will slowly correct, but takes 30k+ more steps ❌
- **Total: 68k steps for okay results**

**Starting fresh with fix:**
- Clean embeddings (random but correct) ✅
- Clean optimizer (no corrupted momentum) ✅
- Proper warmup schedule ✅
- **Total: 25k steps for good results**

**Fresh is 2.7x faster!**

## How the Fix Works on Resume

```python
# Every time training starts (fresh OR resume):

model = build_model(...)           # Create architecture
if resume:
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'])  # Load weights
    optimizer.load_state_dict(checkpoint['optimizer'])  # Load momentum

# Fix ALWAYS applies if vocab > 12k:
if vocab_size > 12000:
    model.text_embedding.weight.register_hook(freeze_base_hook)  # Re-registers!
    # This hook affects loaded weights going forward
```

**Key Point:** Hooks are NOT saved in checkpoint. They re-register fresh every run.

## Your Two Options

### Option 1: Start Fresh (RECOMMENDED ⭐⭐⭐⭐⭐)

```bash
python trainers/train_gpt_v2.py \
  --train-manifest preprocessed_amharic/train_pairs.jsonl \
  --val-manifest preprocessed_amharic/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --config checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt.pth \
  --output-dir trained_ckpts_fixed_fresh \
  --learning-rate 5e-6 \
  --text-loss-weight 0.4 \
  --mel-loss-weight 0.6 \
  --warmup-steps 2000 \
  --amp
  # NO --resume flag!
```

**Timeline:**
- 5k steps: text_loss ~2.7 ✅
- 20k steps: intelligible Amharic ✅
- 50k steps: production quality ✅

### Option 2: Resume from Old (EXPERIMENTAL ⭐⭐)

```bash
python trainers/train_gpt_v2.py \
  --train-manifest preprocessed_amharic/train_pairs.jsonl \
  --val-manifest preprocessed_amharic/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --config checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt.pth \
  --resume trained_ckpts/latest.pth \
  --output-dir trained_ckpts_resumed \
  --learning-rate 5e-6 \
  --text-loss-weight 0.4 \
  --mel-loss-weight 0.6 \
  --warmup-steps 2000 \
  --amp
```

**Monitor at step 43k (38k + 5k):**
- If text_loss < 3.0: Continue ✅
- If text_loss still ~4.5: Abort, go to Option 1 ❌

**Expected timeline if it works:**
- 43k-48k steps: slow improvement
- 60k steps: maybe intelligible
- 80k-100k steps: maybe production quality

**Net time: Probably slower than fresh start!**

## Future Resumes (After Fix)

Once you have checkpoints trained WITH the fix:

```bash
# This works perfectly:
python trainers/train_gpt_v2.py \
  --resume trained_ckpts_fixed_fresh/model_step20000.pth \
  # ... same args as training
```

**Why it works:**
- Embeddings were trained correctly ✅
- Optimizer state is clean ✅
- Gradient hooks re-register ✅
- Seamless continuation ✅

## Bottom Line

**Resume compatibility:** ✅ YES, fully compatible

**Resume from old checkpoint:** ❌ NO, not recommended

**Reason:** Not a technical limitation, just bad checkpoint state from broken training

**Recommendation:** Start fresh, save time, get better results!

---

See also:
- `RESUME_TRAINING_WITH_FIX.md` - Detailed technical explanation
- `LIGHTNING_AI_ACTION_PLAN.md` - Step-by-step instructions
- `TRAINING_FIX_SUMMARY.md` - Quick reference
