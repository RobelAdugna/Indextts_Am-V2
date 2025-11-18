# IndexTTS2 Video Training Guide

**Source:** Official IndexTTS2 Training Tutorial (20:05-30:00 tokenizer, full pipeline)

## Critical Steps

### 1. Tokenizer Extension (NOT Training from Scratch!)
```bash
python tools/tokenizer/extend_bpe.py \
  --base-model checkpoints/bpe.model \
  --manifests dataset/manifest.jsonl \
  --output-model tokenizers/extended_bpe.model \
  --target-size 24000
```
**Why extend?** Preserves base token IDs (0-11999) for cross-lingual transfer!

### 2. Preprocessing
```bash
python tools/preprocess_data.py \
  --manifest dataset/manifest.jsonl \
  --output-dir processed_data \
  --tokenizer tokenizers/extended_bpe.model \
  --language am  # or ja, en, etc.
```
**Auto-detects:** Batch size based on GPU VRAM, workers based on CPU
**OOM Recovery:** Automatic batch size reduction on GPU OOM

### 3. Generate Pairs (CRITICAL!)
```bash
python tools/build_gpt_prompt_pairs.py \
  --manifest processed_data/train_manifest.jsonl \
  --output processed_data/GPT_pairs_train.jsonl \
  --pairs-per-target 2
```
**Required:** GPT training fails without prompt-target pairs!

### 4. Training
```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/GPT_pairs_train.jsonl \
  --val-manifest processed_data/GPT_pairs_val.jsonl \
  --tokenizer tokenizers/extended_bpe.model \
  --epochs 2
```
**Auto-optimized:** Batch size, grad accumulation, workers, AMP
**Resume:** Use `--resume auto` to continue interrupted training

## Batch Size (from video)
- 12GB GPU: batch=2
- 16GB GPU: batch=3  
- 24GB GPU: batch=4-6

**Now:** Auto-detected! Script adjusts based on VRAM.

## Validation Interval
**Video:** 1000 steps (faster validation)
**Default:** 1000 ✓

## Checkpoint Management
**Video:** Last 3 epochs saved
**Default:** 3 checkpoints ✓

## Resume Training
```bash
python trainers/train_gpt_v2.py \
  --resume auto  # Uses latest.pth
  # ... same args as original run
```
**Fixed:** Epoch/batch counting now bulletproof - handles mid-epoch and epoch-boundary resumes correctly

## Monitoring (TensorBoard)
```bash
uv run tensorboard --logdir trained_ckpts
```
**Watch:** 
- Loss should decrease
- mel_top1 should increase
- text_loss should drop to ~1.5

## Vocab Mismatch Fix
**Problem:** Extended vocab (24k) vs base checkpoint (12k) causes stuck training
**Solution:** Automatic gradient hooks freeze base tokens (0-11999), train only new tokens (12000-23999)
**Status:** ✓ Fixed automatically

## Common Issues

**OOM during preprocessing?**
- Auto-reduces batch size (no manual tuning needed)

**Training losses stuck?**
- Check vocab size matches tokenizer
- Ensure using extended tokenizer, not base

**Resume from wrong checkpoint?**
- Use `--resume auto` for latest.pth
- Or specify: `--resume trained_ckpts/model_step5000.pth`

**Epoch counter wrong after resume?**
- Fixed! Now tracks correctly

## Video Timeline Reference
- 20:05-30:00: Tokenizer extension workflow
- Full pipeline: Download → Dataset → Corpus → Tokenizer → Preprocess → Pairs → Train
