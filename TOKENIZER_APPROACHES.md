# Tokenizer Approaches: Extension vs From-Scratch

## Quick Decision Guide

**Use Extension (24k tokens)** if:
- ✅ Want cross-lingual transfer (model learns faster)
- ✅ Have multilingual data (e.g., code-switching)
- ✅ Fine-tuning only (not training from scratch)

**Use From-Scratch (12k tokens)** if:
- ✅ Want simpler resume (no vocab mismatch issues)
- ✅ Pure single-language dataset
- ✅ Script very different from English/Chinese (e.g., Ethiopic, Arabic, Thai)
- ✅ Don't need cross-lingual capability

## Approach 1: Extension (Current - 24,000 tokens)

### How It Works
```bash
python tools/tokenizer/extend_bpe.py \
  --base-model checkpoints/bpe.model \
  --manifests dataset/manifest.jsonl \
  --output-model tokenizers/amharic_extended_bpe.model \
  --target-size 24000
```

### Token Layout
- IDs 0-11999: Base (English/Chinese) - **FROZEN during training**
- IDs 12000-23999: Amharic (new) - **TRAINABLE**

### Pros
✅ Preserves cross-lingual knowledge
✅ Model can handle English/Chinese/Amharic
✅ Faster convergence (transfer learning)
✅ Official IndexTTS2 approach

### Cons
❌ Vocab mismatch breaks resume (24001 vs 24000)
❌ More complex (gradient hooks needed)
❌ Larger model size
❌ 3.4% of parameters frozen

### Resume Issue & Fix
Problem: If checkpoint has 24001 tokens but tokenizer has 24000:
- Optimizer state incompatible
- Training stops learning (losses stuck)

Fix: Code detects mismatch, skips optimizer load, uses fresh optimizer
See: `TRAINING_STUCK_FIX_COMPLETE.md`

---

## Approach 2: From-Scratch (Alternative - 12,000 tokens)

### How It Works
```bash
# 1. Collect corpus
python tools/collect_amharic_corpus.py \
  --manifests dataset/manifest.jsonl \
  --output amharic_corpus.txt

# 2. Train standalone tokenizer
python tools/tokenizer/train_standalone_bpe.py \
  --corpus amharic_corpus.txt \
  --output tokenizers/amharic_standalone_bpe.model \
  --vocab-size 12000 \
  --character-coverage 0.9999 \
  --user-defined-symbols "።,፣,፤,፥,፧,፨,፡"
```

### Token Layout
- IDs 0-11999: Amharic only - **ALL TRAINABLE**

### Pros
✅ No vocab mismatch (always 12000)
✅ Resume works perfectly (no optimizer issues)
✅ Simpler training (no gradient hooks)
✅ 100% of embeddings trainable
✅ Smaller model size

### Cons
❌ No cross-lingual transfer
❌ Can't handle English/Chinese text
❌ Slower initial convergence
❌ Not official IndexTTS2 approach

### When to Use
Perfect for:
- Pure Amharic datasets (no English mixing)
- Avoiding resume complications
- Scripts very different from English/Chinese
- Monolingual TTS applications

---

## Comparison Table

| Feature | Extension (24k) | From-Scratch (12k) |
|---------|----------------|--------------------|
| Vocab Size | 24,000 | 12,000 |
| Cross-lingual | ✅ Yes | ❌ No |
| Resume Issues | ⚠️ Possible | ✅ None |
| Training Speed | ⚡ Fast (transfer) | 🐢 Slower (cold start) |
| Complexity | 🔧 Complex | ✨ Simple |
| Model Size | 📦 Larger | 📦 Smaller |
| Trainable % | 96.6% | 100% |
| Use Case | Multilingual | Monolingual |

---

## Your Specific Case (Amharic)

### Current Situation
- Using Extension approach (24k)
- Hit vocab mismatch bug (24001 vs 24000)
- Training stuck (losses not improving)

### Options

#### Option A: Fix Extension Approach ✅ RECOMMENDED
1. Apply the vocab mismatch fix (already provided)
2. Delete corrupted checkpoints: `rm -rf trained_ckpts_fixed/`
3. Start fresh training with fix
4. Benefits: Keeps cross-lingual transfer, official approach

#### Option B: Switch to From-Scratch 🔄 ALTERNATIVE
1. Create corpus: `python tools/collect_amharic_corpus.py`
2. Train 12k tokenizer: `python tools/tokenizer/train_standalone_bpe.py`
3. Rerun preprocessing with new tokenizer
4. Train from base checkpoint (will work with 12k vocab)
5. Benefits: Simpler, no resume issues

---

## Implementation: From-Scratch Approach

If you choose to try the from-scratch approach:

### Step 1: Create Corpus
```bash
python tools/collect_amharic_corpus.py \
  --input preprocessed_amharic/train_pairs.jsonl \
  --output amharic_corpus.txt \
  --min-length 3
```

### Step 2: Train Standalone Tokenizer
```bash
python tools/tokenizer/train_standalone_bpe.py \
  --corpus amharic_corpus.txt \
  --output tokenizers/amharic_standalone_12k.model \
  --vocab-size 12000 \
  --character-coverage 0.9999 \
  --user-defined-symbols "።,፣,፤,፥,፧,፨,፡"
```

### Step 3: Rerun Preprocessing
```bash
python tools/preprocess_data.py \
  --manifest dataset/manifest.jsonl \
  --output-dir preprocessed_amharic_12k \
  --tokenizer tokenizers/amharic_standalone_12k.model \
  --language am
```

### Step 4: Generate Pairs
```bash
python tools/build_gpt_prompt_pairs.py \
  --input-manifest preprocessed_amharic_12k/train_manifest.jsonl \
  --output-manifest preprocessed_amharic_12k/train_pairs.jsonl \
  --num-pairs 2
```

### Step 5: Train
```bash
python trainers/train_gpt_v2.py \
  --train-manifest preprocessed_amharic_12k/train_pairs.jsonl \
  --val-manifest preprocessed_amharic_12k/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_standalone_12k.model \
  --config checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt.pth \
  --output-dir trained_ckpts_12k \
  --epochs 10 \
  --learning-rate 5e-6 \
  --amp
```

**Important:** The base checkpoint (gpt.pth) has 12k vocab, so it will work!
The model will reinitialize text embeddings for your Amharic tokens.

---

## Recommendation

**For your case:** I recommend **Option A (fix extension approach)**

Why:
1. ✅ You've already done all the work (dataset, preprocessing)
2. ✅ Fix is simple (code already provided)
3. ✅ Keeps cross-lingual transfer benefits
4. ✅ Matches official approach

But Option B is **valid** if:
- You want to avoid any future resume complications
- Don't need English/Chinese support
- Prefer simpler, more robust training

---

**Bottom Line:** Both approaches work! Extension is faster/better for multilingual, from-scratch is simpler/more robust for monolingual.
