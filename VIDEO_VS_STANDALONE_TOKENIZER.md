# Video Tutorial vs Standalone Tokenizer: Side-by-Side Comparison

## Overview

This compares the **official video tutorial approach** (24k extended tokenizer) with the **alternative standalone approach** (12k from-scratch tokenizer).

---

## Tokenizer Stage Comparison

### Video Tutorial (Extension - 24k tokens)

**Video Timestamp:** Around 20:05-30:00

**Command from Video:**
```bash
# Edit extend_bpe.bat
Dataset path: datasets/Spanish_Amelia_dataset/manifest.jsonl
Target size: 24,000 tokens (double original 12,000)
Character coverage: 100%

# Run script
Output: Appends 12,000 new tokens
Creates extended_bp.model and vocab file in checkpoints
```

**What happens:**
- Loads base BPE (12k tokens: English/Chinese)
- Trains temp tokenizer on Spanish dataset
- Extracts new Spanish-only tokens
- Appends to base → 24k total
- Token IDs 0-11999: Base (frozen)
- Token IDs 12000-23999: Spanish (trainable)

### Standalone Approach (From-Scratch - 12k tokens)

**Your Implementation:**
```bash
# Step 1: Collect corpus
python tools/collect_amharic_corpus.py \
  --input datasets/Amharic_Amelia_dataset/manifest.jsonl \
  --output amharic_corpus.txt

# Step 2: Train standalone tokenizer
python tools/tokenizer/train_standalone_bpe.py \
  --corpus amharic_corpus.txt \
  --output tokenizers/amharic_standalone_12k.model \
  --vocab-size 12000 \
  --character-coverage 0.9999
```

**What happens:**
- Trains BPE only on Amharic corpus
- No base model involved
- Token IDs 0-11999: Amharic only (all trainable)
- Simpler, monolingual approach

---

## Complete Pipeline Comparison

| Step | Video Tutorial (24k) | Standalone (12k) |
|------|---------------------|------------------|
| **1. Dataset** | Spanish_Amelia_dataset | Amharic_Amelia_dataset |
| **2. Tokenizer** | Extend base → 24k | Train from scratch → 12k |
| **3. Preprocessing** | ✅ Same process | ✅ Same process |
| **4. Pair JSONL** | ✅ Same process | ✅ Same process |
| **5. Training** | ⚠️ Vocab mismatch risk | ✅ No mismatch issues |
| **6. Resume** | ⚠️ Can break (24001 vs 24000) | ✅ Always works |
| **7. Pruning** | ✅ Same process | ✅ Same process |
| **8. Inference** | ✅ Same process | ✅ Same process |

---

## Detailed Step-by-Step

### STEP 1: Dataset Creation

**Both Approaches: IDENTICAL**

```bash
# Video tutorial example (Spanish)
Audio Collection → dataset-maker → Gradio Interface
Transcription → Spanish_Amelia_dataset/manifest.jsonl

# Your implementation (Amharic)
Audio Collection → create_amharic_dataset.py
Segmentation → Amharic_dataset/manifest.jsonl
```

### STEP 2: Tokenizer Creation

**Video Tutorial:**
```bash
# Edit extend_bpe.bat
Dataset: datasets/Spanish_Amelia_dataset/manifest.jsonl
Target: 24000
Coverage: 100%

# Result
checkpoints/extended_bp.model (24,000 tokens)
- IDs 0-11999: English/Chinese (base)
- IDs 12000-23999: Spanish (new)
```

**Standalone:**
```bash
# Collect corpus
python tools/collect_amharic_corpus.py \
  --input datasets/Amharic_dataset/manifest.jsonl \
  --output amharic_corpus.txt

# Train standalone
python tools/tokenizer/train_standalone_bpe.py \
  --corpus amharic_corpus.txt \
  --output tokenizers/amharic_standalone.model \
  --vocab-size 12000

# Result
tokenizers/amharic_standalone.model (12,000 tokens)
- IDs 0-11999: Amharic only (all trainable)
```

### STEP 3: Preprocessing

**Video Tutorial (train.bat equivalent):**
```bash
Manifests: datasets/Spanish_Amelia_dataset/manifest.jsonl
Output: Spanish_processed_data
Tokenizer: checkpoints/extended_bp.model  # 24k
Config: config_finetune.yaml
GPT checkpoint: checkpoints/gpt.pt
Language: ES
Batch size: 1
```

**Standalone (adapted):**
```bash
python tools/preprocess_data.py \
  --manifest datasets/Amharic_dataset/manifest.jsonl \
  --output-dir Amharic_processed_data \
  --tokenizer tokenizers/amharic_standalone.model  # 12k
  --config checkpoints/config_finetune.yaml \
  --gpt-checkpoint checkpoints/gpt.pt \
  --language am
```

**KEY DIFFERENCE:** Tokenizer path and size

### STEP 4: Generate Pairs

**Both Approaches: IDENTICAL**

```bash
# Video tutorial (pair_jsonl.bat)
Directory: Spanish_processed_data
Pairs per target: 2

# Standalone (same script)
python tools/build_gpt_prompt_pairs.py \
  --input-manifest Amharic_processed_data/train_manifest.jsonl \
  --output-manifest Amharic_processed_data/GPT_pairs_train.jsonl \
  --num-pairs 2
```

### STEP 5: Training

**Video Tutorial (train.bat):**
```bash
Train manifest: Spanish_processed_data/GPT_pairs_train.jsonl
Val manifest: Spanish_processed_data/GPT_pairs_val.jsonl
Tokenizer: checkpoints/extended_bpe.model  # 24k
Base checkpoint: checkpoints/gpt.pt  # Expects 12k, gets 24k!
Batch size: 2-6 (based on VRAM)
Epochs: 2
Validation interval: 1000
```

**Issue:** Base checkpoint (gpt.pt) has 12k vocab, but tokenizer is 24k!
Model reinitializes embeddings for tokens 12000-23999 (random weights).

**Standalone (adapted):**
```bash
python trainers/train_gpt_v2.py \
  --train-manifest Amharic_processed_data/GPT_pairs_train.jsonl \
  --val-manifest Amharic_processed_data/GPT_pairs_val.jsonl \
  --tokenizer tokenizers/amharic_standalone.model  # 12k
  --base-checkpoint checkpoints/gpt.pt  # Also 12k - PERFECT MATCH!
  --output-dir trained_ckpts_12k \
  --batch-size 3 \
  --epochs 2
```

**Advantage:** Vocab sizes match perfectly (both 12k)

### STEP 6: Resume Training

**Video Tutorial:**
```bash
# To resume from checkpoint
Base checkpoint: train_checkpoints_Spanish/38000.pth  # 24k vocab
Tokenizer: checkpoints/extended_bpe.model  # Also 24k
--resume auto

# Problem: If checkpoint saved as 24001 but tokenizer is 24000
# Training breaks (optimizer state incompatible)
```

**Standalone:**
```bash
# To resume
python trainers/train_gpt_v2.py \
  --tokenizer tokenizers/amharic_standalone.model  # Always 12k
  --resume auto

# No problem: Always 12k, no mismatch possible!
```

### STEP 7: Model Pruning

**Both Approaches: IDENTICAL**

```bash
# Video tutorial (prune_model.bat)
Input: Spanish/39002.pth
Output: models/Spanish_39k

# Standalone (same script)
python tools/prune_gpt_checkpoint.py \
  --checkpoint trained_ckpts_12k/final.pth \
  --output models/amharic_12k
```

### STEP 8: Testing/Inference

**Video Tutorial (web_ui_parallel.py):**
```bash
uv run web_ui_parallel.py
Model: Spanish_39k
Tokenizer: extended_bpe.model  # 24k
Generate!
```

**Standalone (same UI):**
```bash
python webui_parallel.py
Model: amharic_12k
Tokenizer: amharic_standalone.model  # 12k
Generate!
```

---

## Hardware Requirements

**Video Tutorial Recommendations:**
- Min: 8GB VRAM
- Recommended: 12GB+
- Batch sizes:
  - 12GB GPU: batch=2
  - 16GB GPU: batch=3
  - 24GB GPU: batch=4-6

**Standalone (Same hardware, potentially better):**
- Same VRAM requirements
- But: 12k vocab uses less VRAM than 24k
- Can potentially use slightly larger batch sizes
- Faster preprocessing (smaller vocab = faster tokenization)

---

## Training Duration

**Video Tutorial (Spanish, 24k vocab):**
- Preprocessing: ~10 samples/second
- 6,000 hours dataset: ~2 days preprocessing
- Training: "Extremely time-consuming"
- Monitor via TensorBoard

**Standalone (Amharic, 12k vocab):**
- Preprocessing: Potentially 10-20% faster (smaller vocab)
- Same dataset size: Similar preprocessing time
- Training: Similar duration
- But: Simpler resume (no vocab mismatch issues)

---

## Key Metrics (TensorBoard)

**Both Approaches Monitor:**
- Loss: Should decrease and flatten
- Mel_top1: Should increase
- Text loss: Should decrease to ~1.5
- Validation graphs should trend similarly

**No difference in monitoring between approaches**

---

## Critical Differences Summary

| Aspect | Video (24k Extension) | Standalone (12k) |
|--------|----------------------|------------------|
| **Tokenizer Training** | Extend base model | Train from scratch |
| **Vocab Size** | 24,000 | 12,000 |
| **Cross-lingual** | ✅ Yes (English/Chinese) | ❌ No (Amharic only) |
| **Resume Safety** | ⚠️ Vocab mismatch risk | ✅ Always safe |
| **Complexity** | 🔧 More complex | ✨ Simpler |
| **Convergence** | ⚡ Faster (transfer) | 🐢 Slower (cold start) |
| **VRAM Usage** | 📦 Higher | 📦 Lower |
| **Trainable %** | 96.6% (3.4% frozen) | 100% (all trainable) |

---

## Which to Choose?

### Choose Video Tutorial Approach (24k) if:
1. ✅ Following official method
2. ✅ Want cross-lingual capability
3. ✅ Have multilingual data
4. ✅ Want faster convergence (transfer learning)
5. ✅ Don't mind vocab mismatch fix

### Choose Standalone Approach (12k) if:
1. ✅ Pure monolingual dataset (Amharic only)
2. ✅ Want simpler, more robust training
3. ✅ Want to avoid resume complications
4. ✅ Script very different from English/Chinese
5. ✅ Prefer 100% trainable parameters

---

## Practical Implementation for Amharic

### Option A: Video Tutorial Approach (Official)

```bash
# 1. Dataset
Amharic_Amelia_dataset/manifest.jsonl

# 2. Extend tokenizer (like video)
python tools/tokenizer/extend_bpe.py \
  --base-model checkpoints/bpe.model \
  --manifests datasets/Amharic_Amelia_dataset/manifest.jsonl \
  --output-model checkpoints/extended_bpe.model \
  --target-size 24000

# 3. Apply vocab mismatch fix (CRITICAL!)
# See TRAINING_STUCK_FIX_COMPLETE.md

# 4. Continue with preprocessing, training, etc.
```

### Option B: Standalone Approach (Alternative)

```bash
# 1. Dataset (same)
Amharic_Amelia_dataset/manifest.jsonl

# 2. Collect corpus
python tools/collect_amharic_corpus.py \
  --input datasets/Amharic_Amelia_dataset/manifest.jsonl \
  --output amharic_corpus.txt

# 3. Train standalone tokenizer (NEW!)
python tools/tokenizer/train_standalone_bpe.py \
  --corpus amharic_corpus.txt \
  --output tokenizers/amharic_standalone.model \
  --vocab-size 12000 \
  --character-coverage 0.9999 \
  --user-defined-symbols "።,፣,፤,፥,፧,፨,፡"

# 4. Continue with preprocessing, training
# (using amharic_standalone.model everywhere)
```

---

## Conclusion

Both approaches are **valid and will work**:

- **Video Tutorial (24k):** Official method, uses transfer learning, but has vocab mismatch risk that needs fixing
- **Standalone (12k):** Alternative method, simpler and more robust, but slower initial convergence

For your Amharic case with the vocab mismatch issue you encountered:
- **Quick fix:** Apply the vocab mismatch fix and continue with 24k
- **Clean slate:** Try standalone 12k approach for simpler training

**Bottom line:** The video tutorial approach is what the creators recommend, but the standalone approach is a viable alternative that avoids the complications you experienced.
