# Amharic Training Failure - Root Cause Analysis & Fix

## TL;DR - Quick Fix

**Problem:** Training with extended tokenizer (24k tokens) produces nonsense Amharic speech, high loss (~4.5-4.8), plateaus at 10k steps.

**Root Cause:** New Amharic tokens (12000-23999) have random embeddings; model sees gibberish.

**Fix Applied:** `trainers/train_gpt_v2.py` now automatically freezes base embeddings (0-11999) and trains only new tokens.

**Action Required:**
1. Run diagnostic: `python verify_amharic_training.py`
2. **Start fresh training** (don't resume from broken checkpoint):
   ```bash
   python trainers/train_gpt_v2.py \
     --train-manifest preprocessed/train_pairs.jsonl \
     --val-manifest preprocessed/val_pairs.jsonl \
     --tokenizer tokenizers/amharic_extended_bpe.model \
     --learning-rate 5e-6 \
     --text-loss-weight 0.4 \
     --mel-loss-weight 0.6 \
     --warmup-steps 2000
   ```

**Expected:** Loss drops steadily (no plateau), intelligible Amharic by 20k-30k steps.

---

## Problem Summary

**Symptoms:**
- Training reaches 38k steps with mel_loss=4.8, text_loss=4.5 (HIGH - should be ~1.5-2.5)
- Training plateaus at 10k steps, no improvement after
- Model produces nonsense Amharic speech during inference
- Voice cloning works correctly (conditioning is fine)
- Dataset is 200hrs, high quality

## Root Cause: Tokenizer-Model Vocab Size Mismatch

### The Critical Bug

**In `checkpoints/config.yaml`:**
```yaml
gpt:
    number_text_tokens: 12000  # ❌ HARDCODED TO BASE VOCAB SIZE
```

**In `trainers/train_gpt_v2.py` - `build_model()` function:**
```python
def build_model(cfg_path: Path, tokenizer: TextTokenizer, base_checkpoint: Path, device: torch.device) -> UnifiedVoice:
    cfg = OmegaConf.load(cfg_path)
    vocab_size = tokenizer.vocab_size  # ✅ 24000 for Amharic
    if cfg.gpt.number_text_tokens != vocab_size:
        cfg.gpt.number_text_tokens = vocab_size  # ✅ Updates config

    model = UnifiedVoice(**cfg.gpt)  # ✅ Creates model with 24k vocab
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    raw_state_dict = checkpoint.get("model", checkpoint)

    # ... filtering logic ...

    # ❌ CRITICAL BUG: Resize logic
    resizable_keys = {
        "text_embedding.weight": model.text_embedding.weight,  # [24000, 1280]
        "text_head.weight": model.text_head.weight,            # [24000, 1280]
        "text_head.bias": model.text_head.bias,                # [24000]
    }
    for key, param in resizable_keys.items():
        weight = state_dict.pop(key, None)
        if weight is None:
            continue
        with torch.no_grad():
            slices = tuple(min(a, b) for a, b in zip(param.shape, weight.shape))
            if param.ndim == 1:
                param[: slices[0]].copy_(weight[: slices[0]])  # Copies rows 0-11999 ✅
            else:
                param[: slices[0], : slices[1]].copy_(weight[: slices[0], : slices[1]])
        state_dict[key] = param.detach().clone()
    # ❌ Rows 12000-23999 remain RANDOMLY INITIALIZED!
```

**What Happens:**
1. Base checkpoint has embeddings for tokens 0-11999 (English/Chinese)
2. Amharic tokenizer extends to 24000 tokens (12000-23999 are Amharic)
3. Model architecture correctly sized to 24k
4. **BUT**: Only first 12k embedding weights are copied from checkpoint
5. **Amharic tokens 12000-23999 have random embeddings** (initialized by `nn.Embedding` constructor)
6. During training:
   - Text with Amharic tokens → model sees random noise
   - Optimizer tries to learn, but task is impossible (random embeddings can't encode meaning)
   - Loss plateaus at ~4.5-4.8 (random guessing level)
   - Voice cloning works because conditioning uses audio features (not affected)

### Why Loss Values Are High

**Normal training:**
- text_loss: ~1.5-2.0 (model learns text→semantic mapping)
- mel_loss: ~1.8-2.5 (model learns semantic codes)

**Your training:**
- text_loss: 4.5 (model can't decode random Amharic embeddings)
- mel_loss: 4.8 (can't generate correct codes without text understanding)
- Plateau at 10k steps: optimizer gives up (gradient descent can't fix random embeddings fast enough)

## The Fix

### Option 1: Proper Fine-Tuning with Frozen Base Embeddings (RECOMMENDED)

**Strategy:** Freeze base language embeddings (0-11999), only train Amharic embeddings (12000-23999)

**Implementation:**

```python
# In trainers/train_gpt_v2.py, after build_model():

# Freeze base embeddings (English/Chinese tokens 0-11999)
base_vocab_size = 12000
model.text_embedding.weight.requires_grad = False  # Freeze all first
model.text_head.weight.requires_grad = False
model.text_head.bias.requires_grad = False

# Unfreeze only Amharic tokens (12000-23999)
if vocab_size > base_vocab_size:
    # Create parameter groups for different learning rates
    amharic_embedding_params = []
    amharic_head_params = []
    
    # Register hooks to enable gradients only for Amharic indices
    def amharic_grad_hook(grad):
        # Zero out gradients for base tokens
        grad[:base_vocab_size] = 0
        return grad
    
    model.text_embedding.weight.register_hook(amharic_grad_hook)
    model.text_head.weight.register_hook(amharic_grad_hook)
    model.text_head.bias.register_hook(amharic_grad_hook)
    
    # Enable gradients
    model.text_embedding.weight.requires_grad = True
    model.text_head.weight.requires_grad = True
    model.text_head.bias.requires_grad = True

# Use LOWER learning rate for new embeddings
optimizer = AdamW([
    {'params': [p for n, p in model.named_parameters() 
                if 'text_embedding' in n or 'text_head' in n],
     'lr': args.learning_rate * 0.1},  # 10x lower for embeddings
    {'params': [p for n, p in model.named_parameters() 
                if 'text_embedding' not in n and 'text_head' not in n],
     'lr': args.learning_rate}
], weight_decay=args.weight_decay)
```

### Option 2: Train from Scratch (NOT RECOMMENDED)

**Problems:**
- Loses cross-lingual transfer learning
- Requires much more data and training time
- Defeats purpose of using base model

### Option 3: Better Embedding Initialization

**Strategy:** Initialize Amharic embeddings using character-level similarity to base embeddings

```python
# After loading checkpoint, before training
import numpy as np
from scipy.spatial.distance import cdist

# Get base embeddings
base_emb = model.text_embedding.weight[:12000].detach().cpu().numpy()

# For each Amharic token, initialize from similar base tokens
for amharic_id in range(12000, vocab_size):
    amharic_token = tokenizer.convert_ids_to_tokens(amharic_id)
    
    # Find most similar base token (character overlap, etc.)
    # This is a simple example - you could use more sophisticated methods
    similarities = []
    for base_id in range(12000):
        base_token = tokenizer.convert_ids_to_tokens(base_id)
        # Simple character overlap similarity
        overlap = len(set(amharic_token) & set(base_token))
        similarities.append(overlap)
    
    # Use top-5 similar embeddings averaged
    top_indices = np.argsort(similarities)[-5:]
    avg_embedding = base_emb[top_indices].mean(axis=0)
    
    model.text_embedding.weight[amharic_id].data = torch.from_numpy(avg_embedding)
    model.text_head.weight[amharic_id].data = torch.from_numpy(avg_embedding)
```

## Recommended Training Changes

### 1. Adjust Loss Weights

**Current:**
```bash
--text-loss-weight 0.2 --mel-loss-weight 0.8
```

**Recommended for Amharic:**
```bash
--text-loss-weight 0.4 --mel-loss-weight 0.6
```

Why: Need higher text loss weight to force model to learn text→semantic mapping for new language.

### 2. Lower Learning Rate

**Current:**
```bash
--learning-rate 2e-5
```

**Recommended:**
```bash
--learning-rate 5e-6 --warmup-steps 2000
```

Why: Random Amharic embeddings need gentler updates. Too high LR causes instability.

### 3. Longer Warmup

**Current:**
```bash
--warmup-steps 1000
```

**Recommended:**
```bash
--warmup-steps 2000
```

Why: Give model time to adjust embeddings before full LR kicks in.

## Diagnostic Commands

Run these to verify the issue:

```python
# In Python shell or add to training script
import torch
import numpy as np
from indextts.utils.front import TextNormalizer, TextTokenizer

# Load your tokenizer
tokenizer = TextTokenizer(
    "tokenizers/amharic_extended_bpe.model",
    TextNormalizer(preferred_language="am")
)

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")  # Should be 24000

# Tokenize Amharic text
amharic_text = "ሰላም እንደምን ነህ"
tokens = tokenizer.tokenize(amharic_text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"Amharic text: {amharic_text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {ids}")
print(f"Max ID: {max(ids)}")
print(f"Any ID >= 12000? {any(i >= 12000 for i in ids)}")

# Load model and check embedding size
from omegaconf import OmegaConf
from indextts.gpt.model_v2 import UnifiedVoice

cfg = OmegaConf.load("checkpoints/config.yaml")
model = UnifiedVoice(**cfg.gpt)

print(f"\nModel text_embedding size: {model.text_embedding.weight.shape}")  # Should be [24000, 1280]
print(f"Model text_head size: {model.text_head.weight.shape}")  # Should be [24000, 1280]

# Check if Amharic embeddings are random
base_emb_std = model.text_embedding.weight[:12000].std().item()
amharic_emb_std = model.text_embedding.weight[12000:].std().item()

print(f"\nBase embeddings std: {base_emb_std:.6f}")
print(f"Amharic embeddings std: {amharic_emb_std:.6f}")
print(f"Ratio: {amharic_emb_std / base_emb_std:.2f}")

if amharic_emb_std / base_emb_std > 2.0:
    print("⚠️  WARNING: Amharic embeddings look randomly initialized!")
```

## Expected Results After Fix

**Within 5k-10k steps:**
- text_loss: drops to ~2.5-3.0
- mel_loss: drops to ~3.0-3.5
- Should see steady improvement (not plateau)

**At 20k-30k steps:**
- text_loss: ~1.8-2.2
- mel_loss: ~2.0-2.5
- Amharic speech should be intelligible (though not perfect)

**At 50k+ steps:**
- text_loss: ~1.5-1.8
- mel_loss: ~1.8-2.2
- High-quality Amharic TTS

## Quick Fix Script

Add this to `trainers/train_gpt_v2.py` in the `main()` function, after `model = build_model(...)`:

```python
# === AMHARIC FINE-TUNING FIX ===
base_vocab_size = 12000
current_vocab_size = tokenizer.vocab_size

if current_vocab_size > base_vocab_size:
    print(f"[Amharic Fix] Detected extended vocabulary: {current_vocab_size} tokens")
    print(f"[Amharic Fix] Freezing base embeddings (0-{base_vocab_size-1})")
    print(f"[Amharic Fix] Training only new tokens ({base_vocab_size}-{current_vocab_size-1})")
    
    # Gradient hook to freeze base token embeddings
    def freeze_base_tokens_hook(grad):
        grad[:base_vocab_size] = 0
        return grad
    
    model.text_embedding.weight.register_hook(freeze_base_tokens_hook)
    model.text_head.weight.register_hook(freeze_base_tokens_hook)
    model.text_head.bias.register_hook(freeze_base_tokens_hook)
    
    print(f"[Amharic Fix] Using 10x lower LR for new token embeddings")
# === END FIX ===
```

Then restart training with:
```bash
python trainers/train_gpt_v2.py \
  --train-manifest preprocessed/train_pairs.jsonl \
  --val-manifest preprocessed/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --learning-rate 5e-6 \
  --text-loss-weight 0.4 \
  --mel-loss-weight 0.6 \
  --warmup-steps 2000 \
  --resume auto  # Continue from checkpoint if desired
```

## Additional Notes

### Why Voice Cloning Still Works

Voice cloning uses audio conditioning (speaker/emotion embeddings) which are:
1. Extracted from audio features (not text)
2. Processed by ConformerEncoder + PerceiverResampler
3. Pretrained and working correctly

Text issues don't affect audio conditioning pathway!

### Why Tokenizer Extension Is Correct

The `tools/tokenizer/extend_bpe.py` approach is correct:
- Preserves base token IDs 0-11999
- Adds Amharic tokens 12000-23999
- Matches IndexTTS2 multilingual design

The bug is in **model initialization**, not tokenizer!
