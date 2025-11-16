# Training Improvements TODO

**Status:** Planned for future integration  
**Last Updated:** 2025-01-15  
**Architecture:** IndexTTS2 GPT-2 + Conformer (dual text+mel loss)  
**Target Language:** Amharic (Ethiopic script, extended tokenizer 24k vocab)

---

## 🔬 Architecture Analysis

### IndexTTS2 Model Structure (model_v2.py)

```python
GPT2Model(
  wte: Embedding(24000, 512),        # Token embeddings [AMHARIC: IDs 12000-23999]
  wpe: Embedding(max_pos, 512),      # Position embeddings [FROZEN]
  h: [GPT2Block × n_layer],          # Transformer decoder
    ├─ attn: GPT2Attention            # c_attn (QKV), c_proj (O)
    ├─ mlp: GPT2MLP                   # c_fc (in), c_proj (out)
    └─ ln_1, ln_2: LayerNorm          # [FROZEN in PEFT]
  ln_f: LayerNorm,                    # Final norm [FROZEN]
)
lm_head: Linear(512, 24000)          # Text predictions [TRAINABLE]
mel_head: Linear(512, mel_vocab)     # Mel predictions [TRAINABLE]
conformer_encoder (optional)         # Speech encoder [OPTIONAL PEFT]
```

### Training Objective (Dual Loss)

```python
loss = text_weight × text_loss + mel_weight × mel_loss
# Both losses backprop through same model params
# Mel_loss is primary TTS quality metric
```

### Amharic Extended Tokenizer

**Vocab Layout (24,000 tokens):**
- **IDs 0–11999:** Base (English/Chinese) - pretrained embeddings
- **IDs 12000–23999:** Amharic (Ethiopic) - random init embeddings

**🔴 CRITICAL:** New Amharic tokens have untrained embeddings in wte!

---

## Safe to Add Mid-Training (Resume-Compatible)

### ✅ Gradient Clipping

**Implementation:**
```python
# After loss.backward() and before optimizer.step()
if use_amp:
    scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if use_amp:
    scaler.step(optimizer)
else:
    optimizer.step()
```

**Benefits:**
- Prevents exploding gradients
- Stabilizes training (especially with long sequences)
- No checkpoint state change

**Compatibility:**
- ✅ Resume-safe (no optimizer/model state change)
- ✅ Works with AMP (call after scaler.unscale_)
- ✅ Amharic-safe (language-agnostic)

---

### ✅ EMA (Exponential Moving Average) Weights

**Implementation:**
```python
# After model initialization
ema_model = copy.deepcopy(model)
ema_decay = 0.9999

# After optimizer.step()
with torch.no_grad():
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1-ema_decay)

# At validation
with torch.no_grad():
    ema_model.eval()
    val_loss = compute_losses(ema_model, val_batch, device)

# At checkpoint save
checkpoint["ema_state"] = ema_model.state_dict()

# At checkpoint load (with fallback)
if "ema_state" in checkpoint:
    ema_model.load_state_dict(checkpoint["ema_state"])
else:
    print("[Info] No EMA state, initializing from current model")
    ema_model = copy.deepcopy(model)
```

**Benefits:**
- Smoother validation metrics (less noise)
- Better inference quality (averaged weights)
- Minimal memory cost (+~5%)

**Compatibility:**
- ✅ Resume-safe (optional checkpoint key with fallback)
- ✅ Dual-loss safe (EMA applies to lm_head + mel_head)
- ✅ Amharic-safe (wte embeddings also averaged)

---

### ✅ Best-by-Val Checkpoint Tracking

**Implementation:**
```python
# Initialize
best_val_mel_loss = float('inf')

# After validation
if val_mel_loss < best_val_mel_loss:
    best_val_mel_loss = val_mel_loss
    torch.save({
        "model": model.state_dict(),
        "ema_state": ema_model.state_dict() if ema_model else None,
        "epoch": epoch,
        "step": global_step,
        "val_mel_loss": val_mel_loss,
    }, output_dir / "best_model.pth")
    print(f"[Info] New best val_mel_loss: {val_mel_loss:.4f}")
```

**Benefits:**
- Easy rollback to optimal checkpoint
- Separate from latest.pth (training continuity)
- Tracks primary TTS quality metric

**Compatibility:**
- ✅ Resume-safe (separate file, no conflict)
- ✅ Dual-loss safe (mel_loss is TTS quality metric)
- ✅ Amharic-safe (language-agnostic validation)

**🔴 Important:** Track by `mel_loss`, NOT `text_loss`
- Text_loss hits ceiling (tokenization limit)
- Mel_loss correlates with speech quality

---

### ✅ Length Bucketing

**Implementation:**
```python
from torch.utils.data import Sampler

class LengthBucketSampler(Sampler):
    def __init__(self, dataset, batch_size, boundaries=[32,64,128,256,512]):
        # Bucket by len(sample['target_semantic']), NOT prompt!
        self.buckets = defaultdict(list)
        for idx, sample in enumerate(dataset):
            target_len = len(sample['target_semantic'])
            bucket_id = bisect.bisect_left(boundaries, target_len)
            self.buckets[bucket_id].append(idx)
        
        # Shuffle within buckets
        for bucket in self.buckets.values():
            random.shuffle(bucket)
    
    def __iter__(self):
        # Yield batches from same bucket
        for bucket_indices in self.buckets.values():
            for i in range(0, len(bucket_indices), self.batch_size):
                yield bucket_indices[i:i+self.batch_size]

# Replace in train_gpt_v2.py
train_loader = DataLoader(
    train_dataset,
    batch_sampler=LengthBucketSampler(train_dataset, batch_size),
    num_workers=num_workers,
    collate_fn=collate_fn,
)
```

**Benefits:**
- 10–20% speedup (less padding waste)
- Smoother loss curves (similar-length batches)
- Better GPU utilization

**Compatibility:**
- ✅ Resume-safe (DataLoader tweak only)
- ✅ Dual-loss safe (applies to both text+mel)
- ✅ Prompt-pairs safe (buckets by target, preserves prompt diversity)

**🔴 Amharic-Specific:**
- **MUST bucket by `target_semantic` length** (not prompt!)
- Amharic syllabary is uniform, but prompts vary by speaker/emotion
- Bucketing by prompt → clusters speakers → poor generalization

---

### ✅ Validation Frequency Increase

**Change:**
```python
# Current: validate every ~5000 steps
# New: validate every 2000-3000 steps
val_interval = 2000  # vs current ~5126 (once per epoch)
```

**Benefits:**
- Catch issues earlier
- Better best-checkpoint selection
- More granular loss tracking

**Compatibility:**
- ✅ Resume-safe (scheduler parameter only)
- ✅ Minimal overhead (~1–2% slower)

---

### ✅ Early Stopping

**Implementation:**
```python
# Initialize
patience = 5
patience_counter = 0
min_improvement = 0.01

# After validation
if val_mel_loss < (best_val_mel_loss - min_improvement):
    best_val_mel_loss = val_mel_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        print(f"[Info] Early stopping: no improvement for {patience} validations")
        break

# Save in checkpoint
checkpoint["patience_counter"] = patience_counter
checkpoint["best_val_mel_loss"] = best_val_mel_loss
```

**Benefits:**
- Prevents overfitting
- Saves compute on plateaued runs

**Compatibility:**
- ✅ Resume-safe (add to checkpoint dict)
- ✅ Optional (can disable for Amharic, which needs 60k+ steps)

---

## Requires Fresh Training (Checkpoint Incompatible)

### ⚠️ DoRA + AdaLoRA (PEFT)

**Why Fresh Training Required:**
- Adds new state_dict keys: `lora_A`, `lora_B`, `magnitude`
- Old checkpoints lack these keys → load fails
- Cannot resume vanilla FT checkpoint with PEFT enabled

**Architecture Targets (GPT-2 Decoder):**

```python
# Apply DoRA to these modules (using PEFT library or custom)
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,                    # Attention rank
    lora_alpha=64,           # 2 × r
    target_modules=[
        "attn.c_attn",       # Q/K/V projection (DoRA r=32)
        "attn.c_proj",       # Output projection (DoRA r=32)
        "mlp.c_fc",          # MLP in (DoRA r=16)
        "mlp.c_proj",        # MLP out (DoRA r=16)
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # AdaLoRA budget allocation
    use_dora=True,           # Enable DoRA (weight decomposition)
)

# Modules to FREEZE
for name, param in model.named_parameters():
    if any(x in name for x in ["wpe", "ln_", "ln_f"]):
        param.requires_grad = False

# Modules to KEEP TRAINABLE
for name, param in model.named_parameters():
    if any(x in name for x in ["wte", "lm_head", "mel_head"]):
        param.requires_grad = True
```

**🔴 CRITICAL: Amharic wte Trainability**

**Problem:**
- Extended tokenizer adds Amharic tokens (IDs 12000–23999)
- These embeddings are randomly initialized
- If wte frozen → Amharic never learns proper semantics!

**Solutions:**

1. **Keep wte Fully Trainable (RECOMMENDED):**
```python
# Allow wte to adapt (all 24k embeddings)
model.transformer.wte.requires_grad = True

# Add weight decay to prevent base token drift
optimizer = AdamW([
    {"params": adapter_params, "weight_decay": 0.05},      # DoRA adapters
    {"params": [model.transformer.wte.weight], "weight_decay": 0.01},  # Embeddings (lighter)
    {"params": [model.lm_head.weight, model.mel_head.weight], "weight_decay": 0.0},  # Heads
], lr=2e-5)
```

2. **Selective Embedding Adapter (Advanced):**
```python
# Freeze base tokens (0-11999), adapt Amharic only (12000-23999)
base_embeddings = model.transformer.wte.weight[:12000].detach()
amharic_adapter = nn.Parameter(torch.randn(12000, 512) * 0.01)

def forward_with_adapter(input_ids):
    base_mask = input_ids < 12000
    amh_mask = input_ids >= 12000
    
    emb = torch.zeros_like(input_ids, dtype=torch.float)
    emb[base_mask] = base_embeddings[input_ids[base_mask]]
    emb[amh_mask] = base_embeddings[input_ids[amh_mask] - 12000] + amharic_adapter[input_ids[amh_mask] - 12000]
    return emb
```

3. **DoRA on wte (Balanced):**
```python
# Apply DoRA to entire wte (rank=8-16, lower than attention)
config.target_modules.append("wte")
# Protects base while adapting Amharic
```

**Recommended for Amharic:** Option 1 (full wte trainable) - simplest, most effective

---

### ⚠️ Batch Size Increase (64 → 72–96)

**Why Fresh Training Required:**
- Optimizer momentum/variance scaled for batch=64
- Changing batch → LR effective scaling changes
- Momentum vectors misaligned with new batch statistics

**Implementation:**
```python
# In hardware_optimizer.py or train_gpt_v2.py
if vram_gb >= 80:  # A100 80GB
    batch_size = 96  # vs current 64
    # Optional: scale LR linearly
    lr = base_lr * (96 / 64)  # e.g., 1e-4 → 1.5e-4
```

**Compatibility:**
- ❌ Resume breaks (optimizer state mismatch)
- ✅ Quality maintained (if LR scaled appropriately)
- ✅ 20–40% faster wall-clock time

---

### ⚠️ Layer-Wise Learning Rate Decay

**Why Fresh Training Required:**
- Creates separate optimizer param groups per layer
- Old checkpoint has single param_group
- State_dict structure incompatible

**Implementation:**
```python
# Group params by layer depth
param_groups = []
num_layers = len(model.transformer.h)
for i, block in enumerate(model.transformer.h):
    layer_lr = base_lr * (decay_rate ** (num_layers - i - 1))
    param_groups.append({
        "params": block.parameters(),
        "lr": layer_lr,
        "weight_decay": 0.05,
    })

# Top layers (lm_head, mel_head) use full LR
param_groups.append({
    "params": list(model.lm_head.parameters()) + list(model.mel_head.parameters()),
    "lr": base_lr,
    "weight_decay": 0.0,
})

optimizer = AdamW(param_groups)
```

**Params:**
- base_lr: 1e-4 (top layers)
- decay_rate: 0.95 (per layer)
- Bottom layer LR ≈ base_lr × 0.95^12 ≈ 0.54 × base_lr

**Compatibility:**
- ❌ Resume breaks (param_groups structure changed)
- ✅ Better stability (protects lower layers)
- ✅ Faster convergence (top layers adapt quickly)

---

## Length Bucketing: Amharic-Specific Design

### 🔴 CRITICAL: Bucket by Target, NOT Prompt

**Why:**
- Prompt-pairs dataset: (prompt_audio, prompt_text) → (target_audio, target_text)
- Prompts vary by speaker, emotion, style
- Bucketing by prompt → clusters speakers → poor generalization

**Correct Approach:**
```python
class LengthBucketSampler(Sampler):
    def __init__(self, dataset, batch_size, boundaries=[32,64,128,256,512]):
        self.buckets = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset[idx]
            # Use TARGET length (main training signal)
            target_len = len(sample['target_semantic'])  # NOT prompt_semantic!
            bucket_id = bisect.bisect_left(boundaries, target_len)
            self.buckets[bucket_id].append(idx)
        
        # Shuffle within buckets (preserve randomness)
        for bucket in self.buckets.values():
            random.shuffle(bucket)
    
    def __iter__(self):
        all_batches = []
        for bucket_indices in self.buckets.values():
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:  # Skip incomplete batches
                    all_batches.append(batch)
        
        random.shuffle(all_batches)  # Randomize bucket order
        for batch in all_batches:
            yield batch
```

**Amharic Characteristics:**
- Ethiopic script: ~231 base syllables (fidels)
- Each character = 1 syllable (uniform density)
- Target length ≈ syllable count ≈ duration predictor
- Bucketing by target → groups similar-duration utterances

---

## DoRA/AdaLoRA: Complete Integration Guide

### Architecture Mapping

**GPT-2 Decoder (Primary PEFT Target):**

| Module | Current Params | DoRA Rank | Adapter Params | Reduction |
|--------|---------------|-----------|----------------|----------|
| attn.c_attn (QKV) | 1536 × 512 | r=32 | 2×32×512 ≈ 33k | 23× fewer |
| attn.c_proj (O) | 512 × 512 | r=32 | 2×32×512 ≈ 33k | 8× fewer |
| mlp.c_fc (in) | 512 × 2048 | r=16 | 2×16×2048 ≈ 66k | 16× fewer |
| mlp.c_proj (out) | 2048 × 512 | r=16 | 2×16×512 ≈ 16k | 66× fewer |

**Per layer:** ~148k adapter params (vs ~2M full params) → **13× reduction**

**Total (12 layers):** ~1.8M adapter params (vs ~24M full decoder) → **13× reduction**

**Conformer Encoder (Optional, if enabled):**
- MultiHeadAttention modules → DoRA r=24
- Budget: 30% of total adapter params

**Frozen (Architecture-Critical):**
- wpe (positional) - shared across languages
- ln_* (all layer norms) - destabilize if adapted
- ln_f (final norm) - critical for head projections

**Trainable (Task/Language-Specific):**
- **wte (token embeddings)** - MUST adapt for Amharic IDs 12000–23999
- lm_head (text predictions) - task-specific output
- mel_head (mel predictions) - task-specific output

### AdaLoRA Budget Allocation

**Rank Distribution:**
```python
# Start with target ranks, AdaLoRA prunes unused capacity
initial_ranks = {
    "attn.c_attn": 32,
    "attn.c_proj": 32,
    "mlp.c_fc": 16,
    "mlp.c_proj": 16,
}

# AdaLoRA dynamically adjusts (some layers may drop to r=8, others keep r=32)
total_budget = sum(initial_ranks.values()) * num_layers
adalora_config = AdaLoRAConfig(
    target_modules=list(initial_ranks.keys()),
    r=initial_ranks,
    total_step=total_training_steps,
    pruning_steps=[10000, 20000, 30000],  # Prune unused ranks
)
```

**Result:**
- Important layers (attn in top layers) keep high rank
- Less important layers (mlp in bottom) prune to r=8–12
- Total params: 1.5–2M (vs 1.8M fixed DoRA)

### Optimizer Setup (PEFT)

```python
# Separate param groups for adapters vs. full-grad modules
optimizer = AdamW([
    # DoRA/AdaLoRA adapters
    {
        "params": [p for n, p in model.named_parameters() if "lora" in n],
        "lr": 2e-5,
        "weight_decay": 0.05,
    },
    # Token embeddings (Amharic IDs need to adapt!)
    {
        "params": [model.transformer.wte.weight],
        "lr": 1e-5,  # Slightly lower to protect base tokens
        "weight_decay": 0.01,
    },
    # Task heads (text + mel)
    {
        "params": list(model.lm_head.parameters()) + list(model.mel_head.parameters()),
        "lr": 3e-5,  # Slightly higher (task-specific)
        "weight_decay": 0.0,
    },
], fused=True)  # Fused AdamW for A100
```

### Merge Script (Export)

```python
# After training completes
from peft import get_peft_model, PeftModel

# Load PEFT checkpoint
peft_model = PeftModel.from_pretrained(base_model, "trained_ckpts/latest")

# Merge adapters into base weights
merged_model = peft_model.merge_and_unload()

# Save single checkpoint (no adapters, zero runtime overhead)
torch.save({
    "model": merged_model.state_dict(),
    "config": config,
}, "trained_ckpts/merged_final.pth")

print("[Info] Merged DoRA adapters → single checkpoint (full capabilities, zero overhead)")
```

---

## Implementation Checklist

### Phase 1: Current Run (Step 28k → 60–100k)

**Add incrementally, test each:**

- [ ] **Gradient clipping** (max_norm=1.0)
  - Insert before optimizer.step()
  - Test: resume from step 28k, train 100 steps, verify no loss spike

- [ ] **EMA weights** (decay=0.9999)
  - Initialize from current model
  - Update after optimizer.step()
  - Save `checkpoint["ema_state"]` with fallback on load
  - Test: checkpoint save/load cycle

- [ ] **Best-by-val tracker** (mel_loss)
  - Save separate `best_model.pth`
  - Update after validation
  - Test: verify file created after first validation

- [ ] **Length bucketing** (by target_semantic)
  - Replace DataLoader sampler
  - Boundaries: [32, 64, 128, 256, 512]
  - Test: verify batch diversity (check speaker IDs)

- [ ] **Validation every 2k steps**
  - Change val_interval parameter
  - Test: verify validation runs at expected steps

- [ ] **Early stopping** (patience=5, optional)
  - Add counter logic
  - Save in checkpoint
  - Test: resume preserves counter

**Validation:** After all Phase 1 improvements:
- Train 1000 steps, compare losses to baseline
- Check VRAM usage (should be +5% max)
- Verify resume from checkpoint works
- Listen to Amharic samples (quality unchanged)

### Phase 2: Next Training Cycle (Fresh Start)

**Full retraining with PEFT:**

- [ ] **DoRA + AdaLoRA setup**
  - Install: `pip install peft`
  - Configure: r=32 (attn), r=16 (mlp), alpha=2r, dropout=0.05
  - **Keep wte trainable** (Amharic IDs 12000–23999 requirement)
  - Apply to: c_attn, c_proj (attn), c_fc, c_proj (mlp)
  - Freeze: wpe, ln_*, ln_f
  - Trainable: wte, lm_head, mel_head

- [ ] **Batch size 72–96** (A100 80GB)
  - Test VRAM with `nvidia-smi`
  - Optional: scale LR linearly (lr × batch_ratio)

- [ ] **Layer-wise LR decay**
  - decay_rate=0.95 per layer (top-to-bottom)
  - Top layers: lr × 1.0
  - Bottom layers: lr × 0.3–0.5

- [ ] **All Phase 1 improvements**
  - EMA, best-by-val, grad clip, bucketing, etc.

- [ ] **Adapter merge script**
  - After convergence, merge DoRA into base
  - Export single checkpoint
  - Test inference (should match adapter quality)

- [ ] **EWC (optional)**
  - Compute Fisher on English/Chinese val set
  - Add regularization: λ × Σ F_i × (θ_i - θ_base)²
  - λ ≈ 1000–10000

**Validation:** After Phase 2 completion:
- Test Amharic quality (mel_top1 ≥0.35, val_mel_loss <4.0)
- Test English/Chinese quality (no regression from EWC)
- Compare merged checkpoint to adapter checkpoint (should match)
- Verify zero runtime overhead (no adapter inference)

---

## Risk Analysis & Mitigations

### Risk 1: EMA + Resume from Old Checkpoint

**Problem:** Old checkpoint lacks "ema_state" key  
**Impact:** KeyError or NoneType exception  
**Mitigation:**
```python
if "ema_state" in checkpoint and ema_model:
    ema_model.load_state_dict(checkpoint["ema_state"])
else:
    print("[Info] No EMA state in checkpoint, initializing from current model")
    ema_model = copy.deepcopy(model)
```

---

### Risk 2: DoRA + Amharic Tokenizer (wte Frozen)

**Problem:** Amharic tokens (12000–23999) never learn if wte frozen  
**Impact:** Catastrophic quality loss (random embeddings)  
**Mitigation:** Keep wte trainable OR selective adapter (see DoRA section)

---

### Risk 3: Length Bucketing + Prompt Diversity

**Problem:** Bucketing by prompt → clusters speakers/styles  
**Impact:** Poor generalization (model overfits to speaker groups)  
**Mitigation:** Bucket by target_semantic length (preserves prompt diversity)

---

### Risk 4: Batch Increase + Resume

**Problem:** Optimizer momentum for batch=64 ≠ batch=96  
**Impact:** Loss spike, unstable training after resume  
**Mitigation:** Restart from step 0 OR keep batch=64 until next cycle

---

### Risk 5: Layer-Wise LR + Resume

**Problem:** Old checkpoint has 1 param_group, new has N param_groups  
**Impact:** State_dict key mismatch, load fails  
**Mitigation:** Restart from step 0 (no workaround)

---

## Testing Protocol

### Before Full Integration

**Incremental Testing:**
1. Add ONE improvement
2. Resume from latest checkpoint (step 28k)
3. Train 100–200 steps
4. Check:
   - No loss spike (±10% of previous)
   - Checkpoint save/load works
   - VRAM within limits
   - Training speed unchanged or faster
5. If stable → add next improvement
6. If issues → revert and debug

**Validation Metrics (Amharic):**
- mel_loss continues downward trend
- mel_top1 stays ≥0.12–0.25 (current range)
- text_loss stable (expected)
- No NaN/Inf losses

### Integration Order (Safest → Riskiest)

1. Gradient clipping (simplest)
2. Best-by-val tracking (separate file)
3. Validation frequency (minimal change)
4. EMA weights (new checkpoint key)
5. Length bucketing (DataLoader change)
6. Early stopping (optional)

**Full Integration Test:**
- All Phase 1 improvements enabled
- Resume from step 28k
- Train to step 30k (2k steps)
- Validate quality:
  - Losses within ±5% of trend
  - Checkpoint size increased by ~5% (EMA)
  - Training time per step unchanged (±5%)

---

## Rollback Plan

### If Improvement Causes Issues

**Symptoms:**
- Loss spikes >50% after adding improvement
- Checkpoint load fails (KeyError, shape mismatch)
- OOM (out of memory)
- Training hangs or crashes

**Action:**
1. Stop training immediately (Ctrl+C)
2. Identify last good checkpoint:
   ```bash
   ls -lth trained_ckpts/  # Find checkpoint before improvement
   ```
3. Revert code changes:
   ```bash
   git diff trainers/train_gpt_v2.py  # Review changes
   git checkout trainers/train_gpt_v2.py  # Revert
   ```
4. Resume from last good checkpoint:
   ```bash
   python trainers/train_gpt_v2.py \
     --resume trained_ckpts/model_step28000.pth \
     # ... other args
   ```
5. Debug improvement in isolation (small test run)

**Checkpoint Safety:**
- Keep ≥3 recent checkpoints (already implemented with `--keep-checkpoints 2`)
- Never delete checkpoints during active training
- Test resume before long runs

---

## Quality Validation (Amharic-Specific)

### Convergence Targets

**Step 50k (Decent Quality):**
- val_mel_loss ≈ 3.8–4.2
- mel_top1 ≈ 0.30–0.35
- Manual: Clear Ethiopic pronunciation, some accent

**Step 80k+ (Near-Native Quality):**
- val_mel_loss ≈ 3.5–3.8
- mel_top1 ≈ 0.35–0.40
- Manual: Natural prosody, minimal accent

### Manual Testing Protocol

**Every 5k steps:**
1. Generate 5 Amharic samples (random from val set)
2. Check:
   - Ethiopic script pronunciation (fidels correct?)
   - Prosody and rhythm (natural flow?)
   - Speaker consistency (matches prompt?)
   - Emotion transfer (if using emotion prompts)
3. Compare to step 25k baseline
4. Document quality progression

**Test Cases (examples/amharic_test_cases.jsonl):**
- Common phrases (greetings, questions)
- Long utterances (multi-sentence)
- Rare fidels (extended Unicode ranges)
- Emotional speech (happy, sad, neutral)

---

## Expected Performance (A100 80GB)

### Current Setup (Step 28k)
- Batch: 64, Grad accum: 1, Workers: 12
- Time per step: ~X seconds (fill in from logs)
- VRAM: ~50–60GB

### After Phase 1 Improvements
- Batch: 64 (unchanged)
- Time per step: ~0.9X seconds (10% faster from bucketing)
- VRAM: ~53–63GB (+5% from EMA)
- Training to 100k: ~Y hours (vs ~1.2Y hours before)

### After Phase 2 (DoRA + Batch 96)
- Batch: 96, Grad accum: 1, Workers: 16
- Time per step: ~0.7X seconds (30% faster)
- VRAM: ~40–50GB (fewer trainable params)
- Training to 80k: ~0.5Y hours (2× faster than baseline)
- Model size: ~500MB (vs ~1.5GB full FT)

---

## References

**Papers:**
- DoRA: Weight-Decomposed Low-Rank Adaptation (2024)
- AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning (2023)
- EMA in Deep Learning: Polyak Averaging

**Libraries:**
- PEFT: https://github.com/huggingface/peft
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- Fused AdamW: torch.optim.AdamW(..., fused=True)

**IndexTTS2:**
- Model architecture: indextts/gpt/model_v2.py
- Training loop: trainers/train_gpt_v2.py
- Hardware optimizer: indextts/utils/hardware_optimizer.py

---

**Last Updated:** 2025-01-15  
**Maintainer:** IndexTTS2 Team  
**Status:** Ready for phased implementation  
**Next Action:** Add Phase 1 improvements to current run (step 28k)
