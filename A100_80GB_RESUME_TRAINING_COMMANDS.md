# A100 80GB Resume Training Commands

## Complete CLI Commands for Optimal Performance

These commands are specifically optimized for your **A100 80GB GPU with 12 CPUs** and include full resume training support.

---

## 🚀 Quick Start: Auto-Resume Training

### Best Practice (Recommended)

**Auto-resume with all optimizations:**

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --amp
```

**What this does:**
- ✅ Auto-detects A100 80GB → batch_size=32, grad_accum=1, workers=12
- ✅ Enables bfloat16 AMP (native A100 support)
- ✅ Enables TF32 (3-8× matmul speedup)
- ✅ Resumes from `training_output/latest.pth` if exists
- ✅ Starts fresh training if no checkpoint found

---

## 📋 Complete Training Commands

### 1. First Training Run (No Resume)

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --config checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt.pth \
  --output-dir training_output \
  --epochs 10 \
  --learning-rate 2e-5 \
  --amp
```

**Auto-optimized settings:**
- Batch size: 32 (A100 80GB)
- Gradient accumulation: 1 (maximum throughput)
- Data workers: 12 (matched to CPUs)
- AMP dtype: bfloat16 (Ampere native)
- TF32: Enabled (automatic)

---

### 2. Auto-Resume from Latest Checkpoint

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --epochs 10 \
  --learning-rate 2e-5 \
  --amp
```

**Resume behavior:**
- Looks for `training_output/latest.pth`
- If found: Resumes from that checkpoint
- If not found: Starts fresh training
- Prints clear message either way

---

### 3. Resume from Specific Checkpoint

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume training_output/model_step5000.pth \
  --epochs 10 \
  --learning-rate 2e-5 \
  --amp
```

**Use cases:**
- Resume from specific milestone (e.g., step 5000)
- Revert to earlier checkpoint if latest has issues
- Continue from backed-up checkpoint

---

### 4. Maximum Performance Configuration

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --batch-size 32 \
  --grad-accumulation 1 \
  --num-workers 12 \
  --epochs 10 \
  --learning-rate 2e-5 \
  --warmup-steps 1000 \
  --max-steps 0 \
  --log-interval 100 \
  --val-interval 500 \
  --amp
```

**Explicit settings (not necessary, but shown for reference):**
- Batch size: 32 (manually set, but auto-detected anyway)
- Workers: 12 (manually set, but auto-detected anyway)
- Validation every 500 steps (instead of per epoch)
- Logs every 100 steps

---

### 5. Conservative Settings (If OOM Occurs)

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --batch-size 24 \
  --grad-accumulation 2 \
  --epochs 10 \
  --amp
```

**Fallback if batch=32 causes OOM:**
- Batch size: 24 (reduced from 32)
- Gradient accumulation: 2 (keep effective batch=48 or reduce to 1 for effective=24)
- Still much faster than smaller GPUs

---

## 🔄 Multi-Dataset Training (Multiple Manifests)

### Train on Multiple Languages Simultaneously

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_amharic/train_pairs.jsonl::am \
  --train-manifest processed_english/train_pairs.jsonl::en \
  --val-manifest processed_amharic/val_pairs.jsonl::am \
  --val-manifest processed_english/val_pairs.jsonl::en \
  --tokenizer tokenizers/multilingual_bpe.model \
  --output-dir training_output \
  --resume auto \
  --epochs 10 \
  --amp
```

**Language hints:**
- `::am` - Amharic language hint
- `::en` - English language hint
- `::ja` - Japanese language hint
- Helps model learn language-specific patterns

---

## 📊 Monitoring Training

### TensorBoard (Recommended)

```bash
# In separate terminal/tmux session
tensorboard --logdir training_output/logs --port 6006
```

Then open: `http://localhost:6006`

### Watch GPU Utilization

```bash
# In separate terminal
watch -n 1 nvidia-smi
```

**Expected A100 80GB utilization:**
- GPU Usage: 90-100%
- VRAM: 50-60GB / 80GB
- Power: 250-400W
- Temp: 60-80°C

---

## 🛠️ Advanced Resume Scenarios

### 1. Change Learning Rate on Resume

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --learning-rate 1e-5 \
  --amp
```

**Note:** Learning rate scheduler resumes from saved state, but you can override initial LR

---

### 2. Resume with Different Batch Size (Not Recommended)

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --batch-size 16 \
  --grad-accumulation 2 \
  --amp
```

**Warning:** Changing batch size affects effective learning rate schedule. Only do this if necessary.

---

### 3. Resume Training with Additional Epochs

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --epochs 20 \
  --amp
```

**Example:** If you stopped at epoch 5, this continues until epoch 20 (15 more epochs)

---

### 4. Resume with Gradient Checkpointing (Save VRAM)

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --grad-checkpointing \
  --amp
```

**Use when:** OOM occurs even with batch_size=24
**Effect:** Reduces VRAM by ~30%, but slows training by ~15%

---

## 📝 Full Parameter Reference (A100 80GB Optimized)

```bash
python trainers/train_gpt_v2.py \
  # Required manifests (MUST use paired manifests from Tab 5.5)
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  
  # Model config
  --config checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt.pth \
  
  # Output
  --output-dir training_output \
  
  # Resume training (KEY FEATURE!)
  --resume auto \
  # OR --resume training_output/model_step5000.pth \
  
  # Training duration
  --epochs 10 \
  --max-steps 0 \
  
  # Optimizer (auto-optimized for A100 80GB)
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-steps 1000 \
  --batch-size 32 \
  --grad-accumulation 1 \
  --grad-clip 1.0 \
  
  # Loss weights
  --text-loss-weight 0.2 \
  --mel-loss-weight 0.8 \
  
  # Data loading (auto-optimized for 12 CPUs)
  --num-workers 12 \
  
  # Mixed precision (A100 native)
  --amp \
  
  # Logging
  --log-interval 100 \
  --val-interval 500 \
  
  # Memory optimization (only if needed)
  # --grad-checkpointing \
  
  # Random seed
  --seed 1234
```

---

## 🎯 Recommended Workflow

### Step 1: Start Training

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --epochs 10 \
  --amp
```

### Step 2: Monitor Progress

```bash
# Terminal 1: Training running
# Terminal 2: TensorBoard
tensorboard --logdir training_output/logs
# Terminal 3: GPU monitoring
watch -n 1 nvidia-smi
```

### Step 3: Resume if Interrupted

```bash
# Same command, just add --resume auto
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --epochs 10 \
  --amp
```

### Step 4: Continue for More Epochs

```bash
# Increase epochs to continue training
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --epochs 20 \
  --amp
```

---

## 🔍 Verify Hardware Detection

### Before Training: Test Detection

```bash
python -m indextts.utils.hardware_optimizer
```

**Expected output for A100 80GB:**

```
======================================================================
[Hardware Detection & Auto-Optimization]
======================================================================

[GPU] NVIDIA A100-SXM4-80GB
   VRAM: 80.0 GB
   AMP dtype: bfloat16
   TF32: Enabled (3-8x matmul speedup)

[CPU] 12 cores

[Optimized Training Settings]
   • Batch size: 32
   • Gradient accumulation: 1
   • Effective batch: 32
   • Data workers: 12
   • Mixed precision: Enabled (bfloat16)

[Recommendations]
   [OK] Detected: NVIDIA A100-SXM4-80GB (80.0GB VRAM)
   [OK] Detected: 12 CPU cores
   [PERF] Using bfloat16 AMP (native GPU support, better stability)
   [PERF] TF32 enabled (3-8x faster matmul)
   [TUNED] A100 80GB: batch=32, grad_accum=1 (effective=32)
   [PERF] Maximum throughput mode for A100 80GB
   [PERF] Using 12 data workers (optimal for 12 CPUs)
```

---

## ⚡ Performance Expectations

### Training Speed (200-hour Dataset, 10 Epochs)

| Metric | A100 80GB | A100 40GB | L4 24GB |
|--------|-----------|-----------|----------|
| **Preprocessing** | 2-4 hours | 3-6 hours | 8-12 hours |
| **Training** | 1-1.5 days | 1.5-2 days | 2-3 days |
| **Total Time** | ~1.3 days | ~1.7 days | ~2.6 days |
| **Speedup** | 2× baseline | 1.5× baseline | 1× baseline |

### GPU Utilization

- **Ideal:** 95-100% GPU usage
- **VRAM:** 50-60GB / 80GB (plenty of headroom)
- **Batch processing:** ~150-200 samples/sec
- **Throughput:** 3-4× faster than L4 GPU

---

## 🐛 Troubleshooting

### "Resume checkpoint not found"

```bash
# Check what checkpoints exist
ls -lh training_output/*.pth

# Use correct checkpoint path
python trainers/train_gpt_v2.py \
  --resume training_output/model_step3000.pth \
  ...
```

### "CUDA out of memory"

```bash
# Reduce batch size to 24
python trainers/train_gpt_v2.py \
  --batch-size 24 \
  --grad-accumulation 2 \
  --resume auto \
  ...
```

### "Training slower than expected"

```bash
# Verify AMP is enabled
python trainers/train_gpt_v2.py \
  --amp \
  --resume auto \
  ...

# Check TensorBoard for GPU utilization
tensorboard --logdir training_output/logs
```

### "Checkpoint loading failed"

```bash
# Try previous checkpoint
python trainers/train_gpt_v2.py \
  --resume training_output/model_step2000.pth \
  ...

# If all fail, restart from base
python trainers/train_gpt_v2.py \
  --output-dir training_output_new \
  ...
```

---

## 💾 Checkpoint Management

### Automatic Cleanup

- Saves checkpoint every 1000 steps
- Keeps only last 3 checkpoints
- Always maintains `latest.pth`

### Files Created

```
training_output/
├── latest.pth              # Always current (for --resume auto)
├── model_step1000.pth      
├── model_step2000.pth      
├── model_step3000.pth      # Only keeps last 3
└── logs/
    └── run_20250120_143022/
        ├── events.out.tfevents.*
        └── ...
```

### Backup Important Checkpoints

```bash
# Copy milestone checkpoint to safe location
cp training_output/model_step5000.pth backups/

# Resume from backup if needed
python trainers/train_gpt_v2.py \
  --resume backups/model_step5000.pth \
  ...
```

---

## 🎉 Summary

**Simplest command for A100 80GB:**

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --output-dir training_output \
  --resume auto \
  --epochs 10 \
  --amp
```

**Everything else auto-optimizes! 🚀**

- ✅ Batch size: 32
- ✅ Workers: 12
- ✅ bfloat16 AMP
- ✅ TF32 enabled
- ✅ Auto-resume
- ✅ Maximum throughput

---

**Date:** 2025-01-20  
**Hardware:** A100 80GB (12 CPUs)  
**Status:** Production Ready ✅
