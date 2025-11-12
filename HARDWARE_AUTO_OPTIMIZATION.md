# Hardware Auto-Optimization System

## Overview

IndexTTS2 now **automatically detects** your hardware and optimizes all training/preprocessing settings for maximum performance. No manual configuration needed!

## Supported Hardware

### GPUs (Auto-Detected)

| GPU Model | VRAM | Batch | Grad Accum | Effective | AMP Type | TF32 |
|-----------|------|-------|------------|-----------|----------|------|
| **A100** | 40-80GB | 16 | 2 | 32 | bfloat16 | Yes |
| **L4** | 24GB | 8 | 4 | 32 | bfloat16 | Yes |
| **RTX 4090** | 24GB | 8 | 4 | 32 | bfloat16 | Yes |
| **RTX 3090** | 24GB | 8 | 4 | 32 | bfloat16 | Yes |
| **V100** | 16-32GB | 6 | 6 | 36 | float16 | No |
| **RTX 4080** | 16GB | 6 | 6 | 36 | bfloat16 | Yes |
| **A10** | 24GB | 8 | 4 | 32 | bfloat16 | Yes |
| **RTX 3060** | 12GB | 4 | 8 | 32 | float16 | No |
| **T4** | 16GB | 6 | 6 | 36 | float16 | No |
| **RTX 3050** | 8GB | 2 | 16 | 32 | float16 | No |
| **CPU** | - | 1 | 32 | 32 | float32 | No |

### Architecture Detection

**Ampere/Ada/Hopper GPUs (bfloat16 + TF32):**
- A100, A10, A30, A40
- L4, L40, L40S
- H100, H800
- RTX 30 series (3060, 3070, 3080, 3090)
- RTX 40 series (4060, 4070, 4080, 4090)

**Older GPUs (float16 only):**
- V100, P100
- T4
- RTX 20 series (2060, 2070, 2080)
- GTX 16 series (1660, 1080 Ti)

## Features

### 1. Automatic VRAM-Based Batch Tuning

**Logic:**
- Detects total GPU VRAM
- Calculates safe batch size (uses ~80-85% VRAM)
- Auto-adjusts gradient accumulation to maintain effective batch=32

**Result:** No more OOM errors, optimal VRAM utilization!

### 2. Smart AMP dtype Selection

**bfloat16 (Ampere+):**
- Same range as FP32 (better stability)
- No gradient scaling overhead (faster)
- Native Tensor Core acceleration
- **Speedup:** 2-3× vs FP32

**float16 (Older GPUs):**
- Narrower range (can underflow/overflow)
- Requires gradient scaling (slight overhead)
- Still provides Tensor Core acceleration
- **Speedup:** 2-3× vs FP32

**float32 (CPU):**
- Full precision
- No acceleration

### 3. TF32 Auto-Enable

**What:** TensorFloat-32 precision for matrix operations
**Speedup:** 3-8× faster matmul (biggest bottleneck in training)
**Quality:** No accuracy loss vs FP32
**Auto-enabled on:** Ampere/Ada/Hopper GPUs

### 4. CPU Worker Auto-Tuning

**Training:**
- Uses 1× CPU core count (capped at 16)
- Ensures GPU never waits for data

**Preprocessing:**
- Uses 1-2× CPU core count (capped at 16)
- I/O-bound workload benefits from more workers

### 5. cuDNN Auto-Tuning

**What:** Automatically finds fastest convolution algorithms
**Benefit:** 5-20% speedup
**Trade-off:** 1-2 min startup time (one-time)

## Usage

### Training (Zero Configuration)

```bash
python trainers/train_gpt_v2.py \
  --train-manifest preprocessed/train_manifest.jsonl \
  --val-manifest preprocessed/val_manifest.jsonl \
  --tokenizer amharic_bpe.model
```

**That's it!** All settings auto-detected:
- ✅ Batch size
- ✅ Gradient accumulation
- ✅ Number of workers
- ✅ AMP dtype (bfloat16/float16/float32)
- ✅ TF32 (if supported)

### Manual Override (Optional)

You can override any auto-detected setting:

```bash
python trainers/train_gpt_v2.py \
  --batch-size 12 \         # Force specific batch size
  --grad-accumulation 3 \   # Force specific grad accum
  --num-workers 4 \         # Force specific workers
  --amp                     # Force enable AMP
```

**Note:** Use `0` for auto-detection:
```bash
--batch-size 0           # Auto-detect
--grad-accumulation 0    # Auto-detect
--num-workers 0          # Auto-detect
```

### Check Detected Settings

```bash
python -m indextts.utils.hardware_optimizer
```

**Output example (L4 GPU):**
```
======================================================================
[Hardware Detection & Auto-Optimization]
======================================================================

[GPU] NVIDIA L4
   VRAM: 24.0 GB
   AMP dtype: bfloat16
   TF32: Enabled (3-8x matmul speedup)

[CPU] 8 cores

[Optimized Training Settings]
   • Batch size: 8
   • Gradient accumulation: 4
   • Effective batch: 32
   • Data workers: 8
   • Mixed precision: Enabled (bfloat16)

[Recommendations]
   [OK] Detected: NVIDIA L4 (24.0GB VRAM)
   [PERF] Using bfloat16 AMP (native GPU support, better stability)
   [PERF] TF32 enabled (3-8x faster matmul)
   [TUNED] Medium GPU: batch=8, grad_accum=4 (effective=32)
   [PERF] Using 8 data workers (optimal for 8 CPUs)
   [OK] Detected: 8 CPU cores

======================================================================
```

## Performance Estimates

### Preprocessing (200hr Amharic dataset)

| Hardware | Time | Workers |
|----------|------|--------|
| A100 40GB | 3-6h | 16 |
| L4 24GB | 8-12h | 8 |
| V100 16GB | 8-16h | 8 |
| T4 16GB | 12-20h | 8 |
| CPU only | 24-48h+ | 4-8 |

### Training (200hr Amharic dataset)

| Hardware | Time | Batch | Grad Accum | AMP |
|----------|------|-------|------------|-----|
| A100 40GB | 1.5-2 days | 16 | 2 | bfloat16 |
| L4 24GB | 2-3 days | 8 | 4 | bfloat16 |
| V100 16GB | 3-4 days | 6 | 6 | float16 |
| T4 16GB | 4-6 days | 6 | 6 | float16 |
| RTX 3060 12GB | 5-7 days | 4 | 8 | float16 |

## Advanced: Gradient Checkpointing

**When to use:** GPU has <12GB VRAM and getting OOM errors

```bash
python trainers/train_gpt_v2.py \
  ... \
  --grad-checkpointing
```

**Effect:**
- Saves ~30% VRAM
- Allows 1.5-2× larger batch sizes
- **Cost:** 20-30% slower training

**Example:** With 8GB GPU:
- Without: batch=2, OOM at batch=3
- With: batch=4 possible, effective batch doubles

## Technical Details

### Optimization Hierarchy

1. **TF32** (Ampere+ GPUs)
   - Automatic 3-8× matmul speedup
   - No code changes needed
   - Zero quality loss

2. **bfloat16 AMP** (Ampere+ GPUs)
   - 2-3× overall training speedup
   - Better stability than float16
   - No gradient scaler overhead

3. **float16 AMP** (Older GPUs)
   - 2-3× overall training speedup
   - Requires gradient scaling
   - Slightly less stable than bfloat16

4. **cuDNN Benchmark**
   - 5-20% additional speedup
   - All CUDA GPUs

5. **Optimal Batch Sizing**
   - Maximizes GPU utilization
   - Prevents OOM errors

6. **Multi-threaded Data Loading**
   - Prevents GPU starvation
   - Scales with CPU count

### Why Auto-Optimization Matters

**Before (Manual):**
```bash
# User has to guess optimal settings
--batch-size 4   # Too small for L4? Too large for V100?
--num-workers 2  # Not using all 8 CPUs!
# No --amp      # Missing 2-3× speedup!
```

**After (Auto):**
```bash
# System detects hardware and optimizes automatically
# L4: batch=8, workers=8, bfloat16 AMP, TF32
# V100: batch=6, workers=8, float16 AMP
# A100: batch=16, workers=16, bfloat16 AMP, TF32
```

**Result:** 2-5× faster training without any configuration!

## Integration with WebUI

The Amharic WebUI automatically uses hardware detection for:
- Tab 5: Preprocessing (optimal workers)
- Tab 6: Training (optimal batch/workers/AMP)

Settings are pre-filled but editable in the UI.

## Troubleshooting

### Detection shows wrong GPU

**Check:**
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

If incorrect, check `CUDA_VISIBLE_DEVICES` environment variable.

### Want to force specific settings

Just specify the parameter - it overrides auto-detection:
```bash
--batch-size 4  # Forces batch=4 (ignores auto-detection)
```

### OOM errors despite auto-tuning

**Solutions:**
1. Enable gradient checkpointing: `--grad-checkpointing`
2. Reduce batch size: `--batch-size 4` (if auto-detected 8)
3. Close other GPU programs
4. Check VRAM usage: `nvidia-smi`

### Training slower than expected

**Checklist:**
1. ✅ Is `--amp` enabled? (should be auto-enabled)
2. ✅ Are workers > 0? (should be auto-set to CPU count)
3. ✅ Is GPU being used? Check `nvidia-smi` during training
4. ✅ Is TF32 enabled? Check startup logs

## Summary

**Key Benefits:**
1. 🚀 **Universal** - Works on any hardware (A100 to RTX 3050 to CPU)
2. ⚡ **Optimal** - Auto-tunes for peak performance
3. 🎯 **Safe** - No OOM errors from conservative VRAM estimates
4. 🔧 **Flexible** - Can override any setting
5. 📊 **Transparent** - Shows detected config at startup

**Before you start training, the system will print:**
```
[Hardware Detection & Auto-Optimization]
[GPU] NVIDIA L4 (24.0GB VRAM)
[PERF] Using bfloat16 AMP
[PERF] TF32 enabled (3-8x faster matmul)
[TUNED] Medium GPU: batch=8, grad_accum=4 (effective=32)
[PERF] Using 8 data workers
```

**You're ready to train with optimal settings!** 🎉
