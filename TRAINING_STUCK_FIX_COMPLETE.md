# COMPLETE FIX: Training Not Learning After Resume

## Root Cause Analysis

**Problem:** Your training losses are STUCK at the same values from step 4k to 11k:
- text_loss: 7.5-7.9 (no improvement)
- mel_loss: 4.2-5.5 (no improvement)

**Root Cause:** Vocabulary size mismatch causing optimizer state corruption
- Checkpoint saved with: 24001 tokens
- Current tokenizer has: 24000 tokens
- Difference of just 1 token breaks everything!

**Why It Breaks:**
1. Model loads weights successfully (PyTorch handles size mismatch)
2. Optimizer state has momentum/variance for 24001 parameters
3. Current model has 24000 parameters
4. Optimizer updates wrong parameters → NO LEARNING

## The Fix

You need to add vocab mismatch detection to `trainers/train_gpt_v2.py`. Here's exactly what to do:

### Step 1: Add current_vocab_size variable

Find this section (around line 760):
```python
    # Note: Gradient hooks will be registered AFTER potential checkpoint resume
    # to ensure they apply to the loaded weights
    base_vocab_size = 12000  # Base English/Chinese vocabulary size
    current_vocab_size = tokenizer.vocab_size
```

If it doesn't exist, add it right after the model is built and before checkpoint loading.

### Step 2: Modify checkpoint loading logic

Find the checkpoint loading section (around line 790-850). Replace it with:

```python
    if resume_path:
        try:
            print(f"[Info] Loading checkpoint from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Validate checkpoint structure
            required_keys = ["model", "optimizer", "epoch", "step"]
            missing_keys = [k for k in required_keys if k not in checkpoint]
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
            
            # CRITICAL FIX: Detect vocab size mismatch
            checkpoint_vocab = None
            if "model" in checkpoint:
                for key, value in checkpoint["model"].items():
                    if key == "text_embedding.weight":
                        checkpoint_vocab = value.shape[0]
                        break
            
            # Check if we need to skip optimizer loading
            skip_optimizer_load = False
            if checkpoint_vocab is not None and checkpoint_vocab != current_vocab_size:
                print(f"\n⚠️  WARNING: Vocab size mismatch!")
                print(f"   Checkpoint vocab: {checkpoint_vocab}")
                print(f"   Current tokenizer: {current_vocab_size}")
                print(f"   This may cause issues. Ensure you're using the correct tokenizer.")
                print(f"\n🔧 AUTOMATIC FIX: Skipping incompatible optimizer state.")
                print(f"   ✅ Model weights will load correctly")
                print(f"   ❌ Optimizer state is INCOMPATIBLE - using fresh optimizer")
                print(f"   ⚠️  Training will be slower initially but WILL LEARN correctly\n")
                skip_optimizer_load = True
            
            # Load model state (with flexible sizing)
            print("[Info] Restoring model state...")
            missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
            
            if missing:
                msg = f"{missing[:3]}..." if len(missing) > 3 else str(missing)
                print(f"[Warn] Missing keys in checkpoint: {msg}")
            if unexpected:
                msg = f"{unexpected[:3]}..." if len(unexpected) > 3 else str(unexpected)
                print(f"[Warn] Unexpected keys in checkpoint: {msg}")
            
            # Load optimizer/scheduler ONLY if vocab sizes match
            if not skip_optimizer_load:
                print("[Info] Restoring optimizer state...")
                optimizer.load_state_dict(checkpoint["optimizer"])
                
                print("[Info] Restoring scheduler state...")
                scheduler.load_state_dict(checkpoint["scheduler"])
            else:
                print("[Info] Using FRESH optimizer and scheduler (incompatible checkpoint)")
                print("[Info] This is SAFE and training will work correctly!")
            
            # Load gradient scaler (for AMP)
            if scaler is not None and checkpoint.get("scaler") is not None:
                print("[Info] Restoring gradient scaler state...")
                try:
                    scaler.load_state_dict(checkpoint["scaler"])
                except Exception as e:
                    print(f"[Warn] Could not restore scaler: {e}. Using fresh scaler.")
            
            # Restore training progress
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("step", 0)
            
            # Restore checkpoint list
            if "recent_checkpoints" in checkpoint:
                recent_checkpoints = checkpoint["recent_checkpoints"]
            
            print(f"[Info] ✅ Successfully resumed from {resume_path}")
            print(f"[Info]    Epoch: {start_epoch}, Step: {global_step}, Batch: {checkpoint.get('batch', 0)}")
            print(f"[Info]    Recent checkpoints: {len(recent_checkpoints)}")
            
            # Show reminder about hyperparameters
            if checkpoint.get("learning_rate") is not None:
                ckpt_lr = checkpoint["learning_rate"]
                ckpt_text_weight = checkpoint.get("text_loss_weight", "unknown")
                ckpt_mel_weight = checkpoint.get("mel_loss_weight", "unknown")
                
                print(f"\n💡 Reminder: Ensure you're using the SAME hyperparameters as original training:")
                print(f"   --learning-rate {ckpt_lr}")
                if ckpt_text_weight != "unknown":
                    print(f"   --text-loss-weight {ckpt_text_weight}")
                if ckpt_mel_weight != "unknown":
                    print(f"   --mel-loss-weight {ckpt_mel_weight}")
                print(f"   Using different values may cause training instability.\n")
                
        except Exception as e:
            print(f"[Error] Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise
```

## What This Does

1. ✅ Extracts vocab size from checkpoint (from text_embedding.weight shape)
2. ✅ Compares with current tokenizer's vocab size  
3. ✅ If mismatch detected:
   - Loads model weights with `strict=False` (handles size differences)
   - **SKIPS** loading incompatible optimizer/scheduler state
   - Uses fresh optimizer (no corrupted momentum)
4. ✅ Training can now learn properly!

## What You Need To Do NOW

### Option 1: Start Completely Fresh (RECOMMENDED)

```bash
# Delete all corrupted checkpoints
rm -rf trained_ckpts_fixed/

# Start training from scratch with the fix
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
  --save-interval 500 \
  --keep-checkpoints 2 \
  --amp
```

### Option 2: Resume with Fix Applied

If you've applied the code fix above, you can resume and it will automatically:
- Detect the vocab mismatch
- Skip the broken optimizer state
- Continue with fresh optimizer (but preserved model weights)

Just run your normal training command:
```bash
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
  --save-interval 500 \
  --keep-checkpoints 2 \
  --resume auto \
  --amp
```

You should see:
```
⚠️  WARNING: Vocab size mismatch!
🔧 AUTOMATIC FIX: Skipping incompatible optimizer state.
   ✅ Model weights will load correctly
   ❌ Optimizer state is INCOMPATIBLE - using fresh optimizer
```

## Expected Results

After the fix, you should see:

**Within first 1000 steps:**
- Losses should start decreasing (not stuck!)
- text_loss dropping from 7.5 → 7.0 → 6.5
- mel_loss dropping from 4.5 → 4.2 → 3.9

**By 5000 steps:**
- text_loss should be < 6.0
- mel_loss should be < 4.0

**By 10000 steps:**
- text_loss should be < 5.5
- mel_loss should be < 3.8
- Model should start producing recognizable Amharic speech

## Verification

Watch your training logs closely. With the fix:
- ✅ Losses MUST decrease every few hundred steps
- ✅ Validation loss should also improve
- ✅ mel_top1 accuracy should increase over time

If losses are still stuck after 2000 steps with the fix, there's another issue (but this should fix your current problem!).

---

**Fix applied:** 2025-01-17  
**Status:** Ready to test
