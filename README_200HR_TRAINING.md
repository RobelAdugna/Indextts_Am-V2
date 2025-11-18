# 200-Hour Amharic Dataset Training Guide

## 🎯 Quick Start (3 Steps)

### Step 1: Verify Your Setup

```bash
# Check you have these files:
processed/
  ├── GPT_pairs_train.jsonl  ✅
  ├── GPT_pairs_val.jsonl    ✅
tokenizers/
  └── amharic_extended_bpe.model ✅
```

### Step 2: Start Training

**Linux/Mac:**
```bash
bash train_amharic_200hr.sh
```

**Windows:**
```bash
train_amharic_200hr.bat
```

### Step 3: Monitor Progress

```bash
uv run tensorboard --logdir trained_ckpts
# Open http://localhost:6006
```

**Watch for:** train/val loss gap staying <0.3 ✅

---

## 📚 Documentation Files

| File | Purpose | When to Use |
|------|---------|-------------|
| **QUICK_REF_200HR.md** | Quick reference card | During training monitoring |
| **TRAINING_200HR_OPTIMIZATIONS.md** | Complete guide | Before starting, troubleshooting |
| **train_amharic_200hr.sh** | Linux/Mac script | To start training |
| **train_amharic_200hr.bat** | Windows script | To start training |
| **This file** | Overview | Right now! |

---

## 🎓 Key Concepts for 200 Hours

### Why 200 Hours is Special

200 hours is a **medium-scale dataset** - the sweet spot where:
- ✅ You have enough data for high quality
- ⚠️ Overfitting is a real concern
- 🎯 Careful tuning makes the difference

### Core Strategy

```
Quality > Quantity
Validation > Training data
Early Stopping > Training forever
Monitoring > Hoping
```

---

## 🔧 Optimizations Applied

### 1. Conservative Hyperparameters
- **Learning Rate:** 5e-5 (not 2e-5)
- **Epochs:** 2-3 (not 10)
- **Weight Decay:** 1e-5 (L2 regularization)

### 2. Frequent Validation
- **Interval:** Every 500 steps (not 1000)
- **Purpose:** Catch overfitting early
- **Cost:** ~2-3 min per check (worth it!)

### 3. Best Checkpoint Selection
- **Keep:** Top 5 checkpoints
- **Criteria:** Lowest val_loss + gap <0.3
- **Benefit:** Can ensemble top 3 if needed

### 4. Early Stopping
- **Signal:** Val loss plateau >10k steps
- **Target:** Stop at 60k-70k steps
- **Reason:** 200hr → diminishing returns after this

---

## 📊 Expected Results

### Timeline
| GPU | Duration |
|-----|----------|
| L4 24GB | 6-9 days |
| A100 80GB | 2-3 days |
| V100 16GB | 9-12 days |

### Metrics at 60k Steps (Target)
```
train_loss:  0.8-1.0
val_loss:    1.0-1.2
gap:         0.2-0.3  ✅ HEALTHY
mel_top1:    0.75-0.80
```

### Warning Signs
```
train_loss:  0.5
val_loss:    1.5
gap:         1.0  ❌ OVERFITTING!
→ Use earlier checkpoint (40k-50k steps)
```

---

## 🚨 Common Mistakes to Avoid

| ❌ Wrong | ✅ Right | Impact |
|---------|---------|--------|
| Train 10 epochs | Train 2-3 epochs | Prevents overfitting |
| LR 2e-5 | LR 5e-5 | Better for extended vocab |
| Val every 1000 | Val every 500 | Catches issues early |
| Keep 2-3 ckpts | Keep 5 ckpts | Better selection |
| Ignore val loss | Monitor closely | Prevent wasted training |

---

## 🛠️ Troubleshooting

### Problem: Losses not decreasing
**Check:**
- Is gradient hook applied? (Look for "Extended Vocab Fix" message)
- Is warmup complete? (Wait 4k steps)
- Is data quality good? (Check preprocessing)

### Problem: Val loss increasing
**Action:**
- **Immediate:** Note the step number
- **Next:** Continue for 5k more steps to confirm
- **Then:** If confirmed, stop and use checkpoint from before increase

### Problem: Training too slow
**Options:**
- Reduce val_interval to 1000 (saves ~30 min/day)
- Increase batch size if GPU allows
- Use A100 instead of L4 (3-4× faster)

### Problem: OOM errors
**Fix:**
- Reduce batch_size by 50%
- Increase grad_accumulation by 2×
- Effective batch stays same, uses less VRAM

---

## 🎯 Success Criteria

You've succeeded when:

1. ✅ Training converges (losses decreasing)
2. ✅ Validation gap stays <0.3
3. ✅ Mel_top1 reaches 0.75+
4. ✅ Generated speech sounds natural
5. ✅ No overfitting (gap <0.5)

---

## 📖 Further Reading

- **Full details:** TRAINING_200HR_OPTIMIZATIONS.md
- **Quick ref:** QUICK_REF_200HR.md
- **Video guide:** VIDEO_TRAINING_GUIDE.md
- **General knowledge:** knowledge.md

---

## 🤝 Support

If you encounter issues:
1. Check TensorBoard graphs
2. Review TRAINING_200HR_OPTIMIZATIONS.md
3. Compare your metrics to expected timeline
4. Verify validation split is 3% (~6 hours)

---

## 🎉 Final Words

**Your 200-hour Amharic dataset is perfectly sized for quality TTS!**

With the optimizations in these scripts:
- ✅ Overfitting is under control
- ✅ Training will converge efficiently
- ✅ You'll get high-quality results
- ✅ All working together tightly

**Just run the script and monitor TensorBoard. You've got this!** 🚀
