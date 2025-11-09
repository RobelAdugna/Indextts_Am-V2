# IndexTTS2 Amharic - Quick Start Guide

This guide will help you set up everything needed to train your Amharic TTS model.

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)
- 20GB+ free disk space
- Stable internet connection for downloads

## Step 1: Download Required Model Checkpoints

**IMPORTANT:** You must download base model checkpoints before training.

### Option A: Automatic Setup (Easiest)

**Windows:**
```bash
double-click download_requirements.bat
```

**Linux/Mac:**
```bash
bash download_requirements.sh
```

This will:
1. Install `huggingface-hub` package
2. Download all 7 required checkpoint files (~3-5GB total)
3. Place them in `checkpoints/` directory

Download time: 10-30 minutes depending on connection speed.

### Option B: Manual Download

```bash
# Install dependency
pip install huggingface-hub

# Download checkpoints
python tools/download_checkpoints.py
```

### Verify Download

Check that these files exist in `checkpoints/`:
- ✓ `gpt.pth` (~500MB)
- ✓ `bpe.model`
- ✓ `s2mel.pth`
- ✓ `wav2vec2bert_stats.pt`
- ✓ `feat1.pt`
- ✓ `feat2.pt`
- ✓ `config.yaml`

## Step 2: Prepare Your Dataset

You mentioned you already have `amharic_dataset/manifest.jsonl` with 0.1k segments. Perfect! ✓

## Step 3: Choose Your Training Method

### Option A: Web Interface (Recommended for Beginners)

```bash
python webui_amharic.py
```

Then:
1. Open http://localhost:7863 in your browser
2. Go to **Tab 3** (Corpus Collection)
3. Enter manifest path: `amharic_dataset/manifest.jsonl`
4. Click through tabs 3→4→5→6 in order
5. Monitor training progress in Tab 6

### Option B: Automated Script (Advanced)

**Windows:**
```powershell
.\scripts\amharic\end_to_end.ps1
```

**Linux/Mac:**
```bash
bash scripts/amharic/end_to_end.sh
```

This runs all steps automatically:
- Corpus collection
- Tokenizer training
- Data preprocessing  
- Model training

## Step 4: Monitor Training

### During Training:
- Watch console for progress updates
- Check GPU usage: `nvidia-smi`
- Training time: 2-6 hours for 0.1k dataset

### If Training Fails:
- **Out of Memory?** Reduce batch size in Tab 6 (try 2 or 4)
- **Checkpoint errors?** Re-run `python tools/download_checkpoints.py`
- **Slow?** Enable AMP (mixed precision) checkbox

## Step 5: Test Your Model

After training completes:
1. Find your model in `checkpoints/amharic/checkpoint_XXXX.pt`
2. Go to **Tab 8** (Inference) in WebUI
3. Load your checkpoint
4. Test with Amharic text
5. Listen to results!

## Troubleshooting

### Download Issues

**Error: "huggingface-hub not installed"**
```bash
pip install huggingface-hub
```

**Error: "Failed to download"**
- Check internet connection
- Try manual download from: https://huggingface.co/IndexTeam/IndexTTS-2
- Resume interrupted download with `python tools/download_checkpoints.py`

### Training Issues

**CUDA Out of Memory**
- Reduce batch size: Tab 6 → change to 2 or 4
- Enable AMP: Tab 6 → check "Enable AMP" box
- Close other GPU programs

**"No such file: gpt.pth"**
```bash
# Re-download checkpoints
python tools/download_checkpoints.py --force
```

**Training too slow**
- Enable AMP (Tab 6)
- Increase batch size if you have VRAM
- Check GPU is actually being used: `nvidia-smi`

## Expected Results

### With 0.1k Dataset:
- Training time: 2-6 hours
- Quality: Basic but functional
- Pronunciation: Decent for common words
- Prosody: Limited naturalness

### To Improve Quality:
- Collect more data (aim for 1k+ segments)
- Use longer training (increase max steps)
- Fine-tune hyperparameters

## Next Steps

1. **Test Your Model:**
   - Use Tab 8 for quick tests
   - Compare with base model
   - Note what sounds good/bad

2. **Improve Dataset:**
   - Download more Amharic content (Tab 1)
   - Create more segments (Tab 2)
   - Re-train with larger dataset

3. **Share Your Results:**
   - Report issues on GitHub
   - Share success stories
   - Help improve Amharic support

## Getting Help

- **WebUI Issues:** Check console for error messages
- **Training Errors:** Look in `checkpoints/amharic/logs/`
- **Dataset Problems:** Review `manifest.jsonl` format
- **General Questions:** See main README.md

## Quick Reference

```bash
# Download checkpoints
python tools/download_checkpoints.py

# Launch WebUI
python webui_amharic.py

# Run full pipeline (advanced)
bash scripts/amharic/end_to_end.sh

# Check GPU
nvidia-smi
```

---

**Ready to start?** Run `download_requirements.bat` (Windows) or `bash download_requirements.sh` (Linux/Mac) now!
