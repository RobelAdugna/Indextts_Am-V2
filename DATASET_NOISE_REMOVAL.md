# Dataset Segment Noise Removal - Quick Reference

## Overview

Post-process existing dataset audio segments to remove background music/noise while maintaining all filenames and manifest structure.

## Use Cases

- ✅ Clean up existing datasets without re-segmenting
- ✅ Apply noise removal after dataset creation  
- ✅ Improve audio quality of legacy datasets
- ✅ Process datasets created before noise removal was available

## Key Features

### In-Place Processing
- Replaces original files with noise-removed versions
- Maintains exact filenames (e.g., `spk000_000001.wav` → `spk000_000001.wav`)
- Manifest.jsonl remains valid
- No need to regenerate features or retrain

### Resume Capability
- Saves progress every N files (default: 10)
- Can interrupt (Ctrl+C) and resume later
- Progress tracked in `.noise_removal_progress.json`
- Skips already-processed files automatically

### Safety Options
- Optional backup of original files (`.backup` extension)
- Validates all segments processed successfully
- Detailed error reporting for failed segments

## Quick Start

### WebUI (Easiest)

1. Launch WebUI: `python webui_amharic.py`
2. Go to Tab 7 "Process Segments"
3. Enter manifest path or audio directory
4. Select noise removal model (UVR-MDX-NET recommended)
5. Click "Process Dataset Segments"

### CLI

```bash
# Process from manifest (recommended)
python tools/process_dataset_segments.py \
  --manifest amharic_dataset/manifest.jsonl \
  --keep-backup

# Process specific audio directory
python tools/process_dataset_segments.py \
  --audio-dir amharic_dataset/audio
```

## Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| **UVR-MDX-NET-Inst_HQ_3** | ⚡⚡⚡ | ⭐⭐⭐⭐ | **Recommended** - Best balance |
| UVR_MDXNET_KARA_2 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Alternative MDX variant |
| htdemucs | ⚡ | ⭐⭐⭐⭐⭐ | Best quality, slowest |

## CLI Options

```bash
python tools/process_dataset_segments.py --help

Input:
  --manifest PATH          Process from manifest.jsonl (recommended)
  --audio-dir PATH         Process audio directory directly

Options:
  --model MODEL            Noise removal model (default: UVR-MDX-NET-Inst_HQ_3)
  --keep-backup           Save originals as .backup files
  --no-resume             Start fresh (ignore previous progress)
  --batch-size N          Save progress every N files (default: 10)
```

## Examples

### Basic Usage

```bash
# Process entire dataset with defaults
python tools/process_dataset_segments.py \
  --manifest amharic_dataset/manifest.jsonl
```

### With Backup

```bash
# Keep backup copies of originals
python tools/process_dataset_segments.py \
  --manifest amharic_dataset/manifest.jsonl \
  --keep-backup
```

### Custom Model

```bash
# Use highest quality model (slower)
python tools/process_dataset_segments.py \
  --manifest amharic_dataset/manifest.jsonl \
  --model htdemucs
```

### Resume After Interruption

```bash
# Automatically resumes from last checkpoint
python tools/process_dataset_segments.py \
  --manifest amharic_dataset/manifest.jsonl
```

### Start Fresh

```bash
# Ignore previous progress
python tools/process_dataset_segments.py \
  --manifest amharic_dataset/manifest.jsonl \
  --no-resume
```

## Workflow

### Standard Pipeline

1. **Create Dataset** (Tabs 1-2)
   - Download YouTube content
   - Segment with subtitles
   - Generate manifest.jsonl

2. **Initial Training** (Optional)
   - Train model to assess quality
   - Identify background noise issues

3. **Process Segments** (Tab 7) ⭐ **NEW**
   - Remove noise from all segments
   - Filenames preserved
   - Manifest stays valid

4. **Continue Training**
   - No need to re-preprocess
   - Resume or start new training

### Improving Existing Dataset

```bash
# 1. Check current dataset
ls amharic_dataset/audio/ | wc -l

# 2. Process segments
python tools/process_dataset_segments.py \
  --manifest amharic_dataset/manifest.jsonl \
  --keep-backup

# 3. Verify results
# Listen to a few processed files

# 4. If satisfied, remove backups to save space
find amharic_dataset/audio -name "*.backup" -delete
```

## GPU Acceleration

### Automatic Detection

The tool automatically detects and uses GPU if available:

```bash
# Check for CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Run processing (automatically uses GPU)
python tools/process_dataset_segments.py --manifest dataset/manifest.jsonl
```

### Performance

| Hardware | Speed (per segment) | Dataset (1000 segments) |
|----------|---------------------|-------------------------|
| CPU only | ~15-30 sec | ~4-8 hours |
| GPU (CUDA) | ~2-5 sec | ~30-90 minutes |

## Troubleshooting

### Progress Not Saved

**Problem:** Process interrupted but resume starts from beginning

**Solution:** Check for `.noise_removal_progress.json` in dataset directory:
```bash
ls amharic_dataset/.noise_removal_progress.json
cat amharic_dataset/.noise_removal_progress.json
```

### Out of Memory

**Problem:** GPU out of memory during processing

**Solution:** Process runs one file at a time, so this is rare. If it occurs:
- Close other GPU-using applications
- Restart the process (it will resume)

### Low Quality Output

**Problem:** Processed audio sounds worse

**Solution:** Try different model:
```bash
# Use highest quality model
python tools/process_dataset_segments.py \
  --manifest dataset/manifest.jsonl \
  --model htdemucs
```

### Files Not Found

**Problem:** "Audio directory not found" error

**Solution:** Ensure manifest points to correct location:
```bash
# Check manifest structure
head -1 amharic_dataset/manifest.jsonl

# Verify audio directory exists
ls amharic_dataset/audio/
```

## Important Notes

⚠️ **Backup First**: Use `--keep-backup` for first run to preserve originals

⚠️ **GPU Recommended**: CPU processing is 5-10x slower

⚠️ **Disk Space**: Processing requires minimal extra space (temp files cleaned automatically)

⚠️ **Manifest Unchanged**: Original manifest.jsonl is not modified (only audio files)

✅ **Safe Interruption**: Can stop anytime with Ctrl+C and resume later

✅ **Idempotent**: Safe to run multiple times (already-processed files skipped)

## Integration with Training

Processed segments can be used immediately:

```bash
# 1. Process segments
python tools/process_dataset_segments.py --manifest dataset/manifest.jsonl

# 2. NO need to re-preprocess (features unchanged)
# Just continue with training

python trainers/train_gpt_v2.py \
  --data-dir dataset \
  --out-dir training_output
```

## Performance Tips

1. **Use GPU**: 5-10x faster than CPU
2. **Batch Size**: Increase `--batch-size` for less frequent progress saves
3. **Resume**: Always enable (default) to recover from interruptions
4. **Model Choice**: Use MDX-NET for speed, Demucs for quality

## Support

For issues or questions:
- Check `knowledge.md` for additional documentation
- Review WebUI Tab 7 for guided interface
- See `tools/process_dataset_segments.py` for implementation details
