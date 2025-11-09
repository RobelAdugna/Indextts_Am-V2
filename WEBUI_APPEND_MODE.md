# WebUI: Incremental Dataset Expansion

## Quick Start

Add more data to your existing dataset without re-processing everything!

### Step-by-Step Guide

1. **Download New Content (Tab 1)**
   - Enter new YouTube URLs or use different source
   - Set output to **new folder** (e.g., `new_downloads_batch2`)
   - Click "Download Content"

2. **Create Additional Segments (Tab 2)**
   - **Input Directory:** Point to new downloads (e.g., `new_downloads_batch2`)
   - **Output Directory:** Point to **existing dataset** (e.g., `amharic_dataset`)
   - **Single Speaker Mode:** Use **same setting** as original dataset
   - ✅ **CHECK:** "📝 Append to Existing Dataset"
   - Click "Create Dataset"

3. **Review Results**
   - Status will show:
     - Previous entries: 3455 (example)
     - New entries: 801 (example)
     - Total entries now: 4256
   - All new files start from `spk000_003456.wav` onwards

## What Happens Behind the Scenes

### Auto-Detection
- Reads existing `manifest.jsonl`
- Finds last segment ID (e.g., `spk000_003455`)
- Calculates next number: `003456`

### Smart Numbering
- **Single Speaker:** Continues spk000_XXXXXX numbering
- **Multi-Speaker:** Increments speaker ID and continues global numbering

### Safe Append
- Opens manifest in append mode (doesn't overwrite)
- Adds new entries to end of file
- Never touches existing audio files
- Preserves all existing data

## Example Workflow

### Initial Dataset (Week 1)
```
Input: downloads_week1/ (5 videos)
Output: amharic_dataset/
Result: spk000_000001.wav → spk000_003455.wav (3455 segments)
```

### Expansion 1 (Week 2)
```
Input: downloads_week2/ (3 videos)
Output: amharic_dataset/ (SAME)
✅ Append: CHECKED
Result: spk000_003456.wav → spk000_005123.wav (+ 1668 segments)
Total: 5123 segments
```

### Expansion 2 (Week 3)
```
Input: downloads_week3/ (4 videos)
Output: amharic_dataset/ (SAME)
✅ Append: CHECKED
Result: spk000_005124.wav → spk000_007234.wav (+ 2111 segments)
Total: 7234 segments
```

## Important Rules

### ✅ DO:
- Use **same Single Speaker setting** as original
- Point output to **existing dataset directory**
- Put new files in **separate input directory**
- Check the append checkbox!

### ❌ DON'T:
- Mix single-speaker and multi-speaker modes
- Process same files twice
- Change output directory between batches
- Forget to check the append box (it will overwrite!)

## Troubleshooting

### Numbers Don't Continue?
**Check:**
- Is append checkbox checked?
- Is output directory correct?
- Does manifest.jsonl exist in output dir?

### "Dataset Created" But Files Missing?
**Likely causes:**
- Input directory empty or wrong
- Quality filters too strict
- Audio/subtitle pairs not found

### Want to Start Fresh Instead?
**Solution:**
- Uncheck append mode
- Use new output directory
- This creates completely new dataset

## Visual Indicator

When append mode is active, the status will show:

```
🔄 APPEND MODE: Continuing from existing dataset...
  📊 Existing dataset info:
     - Total entries: 3455
     - Last segment: spk000_003455
     - Next segment will be: spk000_003456
  ✓ New segments will be appended to existing manifest
```

## After Appending

### Continue Training
Your expanded dataset is ready:
- No need to re-preprocess (unless you want to)
- Can continue training from last checkpoint
- Or start fresh training with larger dataset

### Verify Dataset
Check your expansion:
```bash
# Count total files
dir amharic_dataset\audio\*.wav | find /c ".wav"

# Count manifest entries  
find /c /v "" amharic_dataset\manifest.jsonl

# View last entries
type amharic_dataset\manifest.jsonl | more
```

## FAQ

**Q: Can I append multiple times?**  
A: Yes! Append as many batches as you want.

**Q: Will it affect existing training?**  
A: No. Existing checkpoints remain valid. You can continue or restart.

**Q: What if I forget to check append?**  
A: It will overwrite! Always double-check the checkbox.

**Q: Can I append to multi-speaker dataset?**  
A: Yes! Just use multi-speaker mode (uncheck single-speaker) on append too.

**Q: How do I know which files are new?**  
A: Look at segment numbers. Higher numbers = newer additions.

**Q: Can I delete old segments after appending?**  
A: Yes, but you'd need to update the manifest manually. Not recommended.

## Best Practices

1. **Organize by Batch**
   - `downloads_batch1/`, `downloads_batch2/`, etc.
   - Makes tracking easier

2. **Consistent Quality**
   - Use same quality settings for all batches
   - Ensures dataset uniformity

3. **Backup First**
   - Copy manifest before large appends
   - `copy amharic_dataset\manifest.jsonl amharic_dataset\manifest.backup.jsonl`

4. **Monitor Growth**
   - Track dataset size over time
   - Aim for 1k+ segments for good results
   - 5k+ segments for excellent results

## Summary

**Append mode** lets you grow your dataset incrementally:
- ✅ Automatic number continuation
- ✅ Safe append (no overwriting)
- ✅ Easy to use (just one checkbox)
- ✅ Works for single or multi-speaker
- ✅ Perfect for continuous improvement

**Remember:** Always check that append checkbox! 📝
