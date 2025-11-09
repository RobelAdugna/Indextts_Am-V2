# Incremental Dataset Expansion Guide

## Problem

You have an existing dataset (e.g., 3455 segments) and want to add more data WITHOUT:
- Re-processing existing files
- Breaking the numbering sequence
- Overwriting the manifest

## Solution: Append Mode

Use the `--append` flag to automatically continue from where you left off.

## How It Works

### Step 1: Create Initial Dataset

```bash
python tools/create_amharic_dataset.py \
  --input-dir downloads_batch1 \
  --output-dir amharic_dataset \
  --single-speaker
```

**Result:**
- Files: `spk000_000001.wav` through `spk000_003455.wav`
- Manifest: `amharic_dataset/manifest.jsonl` (3455 entries)

### Step 2: Add More Data Later

**Download new content:**
```bash
python tools/youtube_amharic_downloader.py \
  --url-file new_urls.txt \
  --output-dir downloads_batch2
```

**Process with append mode:**
```bash
python tools/create_amharic_dataset.py \
  --input-dir downloads_batch2 \
  --output-dir amharic_dataset \
  --append \
  --single-speaker
```

**Result:**
- New files: `spk000_003456.wav` through `spk000_005234.wav` (example)
- Manifest: Appended 1779 new entries (now 5234 total)
- Existing files: Completely untouched ✓

### Step 3: Repeat As Needed

You can keep adding more batches:

```bash
# Batch 3
python tools/create_amharic_dataset.py \
  --input-dir downloads_batch3 \
  --output-dir amharic_dataset \
  --append \
  --single-speaker

# Batch 4, 5, etc...
```

Each time:
- ✅ Automatically detects last segment number
- ✅ Continues from next number
- ✅ Appends to manifest
- ✅ Never overwrites existing data

## Key Points

### ✅ DO:
- Use `--append` flag for incremental expansion
- Use same `--single-speaker` mode as original
- Use same `--output-dir` as existing dataset
- Download new content to separate directory first

### ❌ DON'T:
- Mix single-speaker and multi-speaker modes
- Point `--input-dir` to folder with already-processed files
- Change `--output-dir` (must match existing dataset location)

## Automatic Detection

The script automatically:

1. **Reads existing manifest** to find:
   - Last segment number (e.g., 003455)
   - Last speaker ID (e.g., spk000)
   - Total existing entries

2. **Continues numbering:**
   - Next segment: 003456
   - Next speaker: spk001 (only in multi-speaker mode)

3. **Appends to manifest:**
   - Opens manifest in append mode
   - Adds new entries at end
   - Preserves all existing entries

## Example Output

```
🔄 APPEND MODE: Continuing from existing dataset...
  📊 Existing dataset info:
     - Total entries: 3455
     - Last segment: spk000_003455
     - Next segment will be: spk000_003456
  ✓ New segments will be appended to existing manifest

Processing files in: downloads_batch2
Quality filtering: enabled
Speaker mode: single
Text deduplication: enabled

Processing: video1.wav + video1.srt [Speaker 000]
  Generated 234 segments (accepted: 234, rejected: 12)

Processing: video2.wav + video2.srt [Speaker 000]  
  Generated 567 segments (accepted: 567, rejected: 34)

📝 Appending 801 new entries to manifest: amharic_dataset/manifest.jsonl

✓ Dataset updated successfully!
  Files processed: 2
  Files failed: 0
  Total segments checked: 847
  Accepted segments: 801
  Rejected segments: 46

  📊 Dataset totals:
     - Previous entries: 3455
     - New entries: 801
     - Total entries now: 4256
```

## Multi-Speaker Mode

For multi-speaker datasets (different speakers from different sources):

**Initial:**
```bash
python tools/create_amharic_dataset.py \
  --input-dir batch1 \
  --output-dir dataset
# Creates: spk000_*, spk001_*, spk002_*
```

**Append:**
```bash
python tools/create_amharic_dataset.py \
  --input-dir batch2 \
  --output-dir dataset \
  --append
# Creates: spk003_*, spk004_*, spk005_*
# Continues speaker IDs from last one (spk002 → spk003)
```

## Verification

**Check last segment:**
```bash
# Windows
dir amharic_dataset\audio | findstr /E ".wav" | more

# Linux/Mac
ls -1 amharic_dataset/audio/*.wav | tail -5
```

**Check manifest entries:**
```bash
# Count total lines
find /c /v "" amharic_dataset\manifest.jsonl  # Windows
wc -l amharic_dataset/manifest.jsonl  # Linux/Mac

# View last entry
tail -1 amharic_dataset/manifest.jsonl  # Linux/Mac
```

## Troubleshooting

**Numbering doesn't continue correctly?**
- Check you're using `--append` flag
- Verify manifest path is correct
- Check existing manifest format (should have `id` field)

**Getting duplicates?**
- Don't use `--append` if you want to start fresh
- Make sure `--input-dir` points to NEW files only
- Check you're not processing same files twice

**Speaker IDs wrong?**
- Must use same `--single-speaker` mode as original
- Single speaker: always spk000
- Multi-speaker: increments with each new source file

## Best Practices

1. **Organize by batch:**
   - `downloads_batch1/`, `downloads_batch2/`, etc.
   - Makes it easy to track what was processed

2. **Always use append for expansion:**
   - Never re-process entire dataset
   - Only process new files

3. **Backup before major additions:**
   ```bash
   cp amharic_dataset/manifest.jsonl amharic_dataset/manifest.backup.jsonl
   ```

4. **Verify counts:**
   - Check file count matches manifest entries
   - Review last few segments

## CLI Quick Reference

**Create new dataset:**
```bash
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir amharic_dataset \
  --single-speaker
```

**Append to existing:**
```bash
python tools/create_amharic_dataset.py \
  --input-dir new_downloads \
  --output-dir amharic_dataset \
  --append \
  --single-speaker
```

**Check what would be added (dry run not available, but can check input):**
```bash
dir new_downloads\*.wav  # Windows
ls new_downloads/*.wav  # Linux/Mac
```
