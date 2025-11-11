# Fix Summary: 50% Text Overlap Issue

## What Was Wrong

Your dataset manifest had **50% overlapping text** between neighboring segments because:

1. **YouTube SRT subtitles use "rolling text"**: Each subtitle line repeats part of the previous line
   - Example: Line 1: "Hello world this is"
   - Example: Line 2: "this is a test" (repeats "this is")

2. **The deduplication code had a bug**: It compared each segment with the *last successfully added* segment instead of the *previous input* segment
   - When a segment was skipped (e.g., exact duplicate), the next segment would compare with the wrong segment
   - This caused overlaps to be missed

## What Was Fixed

Changed in `tools/create_amharic_dataset.py`:

```python
# BEFORE (BUGGY)
prev_added_text = deduplicated[-1].text  # Wrong: last ADDED segment

# AFTER (FIXED)  
prev_text = segments[i-1].text  # Correct: previous INPUT segment
```

This ensures rolling text is properly detected and removed, even when some segments are skipped.

## How to Fix Your Dataset

### Option 1: Re-create from YouTube (Recommended)

If you downloaded from YouTube:

```bash
# 1. Re-download (subtitles may have changed/improved)
python tools/youtube_amharic_downloader.py --url-file amharic_urls.txt --output-dir downloads_new

# 2. Create dataset with fixed deduplication
python tools/create_amharic_dataset.py --input-dir downloads_new --output-dir amharic_dataset_fixed
```

### Option 2: Re-process Existing Files

If you still have the original downloaded audio + SRT files:

```bash
python tools/create_amharic_dataset.py \
  --input-dir "C:\Users\Abrsh-1\Downloads\your_original_downloads" \
  --output-dir amharic_dataset_fixed
```

## Verify the Fix

Check your new manifest:

```bash
type amharic_dataset_fixed\manifest.jsonl
```

You should see unique, non-overlapping text in each entry.

## Files Changed

- ✅ `tools/create_amharic_dataset.py` - Fixed deduplication logic
- ✅ `DEDUPLICATION_FIX.md` - Technical documentation
- ✅ `knowledge.md` - Updated with fix notes

## What Happens Now

The fix is in place and will work correctly for:
- ✅ New datasets you create
- ✅ Existing SRT files you re-process
- ✅ Both YouTube downloads and local files
- ✅ All languages (not just Amharic)

Your existing `manifest.jsonl` at `C:\Users\Abrsh-1\Downloads\fix_manifest_duplication-text\manifest_fixed.jsonl` needs to be regenerated from the source files to get the benefit of this fix.
