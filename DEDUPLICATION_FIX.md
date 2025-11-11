# Subtitle Text Deduplication Fix

## Problem

The dataset creator and YouTube downloader were producing manifest files with **50% overlapping text** between neighboring segments. This happened because SRT subtitle files commonly use "rolling text" where each line repeats part of the previous line for viewer comprehension.

Example of rolling subtitles:
```
1. "Hello world this is"
2. "this is a test of"     <- repeats "this is"
3. "a test of the emergency" <- repeats "a test of"
```

## Root Cause

The `deduplicate_subtitle_text()` function in `tools/create_amharic_dataset.py` had a logic flaw:

**Before (BUGGY):**
```python
for i, seg in enumerate(segments):
    # ...
    prev_added_text = deduplicated[-1].text  # ❌ WRONG: Last ADDED segment
    
    if current_text == prev_added_text:
        continue
    
    prev_words = prev_added_text.split()
    # ... check for overlap ...
```

**Why this failed:**
1. When segment `i-1` was an exact duplicate and got skipped
2. Segment `i` would compare with segment `i-2` (the last added segment)
3. Since `i` doesn't overlap with `i-2`, it gets added WITH its overlap to `i-1` intact
4. Rolling text is preserved instead of removed

## Solution

Compare with the **previous INPUT segment** instead of the last output segment:

**After (FIXED):**
```python
for i, seg in enumerate(segments):
    # ...
    prev_text = segments[i-1].text  # ✅ CORRECT: Previous INPUT segment
    
    if current_text == prev_text:
        continue
    
    prev_words = prev_text.split()
    # ... check for overlap ...
```

## Impact

This fix ensures:
- ✅ Rolling text overlaps are correctly detected and removed
- ✅ Works even when intermediate segments are skipped (exact duplicates)
- ✅ Each segment contains only unique text
- ✅ No more 50% overlap in dataset manifest files

## What Changed

Modified `tools/create_amharic_dataset.py`:
- Line ~609: Changed `deduplicated[-1].text` → `segments[i-1].text`
- Lines ~618, 543-548: Updated all comparisons to use `prev_text`
- Removed incorrect logic that tried to replace previously added segments

## Testing

To verify the fix works with your data:

```bash
# Re-create your dataset with the fix
python tools/create_amharic_dataset.py \
  --input-dir "C:\Users\Abrsh-1\Downloads\your_downloads" \
  --output-dir amharic_dataset_fixed \
  --no-text-dedup  # Test with this flag to see the difference

# Compare with deduplication enabled (default)
python tools/create_amharic_dataset.py \
  --input-dir "C:\Users\Abrsh-1\Downloads\your_downloads" \
  --output-dir amharic_dataset_deduplicated

# Check the manifest files
type amharic_dataset_deduplicated\manifest.jsonl
```

You should now see unique text in each segment instead of 50% overlap.

## Re-processing Your Existing Dataset

If you have an existing dataset with overlapping text, you'll need to:

1. **Re-download the source files** (if from YouTube):
   ```bash
   python tools/youtube_amharic_downloader.py --url-file urls.txt --output-dir fresh_downloads
   ```

2. **Re-create the dataset** with the fix:
   ```bash
   python tools/create_amharic_dataset.py \
     --input-dir fresh_downloads \
     --output-dir amharic_dataset_fixed
   ```

Or if you still have the original audio and SRT files, just re-run the dataset creator on them.

## Note on the Knowledge File

The knowledge.md mentioned that text deduplication was "enabled by default" and working correctly. This was incorrect - the feature was enabled but had this bug. The fix now makes it work as originally intended.
