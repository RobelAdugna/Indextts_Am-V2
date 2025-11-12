# Corpus Tab - Lightning AI Direct Path Support ✅

## Summary

Updated the Corpus Collection tab (Tab 3) in `webui_amharic.py` to support direct file path input for remote environments like Lightning AI.

## Changes Made

### 1. UI Updates (`webui_amharic.py` lines ~1314-1340)

**Added:**
- New textbox: `corpus_manifest_path` for direct path input
- Clear instructions for Lightning AI users
- Example path: `/teamspace/studios/this_studio/amharic_dataset/manifest.jsonl`

**Modified:**
- Renamed file upload from "Input Files" to "Or Upload Files"
- Updated Markdown with Lightning AI guidance

### 2. Function Updates (`webui_amharic.py` lines ~575-615)

**collect_corpus() function:**
- Added `manifest_path: str` parameter (first parameter)
- Updated docstring for clarity
- Implemented 3-tier priority system:
  1. **Direct path input** (for Lightning AI) - NEW ✅
  2. Uploaded files (existing)
  3. Auto-fill from pipeline state (existing)

**Key Features:**
- Path expansion with `.expanduser()` (supports `~/path`)
- File existence validation
- Clear error messages
- Backward compatible with existing functionality

### 3. Click Handler Update

**Updated inputs:**
```python
collect_corpus_btn.click(
    collect_corpus,
    inputs=[corpus_manifest_path, corpus_input_files, ...],  # Added manifest_path first
    outputs=[corpus_logs, corpus_status, state]
)
```

## How to Use

### For Lightning AI Users:

1. Navigate to Tab 3 "Corpus Collection"
2. Paste your full manifest path in the textbox:
   ```
   /teamspace/studios/this_studio/Indextts_Am-V2/amharic_dataset/manifest.jsonl
   ```
3. Click "📝 Collect Corpus"
4. The corpus will be extracted directly from your dataset

### For Local Users:

Existing workflow unchanged:
- Upload files via "Or Upload Files" section
- Or let it auto-fill from previous dataset creation step

## Validation

The underlying `tools/collect_amharic_corpus.py` script validates:
- **Amharic script:** ≥50% Ethiopic characters (U+1200-137F, U+1380-139F, etc.)
- **Text quality:** Removes duplicates, short lines, non-Amharic text
- **Normalization:** Uses NFC normalization for Amharic

## Testing

To test with your 200hr Amharic dataset:

```bash
# Launch WebUI
python webui_amharic.py

# In Tab 3:
# 1. Paste: /teamspace/studios/this_studio/Indextts_Am-V2/amharic_dataset/manifest.jsonl
# 2. Click "Collect Corpus"
# 3. Verify output shows line count and statistics
```

## Files Modified

- ✅ `webui_amharic.py` - Added direct path support
- ✅ `knowledge.md` - Documented the feature
- ✅ `CORPUS_TAB_LIGHTNING_AI_FIX.md` - This summary

## Benefits

1. **No file uploads** - Saves time and bandwidth on remote servers
2. **Direct access** - Works with files already on Lightning AI storage
3. **Simple UX** - Just copy/paste the path
4. **Backward compatible** - Existing workflows unaffected
5. **Robust validation** - Ensures Amharic text quality

## Next Steps

After collecting corpus:
1. Tab 4: Train BPE tokenizer
2. Tab 5: Preprocess features
3. Tab 6: Start training

Enjoy your 200hrs of Amharic data! 🎉
