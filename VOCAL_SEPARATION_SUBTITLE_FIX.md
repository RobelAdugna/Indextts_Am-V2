# Vocal Separation Subtitle Pairing Fix

## Problem

When using the background music removal feature (audio-separator), the tool creates vocal-separated files with modified names:
- Original: `video.wav` + `video.am.srt`
- After separation: `video_(Vocals)_UVR_MDXNET_KARA_2.wav` (but subtitle stays `video.am.srt`)

The dataset creation process expects audio files to have matching subtitle files, causing it to fail finding subtitles for the separated vocals.

## Solution

### 1. Automatic Fix (WebUI)

The `webui_amharic.py` music removal function now automatically copies subtitle files to match separated vocal filenames:

```python
# Automatically creates:
video_(Vocals)_UVR_MDXNET_KARA_2.am.srt
```

**Features:**
- Detects subtitle files with language codes (am, amh, en, etc.)
- Supports multiple formats (.srt, .vtt, .webvtt)
- Reports copy statistics in logs
- Only copies for Vocals files (not Instrumental)

### 2. Manual Fix Script

For already-separated files, use the standalone fix script:

```bash
# Dry run (preview)
python tools/fix_vocal_subtitles.py \
  --vocal-dir amharic_vocals \
  --original-dir amharic_downloads \
  --dry-run

# Actual fix
python tools/fix_vocal_subtitles.py \
  --vocal-dir amharic_vocals \
  --original-dir amharic_downloads
```

**Features:**
- Processes all vocal-separated files in directory
- Extracts original filename from UVR suffixes
- Finds matching subtitles with language code support
- Dry-run mode for safety
- Detailed statistics and logging

## Implementation Details

### Pattern Matching

The fix identifies vocal files using regex:
```python
# Matches: video_(Vocals)_UVR_MDXNET.wav, video_(Vocals)_UVR_*.wav
pattern = r'_(Vocals|Instrumental)_UVR.*\.(wav|mp3|flac|m4a)$'
```

Extraction of original name:
```python
# Input: 'video_(Vocals)_UVR_MDXNET_KARA_2.wav'
# Output: 'video'
original = re.sub(r'_(Vocals|Instrumental)_UVR.*$', '', stem)
```

### Subtitle Detection

Searches for subtitles in this order:
1. Exact match: `video.srt`
2. Language codes: `video.am.srt`, `video.amh.srt`, `video.en.srt`
3. Multiple formats: `.srt`, `.vtt`, `.webvtt`

## Files Changed

### 1. `tools/fix_vocal_subtitles.py` (NEW)
- Standalone script for fixing already-separated files
- ~250 lines with comprehensive error handling
- Dry-run support
- Statistics tracking

### 2. `webui_amharic.py`
- Updated `remove_background_music()` function
- Added automatic subtitle copying after separation
- Reports subtitle copy statistics
- Logs informative messages

### 3. `knowledge.md`
- Added "Subtitle Pairing for Separated Files" section
- Documented the problem and solutions
- Provided usage examples

## Usage Examples

### Example 1: Using WebUI

1. Go to Tab 1 "Download"
2. Open "Remove Background Music" accordion
3. Set input/output directories:
   - Input: `amharic_downloads` (original files)
   - Output: `amharic_vocals` (separated vocals)
4. Click "Remove Music"
5. Subtitles are automatically copied

### Example 2: Fix Existing Files

```bash
# Check what would be copied
python tools/fix_vocal_subtitles.py \
  --vocal-dir /path/to/vocals \
  --original-dir /path/to/originals \
  --dry-run

# Actually copy
python tools/fix_vocal_subtitles.py \
  --vocal-dir /path/to/vocals \
  --original-dir /path/to/originals
```

Output:
```
📂 Processing vocal files...
  Vocal dir: amharic_vocals
  Original dir: amharic_downloads
  Output dir: amharic_vocals

Found 10 vocal-separated files
  ✅ Copied: video.am.srt -> video_(Vocals)_UVR_MDXNET.am.srt
  ✅ Copied: audio.srt -> audio_(Vocals)_UVR_MDXNET.srt
  ...

📊 Summary:
  Total vocal files: 10
  Subtitles copied: 10
  Already existed: 0
  Not found: 0
  Errors: 0
```

## Testing

### Test Case 1: WebUI Music Removal
1. Place audio + subtitle in `downloads/`:
   - `test.wav`
   - `test.am.srt`
2. Run music removal to `vocals/`
3. Verify files created:
   - `vocals/test_(Vocals)_UVR_MDXNET_KARA_2.wav`
   - `vocals/test_(Vocals)_UVR_MDXNET_KARA_2.am.srt` ✓

### Test Case 2: Manual Fix Script
1. Separate files manually (already done)
2. Run fix script
3. Verify subtitle copies created
4. Run dataset creation - should find subtitles ✓

## Benefits

1. **No Manual Work**: Automatic subtitle pairing
2. **Backward Compatible**: Fix script for old separations
3. **Robust Detection**: Handles language codes and formats
4. **Safe**: Dry-run mode prevents accidents
5. **Informative**: Detailed logging and statistics

## Future Improvements

Potential enhancements:
1. Extract shared subtitle-finding logic to `tools/subtitle_utils.py`
2. Support other separation tools (demucs, spleeter)
3. Handle edge cases (special characters in filenames)
4. Add unit tests
