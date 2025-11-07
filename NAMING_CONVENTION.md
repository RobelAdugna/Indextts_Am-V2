# Dataset Naming Convention

## Overview

All segmented audio files use a consistent, predictable naming scheme regardless of the source filename length or format.

## Format

```
spk{speaker_id:03d}_{segment_number:06d}.wav
```

### Components

1. **Prefix:** `spk` (speaker)
2. **Speaker ID:** 3-digit zero-padded number (000-999)
3. **Separator:** `_`
4. **Segment Number:** 6-digit zero-padded sequential number (000001-999999)
5. **Extension:** `.wav`

### Examples

```
spk000_000001.wav  # Speaker 0, Segment 1
spk000_000042.wav  # Speaker 0, Segment 42
spk001_000234.wav  # Speaker 1, Segment 234
spk015_002893.wav  # Speaker 15, Segment 2893
```

## Modes

### Single Speaker Mode

**Use when:** All audio is from one speaker (e.g., single narrator, one podcast host)

**Behavior:**
- All files use speaker ID `000`
- Segments numbered sequentially across all source files
- Example output from 3 video files:
  ```
  spk000_000001.wav  (from video 1, subtitle 1)
  spk000_000002.wav  (from video 1, subtitle 2)
  ...
  spk000_000045.wav  (from video 2, subtitle 1)
  spk000_000046.wav  (from video 2, subtitle 2)
  ...
  spk000_000127.wav  (from video 3, subtitle 1)
  ```

**Enable in WebUI:** Check "Single Speaker Mode" checkbox

**Enable via CLI:** Add `--single-speaker` flag

### Multi-Speaker Mode (Default)

**Use when:** Audio from multiple speakers or diverse sources

**Behavior:**
- Each source file gets unique speaker ID (000, 001, 002, ...)
- Segments numbered sequentially globally
- Example output from 3 video files:
  ```
  spk000_000001.wav  (video 1, subtitle 1)
  spk000_000002.wav  (video 1, subtitle 2)
  ...
  spk001_000045.wav  (video 2, subtitle 1)
  spk001_000046.wav  (video 2, subtitle 2)
  ...
  spk002_000127.wav  (video 3, subtitle 1)
  ```

**Enable in WebUI:** Uncheck "Single Speaker Mode" (default)

**Enable via CLI:** Default behavior (no flag needed)

## Manifest Integration

Each entry in `manifest.jsonl` includes:

```json
{
  "id": "spk001_000234",
  "text": "ሰላም ልዑል! እንዴት ነዎት?",
  "audio": "audio/spk001_000234.wav",
  "duration": 2.34,
  "language": "am",
  "speaker": "spk001",
  "source_file": "long_original_video_name_from_youtube"
}
```

### Fields

- **`id`**: Segment identifier (filename without .wav)
- **`speaker`**: Speaker ID (e.g., "spk001")
- **`source_file`**: Original filename for reference/debugging
- **`audio`**: Relative path to audio file

## Benefits

### 1. Consistent Length
- All filenames are exactly 18 characters (including .wav)
- Easy to parse and process
- No filesystem issues with long names

### 2. Alphabetical Sorting
- Files naturally sort in processing order
- Speaker grouping when sorted
- Easy to navigate in file browsers

### 3. Clear Organization
- Speaker ID immediately visible
- Segment order preserved
- No ambiguity about file origin

### 4. Scalability
- Supports up to 1,000 speakers (000-999)
- Supports up to 1,000,000 segments per speaker
- More than sufficient for TTS datasets

### 5. No Filename Collisions
- Guaranteed unique filenames
- Sequential numbering prevents duplicates
- Safe for batch processing

## Comparison with Old Naming

### Before (Base Filename)
```
ትረካ_፡_50,000_ሰብስክራይበሮች_-_5_ትርጉም_ታሪኮች_-_Amharic_Audiobook_-_Ethiopia_2024_#tereka_0042_a1b2c3d4.wav
```

**Issues:**
- Very long (100+ characters)
- Special characters (፡, ,)
- Inconsistent length
- Hard to read
- Filename truncation on some systems

### After (Consistent Naming)
```
spk001_000042.wav
```

**Benefits:**
- Short (18 characters)
- Only ASCII characters
- Consistent length
- Easy to read
- Works everywhere

## Usage Examples

### WebUI

1. Navigate to "2️⃣ Dataset" tab
2. Under "Naming Scheme" section:
   - **For single speaker:** Check "Single Speaker Mode"
   - **For multiple speakers:** Leave unchecked (default)
3. Observe filename format description below checkbox
4. Proceed with dataset creation

### Command Line

**Single Speaker:**
```bash
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir dataset \
  --single-speaker
```

**Multi-Speaker (default):**
```bash
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir dataset
```

## Best Practices

### When to Use Single Speaker Mode

✅ **Use for:**
- Audiobook by single narrator
- Single podcast host
- One person's voice recordings
- Consistency across training samples

❌ **Don't use for:**
- Multiple YouTube channels
- Conversational content (multiple speakers)
- Diverse voice dataset
- Mixed gender/age speakers

### When to Use Multi-Speaker Mode

✅ **Use for:**
- Multiple YouTube videos
- Different speakers/channels
- Conversational datasets
- Diverse voice samples
- Production datasets with speaker variety

❌ **Don't use for:**
- Single continuous recording
- When speaker consistency is required

## Migration from Old Format

If you have existing datasets with old naming:

1. **Keep old data separate** - don't mix naming conventions
2. **Reprocess with new tool** - use updated create_amharic_dataset.py
3. **Update manifests** - ensure manifest reflects new naming

## Technical Details

### Counter Management

- **Speaker Counter:** Increments per source file (multi-speaker mode)
- **Segment Counter:** Global, increments per accepted segment
- **Thread-safe:** Sequential processing ensures no race conditions

### Filename Generation

```python
# Format string
segment_id = f"spk{speaker_id:03d}_{segment_counter:06d}"

# Examples
spk_id = 0, seg = 1     → "spk000_000001"
spk_id = 5, seg = 234   → "spk005_000234" 
spk_id = 99, seg = 12345 → "spk099_012345"
```

### Manifest Speaker Field

```python
# Consistent speaker ID formatting
entry["speaker"] = f"spk{speaker_id:03d}"

# Examples
speaker_id = 0   → "spk000"
speaker_id = 15  → "spk015"
speaker_id = 999 → "spk999"
```

## FAQ

**Q: Can I customize the prefix?**
A: Currently "spk" is hardcoded. This ensures consistency across datasets.

**Q: What if I have more than 999 speakers?**
A: Unlikely for TTS datasets. If needed, file a feature request to increase to 4 digits.

**Q: Can I preserve original filenames?**
A: Yes, via `source_file` field in manifest. Original name available for reference.

**Q: Is this compatible with IndexTTS training?**
A: Yes, the manifest format is fully compatible. Filenames don't affect training.

**Q: Can I use custom segment numbering?**
A: No, sequential numbering ensures uniqueness and consistency.

**Q: What about existing datasets?**
A: Keep separate or reprocess. Don't mix old and new naming.

---

**Summary:** The new naming convention provides consistent, short, and predictable filenames that work across all platforms and scale to production datasets. Use single-speaker mode for uniform voice datasets and multi-speaker mode (default) for diverse collections.
