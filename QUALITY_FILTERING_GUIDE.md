# Amharic Dataset Quality Filtering Guide

## Overview

The enhanced dataset creator includes comprehensive quality filtering optimized for Amharic TTS training. All features are accessible through the Gradio WebUI with sensible defaults.

## Quality Checks Applied

### 1. Audio Quality

#### Signal-to-Noise Ratio (SNR)
- **What it checks:** Audio clarity and noise levels
- **Default threshold:** ≥15 dB
- **Recommendation:** 
  - 15-20 dB: Good for most content
  - 20-25 dB: Strict, for high-quality datasets
  - 10-15 dB: Lenient, for low-quality sources

#### Silence Ratio
- **What it checks:** Percentage of segment that is silent
- **Default threshold:** ≤30%
- **Why it matters:** Too much silence wastes training time
- **Recommendation:**
  - 30%: Balanced (default)
  - 20%: Strict, removes pauses
  - 40%: Lenient, keeps natural pauses

#### Clipping Detection
- **What it checks:** Audio distortion from amplitude overflow
- **Default threshold:** ≤1% of samples clipped
- **Why it matters:** Distorted audio teaches bad pronunciation

### 2. Text Quality (Amharic-Specific)

#### Script Validation
- **What it checks:** Percentage of Ethiopic script characters
- **Default threshold:** ≥50% Ethiopic characters
- **Unicode ranges validated:**
  - U+1200-137F: Ethiopic
  - U+1380-139F: Ethiopic Supplement
  - U+2D80-2DDF: Ethiopic Extended
  - U+AB00-AB2F: Ethiopic Extended-A
- **Why it matters:** Ensures text is actually Amharic, not mixed/wrong language

#### Word Count
- **What it checks:** Number of Amharic words in text
- **Default threshold:** ≥3 words
- **Word detection:** Splits on spaces and Ethiopic word separator (፡)
- **Recommendation:**
  - 3 words: Balanced (default)
  - 5 words: Stricter, better for training
  - 1-2 words: Too short, poor context

#### Speech Rate
- **What it checks:** Characters per second (text length / duration)
- **Default range:** 5-20 chars/second
- **Why Amharic-specific:**
  - Amharic is syllabic (each character ≈ 1 syllable)
  - More consistent speech rate than alphabetic languages
  - Typical conversational: 8-15 chars/sec
  - Narration: 5-10 chars/sec
  - Rapid speech: 15-20 chars/sec
- **Filters:**
  - Too slow (<5): Likely poor subtitle alignment or long pauses
  - Too fast (>20): Likely subtitle timing errors

### 3. Subtitle Artifact Removal

Automatically removes:
- **HTML/XML tags:** `<i>`, `<b>`, `<font color="red">`
- **Formatting markers:** `{y:i}`, positioning tags
- **Sound effects (English):** `[Music]`, `[Applause]`, `(background music)`
- **Sound effects (Amharic):** `[ሙዚቃ]`, `(ሙዚቃ)`, `(ድምፅ)`
- **Speaker labels:** `JOHN:`, `SPEAKER 1:`
- **Multiple spaces:** Normalized to single space

**Music-only segments are completely filtered out:**
- Segments containing only `[ሙዚቃ]` or `[Music]` are rejected
- Tracked separately in quality report as `music_or_sound_only`
- Both text cleaning and normalization stages check for empty results

### 4. Boundary Refinement

#### RMS-Based Precision
- **Method:** Analyzes frame-by-frame energy to find quietest point
- **Search window:** ±0.5 seconds around subtitle boundary
- **Validation:** Rejects if boundary change >0.5 seconds
- **Confidence score:** Higher when boundaries change minimally

## Using in Gradio WebUI

### Default Settings (Recommended)

```
Tab 2: Dataset Creation
├── Duration Filters
│   ├── Min Duration: 1.0s
│   └── Max Duration: 30.0s
├── Boundary Refinement: ✓ Enabled
└── Quality Filtering: ✓ Enabled
    ├── Min SNR: 15 dB
    ├── Max Silence: 30%
    ├── Min Words: 3
    ├── Min Speech Rate: 5.0 chars/s
    └── Max Speech Rate: 20.0 chars/s
```

### When to Adjust Settings

**Scenario 1: Low-Quality Source Audio**
- Lower min SNR to 12-13 dB
- Increase max silence ratio to 40%
- Keep other settings default

**Scenario 2: High-Quality Professional Audio**
- Raise min SNR to 20-25 dB
- Lower max silence ratio to 20%
- Increase min words to 5
- Tighter speech rate: 7-15 chars/s

**Scenario 3: Conversational/Natural Speech**
- Keep defaults
- Increase max silence ratio to 35% (natural pauses)
- Speech rate 5-18 chars/s

**Scenario 4: Narration/Audiobooks**
- Keep defaults
- Speech rate 4-12 chars/s (slower, deliberate)
- Min words 5 (complete sentences)

## Quality Report

After processing, check `quality_report.json`:

```json
{
  "total_segments": 1523,
  "accepted": 1287,
  "rejected": 236,
  "rejection_reasons": {
    "Low SNR: 12.3dB": 89,
    "Too much silence: 35%": 67,
    "Too few words: 2": 45,
    "Speech too fast: 22.3 chars/s": 23,
    "Not Amharic: 35% Ethiopic chars": 12
  },
  "files_processed": 15,
  "files_failed": 0
}
```

**Interpreting Results:**

- **High rejection rate (>30%):** Settings may be too strict
  - Check rejection reasons
  - Adjust most common rejection thresholds

- **Low rejection rate (<5%):** Settings may be too lenient
  - Consider stricter thresholds for better quality

- **Balanced (10-20% rejected):** Good sweet spot
  - Getting quality improvement without losing too much data

## Manifest Enhancements

Each entry in `manifest.jsonl` now includes quality metrics:

```json
{
  "id": "video_0042_a1b2c3d4",
  "text": "ሰላም ልዑል! እንዴት ነዎት?",
  "audio": "audio/video_0042_a1b2c3d4.wav",
  "duration": 2.34,
  "language": "am",
  "speaker": "video",
  "quality": {
    "snr": 18.5,
    "speech_rate": 12.3,
    "amharic_ratio": 0.95,
    "boundary_confidence": 0.87
  }
}
```

**Use these metrics to:**
- Filter further during preprocessing
- Analyze dataset composition
- Identify problematic source files
- Tune quality thresholds

## Command-Line Usage

For advanced users, the tool supports all options via CLI:

```bash
# Strict quality settings
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir dataset_strict \
  --min-snr 20.0 \
  --max-silence-ratio 0.2 \
  --min-words 5 \
  --quality-report quality_strict.json

# Lenient settings (maximize data)
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir dataset_lenient \
  --min-snr 12.0 \
  --max-silence-ratio 0.4 \
  --min-words 2 \
  --quality-report quality_lenient.json

# No quality filtering
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir dataset_all \
  --no-quality-check
```

## Best Practices

### 1. Start with Defaults
- Run with default settings first
- Review quality report
- Adjust based on rejection patterns

### 2. Iterate on Thresholds
- Too many rejections? Relax thresholds
- Low quality output? Tighten thresholds
- Different sources may need different settings

### 3. Quality vs Quantity
- **High-quality dataset (stricter filters):**
  - Faster training convergence
  - Better final model quality
  - Less data needed

- **Larger dataset (lenient filters):**
  - More diverse training examples
  - Better coverage of edge cases
  - May need more training time

### 4. Source-Specific Tuning
- **YouTube:**
  - Often compressed audio (lower SNR acceptable)
  - Auto-generated subtitles (more alignment errors)
  - Suggestion: SNR 12-15, lenient speech rate

- **Audiobooks:**
  - Usually high quality audio (strict SNR)
  - Professional narration (consistent speech rate)
  - Suggestion: SNR 18-20, tight speech rate 5-12

- **Podcasts:**
  - Variable quality (moderate settings)
  - Natural pauses (higher silence ratio)
  - Suggestion: SNR 15, silence 35%

## Troubleshooting

### Problem: All segments rejected

**Check:**
1. Quality report for dominant rejection reason
2. Audio files are valid (not corrupted)
3. Subtitles match audio language
4. Thresholds aren't impossibly strict

**Solutions:**
- Disable quality filtering temporarily to see raw output
- Adjust specific threshold causing most rejections
- Verify input audio quality with audio editor

### Problem: Poor quality segments accepted

**Check:**
1. Listen to some generated segments
2. Review quality metrics in manifest
3. Compare with quality report thresholds

**Solutions:**
- Tighten relevant thresholds
- Enable quality filtering if disabled
- Check source audio quality

### Problem: Inconsistent results across files

**Check:**
1. Audio quality varies between source files
2. Different subtitle quality/formats
3. Different speakers/recording conditions

**Solutions:**
- Process high-quality files separately with strict settings
- Use lenient settings for variable-quality sources
- Consider per-file threshold adjustment

## Technical Details

### RMS Calculation
- Frame length: 2048 samples
- Hop length: 512 samples  
- Padding: Reflective at boundaries
- Based on: [audio-slicer](https://github.com/openvpi/audio-slicer)

### Amharic Word Counting
- Splits on: spaces and ፡ (U+1361)
- Counts non-empty tokens
- More accurate than simple space splitting

### Speech Rate for Syllabic Scripts
- Amharic: 1 character ≈ 1 syllable
- English: 1 character ≈ 0.2-0.3 syllables
- Therefore Amharic chars/sec ≈ 0.2-0.3 × English chars/sec
- Typical English: 15-25 chars/sec → Amharic: 5-20 chars/sec

---

**For more information, see:**
- `README_AMHARIC_WEBUI.md` - WebUI usage guide
- `tools/create_amharic_dataset.py` - Implementation details
- `knowledge.md` - Amharic language specifics
