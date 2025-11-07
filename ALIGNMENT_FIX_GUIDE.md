# Audio-Text Alignment Fix Guide

## Problem Description

You reported that segmented audio files don't match their text:
- **Audio starts too late** - missing first few words
- **Audio ends too early** - cutting off final sounds
- **Text is correct** - matches the SRT file
- **Original video + SRT** - perfectly aligned

## Root Cause Analysis

### Why This Happened

The previous boundary refinement algorithm had a fatal flaw:

1. **Searched for "quietest point"** within ±0.5s of subtitle boundaries
2. **Assumed quiet = silence** between utterances
3. **Reality:** Quiet often means:
   - Unvoiced consonants (s, f, h, th)
   - Quiet phonemes at word boundaries  
   - Amharic ejective consonants
   - Geminated consonants (quiet closure phase)

4. **Result:** Algorithm would find a quiet consonant and treat it as the "best" boundary, cutting into actual speech

### Example Problem

**Subtitle:** "ሰላም ልዑል እንዴት ነዎት?" (0:05.0 - 0:08.0)

**Old Algorithm:**
- Searches 0:04.5 - 0:05.5 for quietest point
- Finds unvoiced "s" in "ሰላም" at 0:05.2 (-45dB)
- Moves start to 0:05.2
- **Result:** Misses "ሰላ" at beginning!

**New Algorithm:**
- Expands to 0:04.85 - 0:08.10 (safety margins)
- Searches 0:04.85 - 0:05.0 for **sustained** silence
- Finds silence at 0:04.90 (-55dB for 60ms)
- Uses 0:04.90 as start
- **Result:** Captures full "ሰላም"!

## The Fix

### Two-Stage Approach

#### Stage 1: Safety Margins (Expansion)

**Always expand boundaries beyond subtitle times:**

- **Start Margin:** +0.15 seconds **before** subtitle start
  - Why: Subtitles lag 0.1-0.3s after actual speech onset
  - Accounts for human reaction time in subtitle creation
  - Captures speech onset (often quiet)

- **End Margin:** +0.10 seconds **after** subtitle end
  - Why: Speech trails off gradually
  - Captures final phonemes and intonation
  - Subtitle creators often cut early for readability

#### Stage 2: Sustained Silence Trimming (Optional)

**Only trim if finding TRUE silence:**

- **Threshold:** -50dB (very quiet, stricter than before)
- **Duration:** ≥3 consecutive frames (60ms @ 20ms hop)
- **Direction:**
  - Start: Search backward from subtitle start
  - End: Search forward from subtitle end
- **Constraint:** **NEVER** move inside original subtitle boundaries

### Key Differences from Before

| Aspect | Old Algorithm | New Algorithm |
|--------|---------------|---------------|
| **Approach** | Find quietest point | Find sustained silence |
| **Threshold** | -40dB | -50dB |
| **Duration** | Single frame | 3 consecutive frames |
| **Direction** | Can shrink inside subtitles | Never shrinks inside |
| **Default** | Use quietest found | Use safety margin |
| **Risk** | Cuts speech | Includes extra silence |

## How to Use

### Default Settings (Recommended)

The defaults should work well for most Amharic content:

```bash
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir dataset
  # Uses: start_margin=0.15, end_margin=0.1
```

Or in WebUI:
- ✅ Enable boundary refinement
- Start Safety Margin: 0.15s
- End Safety Margin: 0.10s

### Custom Margins

#### If still hearing cutoffs:

**Increase margins:**
```bash
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir dataset \
  --start-margin 0.25 \
  --end-margin 0.20
```

WebUI: Move sliders to 0.25 and 0.20

#### If getting too much silence:

**Decrease margins:**
```bash
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir dataset \
  --start-margin 0.10 \
  --end-margin 0.05
```

WebUI: Move sliders to 0.10 and 0.05

#### If subtitles are already perfect:

**Disable refinement completely:**
```bash
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir dataset \
  --no-refine
```

WebUI: Uncheck "Enable boundary refinement"

## Validation

### How to Check If Fix Worked

1. **Listen to a few random segments:**
   ```bash
   # Play first segment
   ffplay dataset/audio/spk000_000001.wav
   ```

2. **Check manifest text:**
   ```bash
   head -1 dataset/manifest.jsonl
   ```

3. **Verify alignment:**
   - Does audio start with first word?
   - Does audio end after last word?
   - Any silence at start/end?

4. **Quality report:**
   ```bash
   cat dataset/quality_report.json
   ```
   - Check `rejection_reasons` for issues

### Expected Outcomes

**Good alignment:**
- Audio starts 0.05-0.2s before first phoneme
- Audio ends 0.05-0.15s after last phoneme
- Small amount of leading/trailing silence OK
- All words clearly audible

**Bad alignment (increase margins):**
- First word cut off
- Last word cut off
- Speech starts abruptly

**Too much margin (decrease):**
- Long silence before speech
- Long silence after speech
- Background noise/music included

## Technical Details

### Algorithm Pseudocode

```python
# Stage 1: Expand with safety margins
expanded_start = subtitle_start - start_margin  # e.g., 5.0 - 0.15 = 4.85
expanded_end = subtitle_end + end_margin        # e.g., 8.0 + 0.10 = 8.10

# Stage 2: Search for sustained silence (optional trimming)
# For start: search from expanded_start to subtitle_start
for each window in [expanded_start, subtitle_start]:
    if sustained_silence(window, threshold=-50dB, frames=3):
        refined_start = end_of_silence
        break
else:
    refined_start = expanded_start  # Keep margin if no silence

# For end: search from subtitle_end to expanded_end
for each window in [subtitle_end, expanded_end]:
    if sustained_silence(window, threshold=-50dB, frames=3):
        refined_end = start_of_silence
        break
else:
    refined_end = expanded_end  # Keep margin if no silence

# Safety check: never go inside subtitle bounds
final_start = min(refined_start, subtitle_start)
final_end = max(refined_end, subtitle_end)
```

### RMS Calculation

Uses librosa-compatible RMS:
- Frame length: 2048 samples
- Hop length: 512 samples (20ms @ 24kHz)
- Padding: Reflective
- Convert to dB: `20 * log10(RMS + 1e-10)`

### Sustained Silence Detection

```python
silent_mask = rms_db < threshold_db  # Boolean array

# Find N consecutive True values
for i in range(len(silent_mask) - N + 1):
    if all(silent_mask[i:i+N]):
        # Found sustained silence at frames [i, i+N)
        return True
return False
```

## Troubleshooting

### Still Hearing Cutoffs?

1. **Check original SRT timing:**
   ```bash
   head -20 video.srt
   ```
   - Are timestamps reasonable?
   - Any obvious errors?

2. **Verify audio sample rate:**
   ```bash
   ffprobe video.wav 2>&1 | grep Audio
   ```
   - Should be 24000 Hz after processing
   - Mismatched rates cause drift

3. **Try larger margins:**
   - Start: 0.3s
   - End: 0.2s

4. **Disable refinement:**
   - Use `--no-refine`
   - Trust subtitle timing completely

### Too Much Silence?

1. **Decrease margins:**
   - Start: 0.05s
   - End: 0.05s

2. **Check quality filtering:**
   - May need to increase `max_silence_ratio`
   - Default: 30% silence allowed

3. **Verify source audio:**
   - Original may have silence
   - Not a segmentation issue

### Segments Overlap?

This is OK and expected when:
- Subtitles are very close together (<0.25s gap)
- Safety margins cause overlap
- Model training can handle overlapping prompts

To prevent:
```bash
--start-margin 0.05 --end-margin 0.05
```

## Best Practices

### Content Type Guidelines

**Audiobooks (single narrator):**
- Start margin: 0.15s
- End margin: 0.10s  
- Enable refinement
- Single speaker mode

**YouTube videos (varied quality):**
- Start margin: 0.20s
- End margin: 0.15s
- Enable refinement
- Multi-speaker mode

**Professional broadcasts:**
- Start margin: 0.10s
- End margin: 0.05s
- Enable refinement
- Check quality report

**Conversational content:**
- Start margin: 0.20s
- End margin: 0.15s
- May want to disable refinement
- Rapid exchanges need more margin

### Quality Over Quantity

- **Better:** 1000 perfectly aligned segments
- **Worse:** 5000 segments with cutoffs
- Training on misaligned data teaches bad timing
- Use quality filtering aggressively

## FAQ

**Q: Will this make training take longer?**
A: Slightly (each segment ~0.25s longer), but better alignment improves model quality significantly.

**Q: Do margins affect inference?**
A: No, model learns to generate appropriate timing during training.

**Q: Can I use different margins per file?**
A: Not currently. Process files separately if needed.

**Q: What if my subtitles are auto-generated?**
A: Increase margins to 0.25/0.20 - auto-subs are less accurate.

**Q: Why not use forced alignment?**
A: Requires Amharic phoneme dictionary (not readily available). This is simpler and works well.

**Q: Does this work for other languages?**
A: Yes! The same principles apply. Adjust margins based on language characteristics.

---

**Summary:** The fix adds safety margins to prevent speech cutoff while optionally trimming sustained silence. Default settings (0.15s/0.10s) work well for most Amharic content. Adjust based on your specific source quality.
