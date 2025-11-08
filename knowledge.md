# IndexTTS2 Knowledge Base

## Project Overview

IndexTTS2 is a state-of-the-art Text-to-Speech system supporting multiple languages including English, Chinese, Japanese, and now Amharic.

## Language Support

### Supported Languages
- English (en)
- Chinese (zh, cn)
- Japanese (ja, jp)
- Amharic (am, amh) - **NEW** (Most complete implementation)

### Implementation Quality
- **Amharic:** Most complete with full automation, comprehensive docs, better tooling
- **Japanese:** Reference implementation, less automation
- **English/Chinese:** Base model languages

### Adding New Languages

To add support for a new language, follow the pattern established for Amharic:

1. **Text Processing** (`indextts/utils/front.py`):
   - Add script detection pattern
   - Add punctuation mapping
   - Implement normalize_LANGUAGE() method
   - Update main normalize() method

2. **Syllable/Duration** (`indextts/utils/text_utils.py`):
   - Add script detection
   - Implement syllable counting for script type
   - Adjust duration ratio if needed

3. **Data Collection Tools**:
   - Create YouTube downloader (optional)
   - Create dataset creator from audio+subtitles
   - Create corpus collector

4. **Tokenizer**:
   - Train/extend BPE with new language corpus
   - Add user-defined symbols for language-specific punctuation

5. **Preprocessing**:
   - Add language hint to LANGUAGE_HINT_OVERRIDES

6. **Automation**:
   - Create end-to-end training script

7. **Documentation**:
   - Create comprehensive guide
   - Add test cases
   - Provide examples

## Amharic Implementation Details

### Script Characteristics
- **Type:** Syllabary (abugida)
- **Name:** Ge'ez/Ethiopic script
- **Characters:** ~231 base + labialized forms
- **Property:** Each character = one syllable (fidel)

### Unicode Ranges
- Basic: U+1200–U+137F
- Supplement: U+1380–U+139F
- Extended: U+2D80–U+2DDF
- Extended-A: U+AB00–U+AB2F

### Punctuation
- ። (full stop) → .
- ፣ (comma) → ,
- ፤ (semicolon) → ;
- ፥ (colon) → :
- ፧ (question) → ?
- ፨ (exclamation) → !

### Normalization
- Use NFC (not NFKC) for proper character composition
- Duration ratio: 1.0 (similar to English)
- Each character counts as 1 syllable

## Training Pipeline

### Standard Workflow

1. **Data Collection**
   - Download audio with subtitles
   - Or prepare existing audio+transcript pairs

2. **Dataset Creation**
   - Segment audio by subtitles
   - Refine boundaries with silence detection
   - Normalize text
   - Generate JSONL manifest

3. **Corpus Collection**
   - Extract text from manifests
   - Clean and deduplicate
   - Validate language

4. **Tokenizer Training**
   - Train BPE on collected corpus
   - Set high character coverage (0.9999)
   - Include language-specific symbols

5. **Preprocessing**
   - Extract semantic features
   - Tokenize text
   - Generate conditioning vectors
   - Split train/validation

6. **Generate Prompt Pairs**
   - Create prompt-target combinations
   - Required for GPT training

7. **GPT Training**
   - Fine-tune on new language
   - Monitor validation metrics
   - Save checkpoints

## Lightning AI Deployment

### Best Practices

1. **All paths must be relative** - No absolute paths
2. **Use `uv` for dependencies** - Ensures consistency
3. **Enable `--amp` flag** - Mixed precision saves VRAM
4. **Monitor with TensorBoard** - Track training progress
5. **Keep 3 recent checkpoints** - Automatic in training script

### Typical Command

```bash
cd project-dir
uv sync --all-extras
bash scripts/LANGUAGE/end_to_end.sh
```

## Dataset Statistics

**Feature:** Comprehensive analysis of your dataset manifest
**Location:** Tab 2 accordion "Dataset Statistics"
**Analyzes:**
- Total segments, duration (hours/minutes)
- Number of source videos processed
- Unique speakers
- Duration stats (avg/min/max per segment)
- Text stats (words, characters)
- Language distribution
- Top 10 source files by segment count

**Usage:** Enter manifest path (e.g., `amharic_dataset/manifest.jsonl`) and click "Analyze Dataset"

## WebUI Defaults (Amharic-Optimized)

**Safety Margins:** Start=0.2s, End=0.15s (conservative for YouTube content)
**Quality Filtering:** Disabled by default (accept more segments)
**Text Deduplication:** Enabled by default (remove rolling subtitle text)
**VAD:** Enabled by default (best boundary detection)

## Background Music Removal

**Feature:** Remove music/instruments from downloaded files before dataset creation
**Library:** audio-separator (UVR backend)
**Models:** MDX-Net (fast), Demucs (balanced), Demucs FT (slow/best)
**Install:** `pip install audio-separator` (auto-detects GPU/CPU)
**GPU:** Install base package only (not `[cpu]` extra)
**CPU:** Use `pip install 'audio-separator[cpu]'` to force CPU
**Location:** Tab 1 accordion "Remove Background Music"
**Recommended:** MDX-Net for large batches (8/10 quality, very fast)

## Text Deduplication

**Problem:** SRT subtitles have rolling text (50% overlap between lines)
**Solution:** Auto-detects 3+ word overlap, removes from current segment
**Toggle:** Dataset Creation tab checkbox (enabled by default)
**Result:** Clean, unique text per segment while audio stays aligned

## Segmentation V2: Production-Grade Implementation

**Status:** ✅ COMPLETE - Production ready

**Key Achievement:** Zero audio overlap + minimal speech cutoff through hard boundary enforcement and VAD-based detection.

### Quick Start
1. Install VAD: `pip install webrtcvad` (recommended but optional)
2. Use WebUI Tab 2 or CLI with defaults
3. Verify: Check `boundary_info` in manifest.jsonl

### Documentation
- `SEGMENTATION_V2_GUIDE.md` - Complete technical guide
- `SEGMENTATION_V2_SUMMARY.md` - Executive summary
- `ALIGNMENT_FIX_GUIDE.md` - Original fix documentation

## Critical Fix: Audio-Text Alignment

### Problem: Segmented audio cuts off speech
**Symptoms:** Audio starts too late (missing first words) or ends too early (cutting off final sounds)

**Root Cause:** Previous RMS-based refinement searched for quietest point, which often was:
- Quiet consonants (s, f, h) at word boundaries
- Unvoiced phonemes in speech
- Low-energy geminated consonants in Amharic

**Solution:** Two-stage boundary processing:
1. **Safety Margins** (expand boundaries):
   - Start: +0.15s before subtitle (accounts for subtitle lag)
   - End: +0.1s after subtitle (speech trails off)
2. **Sustained Silence Trimming** (optional):
   - Only trims if ≥3 consecutive frames below -50dB
   - Never moves boundaries inside original subtitle times

**Configuration:**
- CLI: `--start-margin 0.15 --end-margin 0.1`
- WebUI: Sliders under "Boundary Refinement & Safety Margins"
- Increase margins if still hearing cutoffs
- Disable refinement (`--no-refine`) to trust subtitles completely

**Why This Works:**
- Subtitles are typically 0.1-0.3s late at start, early at end
- Safety margins ensure no speech is lost
- Sustained silence requirement prevents cutting into quiet consonants
- Never shrinks inside subtitle bounds (only expands)

**Critical Fix Applied:** Expansion calculation corrected to `(start_time - new_start) + (new_end - end_time)`. Confidence score now properly reflects margin size.

**Production-Grade Boundary Refinement (V2):**

**Hard Boundary Enforcement:**
- Segments can NEVER overlap in audio
- Uses midpoint between adjacent subtitles as absolute limit
- Example: Subtitle A ends 3.5s, Subtitle B starts 4.0s → Hard limit at 3.75s
- Segment A cannot extend beyond 3.75s, Segment B cannot start before 3.75s

**Two-Method Approach (Priority Order):**

1. **VAD-Based (Primary - Recommended):**
   - Uses WebRTC Voice Activity Detection
   - Detects actual speech regions acoustically
   - Finds natural speech start/end within search window
   - Most accurate, language-agnostic
   - Requires: `pip install webrtcvad`

2. **Margin-Based (Fallback):**
   - If VAD unavailable/fails
   - Uses configurable safety margins (0.15s start, 0.1s end)
   - Automatically reduces margins if would cause overlap
   - Guaranteed to respect hard boundaries

**Benefits:**
- Zero audio overlap (mathematically guaranteed)
- No speech cutoff (VAD finds actual speech)
- Works with close subtitles (<0.5s gap)
- Detailed metadata for debugging (`boundary_info` field)

**Configuration:**
- CLI: `--use-vad` (default) or `--no-vad` for margin-only
- Margins: `--start-margin 0.15 --end-margin 0.1`

**Metadata Example:**
```json
"boundary_info": {
  "method": "vad",
  "vad_used": true,
  "constrained": true,
  "start_margin": 0.0,
  "end_margin": 0.0
}
```

## Dataset Naming Convention

### Consistent Segment Naming
Segment files use consistent, sequential naming:
- Format: `spk{speaker_id:03d}_{segment_number:06d}.wav`
- Examples: `spk000_000001.wav`, `spk001_000234.wav`

**Single Speaker Mode** (`--single-speaker`):
- All segments use speaker ID `000`
- Sequential numbering across all files
- Example: `spk000_000001.wav` through `spk000_002345.wav`
- Use for: Single narrator datasets, audiobooks, single podcast host

**Multi-Speaker Mode** (default):
- Each source audio file gets unique speaker ID
- Sequential numbering continues globally
- Example: Video 1 → `spk000_*`, Video 2 → `spk001_*`, etc.
- Use for: Multiple speakers, diverse sources, conversational data

**Manifest Fields:**
- `id`: Segment ID (matches filename without .wav)
- `speaker`: Speaker ID (e.g., "spk000")
- `source_file`: Original filename for reference

This ensures:
- Consistent filename length
- Easy sorting and organization  
- Clear speaker identification
- No long/unwieldy filenames from source videos

## Common Issues & Solutions

### Subtitle Detection Issues
**Problem:** Dataset creator can't find subtitle files for downloaded videos
**Cause:** yt-dlp adds language codes to subtitle filenames (e.g., `video.am.srt`)
**Solution:** The `create_amharic_dataset.py` tool now searches for:
- Exact match: `video.srt`
- Language codes: `video.am.srt`, `video.en.srt`, `video.amh.vtt`

If issues persist, check the actual filenames in the download directory.

### Dataset Quality Improvements (v2)
The `create_amharic_dataset.py` tool includes comprehensive quality filtering:

**RMS-Based Boundary Refinement:**
- Uses slicer2.py approach to find quietest point in search window
- Validates boundaries don't change duration drastically
- Returns confidence score for each refinement

**Amharic Script Validation:**
- Checks text is ≥50% Ethiopic characters (U+1200-137F ranges)
- Filters mixed-script content
- Removes subtitle formatting artifacts

**Audio Quality Checks:**
- SNR filtering (default: ≥15dB)
- Silence ratio limit (default: ≤30%)
- Clipping detection (default: ≤1%)
- Speech rate validation (3-25 chars/second)

**Text Quality Checks:**
- Minimum word count (default: 3 words)
- Automatic subtitle artifact removal ([Music], speaker labels, HTML tags)
- Duplicate detection via content hashing

**Quality Reporting:**
- Use `--quality-report` to save JSON with rejection statistics
- Each segment includes SNR, speech rate, Amharic ratio, and boundary confidence
- Detailed breakdown of rejection reasons

**Example:**
```bash
python tools/create_amharic_dataset.py \
  --input-dir downloads \
  --output-dir dataset \
  --min-snr 20.0 \
  --max-silence-ratio 0.2 \
  --min-words 5 \
  --quality-report quality.json
```

### Music Removal GPU Acceleration
**Issue:** audio-separator using CPU despite GPU available
**Fix:** `use_cuda=True` parameter in Separator() enables GPU
**Note:** Base package auto-detects CUDA. Don't use `[cpu]` extra.

### NumPy 2.x Compatibility Error
**Error:** `ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.4`
**Fix:** `pip install 'numpy<2'`
**Added to pyproject.toml:** numpy<2 constraint

### Out of Memory
- Reduce `--batch-size`
- Increase `--grad-accumulation`
- Use `--amp` flag
- Check VRAM usage with `nvidia-smi`

### Poor Tokenization
- Increase vocab size
- Increase character coverage
- Add more diverse corpus
- Check text normalization

### Slow Training
- Use more GPUs if available
- Increase batch size if VRAM allows
- Check data loading (increase workers)
- Enable AMP if not already

## Code Style

### Imports
- Remove unused imports
- Group: stdlib, third-party, local
- Use absolute imports from project root

### Functions
- Type hints for parameters and return values
- Docstrings for all public functions
- Keep functions focused and single-purpose

### Error Handling
- Catch specific exceptions
- Provide helpful error messages
- Log errors for debugging
- Clean up resources in finally blocks

## Testing

### Unit Tests
- Test normalization functions
- Test syllable counting
- Test tokenization

### Integration Tests
- Test full pipeline on small dataset
- Verify all steps complete
- Check output quality

### Manual Testing
- Listen to generated speech
- Verify pronunciation
- Check emotion transfer
- Compare with base model

## Documentation Standards

### Required Documentation
- Setup guide with examples
- Troubleshooting section
- Best practices
- API/tool references

### Code Comments
- Explain WHY, not WHAT
- Document non-obvious logic
- Include examples in docstrings
- Keep comments up-to-date

## Version Control

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Be specific and descriptive
- Reference issues if applicable

### Branching
- `main` - stable releases
- `training_v2` - current development
- Feature branches for major changes

## Amharic WebUI

### Overview
A comprehensive Gradio web interface (`webui_amharic.py`) integrates all Amharic-specific tools into a single, user-friendly pipeline:

**Features:**
- 7 sequential tabs covering entire pipeline
- Automatic state management and auto-fill
- Real-time progress tracking
- Color-coded status indicators
- Dependency checking (yt-dlp, ffmpeg, CUDA)

**Launch:**
```bash
python webui_amharic.py          # Default port 7863
python webui_amharic.py --share  # Create public link
```

**Pipeline Tabs:**
1. Download - YouTube content collection
2. Dataset - Audio segmentation with subtitles
3. Corpus - Text aggregation and cleaning
4. Tokenizer - BPE model training
5. Preprocess - Feature extraction
6. Training - GPT fine-tuning launcher
7. Inference - Links to existing TTS UIs

### Usage Pattern
- Each tab auto-fills from previous step's output
- Can skip steps if intermediate files exist
- Progress tracked in real-time
- Logs displayed in UI

See `README_AMHARIC_WEBUI.md` for complete documentation.

## Resources

### Documentation
- See `docs/` directory for guides
- Check `examples/` for usage examples
- Read `AMHARIC_IMPLEMENTATION_*.md` for details
- See `README_AMHARIC_WEBUI.md` for WebUI usage

### External References
- [IndexTTS2 Paper](https://arxiv.org/abs/2506.21619)
- [SentencePiece](https://github.com/google/sentencepiece)
- [Unicode Ethiopic](https://unicode.org/charts/PDF/U1200.pdf)

---

**Last Updated:** 2025-01-XX  
**Maintainers:** IndexTTS Team
