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

### Subtitle Pairing for Separated Files

**Problem:** After vocal separation, files are renamed (e.g., `video_(Vocals)_UVR_MDXNET.wav`) but subtitle files keep original names
**Solution:** WebUI automatically copies subtitle files to match separated vocal filenames
**Manual Fix:** For already-separated files, use:
```bash
python tools/fix_vocal_subtitles.py --vocal-dir amharic_vocals --original-dir amharic_downloads
```
**Dry Run:** Add `--dry-run` flag to preview without copying

## Text Deduplication

**Problem:** SRT subtitles have rolling text (50% overlap between lines)
**Solution:** Auto-detects 3+ word overlap, removes from current segment
**Toggle:** Dataset Creation tab checkbox (enabled by default)
**Result:** Clean, unique text per segment while audio stays aligned

**Critical Fixes (2025-01):** 
1. Deduplication compared with last ADDED segment instead of previous INPUT segment, causing overlaps when segments were skipped
2. Text cleaning happened AFTER deduplication, allowing cleaned text to reintroduce overlaps
**Solution:** Clean text BEFORE dedup, and compare with previous INPUT segment. See `DEDUPLICATION_FIX.md` for details.

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

### Incremental Dataset Expansion (--append mode)

**Problem:** Want to add new data without re-processing existing dataset
**Solution:** Use `--append` flag to continue from last segment number

**How it works:**
1. Reads existing `manifest.jsonl` to find last segment ID
2. Continues numbering from next number (e.g., after `spk000_003455` → starts at `spk000_003456`)
3. Appends new entries to manifest (doesn't overwrite)
4. All existing audio files remain untouched

**Example:**
```bash
# First dataset creation
python tools/create_amharic_dataset.py \
  --input-dir downloads_batch1 \
  --output-dir amharic_dataset \
  --single-speaker
# Creates: spk000_000001.wav through spk000_003455.wav

# Add more data later (append mode)
python tools/create_amharic_dataset.py \
  --input-dir downloads_batch2 \
  --output-dir amharic_dataset \
  --append \
  --single-speaker
# Creates: spk000_003456.wav through spk000_005234.wav
# Appends to existing manifest.jsonl
```

**Important:**
- Use same `--single-speaker` or multi-speaker mode as original
- Use same `--output-dir` as existing dataset
- Manifest path auto-detected or specify with `--manifest`
- Safe to run multiple times (won't duplicate existing files)

## Automated Checkpoint Download

**Problem:** Missing base model checkpoints prevents training
**Solution:** Automatic download from HuggingFace

**Quick Setup (Windows):**
```bash
double-click download_requirements.bat
```

**Quick Setup (Linux/Mac):**
```bash
bash download_requirements.sh
```

**Manual Download:**
```bash
pip install huggingface-hub
python tools/download_checkpoints.py
```

**What Gets Downloaded:**
- `gpt.pth` - Base GPT model (~500MB-1GB)
- `bpe.model` - Base BPE tokenizer
- `s2mel.pth` - Semantic-to-mel model
- `wav2vec2bert_stats.pt` - Feature stats
- `feat1.pt` - Speaker embeddings
- `feat2.pt` - Emotion embeddings
- `config.yaml` - Model configuration

**Source:** HuggingFace repository `IndexTeam/IndexTTS-2`

**Automatic Integration:**
- End-to-end scripts check for checkpoints on startup
- Auto-downloads missing files before training
- Safe to interrupt and resume

## Common Issues & Solutions

### YouTube Downloader Enhancements

**URL Validation:**
- Validates YouTube URL format before attempting download
- Supports youtube.com/watch, youtu.be, embed, and /v/ formats
- Provides clear error messages for invalid URLs

**Subtitle Availability Check:**
- Pre-download subtitle checking with `--check-subs-first` (default: enabled)
- Checks for manual and auto-generated subtitles in requested languages
- Skips videos without subtitles to save bandwidth and storage
- Use `--no-check-subs` to disable and download all videos

**Post-Download Cleanup:**
- Automatically removes audio files without matching subtitle pairs
- Searches for exact matches and language-coded subtitles (e.g., `video.am.srt`)
- Also deletes associated .info.json files
- Use `--no-cleanup` to disable

**Temporary Folder Cleanup:**
- Cleans up temp folders after batch downloads complete
- Searches for common temp patterns: temp, tmp, .temp, .tmp, *_temp, *_tmp
- Frees up disk space automatically
- Use `--no-cleanup-temp` to keep temp files

**Background Noise Removal:**
- Integrates audio-separator for vocal extraction
- Removes background music/instruments before dataset creation
- Use `--remove-noise` flag to enable
- Default model: UVR-MDX-NET-Inst_HQ_4.onnx (good balance)
- Requires: `pip install audio-separator`
- Replaces original files with noise-free versions

**Usage Examples:**
```bash
# Basic download with subtitle checking
python tools/youtube_amharic_downloader.py --url-file urls.txt

# Download with noise removal
python tools/youtube_amharic_downloader.py --url-file urls.txt --remove-noise

# Download without cleanup (keep all files)
python tools/youtube_amharic_downloader.py --url-file urls.txt --no-cleanup --no-cleanup-temp

# Skip subtitle check (download all videos)
python tools/youtube_amharic_downloader.py --url-file urls.txt --no-check-subs
```

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
**Fix:** Newer versions (>=0.20.0) auto-detect CUDA. Remove `use_cuda=` parameter.
**Note:** Install base package without `[cpu]` extra for GPU support.

### NumPy 2.x Compatibility Error
**Error:** `ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.4`
**Fix:** `pip install 'numpy<2'`
**Added to pyproject.toml:** numpy<2 constraint

### Filename Too Long Error
**Error:** `OSError: [Errno 36] File name too long`
**Cause:** YouTube video titles with emojis, special chars create 250+ character filenames
**Fix:** Automatically implemented - tool skips files with paths >250 chars
**Warning:** "Filename too long (>250 chars), skipping: ..."
**Solution:** Rename downloaded files to shorter names before processing, or use shorter output paths

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
3. Corpus - Text aggregation and cleaning (supports direct file paths for remote environments)
4. Tokenizer - BPE model training
5. Preprocess - Feature extraction
6. Training - GPT fine-tuning launcher
7. Process Segments - Batch noise removal for existing datasets
8. Inference - Links to existing TTS UIs

**Remote Environment Support (Lightning AI, etc.):**
- Tab 3 (Corpus Collection) supports direct file paths
- No need to upload files from local PC
- Just paste the full path: `/teamspace/studios/this_studio/amharic_dataset/manifest.jsonl`
- Works with both file paths and directory paths

### Usage Pattern
- Each tab auto-fills from previous step's output
- Can skip steps if intermediate files exist
- Progress tracked in real-time
- Logs displayed in UI

### Incremental Dataset Expansion in WebUI

**Location:** Tab 2 "Dataset Creation"
**Checkbox:** "📝 Append to Existing Dataset"

**How to use:**
1. Download new content (Tab 1) to a separate folder
2. Go to Tab 2
3. Set input directory to new downloads
4. Set output directory to existing dataset
5. **Check** "Append to Existing Dataset" checkbox
6. Click "Create Dataset"

**What happens:**
- Auto-detects last segment number (e.g., spk000_003455)
- Continues numbering from next (e.g., spk000_003456)
- Appends new entries to manifest.jsonl
- Shows existing vs new counts in status
- All existing files remain untouched

**Important:**
- Use same "Single Speaker Mode" setting as original
- Point to existing dataset's output directory
- New files are in separate input directory

See `README_AMHARIC_WEBUI.md` for complete documentation.

## Post-Processing Dataset Segments

**Feature:** Remove background music/noise from already-created dataset segments
**Location:** Tab 7 "Process Segments" in webui_amharic.py
**CLI Tool:** `tools/process_dataset_segments.py`

### Use Cases
- Clean up existing datasets without re-segmenting
- Apply noise removal after dataset creation
- Improve audio quality of legacy datasets
- Process datasets created before noise removal was available

### Key Features

**In-Place Processing:**
- Replaces original audio files with noise-removed versions
- Maintains exact filenames (e.g., `spk000_000001.wav` stays `spk000_000001.wav`)
- Manifest.jsonl remains valid (only audio content changes)
- No need to regenerate features or retrain

**Resume Capability:**
- Automatically saves progress every N files (configurable)
- Can safely interrupt and resume later
- Progress tracked in `.noise_removal_progress.json`
- Skips already-processed files on resume

**Safety Options:**
- Optional backup of original files (`.backup` extension)
- Validates all segments processed successfully
- Reports errors for failed segments

**Input Modes:**
1. **From Manifest** (Recommended):
   - Reads `manifest.jsonl` to find dataset directory
   - Processes all files in `dataset/audio/` directory
   - Automatically determines audio location

2. **Audio Directory**:
   - Directly specify audio segment directory
   - Useful for custom directory structures

### WebUI Usage

1. Navigate to Tab 7 "Process Segments"
2. Select input source (manifest or directory)
3. Choose noise removal model:
   - **UVR-MDX-NET-Inst_HQ_4.onnx**: Fast, high quality (recommended)
   - **Demucs**: Slower, best quality
4. Configure options:
   - Keep backup files (optional)
   - Resume from previous run (recommended)
   - Progress save interval (default: 10 files)
5. Click "Process Dataset Segments"

### CLI Usage

```bash
# Process from manifest
python tools/process_dataset_segments.py \
  --manifest amharic_dataset/manifest.jsonl \
  --model UVR-MDX-NET-Inst_HQ_4.onnx \
  --keep-backup

# Process audio directory directly
python tools/process_dataset_segments.py \
  --audio-dir amharic_dataset/audio \
  --model UVR-MDX-NET-Inst_HQ_4.onnx

# Resume after interruption (default behavior)
python tools/process_dataset_segments.py \
  --manifest amharic_dataset/manifest.jsonl

# Start fresh (ignore previous progress)
python tools/process_dataset_segments.py \
  --manifest amharic_dataset/manifest.jsonl \
  --no-resume
```

### Models

- **UVR-MDX-NET-Inst_HQ_4.onnx**: Balanced speed/quality (recommended for large datasets)
- **UVR_MDXNET_KARA_2.onnx**: Alternative MDX-Net variant for karaoke-style separation
- **htdemucs**: Best quality, slower processing

### Important Notes

1. **GPU Acceleration**: Automatically detects and uses GPU if available (much faster)
   - Use `--mdx-batch-size 8` for GPUs with 16GB+ VRAM (default: 4)
   - Use `--no-autocast` to disable mixed precision if experiencing issues
   - Use `--normalization 0.9` to normalize audio levels (default: 0.9)
2. **Backup Recommended**: Use `--keep-backup` for first run to preserve originals
3. **Progress Tracking**: Progress saved in dataset directory as `.noise_removal_progress.json`
4. **Manifest Compatibility**: Original manifest.jsonl remains valid after processing
5. **Memory Efficient**: Processes files sequentially with GPU optimization
6. **Safe Interruption**: Can stop (Ctrl+C) and resume anytime
7. **Chunk Processing**: `--chunk-size` groups files for progress tracking (not parallel processing)

### Workflow

**Typical workflow for improving existing dataset:**
1. Create dataset using standard pipeline (Tabs 1-2)
2. Train initial model to test quality
3. If background noise is an issue, run Tab 7 to clean segments
4. No need to re-preprocess (features unchanged)
5. Can resume training or train new model

**When to use:**
- Dataset has noticeable background music
- Audio quality inconsistent across segments
- Want to improve existing dataset without re-collecting data
- Legacy datasets created before noise removal was available

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
