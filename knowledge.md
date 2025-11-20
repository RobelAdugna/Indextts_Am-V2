# IndexTTS2 Knowledge Base

## Project Overview

IndexTTS2 is a state-of-the-art Text-to-Speech system supporting multiple languages including English, Chinese, Japanese, and now Amharic.

**🎯 PRIMARY INTERFACE: Amharic WebUI (`webui_amharic.py`)**

Always use the Amharic WebUI as your first option for any Amharic training tasks. It provides:
- ✅ Complete 8-tab pipeline automation
- ✅ Amharic-optimized defaults throughout
- ✅ Real-time progress tracking
- ✅ Error handling and validation
- ✅ State management across tabs

**Launch:** `python webui_amharic.py --share`

**Only use CLI tools directly when:**
- Debugging specific steps
- Automation/scripting requirements
- Remote environments without browser access

## Language Support

### Supported Languages
- English (en)
- Chinese (zh, cn)
- Japanese (ja, jp)
- Amharic (am, amh) - **NEW** (Most complete implementation)

### Implementation Quality
- **Amharic:** Most complete with full automation, comprehensive docs, better tooling, enterprise-grade quality controls
- **Japanese:** Reference implementation, less automation, dodged vocab mismatch bug by staying at 12k tokens
- **English/Chinese:** Base model languages

**Critical Discovery:** Amharic implementation is SUPERIOR to Japanese in every aspect (quality filtering, error handling, automation) except it exposed a critical bug in the training script that Japanese avoided by using 12k vocab instead of extending to 24k. The bug has been fixed with gradient hooks.

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
- ፡ (word separator) → " " (space) **[CRITICAL - Added to fix tokenization]**

### Normalization
- Use NFC (not NFKC) for proper character composition
- Duration ratio: 1.0 (similar to English)
- Each character counts as 1 syllable

## Training Pipeline

### Dataset Size Considerations

**Your Dataset: 200 hours (Medium-Scale)**

**Key Points:**
- ✅ Enough data for high quality
- ⚠️ Overfitting is a real concern (not big data)
- 🎯 Needs careful regularization and monitoring

**Specific Recommendations:**
1. **Epochs:** 2-3 (NOT 10!) - prevents memorization
2. **Validation:** Every 500 steps - catches overfitting early
3. **Checkpoints:** Keep top 5 - allows best model selection
4. **Early Stopping:** Stop at 60k-70k steps if val loss plateaus
5. **Weight Decay:** 1e-5 (L2 regularization active)
6. **Learning Rate:** 5e-5 (conservative for extended vocab)
7. **Validation Split:** 3% (~6 hours) for 200hr datasets, minimum 10 samples for reliable metrics

**See `TRAINING_200HR_OPTIMIZATIONS.md` for complete guide!**

**Critical Validations (FIXED 2025-01):**
- ✅ Base vocab size auto-detected from checkpoint (not hardcoded 12000)
- ✅ Tokenizer path validated on resume (prevents wrong token mappings)
- ✅ Val split quality checked (warns if actual != expected)
- ✅ Pair distribution analyzed (detects speaker imbalance)
- ✅ Tokenizer extension size verified (warns if target not reached)

### CRITICAL: Extended Vocabulary Training Fix

**Problem:** When using extended tokenizers (e.g., Amharic 24k tokens vs base 12k), new tokens (12000-23999) are randomly initialized but base model only has pretrained embeddings for 0-11999. This causes:
- High loss values (mel_loss=4.5-4.8, text_loss=4.5) that plateau
- Model produces nonsense speech for new language
- Training appears stuck after ~10k steps

**Root Cause:** In `build_model()`, only first 12k embedding weights are copied from checkpoint. New language tokens 12000-23999 remain randomly initialized, causing model to see random noise for all new language text.

**Fix Applied:** `trainers/train_gpt_v2.py` now automatically:
1. Detects extended vocabularies (vocab_size > 12000)
2. Freezes base token embeddings (0-11999) via gradient hooks
3. Allows only new tokens (12000+) to train
4. Prints diagnostic info on startup

**Expected Results After Fix:**
- Loss should drop steadily (not plateau)
- Within 10k steps: text_loss ~2.5-3.0, mel_loss ~3.0-3.5
- At 30k steps: text_loss ~1.8-2.2, mel_loss ~2.0-2.5
- Intelligible speech from new language

**Recommended Settings for Extended Vocab:**

**For Small Datasets (<50 hours):**
```bash
python trainers/train_gpt_v2.py \
  --learning-rate 5e-6 \              # Very conservative
  --text-loss-weight 0.4 \            # Higher text weight
  --mel-loss-weight 0.6 \
  --warmup-steps 2000 \
  --epochs 5-10
```

**For Medium Datasets (50-300 hours):** ✅ **YOUR CASE (200hr)**
```bash
python trainers/train_gpt_v2.py \
  --learning-rate 5e-5 \              # Conservative but effective
  --text-loss-weight 0.3 \            # Balanced for quality
  --mel-loss-weight 0.7 \
  --warmup-steps 4000 \
  --val-interval 500 \                # Frequent validation
  --keep-checkpoints 5 \              # More selection options
  --epochs 2-3                         # Avoid overfitting!
```

**For Large Datasets (>300 hours):**
```bash
python trainers/train_gpt_v2.py \
  --learning-rate 1e-4 \              # Can go higher
  --text-loss-weight 0.2 \
  --mel-loss-weight 0.8 \
  --warmup-steps 8000 \
  --epochs 3-5
```

**See `AMHARIC_TRAINING_FIX.md` for complete analysis and diagnostics.**

### CRITICAL: Inference Vocab Size Bug (FIXED 2025-01)

**Problem:** Extended vocab models (24k tokens) produce nonsense speech during inference even though training works correctly.

**Root Cause:** 
- Tokenizer: 24000 tokens
- Model reserves STOP_TEXT_TOKEN at position `vocab_size + 1`
- Training correctly uses 24001 embeddings (24000 + 1 STOP)
- **Inference bug:** Loads 24001 from checkpoint, then adds +1 again → creates 24002 embeddings
- Reshaping from 24001→24002 scrambles learned Amharic weights (IDs 12000-23999)

**Fix Applied:** `indextts/infer_v2_modded.py` now subtracts 1 when setting config from checkpoint, preventing double-counting of STOP token.

**Evidence of bug:** Inference log showed `"Reshaping from [24001, 1280] to [24002, 1280]"` - this scrambled the embeddings!

**See `INFERENCE_VOCAB_FIX.md` for complete details.**

### Resume Training

**Feature:** Automatically resume interrupted training from checkpoints

**CLI Usage:**
```bash
# Auto-resume (uses output_dir/latest.pth)
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --resume auto

# Resume from specific checkpoint
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/train_pairs.jsonl \
  --val-manifest processed_data/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model \
  --resume training_output/model_step5000.pth
```

**WebUI Usage (Tab 6):**
1. **Choose preset** (200hr-optimized): Balanced/Conservative/Aggressive
2. **Or customize:** LR (default 5e-5), Batch (0=auto), Epochs (default 3)
3. **Resume:** Check box + leave path empty for auto-resume
4. Click "🚀 Start Training"

**New in WebUI (2025-01):**
- ✅ 200hr-optimized defaults (3 epochs, LR 5e-5, auto-batch)
- ✅ Quick presets (Balanced/Conservative/Aggressive)
- ✅ Integrated optimizations (val_interval=500, keep_ckpts=5, warmup=4k)
- ✅ Monitoring guide (TensorBoard, expected metrics, stop criteria)
- ✅ All CLI optimizations auto-injected

**What Gets Restored:**
- ✅ Model weights
- ✅ Optimizer state (momentum, etc.)
- ✅ Learning rate scheduler
- ✅ Gradient scaler (AMP)
- ✅ Training step counter
- ✅ Epoch counter
- ✅ Recent checkpoint list

**A100 GPU Benefits (80GB Model):**
- 80GB VRAM allows batch_size=32 (maximum throughput)
- Supports bfloat16 AMP (no gradient scaling needed)
- TF32 matmul acceleration (3-8× speedup)
- 12 CPUs enable 12-24 data workers for faster loading
- Enhanced TFLOPs for bfloat16/float16 operations
- Fastest training configuration available

**Troubleshooting:**
- If checkpoint not found, check `--output-dir` matches previous run
- If incompatible checkpoint, ensure same tokenizer/config
- If OOM after resume, reduce `--batch-size`
- **CRITICAL: Vocab mismatch breaks training!** If checkpoint has different vocab size than current tokenizer (even 1 token difference), optimizer state becomes incompatible. Symptoms: losses completely stuck, no learning. Fix: Code detects mismatch and skips incompatible optimizer state automatically (uses fresh optimizer but preserves model weights). See `TRAINING_STUCK_FIX_COMPLETE.md` and `VOCAB_MISMATCH_FIX.md` for details.

**CRITICAL for Extended Vocabularies (Amharic, Korean, Arabic, etc.):**
- ✅ Resume works with gradient hook fix (hooks re-register automatically)
- ❌ **Don't resume from pre-fix checkpoints** - optimizer state is corrupted from training with random embeddings
- ✅ **Start fresh after applying fix** - much faster to good results
- ✅ **Future resumes work perfectly** - once trained with fix, resume is seamless
- See `RESUME_TRAINING_WITH_FIX.md` for detailed explanation

**Epoch Tracking on Resume (FIXED 2025-01):**
- Bug: Epoch incremented on every resume, causing epoch number to be incorrect
- Fix: Checkpoint saves NEXT epoch/batch to resume from. Handles epoch boundaries correctly (when next_batch >= len(loader), increments epoch and resets batch to 0)
- Video Best Practice: Keep last 3 checkpoints (every 1000 steps), validate every 1000 steps
- Result: Bulletproof continuity - interrupt/resume works perfectly at any point (mid-epoch or epoch boundary)

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

4. **Tokenizer Extension** 🔤
   - **CRITICAL:** Extend base tokenizer, DON'T train from scratch!
   - Uses `tools/tokenizer/extend_bpe.py` (matches video at 20:05-30:00)
   - Preserves base English/Chinese tokens (IDs 0-11999)
   - Adds Amharic tokens (IDs 12000-23999)
   - **Target size:** 24,000 (Amharic default)
   - **Character coverage:** 0.9999 (high for Ethiopic script)
   - **Input:** Base model + manifest.jsonl (NOT corpus.txt!)
   - **WebUI:** Tab 4 with Amharic-optimized defaults

5. **Preprocessing**
   - Extract semantic features (GPU-accelerated)
   - Tokenize text
   - Generate conditioning vectors (GPU-accelerated)
   - Split train/validation
   - **Performance:** Auto-detects batch size (L4 22GB→16, 24GB→32, V100→16) - trust auto-detection!
   - **GPU utilization:** Preprocessing is I/O-bound, not compute-bound. 30-60% GPU is normal.
   - **Models use 12-16GB:** Large pretrained models (SeamlessM4T, GPT) occupy most VRAM before batching

6. **Generate Prompt Pairs** 🔗
   - Create prompt-target combinations from same speaker
   - **Critical:** Enables voice cloning and emotion transfer
   - Pairs teach model to apply voice A to text B
   - **Required for GPT training** - model will fail without this step!
   - **Location:** Tab 5.5 in WebUI or `build_gpt_prompt_pairs.py`
   - **Default:** 2 pairs per target utterance

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

### Hardware Auto-Optimization (Universal)

**🎯 Automatic Hardware Detection:**

The training pipeline now **automatically detects** your hardware and optimizes settings:

- ✅ **GPU Detection**: VRAM-based batch size tuning
- ✅ **AMP dtype**: Auto-selects bfloat16 (Ampere+), float16 (older), or FP32 (CPU)
- ✅ **TF32**: Auto-enabled on Ampere/Ada/Hopper GPUs (3-8× matmul speedup)
- ✅ **CPU Workers**: Optimized based on CPU core count
- ✅ **cuDNN**: Auto-tuning enabled for all CUDA GPUs

**Supported Hardware:**

| GPU VRAM | Batch Size | Grad Accum | Effective | AMP dtype | Workers |
|----------|------------|------------|-----------|----------|----------|
| 80GB (A100 80GB) | 64 | 1 | 64 | bfloat16 | 12-24 |
| 40GB+ (A100 40GB, H100) | 16 | 2 | 32 | bfloat16 | 16 |
| 24GB (L4, 3090, 4090) | 8 | 4 | 32 | bfloat16 | 8-16 |
| 16GB (V100, 4080) | 6 | 6 | 36 | float16/bfloat16 | 8 |
| 12GB (3060, T4) | 4 | 8 | 32 | float16 | 4-8 |
| 8GB (3050) | 2 | 16 | 32 | float16 | 4 |
| CPU only | 1 | 32 | 32 | float32 | 2-4 |

**Simple Command (Auto-Optimized):**
```bash
python trainers/train_gpt_v2.py \
  --train-manifest preprocessed/train_pairs.jsonl \
  --val-manifest preprocessed/val_pairs.jsonl \
  --tokenizer tokenizers/amharic_extended_bpe.model
  # All settings auto-detected! 🎉
```

**Manual Override (if needed):**
```bash
python trainers/train_gpt_v2.py \
  --batch-size 8 \         # Override auto-detection
  --grad-accumulation 4 \  # Override auto-detection
  --num-workers 8 \        # Override auto-detection
  --amp                    # Force enable AMP
```

**Hardware Detection Test:**
```bash
python -m indextts.utils.hardware_optimizer
# Shows detected hardware + optimal settings
```

**Expected Performance Examples:**

**L4 GPU (24GB VRAM, 8 vCPUs):**
- Preprocessing: 8-12 hours
- Training: 2-3 days (200hr dataset)
- Settings: batch=8, grad_accum=4, workers=8, bfloat16

**A100 80GB GPU (80GB VRAM, 12 CPUs):**
- Preprocessing: 2-4 hours
- Training: 0.75-1 day (200hr dataset)
- Settings: batch=64, grad_accum=1, workers=12, bfloat16
- VRAM usage: 50-60GB (optimal utilization)
- Peak throughput: ~3-4× faster than L4 GPU

**A100 40GB GPU (40GB VRAM):**
- Preprocessing: 3-6 hours
- Training: 1.5-2 days (200hr dataset)
- Settings: batch=16, grad_accum=2, workers=16, bfloat16

**V100 GPU (16GB VRAM):**
- Preprocessing: 8-16 hours
- Training: 3-4 days (200hr dataset)
- Settings: batch=6, grad_accum=6, workers=8, float16

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

## Preprocessing

### Resume Capability
**Feature:** Automatic checkpoint/resume for interrupted preprocessing runs
**How it works:**
- Progress saved to `.preprocessing_progress.txt` after each sample
- Automatically skips already-processed samples on restart
- Handles crashes, OOM errors, and Ctrl+C gracefully
- Flushes manifest files immediately (no data loss)

**Usage:**
```bash
# Start preprocessing
python tools/preprocess_data.py --manifest dataset/manifest.jsonl --output-dir preprocessed --tokenizer tokenizers/bpe.model --language am

# If interrupted/crashed, just rerun same command - it will resume automatically!
```

**Manual cleanup (if needed):**
```bash
# Remove checkpoint to start fresh
rm preprocessed/.preprocessing_progress.txt
```

### Dynamic OOM Handling (Auto-Recovery)
**Feature:** Intelligent OOM detection and automatic batch size adjustment
**How it works:**
- Detects GPU out-of-memory errors during preprocessing
- Automatically reduces batch size by 50% and retries
- Falls back to single-sample processing if needed
- Clears GPU cache between batches for memory hygiene
- Continues processing without manual intervention

**Example:**
- Start: batch_size=16 (24GB GPU)
- OOM detected → reduce to 8
- OOM again → reduce to 4
- OOM again → reduce to 2
- OOM again → process one sample at a time

**Benefits:**
- No manual `--batch-size` tuning needed
- Automatically finds optimal batch size for your GPU
- Prevents preprocessing failures from OOM
- Maximizes throughput while staying within VRAM limits

**Conservative Starting Points (accounting for 12-16GB model overhead):**
- **40GB+ (A100):** batch_size=24
- **24GB (L4, RTX 3090/4090):** batch_size=16
- **16GB (V100, RTX 4080):** batch_size=8
- **12GB (T4, RTX 3060):** batch_size=6
- **8GB (RTX 3050):** batch_size=4

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
- **Auto-cleanup:** Deletes source audio+srt files after successful processing (use `--keep-source` to disable)
- **Temp cleanup:** Automatically removes instrumental files and temp directories

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

### Music Removal GPU Acceleration (Enhanced)
**Auto-Detection:** Both `youtube_amharic_downloader.py` and `process_dataset_segments.py` now use `get_optimal_mdx_batch_size()` from `hardware_optimizer.py`:
- **24GB+ (L4, 3090, 4090):** batch_size=16 (maximum throughput)
- **16GB (V100, 4080):** batch_size=12
- **12GB (T4, 3060):** batch_size=8
- **8GB (3050):** batch_size=6
- **<8GB or CPU:** batch_size=4 or 1

**CLI Flags:**
- `--mdx-batch-size N` - Override auto-detection (not recommended, may cause OOM)
- `--no-autocast` - Disable mixed precision if issues occur

**Installation:**
- GPU: `pip install audio-separator` (auto-detects CUDA)
- For ONNX GPU acceleration: `pip uninstall onnxruntime -y && pip install onnxruntime-gpu`
- CPU-only: `pip install 'audio-separator[cpu]'`

**Performance:** GPU is 50-100x faster than CPU. Auto-batch sizing maximizes GPU utilization without manual tuning.

**Implementation:** Shared utility in `indextts/utils/hardware_optimizer.py` eliminates code duplication and ensures consistent behavior.

### NumPy 2.x Compatibility Error
**Error:** `ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.4`
**Fix:** `pip install 'numpy<2'`
**Added to pyproject.toml:** numpy<2 constraint

### Protobuf Version Error (TensorBoard)
**Error:** `TypeError: Descriptors cannot be created directly... your generated code is out of date and must be regenerated with protoc >= 3.19.0`
**Cause:** TensorBoard incompatibility with protobuf 4+
**Fix:** `pip install 'protobuf<4.0.0'` or `pip install 'protobuf==3.20.3'`
**Why:** TensorBoard's .proto files compiled with protobuf 3.x, incompatible with 4+ API

### QwenEmotion Model Missing Error
**Error:** `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'checkpoints/qwen0.6bemo4-merge/'`
**OR:** `TypeError: join() argument must be str, bytes, or os.PathLike object, not 'NoneType'`
**Cause:** Config references optional QwenEmotion model that doesn't exist
**Fix:** Two changes required:
  1. Edit `checkpoints/config.yaml` and set `qwen_emo_path: null`
  2. Updated `indextts/infer_v2_modded.py` to handle null path gracefully
**Why:** QwenEmotion is an optional emotion model for text-based emotion control. When disabled, you can still use emotion reference audio or emotion vectors

### WebUI Parallel Not Finding Trained Checkpoints
**Problem:** webui_parallel.py dropdowns don't show trained checkpoints or custom tokenizers
**Cause:** Discovery functions only scan checkpoints/ and models/, not trained_ckpts/ or tokenizers/
**Fix:** Updated _discover_gpt_checkpoints() and _discover_bpe_models() to include:
  - `trained_ckpts/` directory for GPT checkpoints
  - Project-level `tokenizers/` directory for BPE models
**Result:** All checkpoints now appear in dropdowns automatically

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

## BPE Tokenizer Extension (Video Workflow - CRITICAL!)

**Source:** IndexTTS2 video at timestamp 20:05-30:00
**Status:** ✓ Implemented correctly in extend_bpe.py

### Why Extension Not Training?

**Extension Approach (CORRECT - from video):**
```bash
python tools/tokenizer/extend_bpe.py \
  --base-model checkpoints/bpe.model \
  --manifests dataset/manifest.jsonl \
  --output-model tokenizers/amharic_extended_bpe.model \
  --target-size 24000
```

**Benefits:**
1. ✅ Preserves base token IDs → maintains cross-lingual transfer
2. ✅ Base model already knows English/Chinese patterns
3. ✅ Only adds new Amharic-specific tokens
4. ✅ Smaller final model (no redundant tokens)
5. ✅ Matches official IndexTTS2 multilingual approach

**How It Works:**
1. Loads base tokenizer (12k tokens: English/Chinese)
2. Trains temporary tokenizer on Amharic manifest
3. Extracts new Amharic-only tokens (not in base)
4. Appends ~12k Amharic tokens to base
5. Final vocab: 24k tokens (12k base + 12k Amharic)

**Token ID Layout:**
- IDs 0-11999: Base (English/Chinese) - **PRESERVED**
- IDs 12000-23999: Amharic (new) - **ADDED**

**DO NOT train from scratch** - you'll lose cross-lingual capability!

## Amharic WebUI (PRIMARY INTERFACE)

### Overview
**🎯 This is your #1 tool for all Amharic training!**

A comprehensive Gradio web interface (`webui_amharic.py`) integrates all Amharic-specific tools into a single, user-friendly pipeline:

### Troubleshooting Tab 4 (Tokenizer Extension)

**Common Errors:**
- **Empty manifest path:** Complete Tab 2 (Dataset Creation) first - the manifest auto-fills from there
- **Base model not found:** Run `download_requirements.bat` (Windows) or `download_requirements.sh` (Linux/Mac) to download checkpoints
- **extend_bpe.py not found:** Ensure `tools/tokenizer/extend_bpe.py` exists in your project
- **Unclear error messages:** Check the "Extension Logs" tab - it now shows:
  - Full command being run
  - Complete stdout/stderr output
  - Exit code
  - Full Python traceback for exceptions

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
5.5. **Pairs - Generate prompt-target pairs (CRITICAL STEP!)** 🔗
6. Training - GPT fine-tuning launcher (uses paired manifests)
7. Post-Process - Batch noise removal for existing datasets
8. Inference - Links to existing TTS UIs

**Remote Environment Support (Lightning AI, etc.):**
- Tab 3 (Corpus Collection) supports direct file paths ✅
- No need to upload files from local PC
- Just paste the full path: `/teamspace/studios/this_studio/amharic_dataset/manifest.jsonl`
- Supports `~` expansion for home directory paths
- Priority: Direct path → File upload → Auto-fill from pipeline
- Amharic validation: Requires ≥50% Ethiopic characters (U+1200-137F ranges)

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

## Hardware Auto-Optimization

**New Feature:** Automatic hardware detection and optimization!

- See `HARDWARE_AUTO_OPTIMIZATION.md` for complete guide
- See `L4_OPTIMIZATIONS.md` for L4-specific details
- Training/preprocessing now auto-tune based on your GPU/CPU
- No manual configuration needed - just use default settings!

**Key improvement:** Training is now 2-5× faster with zero configuration.

## Resources

### Documentation
- See `docs/` directory for guides
- Check `examples/` for usage examples
- Read `AMHARIC_IMPLEMENTATION_*.md` for details
- See `README_AMHARIC_WEBUI.md` for WebUI usage
- See `HARDWARE_AUTO_OPTIMIZATION.md` for hardware optimization

### External References
- [IndexTTS2 Paper](https://arxiv.org/abs/2506.21619)
- [SentencePiece](https://github.com/google/sentencepiece)
- [Unicode Ethiopic](https://unicode.org/charts/PDF/U1200.pdf)

---

**Last Updated:** 2025-01-XX  
**Maintainers:** IndexTTS Team
