# Amharic TTS Pipeline WebUI

Comprehensive Gradio interface for the complete Amharic TTS training pipeline in IndexTTS2.

## Features

🎯 **Complete End-to-End Pipeline:**
- **Download** - Collect Amharic content from YouTube
- **Dataset Creation** - Segment audio using subtitles with silence detection
- **Corpus Collection** - Clean and aggregate Amharic text
- **Tokenizer Training** - Train multilingual BPE models
- **Preprocessing** - Extract semantic features and conditioning vectors
- **Training** - Fine-tune GPT models (launcher)
- **Inference** - Links to existing TTS generation UIs

✨ **Key Highlights:**
- Modern, clean, organized interface with emoji navigation
- Sequential tab-based workflow with auto-fill from previous steps
- Real-time progress tracking for all operations
- Comprehensive error handling and status messages
- Automatic dependency checking (yt-dlp, ffmpeg, CUDA)
- Pipeline state management across all stages
- Color-coded status indicators (green for success, red for errors)

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install gradio sentencepiece librosa soundfile tqdm

# Install yt-dlp (for YouTube downloads)
pip install yt-dlp

# Ensure ffmpeg is installed (for audio processing)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

### 2. Launch the WebUI

```bash
python webui_amharic.py
```

The interface will be available at http://localhost:7863

### 3. Optional Arguments

```bash
python webui_amharic.py --port 8000     # Custom port
python webui_amharic.py --share         # Create public link
python webui_amharic.py --host 127.0.0.1  # Localhost only
```

## Usage Guide

### Tab 1: Download YouTube Videos

**Purpose:** Download Amharic content from YouTube with audio and subtitles.

**Steps:**
1. Paste YouTube URLs (one per line) or upload a text file with URLs
2. Set output directory (default: `amharic_downloads`)
3. Configure subtitle options:
   - Enable/disable subtitle download
   - Specify subtitle languages (default: `am en amh`)
4. Choose audio format (wav, mp3, flac)
5. Click "🔽 Download Videos"

**Output:** Audio files + subtitle files (SRT/VTT) in the output directory

**Tips:**
- Create a file with one URL per line for batch downloads
- Use `am` for Amharic, `en` for English subtitles
- WAV format recommended for best quality
- Each video has a 10-minute timeout

### Tab 2: Create Dataset

**Purpose:** Segment audio files using subtitle timestamps with boundary refinement.

**Steps:**
1. Input directory auto-fills from download step (or specify manually)
2. Set output directory (default: `amharic_dataset`)
3. Configure duration filters:
   - Minimum duration: 1.0s (recommended)
   - Maximum duration: 30.0s (recommended)
4. Enable boundary refinement for better segmentation
5. Click "🎵 Create Dataset"

**Output:** 
- `manifest.jsonl` - Dataset manifest with metadata
- `audio/` directory - Segmented WAV files

**Tips:**
- Boundary refinement uses silence detection for precise cuts
- Duration filters help remove very short or very long segments
- Process takes ~1 minute per hour of audio

### Tab 3: Collect Corpus

**Purpose:** Aggregate and clean Amharic text for tokenizer training.

**Steps:**
1. Upload JSONL/text files or use manifest from previous step
2. Set output corpus file (default: `amharic_corpus.txt`)
3. Configure minimum text length (default: 5 characters)
4. Enable character statistics for analysis
5. Click "📝 Collect Corpus"

**Output:** Clean Amharic text corpus file (one sentence per line)

**Tips:**
- Automatically validates Amharic content (min 50% Amharic characters)
- Removes duplicates and normalizes text
- Statistics help verify character coverage

### Tab 4: Train Tokenizer

**Purpose:** Train multilingual BPE tokenizer with Amharic support.

**Steps:**
1. Upload corpus files or use from previous step
2. Set model prefix (default: `amharic_bpe`)
3. Configure parameters:
   - Vocabulary size: 32000 (recommended for Amharic)
   - Character coverage: 0.9995 (high for Amharic script)
4. Optionally add test text for validation
5. Click "🔤 Train Tokenizer"

**Output:** 
- `{prefix}.model` - SentencePiece model
- `{prefix}.vocab` - Vocabulary file
- Test tokenization result

**Tips:**
- Higher vocab size = better coverage, larger model
- Character coverage 0.9995+ recommended for Amharic's large character set
- Test with Amharic text to verify tokenization quality
- Training takes 5-30 minutes depending on corpus size

### Tab 5: Preprocess Data

**Purpose:** Extract semantic features and conditioning vectors for training.

**Steps:**
1. Paths auto-fill from previous steps
2. Verify manifest path, tokenizer model, and checkpoints
3. Configure:
   - Language: `am` (Amharic)
   - Validation ratio: 0.01 (1%)
   - Batch size: 4 (adjust based on GPU memory)
4. Click "⚙️ Preprocess Data"

**Output:**
- `train_manifest.jsonl` & `val_manifest.jsonl` - Split manifests
- `codes/` - Semantic codes (.npy files)
- `condition/` - Conditioning vectors
- `emo_vec/` - Emotion vectors
- `text_ids/` - Tokenized text

**Tips:**
- GPU strongly recommended (CPU is very slow)
- Batch size 4 suitable for 8GB VRAM
- Processing takes ~5 seconds per sample
- Monitor terminal for detailed progress

### Tab 6: Training

**Purpose:** Launch GPT model fine-tuning.

**Steps:**
1. Paths auto-fill from preprocessing
2. Configure training parameters:
   - Learning rate: 1e-5 (recommended)
   - Batch size: 8
   - Epochs: 10
3. Click "🚀 Start Training"

**Output:** Training starts in background

**Monitoring:**
```bash
tensorboard --logdir training_output
```

**Tips:**
- Training runs in background - check terminal for progress
- Use TensorBoard for metrics visualization
- Checkpoints saved every epoch
- Training time: ~1-3 days for 100+ hours of data

### Tab 7: Inference

**Purpose:** Generate speech with trained models.

**Use existing WebUIs:**

```bash
# Single generation
python webui.py --model_dir training_output

# Batch generation
python webui_parallel.py --model_dir training_output
```

## Pipeline State Management

The WebUI automatically tracks your progress across tabs:

**Overview Tab** shows current state:
```
✅ Downloads: amharic_downloads
✅ Dataset: amharic_dataset
✅ Corpus: amharic_corpus.txt
✅ Tokenizer: amharic_bpe.model
✅ Processed: processed_data
```

**Auto-fill behavior:**
- Dataset tab uses download directory
- Corpus tab uses dataset manifest
- Tokenizer tab uses corpus file
- Preprocessing uses dataset + tokenizer
- Training uses preprocessed manifests

## Troubleshooting

### Dependencies Not Found

**Symptom:** Red ❌ for yt-dlp, ffmpeg, or CUDA in system status

**Solution:**
```bash
# Install yt-dlp
pip install -U yt-dlp

# Install ffmpeg (varies by OS)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Verify CUDA
import torch
print(torch.cuda.is_available())
```

### Download Failures

**Symptom:** Videos fail to download

**Solutions:**
- Check internet connection
- Verify YouTube URLs are valid
- Some videos may have download restrictions
- Try updating yt-dlp: `pip install -U yt-dlp`

### Subtitle Files Not Detected

**Symptom:** "Warning: No subtitle found for video.wav"

**Cause:** yt-dlp downloads subtitles with language codes (e.g., `video.am.srt`, `video.en.srt`)

**Solution:** The dataset creator now automatically searches for subtitles with language codes. Supported patterns:
- `video.srt` (exact match)
- `video.am.srt` (Amharic)
- `video.en.srt` (English)
- `video.amh.vtt` (Amharic VTT)

If still not detected, manually rename subtitle files to match audio filenames exactly.

### Out of Memory

**Symptom:** Preprocessing crashes with CUDA OOM

**Solutions:**
- Reduce batch size (try 2 or 1)
- Close other GPU applications
- Use CPU mode (much slower): edit code to set device='cpu'

### Tokenizer UNK Tokens

**Symptom:** High percentage of unknown tokens

**Solutions:**
- Increase vocabulary size (e.g., 64000)
- Increase character coverage to 0.9999
- Add more diverse corpus data
- Check corpus has sufficient Amharic text

### Preprocessing Errors

**Symptom:** Missing files or invalid paths

**Solutions:**
- Verify all required files exist:
  - `checkpoints/config.yaml`
  - `checkpoints/gpt.pth`
  - `checkpoints/wav2vec2bert_stats.pt`
- Check manifest paths are correct
- Ensure tokenizer model was created successfully

## Advanced Features

### Skip Steps

You can start at any step if you have intermediate files:

1. **Start at Dataset Creation:** If you already have audio + subtitles
2. **Start at Corpus Collection:** If you have a dataset manifest
3. **Start at Tokenizer Training:** If you have a corpus file
4. **Start at Preprocessing:** If you have a tokenizer
5. **Start at Training:** If you have preprocessed data

### Custom Parameters

All parameters have sensible defaults but can be customized:

**Dataset Creation:**
- Adjust duration filters for your content
- Disable boundary refinement for speed

**Tokenizer:**
- Larger vocab for diverse content
- Lower vocab for simpler language

**Preprocessing:**
- Batch size based on GPU memory
- Val ratio for larger/smaller validation sets

### Batch Operations

For processing many files:

**Download:** Prepare URL file:
```
https://www.youtube.com/watch?v=video1
https://www.youtube.com/watch?v=video2
# Add as many as needed
```

**Corpus:** Upload multiple JSONL files in Tab 3

## Best Practices

### Data Quality

1. **YouTube Source Selection:**
   - News broadcasts (clear speech)
   - Podcasts (conversational)
   - Audiobooks (narrative)
   - Avoid music videos or noisy content

2. **Subtitle Quality:**
   - Manual subtitles > auto-generated
   - Verify synchronization
   - Check for transcription errors

3. **Corpus Diversity:**
   - Mix formal and informal speech
   - Various speakers and accents
   - Different topics

### Resource Management

1. **Disk Space:**
   - Downloads: ~100MB per video
   - Dataset: ~2x download size
   - Preprocessed: ~5x dataset size
   - Plan for 1TB+ for large projects

2. **GPU Memory:**
   - 8GB: Batch size 2-4
   - 16GB: Batch size 8-12
   - 24GB+: Batch size 16+

3. **Processing Time:**
   - Download: Depends on network
   - Dataset: ~1 min/hour audio
   - Tokenizer: 5-30 minutes
   - Preprocessing: ~5 sec/sample
   - Training: 1-3 days

## File Structure

After running the complete pipeline:

```
project/
├── amharic_downloads/          # Downloaded files
│   ├── video1.wav
│   ├── video1.srt
│   └── ...
├── amharic_dataset/            # Segmented dataset
│   ├── manifest.jsonl
│   └── audio/
│       ├── segment_001.wav
│       └── ...
├── amharic_corpus.txt          # Text corpus
├── amharic_bpe.model           # Tokenizer
├── amharic_bpe.vocab
├── processed_data/             # Features
│   ├── train_manifest.jsonl
│   ├── val_manifest.jsonl
│   ├── codes/
│   ├── condition/
│   ├── emo_vec/
│   └── text_ids/
└── training_output/            # Training checkpoints
    ├── checkpoint_epoch_1.pth
    └── ...
```

## Integration with Existing Tools

This WebUI wraps the following command-line tools:

1. `tools/youtube_amharic_downloader.py`
2. `tools/create_amharic_dataset.py`
3. `tools/collect_amharic_corpus.py`
4. `tools/train_multilingual_bpe.py`
5. `tools/preprocess_data.py`
6. `trainers/train_gpt_v2.py`

You can still use these directly via CLI if preferred.

## Comparison with CLI

**WebUI Advantages:**
- Visual progress tracking
- Automatic state management
- Parameter validation
- Error messages in UI
- No command-line knowledge needed

**CLI Advantages:**
- Scriptable/automatable
- Better for HPC/cluster environments
- More control over advanced options
- Easier to integrate with workflows

## Contributing

To add new features:

1. Add a new tab in `create_ui()`
2. Create handler function for the operation
3. Update pipeline state management
4. Add progress tracking with `gr.Progress()`
5. Include error handling

## License

Same as IndexTTS2 main project.

## Support

For issues:
1. Check troubleshooting section above
2. Verify all dependencies installed
3. Check terminal output for detailed errors
4. Refer to main IndexTTS2 documentation

---

**Happy Training! 🎙️**
