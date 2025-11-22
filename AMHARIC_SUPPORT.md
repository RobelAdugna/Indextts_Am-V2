# Amharic Language Support for IndexTTS2

This document provides comprehensive guidance for training IndexTTS2 with Amharic language support.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Training](#training)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Overview

Amharic is a Semitic language spoken in Ethiopia, written in the Ge'ez (Ethiopic) script. IndexTTS2 now supports Amharic through:

- Ge'ez script normalization and tokenization
- Amharic-specific text processing
- Syllable-based duration estimation (fidel system)
- Multilingual BPE tokenizer with Amharic support

## Quick Start

### Automated Pipeline

The easiest way to train an Amharic model is using our end-to-end script:

**Linux/Mac:**
```bash
cd index-tts2
bash scripts/amharic/end_to_end.sh
```

**Windows:**
```powershell
cd index-tts2
.\scripts\amharic\end_to_end.ps1
```

This script will:
1. Download Amharic YouTube content
2. Create dataset from audio + subtitles
3. Collect and clean text corpus
4. Train multilingual BPE tokenizer
5. Preprocess data
6. Generate prompt-target pairs
7. Train the GPT model

## Detailed Setup

### 1. Prepare URL List

Create or edit `examples/amharic_youtube_urls.txt` with Amharic YouTube URLs:

```text
https://www.youtube.com/watch?v=EXAMPLE1
https://www.youtube.com/watch?v=EXAMPLE2
```

Recommended sources:
- Amharic news channels
- Educational content
- Audiobooks
- Podcasts
- Public speeches

### 2. Download Content

```bash
uv run python tools/youtube_amharic_downloader.py \
    --url-file examples/amharic_youtube_urls.txt \
    --output-dir amharic_data/downloads \
    --subtitle-langs am en amh
```

This downloads:
- Audio in WAV format
- Subtitles in SRT format
- Metadata (duration, title, etc.)

### 3. Create Dataset

```bash
uv run python tools/create_amharic_dataset.py \
    --input-dir amharic_data/downloads \
    --output-dir amharic_data/raw_dataset \
    --manifest amharic_data/raw_dataset/manifest.jsonl
```

This script:
- Parses SRT/VTT subtitles
- Segments audio at subtitle boundaries
- Refines boundaries using silence detection
- Normalizes Amharic text
- Generates JSONL manifest

## Dataset Preparation

### Collecting Text Corpus

For tokenizer training, collect Amharic text:

```bash
uv run python tools/collect_amharic_corpus.py \
    --input amharic_data/raw_dataset/manifest.jsonl \
    --output amharic_data/amharic_corpus.txt \
    --stats
```

This:
- Extracts text from manifests
- Normalizes and cleans text
- Removes duplicates
- Filters non-Amharic content
- Shows character statistics

### Training Tokenizer

Train a BPE tokenizer with Amharic support:

```bash
uv run python tools/train_multilingual_bpe.py \
    --corpus amharic_data/amharic_corpus.txt \
    --model-prefix amharic_output/amharic_bpe \
    --vocab-size 32000 \
    --character-coverage 0.9999
```

This generates:
- `amharic_bpe.model` - SentencePiece model
- `amharic_bpe.vocab` - Vocabulary file

**Note:** For multilingual models, include corpora from all languages:

```bash
uv run python tools/train_multilingual_bpe.py \
    --corpus corpus_en.txt corpus_zh.txt corpus_ja.txt corpus_am.txt \
    --model-prefix multilingual_bpe \
    --vocab-size 40000
```

### Preprocessing Data

Extract features and generate training manifests:

```bash
uv run python tools/preprocess_data.py \
    --manifest amharic_data/raw_dataset/manifest.jsonl \
    --output-dir amharic_data/processed \
    --tokenizer amharic_output/amharic_bpe.model \
    --language am \
    --val-ratio 0.01 \
    --batch-size 4
```

This creates:
- `text_ids/*.npy` - Tokenized text
- `codes/*.npy` - Semantic codes
- `condition/*.npy` - Conditioning latents
- `emo_vec/*.npy` - Emotion vectors
- `train_manifest.jsonl` - Training manifest
- `val_manifest.jsonl` - Validation manifest

## Training

### Generate Prompt Pairs

IndexTTS2 GPT training requires prompt-target pairs:

```bash
uv run python tools/build_gpt_prompt_pairs.py \
    --manifest amharic_data/processed/train_manifest.jsonl \
    --output amharic_data/processed/train_pairs.jsonl
```

### Train GPT Model

```bash
uv run python trainers/train_gpt_v2.py \
    --train-manifest amharic_data/processed/train_pairs.jsonl \
    --val-manifest amharic_data/processed/val_pairs.jsonl \
    --tokenizer amharic_output/amharic_bpe.model \
    --config checkpoints/config.yaml \
    --base-checkpoint checkpoints/gpt.pth \
    --output-dir amharic_output/trained_ckpts \
    --batch-size 4 \
    --grad-accumulation 4 \
    --epochs 10 \
    --learning-rate 2e-5 \
    --amp
```

### Monitor Training

Use TensorBoard to monitor training:

```bash
uv run tensorboard --logdir amharic_output/trained_ckpts/logs
```

Open http://localhost:6006 in your browser.

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Symptoms:** CUDA out of memory errors

**Solutions:**
- Reduce `--batch-size`
- Increase `--grad-accumulation`
- Use `--amp` for mixed precision
- Use smaller model or reduce sequence length

#### 2. Slow Download

**Symptoms:** YouTube download is very slow

**Solutions:**
- Use smaller quality settings
- Download audio only (default)
- Use VPN if geo-restricted
- Download in batches

#### 3. Poor Tokenization

**Symptoms:** High UNK token rate

**Solutions:**
- Increase `--vocab-size`
- Increase `--character-coverage` to 0.9999
- Add more diverse Amharic corpus
- Check text normalization

#### 4. Subtitle Sync Issues

**Symptoms:** Audio segments don't match text

**Solutions:**
- Enable boundary refinement (default)
- Adjust `--min-duration` and `--max-duration`
- Manually review subtitle files
- Use better quality source videos

## Best Practices

### Data Quality

1. **Audio Quality**
   - Use high-quality recordings
   - Minimize background noise
   - Prefer professional narration
   - Ensure consistent volume

2. **Speaker Diversity**
   - Include multiple speakers
   - Mix genders and ages
   - Include various accents/dialects
   - Balance speaker representation

3. **Text Quality**
   - Verify subtitle accuracy
   - Fix transcription errors
   - Normalize spelling variations
   - Remove non-speech segments

### Training Strategy

1. **Start Small**
   - Test with 1-10 hours of data
   - Verify pipeline works
   - Iterate on data quality
   - Scale up gradually

2. **Incremental Training**
   - Start from base model
   - Fine-tune on Amharic
   - Monitor validation metrics
   - Stop if overfitting

3. **Evaluation**
   - Test on held-out samples
   - Listen to generated speech
   - Check pronunciation accuracy
   - Measure emotion preservation

### Lightning AI Deployment

1. **Prepare Repository**
   ```bash
   cd index-tts2
   git add .
   git commit -m "Add Amharic training support"
   git push origin main
   ```

2. **Pull in Lightning AI**
   ```bash
   git clone https://github.com/YOUR_USERNAME/index-tts2.git
   cd index-tts2
   git checkout training_v2
   ```

3. **Set Up Environment**
   ```bash
   uv sync --all-extras
   ```

4. **Run Training**
   ```bash
   bash scripts/amharic/end_to_end.sh
   ```

### Resource Requirements

**Minimum (1-10 hours of data):**
- GPU: 16GB VRAM (RTX 4080, A4000)
- RAM: 32GB
- Disk: 100GB
- Time: 1-3 days

**Recommended (100+ hours):**
- GPU: 24-40GB VRAM (A5000, A100)
- RAM: 64GB+
- Disk: 500GB+
- Time: 1-2 weeks

## Amharic Language Specifics

### Ge'ez Script

Amharic uses the Ge'ez (Ethiopic) script:
- Syllabary system (abugida)
- Each character represents a syllable
- 33 consonants × 7 vowel forms = 231 base characters
- Additional characters for labialized consonants

### Unicode Ranges

- Basic Ethiopic: U+1200–U+137F
- Ethiopic Supplement: U+1380–U+139F
- Ethiopic Extended: U+2D80–U+2DDF
- Ethiopic Extended-A: U+AB00–U+AB2F

### Punctuation

- Word separator: ፡ (U+1361)
- Comma: ፣ (U+1363)
- Full stop: ። (U+1362)
- Semicolon: ፤ (U+1364)
- Colon: ፥ (U+1365)
- Question mark: ፧ (U+1367)

### Number System

Amharic has its own numerals (1-9, 10, 20, ..., 100):
- ፩ (1), ፪ (2), ፫ (3), etc.
- But modern usage also includes Arabic numerals

## References

- [Amharic Language (Wikipedia)](https://en.wikipedia.org/wiki/Amharic)
- [Ge'ez Script (Wikipedia)](https://en.wikipedia.org/wiki/Ge%CA%BDez_script)
- [Unicode Ethiopic](https://unicode.org/charts/PDF/U1200.pdf)
- [SentencePiece Documentation](https://github.com/google/sentencepiece)

## Support

For issues or questions:
1. Check this documentation
2. Review troubleshooting section
3. Open an issue on GitHub
4. Join the Discord community

---

**Last Updated:** 2025-01-XX  
**Version:** IndexTTS2 with Amharic Support
