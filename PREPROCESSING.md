# Preprocessing: What It Does and How It Works

This step converts a raw dataset (audio + text manifest) into compact features the GPT model can train on. It tokenizes text, extracts semantic features, quantizes codes, builds conditioning/emo vectors, and writes train/val manifests — with automatic VRAM-aware batching, OOM recovery, and resume support.

## Inputs
- Manifest (JSONL): One line per sample with fields:
  - id, text, audio (path), speaker (optional), language (optional), duration (optional)
- Tokenizer: SentencePiece model (.model)
- Config: Model config (YAML) with paths (e.g., w2v stats)
- GPT checkpoint: Base UnifiedVoice checkpoint (gpt.pth)

## Outputs
- Directory layout:
  - text_ids/<id>.npy — tokenized text ids (int32)
  - codes/<id>.npy — semantic codes (int32) via MaskGCT
  - condition/<id>.npy — conditioning latents (float32)
  - emo_vec/<id>.npy — emotion vectors (float32)
  - train_manifest.jsonl, val_manifest.jsonl — feature paths and lengths
  - stats.json — counts + tokenizer/checkpoint provenance
  - .preprocessing_progress.txt — resume checkpoint

## How It Works (Pipeline)
1. Normalizes + tokenizes text using TextNormalizer/TextTokenizer (language hint aware)
2. Loads audio, ensures mono + target SR
3. Extracts semantic features with Wav2Vec2Bert features (SeamlessM4T feature extractor) and project stats
4. Quantizes semantic codes using MaskGCT
5. Runs UnifiedVoice to produce:
   - conditioning (from features)
   - emo vector (from features)
6. Saves .npy artifacts and appends records to train/val manifests (deterministic split by id hash)

## VRAM & Batch Size (Auto + OOM Recovery)
- Auto batch size when `--batch-size 0` (default):
  - ~40GB → 24
  - ~24GB (L4/3090/4090/A10) → 16
  - ~16GB → 8
  - ~12GB → 6
  - ~8GB → 4
  - CPU → 1
- Dynamic recovery:
  - Detects CUDA OOM during a batch
  - Halves batch size and retries automatically
  - Falls back to single-sample processing if needed
  - Clears GPU cache between retries/batches
- Workers auto-detected based on CPU cores for faster audio I/O

## Resume Capability
- Safe to interrupt; rerun the same command to continue
- Uses `.preprocessing_progress.txt` and existing artifacts to skip completed samples
- Manifests flush on every write to avoid data loss

## Quick Start (Single Dataset)
```bash
uv run python tools/preprocess_data.py \
  --manifest dataset/manifest.jsonl \
  --output-dir preprocessed \
  --tokenizer tokenizers/bpe.model \
  --language am \
  --val-ratio 0.01 \
  --batch-size 0 \
  --workers 0
```
Notes:
- `--batch-size 0` enables auto-detection; you can set a fixed number if preferred
- `--workers 0` auto-detects optimal worker threads
- `--skip-existing` will avoid recomputing artifacts already on disk
- `--max-samples N` caps processing for debugging

## Multiple Datasets in One Run (Optional)
```bash
uv run python tools/preprocess_data.py \
  --dataset am=am/manifest.jsonl=am_processed \
  --dataset ja=ja/manifest.jsonl=ja_processed \
  --output-root preprocessed_root \
  --tokenizer tokenizers/multilingual_bpe.model \
  --batch-size 0 --workers 0
```
Each dataset gets its own output directory under `--output-root` (or use explicit OUTPUT paths per dataset).

## Parallel Variant (Advanced)
`tools/preprocess_multiproc.py` runs multiple `preprocess_data.py` workers in parallel by sharding the manifest, then merges results.

```bash
uv run python tools/preprocess_multiproc.py \
  --manifest dataset/manifest.jsonl \
  --output-dir preprocessed \
  --tokenizer tokenizers/bpe.model \
  --language am \
  --num-processes 4 \
  --batch-size 4 \
  --workers 8 \
  --skip-existing
```
- Creates `worker_*` directories, then merges artifacts and manifests
- Uses a shared HF cache; can run offline once populated

## Tips & Troubleshooting
- OOM: Let auto-batch handle it (`--batch-size 0`), or set a smaller batch (e.g., 8→4)
- Missing models: Ensure checkpoints are downloaded (see README and download scripts)
- Slow I/O: Increase `--workers`; store data on a fast disk
- Bad manifest lines: The script skips malformed JSON and logs warnings
- Audio paths: Relative paths are normalized; absolute files also work; prefer stable relative paths for portability

## Where This Is Used Next
- Training script reads these artifacts and manifests to train the GPT model
- If you change tokenization later, you can regenerate only text IDs with `tools/process_text_ids.py`
