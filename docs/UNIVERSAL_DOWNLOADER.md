# Universal Media Downloader Guide

## Overview

The universal media downloader supports:
- ✅ Direct media URLs (MP4, WAV, MP3, etc.)
- ✅ Auto-pairing with SRT files from different sources
- ✅ Local files + remote URLs (hybrid mode)
- ✅ Batch processing from manifest files
- ✅ Integration with dataset creation pipeline

## Quick Examples

### Example 1: Direct URLs (Your Use Case)

```bash
cd index-tts2

# Single pair
uv run python tools/universal_media_downloader.py \
    --media-url "https://www.limboour.com/download-media/tereffe.mp4" \
    --srt-url "https://www.germew.net/emerged/downloads/tereffe(am).srt" \
    --output-dir "my_downloads"
```

### Example 2: Local Files

```bash
uv run python tools/universal_media_downloader.py \
    --media-local "path/to/my_video.mp4" \
    --srt-local "path/to/my_subtitles.srt" \
    --output-dir "my_dataset"
```

### Example 3: Hybrid (Remote Media + Local SRT)

```bash
uv run python tools/universal_media_downloader.py \
    --media-url "https://example.com/video.mp4" \
    --srt-local "local/subtitles.srt" \
    --output-dir "hybrid_data"
```

### Example 4: Batch Processing

Create `examples/media_manifest.jsonl`:
```jsonl
{"media_url": "https://www.limboour.com/download-media/tereffe.mp4", "srt_url": "https://www.germew.net/emerged/downloads/tereffe(am).srt"}
{"media_local": "local/video2.mp4", "srt_local": "local/video2.srt"}
{"media_url": "https://example.com/video3.mp4", "srt_local": "subs/video3.srt"}
```

Then run:
```bash
uv run python tools/universal_media_downloader.py \
    --manifest examples/media_manifest.jsonl \
    --output-dir "batch_downloads"
```

### Example 5: Auto-Create Dataset

```bash
uv run python tools/universal_media_downloader.py \
    --manifest examples/media_manifest.jsonl \
    --output-dir "downloads" \
    --create-dataset \
    --dataset-output "amharic_dataset"
```

This will:
1. Download all media/SRT pairs
2. Automatically segment audio by subtitles
3. Create training-ready manifest

## Manifest Format

### Fields

Each line in the manifest is a JSON object with optional fields:

```json
{
  "media_url": "https://example.com/video.mp4",    // Optional: URL to media
  "srt_url": "https://example.com/subs.srt",       // Optional: URL to SRT
  "media_local": "path/to/local/video.mp4",        // Optional: Local media path
  "srt_local": "path/to/local/subs.srt",           // Optional: Local SRT path
  "base_name": "custom_name"                       // Optional: Output filename base
}
```

### Rules

- Must have either `media_url` OR `media_local`
- Can mix local and remote for media and SRT
- If `base_name` not provided, uses filename from URL/path
- Lines starting with `#` are comments

### Examples

```jsonl
# Remote only
{"media_url": "https://example.com/vid.mp4", "srt_url": "https://example.com/sub.srt"}

# Local only
{"media_local": "my_video.mp4", "srt_local": "my_video.srt"}

# Hybrid 1: Remote media + Local SRT
{"media_url": "https://example.com/video.mp4", "srt_local": "local/subs.srt"}

# Hybrid 2: Local media + Remote SRT
{"media_local": "local/video.wav", "srt_url": "https://example.com/subs.srt"}

# Media only (no subtitles)
{"media_url": "https://example.com/audio.mp3"}

# Custom output name
{"media_url": "https://example.com/v.mp4", "base_name": "episode_01"}
```

## Integration with Pipeline

### Option 1: Manual Two-Step

```bash
# Step 1: Download
uv run python tools/universal_media_downloader.py \
    --manifest examples/media_manifest.jsonl \
    --output-dir "downloads"

# Step 2: Create dataset
uv run python tools/create_amharic_dataset.py \
    --input-dir "downloads" \
    --output-dir "amharic_dataset"
```

### Option 2: Automatic (Recommended)

```bash
uv run python tools/universal_media_downloader.py \
    --manifest examples/media_manifest.jsonl \
    --output-dir "downloads" \
    --create-dataset \
    --dataset-output "amharic_dataset"
```

### Option 3: Use End-to-End Script

The automated pipeline now checks for `examples/media_manifest.jsonl` first:

```bash
# Create manifest
cat > examples/media_manifest.jsonl << EOF
{"media_url": "https://www.limboour.com/download-media/tereffe.mp4", "srt_url": "https://www.germew.net/emerged/downloads/tereffe(am).srt"}
EOF

# Run pipeline
bash scripts/amharic/end_to_end.sh
```

## Supported Media Formats

### Audio
- WAV
- MP3
- FLAC
- M4A
- OGG

### Video
- MP4
- MKV
- AVI
- WEBM

(Audio will be extracted automatically)

### Subtitles
- SRT
- VTT
- WEBVTT

## Error Handling

### Missing Files

```
Error: Local media file not found: path/to/file.mp4
```

**Solution:** Check path is correct relative to current directory

### Download Failures

```
Error downloading URL: Connection timeout
```

**Solutions:**
- Check internet connection
- Verify URL is accessible
- Try downloading file manually first

### No Subtitles

```
Warning: Local SRT file not found: subs.srt
```

**Note:** Media will still be downloaded, just without subtitles

## Advanced Usage

### Different Sources for Media and SRT

Your exact use case:

```jsonl
{"media_url": "https://www.limboour.com/download-media/tereffe.mp4", "srt_url": "https://www.germew.net/emerged/downloads/tereffe(am).srt", "base_name": "tereffe"}
```

### Mixed Local and Remote Batch

```jsonl
{"media_url": "https://site1.com/video1.mp4", "srt_url": "https://site2.com/subs1.srt"}
{"media_local": "local/video2.mp4", "srt_url": "https://site3.com/subs2.srt"}
{"media_url": "https://site4.com/video3.mp4", "srt_local": "local/subs3.srt"}
{"media_local": "local/video4.mp4", "srt_local": "local/subs4.srt"}
```

### Programmatic Usage

```python
from pathlib import Path
from tools.universal_media_downloader import MediaDownloader

downloader = MediaDownloader(output_dir=Path("my_data"))

# Download pair
media_path, srt_path = downloader.download_media_srt_pair(
    media_url="https://example.com/video.mp4",
    srt_url="https://example.com/subs.srt",
    base_name="my_video"
)

print(f"Media: {media_path}")
print(f"SRT: {srt_path}")
```

## Complete Workflow

### From Scratch to Trained Model

1. **Create manifest:**
   ```bash
   nano examples/media_manifest.jsonl
   ```

2. **Add your sources:**
   ```jsonl
   {"media_url": "https://www.limboour.com/download-media/file1.mp4", "srt_url": "https://www.germew.net/subs/file1.srt"}
   {"media_local": "my_recordings/file2.wav", "srt_local": "my_recordings/file2.srt"}
   ```

3. **Run pipeline:**
   ```bash
   bash scripts/amharic/end_to_end.sh
   ```

4. **Done!** Model will train on your custom data.

## Comparison with YouTube Downloader

| Feature | YouTube Downloader | Universal Downloader |
|---------|:------------------:|:--------------------:|
| YouTube URLs | ✅ | ❌ |
| Direct media URLs | ❌ | ✅ |
| Local files | ❌ | ✅ |
| Hybrid mode | ❌ | ✅ |
| Auto SRT pairing | ✅ (same video) | ✅ (any source) |
| Batch processing | ✅ | ✅ |

**Use Universal Downloader when:**
- Media files are on direct URLs (not YouTube)
- SRT files are on different servers
- You have local recordings
- You want to mix local and remote sources

**Use YouTube Downloader when:**
- All content is on YouTube
- Subtitles are YouTube captions

## Tips

1. **Test with one file first** before batch processing
2. **Check URLs are public** and accessible
3. **Use base_name** for consistent naming
4. **Validate SRT files** match media timing
5. **Start small** (5-10 files) then scale up

---

**Created:** 2025-01-XX  
**Status:** Production Ready  
**See Also:** `docs/AMHARIC_SUPPORT.md`
