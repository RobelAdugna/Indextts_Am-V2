# Quick Start: Amharic Training

## ⚡ 5-Minute Setup

### 1. Add Your Data Sources

Edit `examples/amharic_youtube_urls.txt`:
```
https://www.youtube.com/watch?v=YOUR_VIDEO_1
https://www.youtube.com/watch?v=YOUR_VIDEO_2
```

### 2. Run the Pipeline

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

### 3. Wait for Completion

The script will:
1. ✅ Download videos with subtitles
2. ✅ Create dataset from audio+srt
3. ✅ Collect text corpus
4. ✅ Train tokenizer
5. ✅ Preprocess data
6. ✅ Generate training pairs
7. ✅ Train the model

## 📁 Output Structure

```
amharic_data/
├── downloads/          # Downloaded videos
├── raw_dataset/        # Segmented audio + manifest
├── processed/          # Features for training
└── amharic_corpus.txt  # Text corpus

amharic_output/
├── amharic_bpe.model   # Trained tokenizer
└── trained_ckpts/      # Model checkpoints
```

## 🎤 Test Your Model

After training completes:

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True
)

# Update paths to your trained model:
# - Tokenizer: amharic_output/amharic_bpe.model
# - GPT: amharic_output/trained_ckpts/model_stepXXXX.pth

text = "ሰላም ልዑል! እንዴት ነዎት?"
tts.infer(
    spk_audio_prompt='path/to/amharic_voice.wav',
    text=text,
    output_path="output.wav"
)
```

## 🚀 Lightning AI Deployment

### On Your Local Machine:
```bash
git add .
git commit -m "Add Amharic support"
git push origin training_v2
```

### On Lightning AI:
```bash
git clone https://github.com/YOUR_USERNAME/index-tts2.git
cd index-tts2
git checkout training_v2
uv sync --all-extras
bash scripts/amharic/end_to_end.sh
```

## 💡 Tips

### Data Quality
- Use videos with clear speech
- Prefer professional narration
- Verify subtitle accuracy
- Aim for 50-100+ hours of data

### Performance
- Start with 5-10 hours for testing
- Use `--amp` flag to save VRAM
- Monitor with TensorBoard
- Keep 3 recent checkpoints

### Troubleshooting

See `docs/AMHARIC_SUPPORT.md` for:
- Common issues and solutions
- Best practices
- Detailed documentation

## 📊 Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 5 min | Add URLs, install deps |
| Download | 1-2 hours | Download 10 hours of videos |
| Processing | 2-4 hours | Create dataset, train tokenizer |
| Training | 1-3 days | Train model (depends on data size) |

## ✅ Verification Checklist

- [ ] Added YouTube URLs
- [ ] Ran end-to-end script
- [ ] Checked output directory
- [ ] Verified manifest files created
- [ ] Tokenizer trained successfully
- [ ] Training started without errors
- [ ] TensorBoard accessible
- [ ] Checkpoints being saved

## 🆘 Support

If you encounter issues:
1. Check `docs/AMHARIC_SUPPORT.md`
2. Review `IMPLEMENTATION_STATUS.md`
3. Check script output for errors
4. Verify all dependencies installed

---

**Status:** ✅ Ready to Use
**Estimated Time to First Model:** 1-3 days
**Recommended Data:** 50-100 hours
