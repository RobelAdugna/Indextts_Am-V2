# Deployment Checklist for Lightning AI

## Pre-Push Verification (Local)

### 1. File Structure Check
- [ ] `webui_amharic.py` exists and is complete
- [ ] `README_AMHARIC_WEBUI.md` exists
- [ ] All Amharic tools exist:
  - [ ] `tools/youtube_amharic_downloader.py`
  - [ ] `tools/create_amharic_dataset.py`
  - [ ] `tools/collect_amharic_corpus.py`
  - [ ] `tools/train_multilingual_bpe.py`
  - [ ] `tools/preprocess_data.py`
- [ ] `scripts/amharic/end_to_end.sh` exists
- [ ] `scripts/amharic/end_to_end.ps1` exists
- [ ] `knowledge.md` updated with WebUI info

### 2. Python Syntax Check
```bash
python -m py_compile webui_amharic.py
python -m py_compile tools/youtube_amharic_downloader.py
python -m py_compile tools/create_amharic_dataset.py
python -m py_compile tools/collect_amharic_corpus.py
python -m py_compile tools/train_multilingual_bpe.py
```

### 3. Import Verification
Test that all imports work:
```bash
python -c "import sys; sys.path.append('.'); from webui_amharic import check_dependencies; print('Imports OK')"
```

### 4. Dependency Check
Verify all required packages:
```bash
pip list | grep -E "gradio|sentencepiece|librosa|soundfile|tqdm|torch"
```

### 5. Basic Functionality Test (Optional)
If you want to test locally:
```bash
python webui_amharic.py --host 127.0.0.1 --port 7863
# Open http://127.0.0.1:7863 in browser
# Check that all tabs load without errors
# Ctrl+C to stop
```

## Git Preparation

### 6. Review Changes
```bash
# Check what will be committed
git status

# Review new files
git diff --cached
```

### 7. Stage Files
```bash
# Stage the WebUI and documentation
git add webui_amharic.py
git add README_AMHARIC_WEBUI.md
git add DEPLOYMENT_CHECKLIST.md
git add knowledge.md

# Verify staging
git status
```

### 8. Commit
```bash
git commit -m "Add comprehensive Amharic TTS pipeline WebUI

- Create webui_amharic.py with 7-tab interface
- Integrate all Amharic tools (download, dataset, corpus, tokenizer, preprocess, training)
- Add automatic state management and progress tracking
- Include dependency checking and error handling
- Create comprehensive README_AMHARIC_WEBUI.md
- Update knowledge.md with WebUI documentation

Features:
- Sequential workflow with auto-fill
- Real-time progress for all operations
- Color-coded status messages
- Modern, clean UI design
- Complete end-to-end pipeline

🤖 Generated with Codebuff
Co-Authored-By: Codebuff <noreply@codebuff.com>"
```

### 9. Push to GitHub
```bash
# Push to your branch (adjust branch name as needed)
git push origin main
# OR
git push origin training_v2
```

## Lightning AI Setup

### 10. Clone Repository
```bash
# On Lightning AI
git clone https://github.com/YOUR_USERNAME/index-tts2.git
cd index-tts2

# Checkout the correct branch if needed
git checkout training_v2  # or main
```

### 11. Install Dependencies
```bash
# Use uv for consistency (recommended)
uv sync --all-extras

# OR use pip
pip install gradio sentencepiece librosa soundfile tqdm yt-dlp
```

### 12. Verify Installation
```bash
# Check Python version (should be 3.8+)
python --version

# Verify imports
python -c "import gradio; import sentencepiece; import librosa; print('✓ All imports OK')"

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda if torch.cuda.is_available() else None}')"
```

### 13. Install System Dependencies
```bash
# Install yt-dlp (if not already installed)
pip install -U yt-dlp

# Verify yt-dlp
yt-dlp --version

# ffmpeg should be pre-installed on Lightning AI
# Verify:
ffmpeg -version
```

### 14. Create Required Directories
```bash
mkdir -p checkpoints
mkdir -p outputs
mkdir -p prompts
```

### 15. Download Base Checkpoints
```bash
# You'll need to download the base model checkpoints
# Follow the instructions in the main README
# Ensure these files exist in checkpoints/:
# - config.yaml
# - gpt.pth
# - bpe.model
# - s2mel.pth
# - wav2vec2bert_stats.pt
```

### 16. Test WebUI Launch
```bash
# Launch the WebUI
python webui_amharic.py --host 0.0.0.0 --port 7863

# The UI should be accessible at the Lightning AI URL
# Check the logs for any errors
```

### 17. Verify Each Tab

#### Tab 1: Overview
- [ ] Shows system status (yt-dlp, ffmpeg, CUDA)
- [ ] Pipeline status shows "No steps completed yet"

#### Tab 2: Download
- [ ] URL input field works
- [ ] File upload works
- [ ] All settings are visible
- [ ] Button is clickable

#### Tab 3: Dataset
- [ ] Input directory field works
- [ ] Sliders are functional
- [ ] Checkbox works

#### Tab 4: Corpus
- [ ] File upload works
- [ ] Settings are functional

#### Tab 5: Tokenizer
- [ ] File upload works
- [ ] Sliders work
- [ ] Test text input works

#### Tab 6: Preprocess
- [ ] All fields are present
- [ ] Batch size slider works

#### Tab 7: Training
- [ ] All configuration fields present
- [ ] Sliders functional

### 18. Test Basic Workflow (Optional)

If you have test data:

1. **Download Test**: 
   - Use a single YouTube URL
   - Verify download completes
   - Check output directory

2. **Dataset Test**:
   - Use downloaded files
   - Verify segmentation works
   - Check manifest.jsonl

3. **Corpus Test**:
   - Use dataset manifest
   - Verify corpus file created

4. **Tokenizer Test**:
   - Use corpus file
   - Verify .model file created
   - Test tokenization result

## Common Issues & Solutions

### Issue: Import errors
**Solution**: 
```bash
pip install --upgrade gradio sentencepiece librosa soundfile tqdm
```

### Issue: yt-dlp not found
**Solution**:
```bash
pip install -U yt-dlp
```

### Issue: CUDA not available
**Check**:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Port already in use
**Solution**: Use a different port:
```bash
python webui_amharic.py --port 7864
```

### Issue: Permission denied
**Solution**: Check file permissions:
```bash
chmod +x webui_amharic.py
chmod +x scripts/amharic/end_to_end.sh
```

## Performance Verification

### 19. Check Resource Usage
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h

# Check memory
free -h
```

### 20. Verify Paths Are Relative
```bash
# Search for absolute paths (should find none or minimal)
grep -r "/home/" webui_amharic.py || echo "✓ No absolute paths found"
grep -r "C:\\" webui_amharic.py || echo "✓ No Windows paths found"
```

## Final Checks

- [ ] WebUI launches without errors
- [ ] All tabs are accessible
- [ ] System status shows correct information
- [ ] No Python errors in terminal
- [ ] Can navigate between tabs smoothly
- [ ] UI is responsive

## Documentation

- [ ] `README_AMHARIC_WEBUI.md` is complete
- [ ] `knowledge.md` updated
- [ ] All example files exist
- [ ] Scripts are executable

## Ready for Production

Once all checks pass:
- ✅ Code is tested locally
- ✅ Committed to git
- ✅ Pushed to GitHub
- ✅ Deployed to Lightning AI
- ✅ Verified on remote server
- ✅ Documentation complete

---

## Quick Command Reference

### Local Testing
```bash
# Syntax check
python -m py_compile webui_amharic.py

# Import test
python -c "from webui_amharic import check_dependencies; print('OK')"

# Launch
python webui_amharic.py
```

### Git Commands
```bash
git add webui_amharic.py README_AMHARIC_WEBUI.md DEPLOYMENT_CHECKLIST.md knowledge.md
git commit -m "Add Amharic TTS Pipeline WebUI"
git push origin main
```

### Lightning AI Commands
```bash
git clone <repo-url>
cd index-tts2
uv sync --all-extras
python webui_amharic.py --host 0.0.0.0
```

---

**Last Updated**: 2025-01-15
