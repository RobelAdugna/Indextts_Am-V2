# Git Commands for WebUI Deployment

## Quick Copy-Paste Commands

### 1. Check Current Status
```bash
git status
```

### 2. Stage All New Files
```bash
git add webui_amharic.py
git add README_AMHARIC_WEBUI.md
git add DEPLOYMENT_CHECKLIST.md
git add WEBUI_IMPLEMENTATION_SUMMARY.md
git add GIT_COMMANDS.md
git add verify_webui.py
git add knowledge.md
```

### 3. Verify Staged Files
```bash
git status
```

You should see:
- `webui_amharic.py` (new file)
- `README_AMHARIC_WEBUI.md` (new file)
- `DEPLOYMENT_CHECKLIST.md` (new file)
- `WEBUI_IMPLEMENTATION_SUMMARY.md` (new file)
- `GIT_COMMANDS.md` (new file)
- `verify_webui.py` (new file)
- `knowledge.md` (modified)

### 4. Commit Changes
```bash
git commit -m "Add comprehensive Amharic TTS pipeline WebUI

- Create webui_amharic.py with 7-tab interface
- Integrate all Amharic tools (download, dataset, corpus, tokenizer, preprocess, training)
- Add automatic state management and progress tracking
- Include dependency checking and error handling
- Create comprehensive README_AMHARIC_WEBUI.md
- Add verification script and deployment checklist
- Update knowledge.md with WebUI documentation

Features:
- Sequential workflow with auto-fill
- Real-time progress for all operations
- Color-coded status messages
- Modern, clean UI design
- Complete end-to-end pipeline from download to inference

🤖 Generated with Codebuff
Co-Authored-By: Codebuff <noreply@codebuff.com>"
```

### 5. Push to GitHub

**Option A: Push to main branch**
```bash
git push origin main
```

**Option B: Push to training_v2 branch**
```bash
git push origin training_v2
```

**Option C: Create new branch for review**
```bash
git checkout -b feature/amharic-webui
git push origin feature/amharic-webui
```

## Post-Push: Lightning AI Commands

### On Lightning AI Terminal

```bash
# 1. Clone repository (replace with your URL)
git clone https://github.com/YOUR_USERNAME/index-tts2.git
cd index-tts2

# 2. Checkout branch (if not main)
git checkout training_v2  # or feature/amharic-webui

# 3. Pull latest changes
git pull

# 4. Install dependencies
uv sync --all-extras

# 5. Launch WebUI
python webui_amharic.py --host 0.0.0.0 --port 7863
```

The WebUI will be accessible at the Lightning AI provided URL.

## Verification After Push

### Check on GitHub
1. Navigate to your repository
2. Verify all files are present
3. Check commit message is correct
4. Ensure files have proper content

### Test on Lightning AI
1. Clone and navigate to repo
2. Install dependencies
3. Launch WebUI
4. Access via browser
5. Check all tabs load
6. Verify system status shows correctly

## Rollback (If Needed)

```bash
# Undo last commit (if not pushed)
git reset --soft HEAD~1

# Undo last commit (if already pushed - creates new commit)
git revert HEAD
git push origin main
```

## Summary

✅ **All files verified and ready**
✅ **Python syntax validated**  
✅ **Ready to push to GitHub**  
✅ **Ready for Lightning AI deployment**

---

**Quick Reference:**
- Local WebUI: `python webui_amharic.py`
- Lightning AI: `python webui_amharic.py --host 0.0.0.0`
- Documentation: `README_AMHARIC_WEBUI.md`
- Deployment Guide: `DEPLOYMENT_CHECKLIST.md`
