# Tab 4 Tokenizer Extension - Troubleshooting Guide

**Status:** ✅ Error handling improvements COMPLETE

## Problem

User reported "unclear error message" when clicking "Extend Tokenizer" button in Tab 4.

## Root Cause Analysis

### Primary Issue: NumPy 2.x Incompatibility

The terminal logs show:
```
ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.4
```

This crashes the **entire WebUI** on startup before you can even reach Tab 4!

### Secondary Issue: Vague Error Messages

The original `train_tokenizer()` function had minimal error reporting:
- Generic "Error: {str(e)}" messages
- No command logging
- No full traceback
- No exit code reporting

## Solution Applied

### ✅ Enhanced Error Handling (webui_amharic.py)

**Changes made:**

1. **Full Traceback Logging:**
   ```python
   import traceback
   error_details = traceback.format_exc()
   log_text = f"Exception occurred:\n{error_details}\n\nError: {str(e)}"
   ```

2. **Command Logging:**
   ```python
   cmd_str = ' '.join(cmd)
   print(f"Running command: {cmd_str}")  # Shows in terminal
   ```

3. **Better Validation Messages:**
   ```python
   # Before: "Error: Base model not found: {path}"
   # After: "Error: Base model not found: {path}\n\nPlease run download_requirements.bat first!"
   ```

4. **Exit Code + Full Output:**
   ```python
   log_text = f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nExit code: {result.returncode}"
   ```

### ✅ Documentation (knowledge.md)

Added troubleshooting section for common Tab 4 errors.

## How to Use the Fix

### Step 1: Fix NumPy (Required)

```bash
# Install NumPy 1.x
pip install 'numpy<2'
```

### Step 2: Restart WebUI

```bash
# Stop current WebUI (Ctrl+C)
# Then restart:
python webui_amharic.py --share
```

### Step 3: Try Tab 4 Again

Now when you click "Extend Tokenizer", you'll see:
- **Full command** being run (in terminal logs)
- **Complete stdout/stderr** (in WebUI logs tab)
- **Exit code** (in error message)
- **Full Python traceback** (if exception occurs)
- **Helpful instructions** (e.g., "Run download_requirements.bat first!")

## Common Error Messages (Now Clear!)

### Error: "Base model not found"

**Message:**
```
❌ Error: Base model not found: checkpoints/bpe.model

Please run download_requirements.bat first!
```

**Solution:**
```bash
# Windows:
double-click download_requirements.bat

# Linux/Mac:
bash download_requirements.sh
```

### Error: "Manifest not found"

**Message:**
```
❌ Error: Manifest not found: amharic_dataset/manifest.jsonl

Please create a dataset in Tab 2 first.
```

**Solution:**
1. Go to Tab 2 (Dataset Creation)
2. Complete dataset creation
3. Return to Tab 4 (auto-fills manifest path)

### Error: "extend_bpe.py not found"

**Message:**
```
❌ Error: extend_bpe.py not found
```

**Solution:**
Verify file exists:
```bash
dir tools\tokenizer\extend_bpe.py  # Windows
ls tools/tokenizer/extend_bpe.py   # Linux/Mac
```

If missing, re-clone the repository.

### Error: Script fails with exit code ≠ 0

**Message:**
```
❌ Error extending tokenizer (exit code: 1)

Command: python tools/tokenizer/extend_bpe.py --base-model ...

STDOUT:
[extend_bpe] Collected 0 samples...

STDERR:
RuntimeError: No usable text samples found in manifests.

Exit code: 1
```

**Solution:**
Check manifest.jsonl has `text` field:
```bash
head amharic_dataset/manifest.jsonl
```

Should see:
```json
{"id":"spk000_000001","text":"ሰላም እንዴት ነሽ?","audio":"audio/spk000_000001.wav",...}
```

## Verification

To verify the fix is working:

1. **Check Python syntax:**
   ```bash
   python -m py_compile webui_amharic.py
   # Should return with exit code 0 (no output)
   ```

2. **Check script exists:**
   ```bash
   python tools/tokenizer/extend_bpe.py --help
   # Should show help message
   ```

3. **Test with fake inputs:**
   Create a test manifest:
   ```bash
   echo '{"text":"ሰላም"}' > test_manifest.jsonl
   ```
   
   Then in WebUI:
   - Base Model: `checkpoints/bpe.model` (must exist)
   - Manifest: `test_manifest.jsonl`
   - Click "Extend Tokenizer"
   
   Should see detailed logs showing what went wrong!

## Technical Details

### Command Structure

The WebUI runs:
```bash
python tools/tokenizer/extend_bpe.py \
  --base-model checkpoints/bpe.model \
  --manifests amharic_dataset/manifest.jsonl \
  --output-model tokenizers/amharic_extended_bpe.model \
  --target-size 24000 \
  --character-coverage 0.9999
```

### Script Requirements

`extend_bpe.py` needs:
- `--base-model`: Existing bpe.model file
- `--manifests`: One or more JSONL files with `text` field
- Valid manifest format: `{"text": "some text", ...}`

### Output Files

- `tokenizers/amharic_extended_bpe.model` - Extended SentencePiece model
- `tokenizers/amharic_extended_bpe.vocab` - Human-readable vocabulary

## Summary

✅ **Error handling improvements applied**
✅ **Documentation updated**  
✅ **Syntax validated**

**User action required:**
1. Fix NumPy: `pip install 'numpy<2'`
2. Restart WebUI
3. Try Tab 4 - now shows detailed errors!

---

**Date:** 2025-01-XX  
**Status:** Production Ready
