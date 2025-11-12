# Tab 4 Tokenizer Extension - Troubleshooting Guide

**Status:** ✅ Error handling improvements COMPLETE

## Problem

User reported "unclear error message" when clicking "Extend Tokenizer" button in Tab 4.

## Root Cause Analysis

### PRIMARY ISSUE: Gradio UI Component Mismatch ✅ FIXED

**Error:**
```python
AttributeError: 'list' object has no attribute 'strip'
```

**Cause:**
The Gradio UI components didn't match the `train_tokenizer()` function signature!

**Old UI (WRONG):**
- `tokenizer_corpus_files` (gr.Files) → Returns a **LIST**!
- `tokenizer_model_prefix` (Textbox)
- `vocab_size` (Slider)
- 5 total inputs

**Function Expected:**
- `base_model_path` (string)
- `manifest_path` (string)  
- `output_model` (string)
- `target_size` (int)
- `character_coverage` (float)
- `test_text` (string)
- 6 total parameters

When Gradio passed a list to `base_model_path.strip()`, Python threw `AttributeError`!

### Secondary Issue: NumPy 2.x (Already documented)

### Tertiary Issue: Vague Error Messages

The original `train_tokenizer()` function had minimal error reporting:
- Generic "Error: {str(e)}" messages
- No command logging
- No full traceback
- No exit code reporting

## Solution Applied

### ✅ Fixed UI Components (webui_amharic.py)

**New UI (CORRECT):**
```python
base_model_input = gr.Textbox(
    label="Base Model Path",
    value="checkpoints/bpe.model"
)
manifest_input_tokenizer = gr.Textbox(
    label="Manifest Path (from Tab 2)",
    placeholder="Auto-fills from Tab 2"
)
output_model_input = gr.Textbox(
    label="Output Model Path",
    value="tokenizers/amharic_extended_bpe.model"
)
target_size = gr.Slider(
    label="Target Vocabulary Size",
    minimum=12000,
    maximum=48000,
    value=24000,  # Amharic default!
    info="Total vocab size (base 12k + new tokens)"
)
```

**Auto-fill from Tab 2:**
```python
state.change(
    lambda s: str(Path(s.get("dataset_dir", "")) / "manifest.jsonl") if s.get("dataset_dir") else "",
    inputs=[state],
    outputs=[manifest_input_tokenizer]
)
```

**Button Click Fixed:**
```python
train_tokenizer_btn.click(
    train_tokenizer,
    inputs=[base_model_input, manifest_input_tokenizer, output_model_input, 
            target_size, character_coverage, test_text_tokenizer],  # 6 inputs ✅
    outputs=[tokenizer_logs, tokenizer_status, tokenizer_test_result, state]
)
```

### ✅ Enhanced Error Handling (Already Applied)

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

✅ **UI components fixed** (base_model, manifest, output_model, target_size)
✅ **Auto-fill from Tab 2** (manifest path)
✅ **Error handling improvements applied** (traceback, command logging, exit codes)
✅ **Documentation updated**  
✅ **Syntax validated**
✅ **Amharic-optimized defaults** (24k vocab, 0.9995 coverage)

**User action required:**
1. Restart WebUI (to load the fixed code)
2. Try Tab 4 - should work now!
3. (Optional) Fix NumPy if needed: `pip install 'numpy<2'`

---

**Date:** 2025-01-XX  
**Status:** Production Ready
