# IndexTTS2 Knowledge Base

## Project Overview

IndexTTS2 is a state-of-the-art Text-to-Speech system supporting multiple languages including English, Chinese, Japanese, and now Amharic.

## Language Support

### Supported Languages
- English (en)
- Chinese (zh, cn)
- Japanese (ja, jp)
- Amharic (am, amh) - **NEW** (Most complete implementation)

### Implementation Quality
- **Amharic:** Most complete with full automation, comprehensive docs, better tooling
- **Japanese:** Reference implementation, less automation
- **English/Chinese:** Base model languages

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

### Normalization
- Use NFC (not NFKC) for proper character composition
- Duration ratio: 1.0 (similar to English)
- Each character counts as 1 syllable

## Training Pipeline

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

4. **Tokenizer Training**
   - Train BPE on collected corpus
   - Set high character coverage (0.9999)
   - Include language-specific symbols

5. **Preprocessing**
   - Extract semantic features
   - Tokenize text
   - Generate conditioning vectors
   - Split train/validation

6. **Generate Prompt Pairs**
   - Create prompt-target combinations
   - Required for GPT training

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

### Typical Command

```bash
cd project-dir
uv sync --all-extras
bash scripts/LANGUAGE/end_to_end.sh
```

## Common Issues & Solutions

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

## Amharic WebUI

### Overview
A comprehensive Gradio web interface (`webui_amharic.py`) integrates all Amharic-specific tools into a single, user-friendly pipeline:

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
3. Corpus - Text aggregation and cleaning
4. Tokenizer - BPE model training
5. Preprocess - Feature extraction
6. Training - GPT fine-tuning launcher
7. Inference - Links to existing TTS UIs

### Usage Pattern
- Each tab auto-fills from previous step's output
- Can skip steps if intermediate files exist
- Progress tracked in real-time
- Logs displayed in UI

See `README_AMHARIC_WEBUI.md` for complete documentation.

## Resources

### Documentation
- See `docs/` directory for guides
- Check `examples/` for usage examples
- Read `AMHARIC_IMPLEMENTATION_*.md` for details
- See `README_AMHARIC_WEBUI.md` for WebUI usage

### External References
- [IndexTTS2 Paper](https://arxiv.org/abs/2506.21619)
- [SentencePiece](https://github.com/google/sentencepiece)
- [Unicode Ethiopic](https://unicode.org/charts/PDF/U1200.pdf)

---

**Last Updated:** 2025-01-XX  
**Maintainers:** IndexTTS Team
