# End-to-End Amharic Training Pipeline for IndexTTS2 (Windows PowerShell)
# This script automates the entire process from data collection to model training

$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$DATA_DIR = Join-Path $PROJECT_ROOT "amharic_data"
$OUTPUT_DIR = Join-Path $PROJECT_ROOT "amharic_output"
$CHECKPOINTS_DIR = Join-Path $PROJECT_ROOT "checkpoints"

# Create directories
New-Item -ItemType Directory -Force -Path $DATA_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IndexTTS2 Amharic Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project root: $PROJECT_ROOT"
Write-Host "Data directory: $DATA_DIR"
Write-Host "Output directory: $OUTPUT_DIR"
Write-Host ""

# Step 0: Check and download required checkpoints
Write-Host "[Step 0/7] Checking for required model checkpoints..." -ForegroundColor Yellow
$MISSING_CHECKPOINTS = $false

$requiredCheckpoints = @("gpt.pth", "bpe.model", "s2mel.pth", "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt")
foreach ($checkpoint in $requiredCheckpoints) {
    $checkpointPath = Join-Path $CHECKPOINTS_DIR $checkpoint
    if (-not (Test-Path $checkpointPath)) {
        Write-Host "  ✗ Missing: $checkpoint" -ForegroundColor Red
        $MISSING_CHECKPOINTS = $true
    }
}

if ($MISSING_CHECKPOINTS) {
    Write-Host "  ⬇ Downloading missing checkpoints from HuggingFace..." -ForegroundColor Yellow
    uv run python tools/download_checkpoints.py --output-dir "$CHECKPOINTS_DIR"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Failed to download checkpoints. Exiting." -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ Checkpoints downloaded successfully" -ForegroundColor Green
} else {
    Write-Host "  ✓ All required checkpoints present" -ForegroundColor Green
}
Write-Host ""

# Step 1: Download content (YouTube OR direct URLs)
Write-Host "[Step 1/8] Downloading Amharic content..." -ForegroundColor Yellow

# Check for media manifest (direct URLs)
$manifestFile = Join-Path $PROJECT_ROOT "examples\media_manifest.jsonl"
if (Test-Path $manifestFile) {
    Write-Host "  Using media manifest for direct URL downloads"
    uv run python tools/universal_media_downloader.py `
        --manifest "$manifestFile" `
        --output-dir "$DATA_DIR\downloads"
    Write-Host "✓ Step 1 complete (direct URLs)" -ForegroundColor Green

# Fall back to YouTube
} elseif (Test-Path (Join-Path $PROJECT_ROOT "examples\amharic_youtube_urls.txt")) {
    Write-Host "  Using YouTube downloader"
    uv run python tools/youtube_amharic_downloader.py `
        --url-file "$PROJECT_ROOT\examples\amharic_youtube_urls.txt" `
        --output-dir "$DATA_DIR\downloads" `
        --subtitle-langs am en amh `
        --audio-format wav
    Write-Host "✓ Step 1 complete (YouTube)" -ForegroundColor Green

} else {
    Write-Host "⚠ Skipping: No source file found" -ForegroundColor Yellow
    Write-Host "  Create either:"
    Write-Host "    - examples/media_manifest.jsonl (for direct URLs/local files)"
    Write-Host "    - examples/amharic_youtube_urls.txt (for YouTube)"
}
Write-Host ""

# Step 2: Create dataset from media files
Write-Host "[Step 2/8] Creating dataset from audio and subtitle files..." -ForegroundColor Yellow
$downloadsDir = Join-Path $DATA_DIR "downloads"
if (Test-Path $downloadsDir) {
    uv run python tools/create_amharic_dataset.py `
        --input-dir "$downloadsDir" `
        --output-dir "$DATA_DIR\raw_dataset" `
        --manifest "$DATA_DIR\raw_dataset\manifest.jsonl" `
        --min-duration 1.0 `
        --max-duration 30.0
    Write-Host "✓ Step 2 complete" -ForegroundColor Green
} else {
    Write-Host "⚠ Skipping: No downloads directory found" -ForegroundColor Yellow
}
Write-Host ""

# Step 3: Collect and clean corpus
Write-Host "[Step 3/8] Collecting and cleaning Amharic corpus..." -ForegroundColor Yellow
$manifestFile = Join-Path $DATA_DIR "raw_dataset\manifest.jsonl"
if (Test-Path $manifestFile) {
    uv run python tools/collect_amharic_corpus.py `
        --input "$manifestFile" `
        --output "$DATA_DIR\amharic_corpus.txt" `
        --text-field text `
        --stats
    Write-Host "✓ Step 3 complete" -ForegroundColor Green
} else {
    Write-Host "⚠ Skipping: No manifest file found" -ForegroundColor Yellow
}
Write-Host ""

# Step 4: Extend BPE tokenizer (following video workflow at 20:05-30:00)
Write-Host "[Step 4/8] Extending base BPE tokenizer with Amharic..." -ForegroundColor Yellow
$manifestFile = Join-Path $DATA_DIR "raw_dataset\manifest.jsonl"
if (Test-Path $manifestFile) {
    $baseTokenizer = Join-Path $CHECKPOINTS_DIR "bpe.model"
    if (-not (Test-Path $baseTokenizer)) {
        Write-Host "  ✗ Base tokenizer not found: $baseTokenizer" -ForegroundColor Red
        Write-Host "  Run download_requirements.bat first!" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "  ✓ Base tokenizer found (English/Chinese)" -ForegroundColor Green
    Write-Host "  📊 Extending with Amharic tokens..."
    
    # Use extend_bpe.py (matches video exactly!)
    uv run python tools/tokenizer/extend_bpe.py `
        --base-model "$baseTokenizer" `
        --manifests "$manifestFile" `
        --output-model "$OUTPUT_DIR\amharic_extended_bpe.model" `
        --target-size 24000 `
        --character-coverage 0.9999
    
    Write-Host "  ✓ Extension complete: base vocab preserved + Amharic tokens added" -ForegroundColor Green
    Write-Host "✓ Step 4 complete" -ForegroundColor Green
} else {
    Write-Host "⚠ Skipping: No manifest file found" -ForegroundColor Yellow
}
Write-Host ""

# Step 5: Preprocess data
Write-Host "[Step 5/8] Preprocessing Amharic dataset..." -ForegroundColor Yellow
$tokenizerFile = Join-Path $OUTPUT_DIR "amharic_extended_bpe.model"
if ((Test-Path $manifestFile) -and (Test-Path $tokenizerFile)) {
    uv run python tools/preprocess_data.py `
        --manifest "$manifestFile" `
        --output-dir "$DATA_DIR\processed" `
        --tokenizer "$tokenizerFile" `
        --language am `
        --val-ratio 0.01 `
        --batch-size 4
    Write-Host "✓ Step 5 complete" -ForegroundColor Green
} else {
    Write-Host "⚠ Skipping: Missing manifest or tokenizer" -ForegroundColor Yellow
}
Write-Host ""

# Step 6: Generate prompt-target pairs
Write-Host "[Step 6/8] Generating GPT prompt-target pairs..." -ForegroundColor Yellow
$trainManifest = Join-Path $DATA_DIR "processed\train_manifest.jsonl"
if (Test-Path $trainManifest) {
    $pairScript = Join-Path $PROJECT_ROOT "tools\build_gpt_prompt_pairs.py"
    if (Test-Path $pairScript) {
        uv run python tools/build_gpt_prompt_pairs.py `
            --manifest "$trainManifest" `
            --output "$DATA_DIR\processed\train_pairs.jsonl"
        
        uv run python tools/build_gpt_prompt_pairs.py `
            --manifest "$DATA_DIR\processed\val_manifest.jsonl" `
            --output "$DATA_DIR\processed\val_pairs.jsonl"
        Write-Host "✓ Step 6 complete" -ForegroundColor Green
    } else {
        Write-Host "⚠ Warning: build_gpt_prompt_pairs.py not found" -ForegroundColor Yellow
        Write-Host "  Continuing with single-sample manifests..."
        Copy-Item "$trainManifest" "$DATA_DIR\processed\train_pairs.jsonl"
        Copy-Item "$DATA_DIR\processed\val_manifest.jsonl" "$DATA_DIR\processed\val_pairs.jsonl"
    }
} else {
    Write-Host "⚠ Skipping: No processed manifests found" -ForegroundColor Yellow
}
Write-Host ""

# Step 7: Train GPT model
Write-Host "[Step 7/8] Training GPT model (this will take a long time)..." -ForegroundColor Yellow
$trainPairs = Join-Path $DATA_DIR "processed\train_pairs.jsonl"
if (Test-Path $trainPairs) {
    Write-Host "  Starting training..."
    Write-Host "  You can monitor progress in: $OUTPUT_DIR\trained_ckpts\logs\"
    Write-Host ""
    
    uv run python trainers/train_gpt_v2.py `
        --train-manifest "$trainPairs" `
        --val-manifest "$DATA_DIR\processed\val_pairs.jsonl" `
        --tokenizer "$OUTPUT_DIR\amharic_bpe.model" `
        --config "$CHECKPOINTS_DIR\config.yaml" `
        --base-checkpoint "$CHECKPOINTS_DIR\gpt.pth" `
        --output-dir "$OUTPUT_DIR\trained_ckpts" `
        --batch-size 4 `
        --grad-accumulation 4 `
        --epochs 10 `
        --learning-rate 2e-5 `
        --warmup-steps 1000 `
        --log-interval 100 `
        --val-interval 500 `
        --amp
    
    Write-Host "✓ Step 7 complete" -ForegroundColor Green
} else {
    Write-Host "⚠ Skipping: No paired manifests found" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pipeline Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Output directory: $OUTPUT_DIR"
Write-Host "Trained checkpoints: $OUTPUT_DIR\trained_ckpts\"
Write-Host ""
Write-Host "To use the trained model, update your config to point to:"
Write-Host "  Tokenizer: $OUTPUT_DIR\amharic_extended_bpe.model"
Write-Host "  GPT: $OUTPUT_DIR\trained_ckpts\model_stepXXXX.pth"
Write-Host ""
