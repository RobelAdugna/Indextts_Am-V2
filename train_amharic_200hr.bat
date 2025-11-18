@echo off
REM Optimized Training Script for 200-Hour Amharic Dataset (Windows)
REM Features: Overfitting protection, early stopping, quality monitoring

setlocal enabledelayedexpansion

echo 🚀 Starting Optimized Amharic Training (200hr dataset)
echo =================================================
echo.

REM Configuration
set TRAIN_MANIFEST=processed\GPT_pairs_train.jsonl
set VAL_MANIFEST=processed\GPT_pairs_val.jsonl
set TOKENIZER=tokenizers\amharic_extended_bpe.model
set OUTPUT_DIR=trained_ckpts

REM Verify files exist
if not exist "%TRAIN_MANIFEST%" (
    echo ❌ Error: Training manifest not found: %TRAIN_MANIFEST%
    exit /b 1
)

if not exist "%VAL_MANIFEST%" (
    echo ❌ Error: Validation manifest not found: %VAL_MANIFEST%
    exit /b 1
)

if not exist "%TOKENIZER%" (
    echo ❌ Error: Tokenizer not found: %TOKENIZER%
    exit /b 1
)

echo ✅ All required files found
echo.

REM Count samples (approximate using findstr)
for /f %%i in ('findstr /r /c:"^" "%TRAIN_MANIFEST%"') do set TRAIN_SAMPLES=%%i
for /f %%i in ('findstr /r /c:"^" "%VAL_MANIFEST%"') do set VAL_SAMPLES=%%i

echo 📊 Dataset Statistics:
echo    Training samples: %TRAIN_SAMPLES%
echo    Validation samples: %VAL_SAMPLES%
echo.

echo 🎯 Training Configuration:
echo    Epochs: 3
echo    Learning rate: 5e-5 (conservative for extended vocab)
echo    Weight decay: 1e-5 (L2 regularization)
echo    Warmup: 4000 steps
echo    Validation: Every 500 steps
echo    Checkpoints: Save every 1000 steps, keep best 5
echo    Loss weights: Text=0.3, Mel=0.7
echo.

set /p CONFIRM="▶️  Start training? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo ❌ Training cancelled
    exit /b 0
)

echo.
echo 🏋️  Starting training...
echo 💡 Monitor with: uv run tensorboard --logdir %OUTPUT_DIR%
echo 🛑 Stop early if validation loss plateaus for 10k steps!
echo.

REM Start training with optimal settings
python trainers\train_gpt_v2.py ^
  --train-manifest "%TRAIN_MANIFEST%" ^
  --val-manifest "%VAL_MANIFEST%" ^
  --tokenizer "%TOKENIZER%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --epochs 3 ^
  --learning-rate 5e-5 ^
  --weight-decay 1e-5 ^
  --warmup-steps 4000 ^
  --val-interval 500 ^
  --save-interval 1000 ^
  --keep-checkpoints 5 ^
  --text-loss-weight 0.3 ^
  --mel-loss-weight 0.7 ^
  --grad-clip 1.0 ^
  --resume auto ^
  --amp

echo.
echo ✅ Training complete!
echo.
echo 📦 Next steps:
echo    1. Check TensorBoard for validation metrics
echo    2. Select best checkpoint (lowest val_loss, gap ^<0.3)
echo    3. Test with inference on validation samples
echo    4. If overfitting (gap ^>0.5), use earlier checkpoint

pause
