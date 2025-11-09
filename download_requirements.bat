@echo off
REM Quick setup script for IndexTTS2 - Downloads all requirements

echo ========================================
echo IndexTTS2 Setup - Downloading Requirements
echo ========================================
echo.

echo [1/2] Installing Python dependencies...
echo.
pip install huggingface-hub
if %errorlevel% neq 0 (
    echo ERROR: Failed to install huggingface-hub
    echo Please run: pip install huggingface-hub
    pause
    exit /b 1
)

echo.
echo [2/2] Downloading pretrained model checkpoints...
echo This may take 10-30 minutes depending on your connection.
echo.
python tools\download_checkpoints.py --output-dir checkpoints

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Download failed!
    echo.
    echo Troubleshooting:
    echo 1. Check your internet connection
    echo 2. Make sure you have enough disk space (~5GB)
    echo 3. Try running manually: python tools\download_checkpoints.py
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo All requirements downloaded successfully.
echo You can now start training with:
echo   python webui_amharic.py
echo.
echo Or run the end-to-end pipeline:
echo   scripts\amharic\end_to_end.ps1
echo.
pause
