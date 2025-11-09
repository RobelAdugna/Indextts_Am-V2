@echo off
REM Example: Append new data to existing Amharic dataset
REM This continues numbering from where you left off

echo ========================================
echo Incremental Dataset Expansion Example
echo ========================================
echo.

REM Check if existing dataset exists
if not exist "amharic_dataset\manifest.jsonl" (
    echo ERROR: No existing dataset found at amharic_dataset/manifest.jsonl
    echo.
    echo Create initial dataset first:
    echo   python tools\create_amharic_dataset.py --input-dir downloads --output-dir amharic_dataset --single-speaker
    echo.
    pause
    exit /b 1
)

echo Found existing dataset ✓
echo.

REM Check if new downloads directory exists
if not exist "new_downloads" (
    echo ERROR: No new_downloads directory found
    echo.
    echo Download new content first:
    echo   python tools\youtube_amharic_downloader.py --url-file new_urls.txt --output-dir new_downloads
    echo.
    pause
    exit /b 1
)

echo Found new downloads directory ✓
echo.

echo Starting append process...
echo This will:
echo   1. Read existing manifest to find last segment number
echo   2. Continue numbering from next number
echo   3. Process new files from new_downloads/
echo   4. Append new entries to manifest
echo   5. Keep all existing files untouched
echo.

pause

REM Run dataset creation in append mode
python tools\create_amharic_dataset.py ^
  --input-dir new_downloads ^
  --output-dir amharic_dataset ^
  --append ^
  --single-speaker

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Dataset append failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Dataset Expansion Complete!
echo ========================================
echo.
echo Your dataset has been expanded with new segments.
echo All existing segments remain untouched.
echo.
echo Next step: Continue training or re-train with larger dataset
echo.
pause
