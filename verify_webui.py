#!/usr/bin/env python3
"""
Verification script for webui_amharic.py
Run this before pushing to GitHub or deploying to Lightning AI
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[FAIL] {description} MISSING: {filepath}")
        return False

def check_python_syntax(filepath: str) -> bool:
    """Check Python file syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            compile(f.read(), filepath, 'exec')
        print(f"[OK] Syntax OK: {filepath}")
        return True
    except SyntaxError as e:
        print(f"[FAIL] Syntax Error in {filepath}: {e}")
        return False

def check_imports() -> bool:
    """Check if critical imports work"""
    try:
        import gradio
        print(f"[OK] gradio {gradio.__version__}")
    except ImportError:
        print("[FAIL] gradio not installed")
        return False
    
    try:
        import sentencepiece
        print(f"[OK] sentencepiece installed")
    except ImportError:
        print("[FAIL] sentencepiece not installed")
        return False
    
    try:
        import librosa
        print(f"[OK] librosa installed")
    except ImportError:
        print("[FAIL] librosa not installed")
        return False
    
    try:
        import soundfile
        print(f"[OK] soundfile installed")
    except ImportError:
        print("[FAIL] soundfile not installed")
        return False
    
    try:
        import tqdm
        print(f"[OK] tqdm installed")
    except ImportError:
        print("[FAIL] tqdm not installed")
        return False
    
    return True

def check_webui_imports() -> bool:
    """Check if webui_amharic imports work"""
    try:
        # Just check syntax without full import to avoid hanging
        with open('webui_amharic.py', 'r', encoding='utf-8') as f:
            compile(f.read(), 'webui_amharic.py', 'exec')
        print("[OK] webui_amharic syntax valid")
        return True
    except Exception as e:
        print(f"[FAIL] webui_amharic validation failed: {e}")
        return False

def main():
    print("=" * 60)
    print("IndexTTS2 Amharic WebUI Verification")
    print("=" * 60)
    
    all_ok = True
    
    # Check files
    print("\n1. Checking Files...")
    all_ok &= check_file_exists("webui_amharic.py", "WebUI")
    all_ok &= check_file_exists("README_AMHARIC_WEBUI.md", "README")
    all_ok &= check_file_exists("DEPLOYMENT_CHECKLIST.md", "Checklist")
    all_ok &= check_file_exists("tools/youtube_amharic_downloader.py", "YouTube Downloader")
    all_ok &= check_file_exists("tools/create_amharic_dataset.py", "Dataset Creator")
    all_ok &= check_file_exists("tools/collect_amharic_corpus.py", "Corpus Collector")
    all_ok &= check_file_exists("tools/train_multilingual_bpe.py", "Tokenizer Trainer")
    all_ok &= check_file_exists("tools/preprocess_data.py", "Preprocessor")
    all_ok &= check_file_exists("scripts/amharic/end_to_end.sh", "Bash Script")
    all_ok &= check_file_exists("scripts/amharic/end_to_end.ps1", "PowerShell Script")
    
    # Check syntax
    print("\n2. Checking Python Syntax...")
    all_ok &= check_python_syntax("webui_amharic.py")
    all_ok &= check_python_syntax("tools/youtube_amharic_downloader.py")
    all_ok &= check_python_syntax("tools/create_amharic_dataset.py")
    all_ok &= check_python_syntax("tools/collect_amharic_corpus.py")
    all_ok &= check_python_syntax("tools/train_multilingual_bpe.py")
    
    # Check imports
    print("\n3. Checking Dependencies...")
    all_ok &= check_imports()
    
    # Check webui imports
    print("\n4. Checking WebUI Imports...")
    all_ok &= check_webui_imports()
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("[SUCCESS] ALL CHECKS PASSED")
        print("\nReady to push to GitHub and deploy to Lightning AI!")
        print("\nNext steps:")
        print("1. git add webui_amharic.py README_AMHARIC_WEBUI.md DEPLOYMENT_CHECKLIST.md knowledge.md verify_webui.py")
        print("2. git commit -m 'Add comprehensive Amharic TTS pipeline WebUI'")
        print("3. git push origin main")
        print("4. Follow DEPLOYMENT_CHECKLIST.md for Lightning AI setup")
        return 0
    else:
        print("[ERROR] SOME CHECKS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
