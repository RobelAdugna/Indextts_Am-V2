#!/usr/bin/env python3
"""
Automatic Checkpoint Downloader for IndexTTS2

Downloads all required pretrained model checkpoints from HuggingFace.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("ERROR: huggingface-hub not installed")
    print("Install with: pip install huggingface-hub")
    sys.exit(1)


REQUIRED_FILES = {
    "gpt.pth": "GPT model checkpoint",
    "bpe.model": "Base BPE tokenizer",
    "s2mel.pth": "Semantic-to-mel model",
    "wav2vec2bert_stats.pt": "Feature extraction statistics",
    "feat1.pt": "Speaker embedding matrix",
    "feat2.pt": "Emotion embedding matrix",
    "config.yaml": "Model configuration"
}


MODEL_REPO = "IndexTeam/IndexTTS-2"


def check_existing_files(output_dir: Path) -> dict:
    """Check which files already exist
    
    Returns:
        Dict of {filename: exists}
    """
    status = {}
    for filename in REQUIRED_FILES.keys():
        file_path = output_dir / filename
        status[filename] = file_path.exists()
    return status


def download_file(repo_id: str, filename: str, output_dir: Path, force: bool = False) -> bool:
    """Download a single file from HuggingFace
    
    Args:
        repo_id: HuggingFace repository ID
        filename: Name of file to download
        output_dir: Directory to save to
        force: Re-download even if exists
    
    Returns:
        True if successful
    """
    file_path = output_dir / filename
    
    if file_path.exists() and not force:
        print(f"✓ {filename} already exists, skipping")
        return True
    
    try:
        print(f"⬇ Downloading {filename} ({REQUIRED_FILES[filename]})...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False
        )
        print(f"✓ {filename} downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False


def download_all_checkpoints(
    output_dir: Path,
    force: bool = False,
    repo_id: str = MODEL_REPO
) -> tuple:
    """Download all required checkpoints
    
    Returns:
        (success_count, failed_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("IndexTTS2 Checkpoint Downloader")
    print("="*60)
    print(f"Repository: {repo_id}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check existing files
    existing = check_existing_files(output_dir)
    existing_count = sum(existing.values())
    
    if existing_count > 0 and not force:
        print(f"Found {existing_count}/{len(REQUIRED_FILES)} existing files:")
        for filename, exists in existing.items():
            if exists:
                print(f"  ✓ {filename}")
        print()
    
    # Download missing/all files
    success = 0
    failed = 0
    
    for filename in REQUIRED_FILES.keys():
        if download_file(repo_id, filename, output_dir, force):
            success += 1
        else:
            failed += 1
    
    print()
    print("="*60)
    print("Download Summary")
    print("="*60)
    print(f"Successfully downloaded: {success}/{len(REQUIRED_FILES)}")
    print(f"Failed: {failed}/{len(REQUIRED_FILES)}")
    
    if failed == 0:
        print("\n✓ All checkpoints ready!")
        print(f"\nYou can now start training with:")
        print(f"  python webui_amharic.py")
    else:
        print("\n✗ Some downloads failed. Check errors above.")
    
    return success, failed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download IndexTTS2 pretrained checkpoints from HuggingFace"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Output directory for checkpoints (default: checkpoints)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they exist"
    )
    
    parser.add_argument(
        "--repo-id",
        default=MODEL_REPO,
        help=f"HuggingFace repository ID (default: {MODEL_REPO})"
    )
    
    return parser.parse_args()


def main():
    if not HF_AVAILABLE:
        sys.exit(1)
    
    args = parse_args()
    
    success, failed = download_all_checkpoints(
        output_dir=args.output_dir,
        force=args.force,
        repo_id=args.repo_id
    )
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
