#!/usr/bin/env python3
"""
YouTube Amharic Content Downloader

Downloads Amharic videos from YouTube with audio and subtitles.
Supports batch processing from URL lists.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import re


def check_yt_dlp() -> bool:
    """Check if yt-dlp is installed"""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_yt_dlp():
    """Install yt-dlp if not present"""
    print("yt-dlp not found. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp"], check=True)
    print("yt-dlp installed successfully.")


def download_video(
    url: str,
    output_dir: Path,
    download_subtitles: bool = True,
    subtitle_langs: Optional[List[str]] = None,
    audio_only: bool = True,
    audio_format: str = "wav",
    audio_quality: str = "0",  # 0 is best
) -> bool:
    """Download a single YouTube video
    
    Args:
        url: YouTube video URL
        output_dir: Output directory
        download_subtitles: Whether to download subtitles
        subtitle_langs: List of subtitle languages to download (e.g., ['am', 'en'])
        audio_only: Extract audio only
        audio_format: Output audio format (wav, mp3, etc.)
        audio_quality: Audio quality (0-9, 0 is best)
    
    Returns:
        True if successful, False otherwise
    """
    if subtitle_langs is None:
        subtitle_langs = ['am', 'en', 'amh']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build yt-dlp command
    cmd = [
        "yt-dlp",
        url,
        "--output", str(output_dir / "%(title)s.%(ext)s"),
    ]
    
    if audio_only:
        cmd.extend([
            "--extract-audio",
            "--audio-format", audio_format,
            "--audio-quality", audio_quality,
        ])
    
    if download_subtitles:
        cmd.extend([
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs", ",".join(subtitle_langs),
            "--convert-subs", "srt",
        ])
    
    # Add metadata
    cmd.extend([
        "--write-info-json",
        "--no-playlist",
    ])
    
    try:
        print(f"Downloading: {url}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading {url}: {e}")
        print(f"stderr: {e.stderr}")
        return False


def download_from_file(
    url_file: Path,
    output_dir: Path,
    download_subtitles: bool = True,
    subtitle_langs: Optional[List[str]] = None,
    audio_format: str = "wav",
) -> dict:
    """Download videos from a file containing URLs
    
    Args:
        url_file: Path to file containing YouTube URLs (one per line)
        output_dir: Output directory
        download_subtitles: Whether to download subtitles
        subtitle_langs: List of subtitle languages
        audio_format: Audio format
    
    Returns:
        Dict with success/failure counts
    """
    if not url_file.exists():
        raise FileNotFoundError(f"URL file not found: {url_file}")
    
    urls = []
    with open(url_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    
    print(f"Found {len(urls)} URLs to download")
    
    results = {'success': 0, 'failed': 0, 'total': len(urls)}
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Processing: {url}")
        if download_video(
            url,
            output_dir,
            download_subtitles=download_subtitles,
            subtitle_langs=subtitle_langs,
            audio_format=audio_format,
        ):
            results['success'] += 1
        else:
            results['failed'] += 1
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Amharic content from YouTube with subtitles"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url",
        type=str,
        help="Single YouTube URL to download"
    )
    group.add_argument(
        "--url-file",
        type=Path,
        help="File containing YouTube URLs (one per line)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("amharic_downloads"),
        help="Output directory for downloaded files"
    )
    
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Don't download subtitles"
    )
    
    parser.add_argument(
        "--subtitle-langs",
        nargs="+",
        default=["am", "en", "amh"],
        help="Subtitle languages to download (default: am en amh)"
    )
    
    parser.add_argument(
        "--audio-format",
        default="wav",
        choices=["wav", "mp3", "flac", "m4a"],
        help="Audio format (default: wav)"
    )
    
    parser.add_argument(
        "--video",
        action="store_true",
        help="Download video instead of audio only"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check for yt-dlp
    if not check_yt_dlp():
        install_yt_dlp()
    
    download_subtitles = not args.no_subtitles
    audio_only = not args.video
    
    if args.url:
        # Single URL
        success = download_video(
            args.url,
            args.output_dir,
            download_subtitles=download_subtitles,
            subtitle_langs=args.subtitle_langs,
            audio_only=audio_only,
            audio_format=args.audio_format,
        )
        sys.exit(0 if success else 1)
    
    elif args.url_file:
        # Batch download
        results = download_from_file(
            args.url_file,
            args.output_dir,
            download_subtitles=download_subtitles,
            subtitle_langs=args.subtitle_langs,
            audio_format=args.audio_format,
        )
        
        print("\n" + "="*50)
        print("Download Summary:")
        print(f"  Total:   {results['total']}")
        print(f"  Success: {results['success']}")
        print(f"  Failed:  {results['failed']}")
        print("="*50)
        
        sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
