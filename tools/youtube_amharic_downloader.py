#!/usr/bin/env python3
"""
YouTube Amharic Content Downloader - Enhanced Version

Downloads Amharic videos from YouTube with audio and subtitles.
Supports batch processing from URL lists.

Enhancements:
- URL validation before download
- Subtitle availability checking
- Post-download cleanup of files without subtitles
- Temporary folder cleanup
- Background music/noise removal integration
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
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


def check_audio_separator() -> bool:
    """Check if audio-separator is installed"""
    try:
        import audio_separator
        return True
    except ImportError:
        return False


def validate_youtube_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validate YouTube URL format
    
    Args:
        url: URL to validate
    
    Returns:
        (is_valid, error_message) tuple
    """
    # YouTube URL patterns
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return True, None
    
    return False, f"Invalid YouTube URL format: {url}"


def check_subtitle_availability(
    url: str,
    subtitle_langs: List[str]
) -> Tuple[bool, Dict[str, bool]]:
    """Check if subtitles are available for the video
    
    Args:
        url: YouTube video URL
        subtitle_langs: List of language codes to check
    
    Returns:
        (has_any_subtitle, lang_availability_dict)
    """
    try:
        # Get video info
        cmd = [
            "yt-dlp",
            "--list-subs",
            "--skip-download",
            url
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        
        output = result.stdout.lower()
        
        # Check for each requested language
        lang_availability = {}
        for lang in subtitle_langs:
            # Check for both manual and auto-generated subtitles
            lang_availability[lang] = f"{lang.lower()}" in output
        
        has_any = any(lang_availability.values())
        return has_any, lang_availability
    
    except subprocess.TimeoutExpired:
        print(f"  Warning: Timeout checking subtitles for {url}")
        return False, {}
    except subprocess.CalledProcessError as e:
        print(f"  Warning: Error checking subtitles: {e}")
        return False, {}


def download_video(
    url: str,
    output_dir: Path,
    download_subtitles: bool = True,
    subtitle_langs: Optional[List[str]] = None,
    audio_only: bool = True,
    audio_format: str = "wav",
    audio_quality: str = "0",
    check_subs_first: bool = True,
    temp_dir: Optional[Path] = None
) -> Tuple[bool, Optional[Path], Optional[Path]]:
    """Download a single YouTube video
    
    Args:
        url: YouTube video URL
        output_dir: Output directory
        download_subtitles: Whether to download subtitles
        subtitle_langs: List of subtitle languages to download
        audio_only: Extract audio only
        audio_format: Output audio format
        audio_quality: Audio quality (0-9, 0 is best)
        check_subs_first: Check subtitle availability before download
        temp_dir: Temporary directory for downloads
    
    Returns:
        (success, audio_file_path, subtitle_file_path) tuple
    """
    if subtitle_langs is None:
        subtitle_langs = ['am', 'en', 'amh']
    
    # Validate URL
    is_valid, error = validate_youtube_url(url)
    if not is_valid:
        print(f"✗ {error}")
        return False, None, None
    
    # Check subtitle availability if requested
    if check_subs_first and download_subtitles:
        print(f"  Checking subtitle availability...")
        has_subs, lang_avail = check_subtitle_availability(url, subtitle_langs)
        
        if not has_subs:
            print(f"✗ No subtitles found in requested languages: {subtitle_langs}")
            print(f"  Skipping download.")
            return False, None, None
        else:
            available_langs = [lang for lang, avail in lang_avail.items() if avail]
            print(f"  ✓ Subtitles available: {', '.join(available_langs)}")
    
    # Use temp directory if provided, otherwise use output_dir
    download_dir = temp_dir if temp_dir else output_dir
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Build yt-dlp command
    cmd = [
        "yt-dlp",
        url,
        "--output", str(download_dir / "%(title)s.%(ext)s"),
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
        
        # Find downloaded files
        audio_files = list(download_dir.glob(f"*.{audio_format}"))
        subtitle_files = list(download_dir.glob("*.srt"))
        
        # Get the most recently created files
        audio_file = max(audio_files, key=lambda p: p.stat().st_mtime) if audio_files else None
        subtitle_file = None
        
        # Find matching subtitle
        if audio_file and subtitle_files:
            audio_stem = audio_file.stem
            for srt in subtitle_files:
                if srt.stem.startswith(audio_stem):
                    subtitle_file = srt
                    break
        
        # Move from temp to output if using temp dir
        if temp_dir and audio_file:
            final_audio = output_dir / audio_file.name
            shutil.move(str(audio_file), str(final_audio))
            audio_file = final_audio
            
            if subtitle_file:
                final_subtitle = output_dir / subtitle_file.name
                shutil.move(str(subtitle_file), str(final_subtitle))
                subtitle_file = final_subtitle
        
        print(f"✓ Downloaded successfully")
        if subtitle_file:
            print(f"  Audio: {audio_file.name}")
            print(f"  Subtitle: {subtitle_file.name}")
        else:
            print(f"  Audio: {audio_file.name}")
            print(f"  Warning: No subtitle file found")
        
        return True, audio_file, subtitle_file
    
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading {url}: {e}")
        print(f"stderr: {e.stderr}")
        return False, None, None


def cleanup_files_without_subtitles(
    output_dir: Path,
    audio_exts: Optional[List[str]] = None,
    subtitle_exts: Optional[List[str]] = None,
    dry_run: bool = False
) -> Dict[str, int]:
    """Remove audio files that don't have matching subtitle files
    
    Args:
        output_dir: Directory to clean
        audio_exts: Audio file extensions to check
        subtitle_exts: Subtitle file extensions to look for
        dry_run: If True, only report what would be deleted
    
    Returns:
        Dict with cleanup statistics
    """
    if audio_exts is None:
        audio_exts = ['.wav', '.mp3', '.m4a', '.flac']
    if subtitle_exts is None:
        subtitle_exts = ['.srt', '.vtt', '.webvtt']
    
    stats = {
        'total_audio': 0,
        'with_subtitles': 0,
        'without_subtitles': 0,
        'deleted_files': 0,
        'deleted_bytes': 0
    }
    
    # Find all audio files
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(output_dir.glob(f"*{ext}"))
    
    stats['total_audio'] = len(audio_files)
    
    for audio_file in audio_files:
        # Look for matching subtitle
        has_subtitle = False
        
        # Try exact match first
        for ext in subtitle_exts:
            subtitle_file = audio_file.with_suffix(ext)
            if subtitle_file.exists():
                has_subtitle = True
                break
        
        # Try with language codes
        if not has_subtitle:
            lang_codes = ['am', 'amh', 'en', 'en-US']
            for lang_code in lang_codes:
                for ext in subtitle_exts:
                    subtitle_file = audio_file.parent / f"{audio_file.stem}.{lang_code}{ext}"
                    if subtitle_file.exists():
                        has_subtitle = True
                        break
                if has_subtitle:
                    break
        
        if has_subtitle:
            stats['with_subtitles'] += 1
        else:
            stats['without_subtitles'] += 1
            file_size = audio_file.stat().st_size
            
            if dry_run:
                print(f"Would delete: {audio_file.name} ({file_size / 1024 / 1024:.2f} MB)")
            else:
                print(f"Deleting: {audio_file.name} (no subtitle found)")
                audio_file.unlink()
                stats['deleted_files'] += 1
                stats['deleted_bytes'] += file_size
                
                # Also delete associated .info.json if exists
                info_file = audio_file.with_suffix('.info.json')
                if info_file.exists():
                    info_file.unlink()
    
    return stats


def cleanup_temp_folders(
    base_dir: Path,
    temp_patterns: Optional[List[str]] = None,
    dry_run: bool = False
) -> Dict[str, int]:
    """Clean up temporary folders after processing
    
    Args:
        base_dir: Base directory to search for temp folders
        temp_patterns: Patterns to match temp folder names
        dry_run: If True, only report what would be deleted
    
    Returns:
        Dict with cleanup statistics
    """
    if temp_patterns is None:
        temp_patterns = ['temp', 'tmp', '.temp', '.tmp', '*_temp', '*_tmp']
    
    stats = {
        'folders_found': 0,
        'folders_deleted': 0,
        'bytes_freed': 0
    }
    
    for pattern in temp_patterns:
        temp_folders = list(base_dir.glob(pattern))
        
        for folder in temp_folders:
            if not folder.is_dir():
                continue
            
            stats['folders_found'] += 1
            
            # Calculate folder size
            folder_size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
            
            if dry_run:
                print(f"Would delete temp folder: {folder.name} ({folder_size / 1024 / 1024:.2f} MB)")
            else:
                print(f"Deleting temp folder: {folder.name}")
                shutil.rmtree(folder)
                stats['folders_deleted'] += 1
                stats['bytes_freed'] += folder_size
    
    return stats


def remove_background_noise(
    audio_file: Path,
    output_dir: Path,
    model_name: str = "UVR-MDX-NET-Inst_HQ_3",
    keep_original: bool = False
) -> Optional[Path]:
    """Remove background music/noise from audio file
    
    Args:
        audio_file: Input audio file
        output_dir: Output directory for processed audio
        model_name: audio-separator model to use
        keep_original: If False, replace original with vocal-only version
    
    Returns:
        Path to processed audio file or None on failure
    """
    try:
        from audio_separator.separator import Separator
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Removing background noise from: {audio_file.name}")
        
        # Initialize separator
        separator = Separator(
            output_dir=str(output_dir),
            output_format="WAV"
        )
        
        # Load model
        separator.load_model(model_filename=model_name)
        
        # Separate vocals
        output_files = separator.separate(str(audio_file))
        
        # Find vocal file
        vocal_file = None
        for output_file in output_files:
            if "Vocals" in output_file or "vocals" in output_file:
                vocal_file = Path(output_file)
                break
        
        if vocal_file and vocal_file.exists():
            print(f"  ✓ Noise removed: {vocal_file.name}")
            
            if not keep_original:
                # Replace original with vocal version
                shutil.move(str(vocal_file), str(audio_file))
                print(f"  Replaced original with noise-free version")
                return audio_file
            else:
                return vocal_file
        else:
            print(f"  Warning: Could not find vocal output file")
            return None
    
    except ImportError:
        print(f"  Warning: audio-separator not installed. Skipping noise removal.")
        print(f"  Install with: pip install audio-separator")
        return None
    except Exception as e:
        print(f"  Error removing noise: {e}")
        return None


def download_from_file(
    url_file: Path,
    output_dir: Path,
    download_subtitles: bool = True,
    subtitle_langs: Optional[List[str]] = None,
    audio_format: str = "wav",
    check_subs_first: bool = True,
    cleanup_no_subs: bool = True,
    cleanup_temp: bool = True,
    remove_noise: bool = False,
    noise_model: str = "UVR-MDX-NET-Inst_HQ_3"
) -> dict:
    """Download videos from a file containing URLs
    
    Args:
        url_file: Path to file containing YouTube URLs (one per line)
        output_dir: Output directory
        download_subtitles: Whether to download subtitles
        subtitle_langs: List of subtitle languages
        audio_format: Audio format
        check_subs_first: Check subtitle availability before download
        cleanup_no_subs: Remove files without subtitle pairs after download
        cleanup_temp: Clean up temporary folders after completion
        remove_noise: Remove background music/noise
        noise_model: audio-separator model name
    
    Returns:
        Dict with download statistics
    """
    if not url_file.exists():
        raise FileNotFoundError(f"URL file not found: {url_file}")
    
    # Read URLs
    urls = []
    with open(url_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    
    print(f"Found {len(urls)} URLs to download")
    
    # Create temp directory
    temp_dir = output_dir / ".temp_downloads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'success': 0,
        'failed': 0,
        'total': len(urls),
        'skipped_no_subs': 0,
        'noise_removed': 0
    }
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Processing: {url}")
        
        success, audio_file, subtitle_file = download_video(
            url,
            output_dir,
            download_subtitles=download_subtitles,
            subtitle_langs=subtitle_langs,
            audio_format=audio_format,
            check_subs_first=check_subs_first,
            temp_dir=temp_dir
        )
        
        if success:
            results['success'] += 1
            
            # Remove background noise if requested
            if remove_noise and audio_file and check_audio_separator():
                vocal_file = remove_background_noise(
                    audio_file,
                    output_dir,
                    model_name=noise_model,
                    keep_original=False
                )
                if vocal_file:
                    results['noise_removed'] += 1
        else:
            if check_subs_first and download_subtitles:
                results['skipped_no_subs'] += 1
            else:
                results['failed'] += 1
    
    # Cleanup phase
    print("\n" + "="*50)
    print("Cleanup Phase")
    print("="*50)
    
    # Remove files without subtitle pairs
    if cleanup_no_subs and download_subtitles:
        print("\nCleaning up files without subtitle pairs...")
        cleanup_stats = cleanup_files_without_subtitles(output_dir)
        print(f"  Total audio files: {cleanup_stats['total_audio']}")
        print(f"  With subtitles: {cleanup_stats['with_subtitles']}")
        print(f"  Deleted: {cleanup_stats['deleted_files']} files ({cleanup_stats['deleted_bytes'] / 1024 / 1024:.2f} MB)")
    
    # Clean temp folders
    if cleanup_temp:
        print("\nCleaning up temporary folders...")
        temp_stats = cleanup_temp_folders(output_dir)
        print(f"  Deleted: {temp_stats['folders_deleted']} temp folders ({temp_stats['bytes_freed'] / 1024 / 1024:.2f} MB freed)")
        
        # Also clean up the .temp_downloads folder if it exists
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print(f"  Cleaned up temp download directory")
            except Exception as e:
                print(f"  Warning: Could not remove temp directory: {e}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Amharic content from YouTube with subtitles (Enhanced)"
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
    
    parser.add_argument(
        "--no-check-subs",
        action="store_true",
        help="Skip subtitle availability check before download"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up files without subtitle pairs"
    )
    
    parser.add_argument(
        "--no-cleanup-temp",
        action="store_true",
        help="Don't clean up temporary folders after completion"
    )
    
    parser.add_argument(
        "--remove-noise",
        action="store_true",
        help="Remove background music/noise from downloaded audio"
    )
    
    parser.add_argument(
        "--noise-model",
        default="UVR-MDX-NET-Inst_HQ_3",
        help="audio-separator model for noise removal (default: UVR-MDX-NET-Inst_HQ_3)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check for yt-dlp
    if not check_yt_dlp():
        install_yt_dlp()
    
    # Check for audio-separator if noise removal requested
    if args.remove_noise and not check_audio_separator():
        print("Warning: audio-separator not installed. Noise removal will be skipped.")
        print("Install with: pip install audio-separator")
    
    download_subtitles = not args.no_subtitles
    audio_only = not args.video
    check_subs_first = not args.no_check_subs
    cleanup_no_subs = not args.no_cleanup
    cleanup_temp = not args.no_cleanup_temp
    
    if args.url:
        # Single URL
        success, audio_file, subtitle_file = download_video(
            args.url,
            args.output_dir,
            download_subtitles=download_subtitles,
            subtitle_langs=args.subtitle_langs,
            audio_only=audio_only,
            audio_format=args.audio_format,
            check_subs_first=check_subs_first
        )
        
        if success and args.remove_noise and audio_file:
            remove_background_noise(
                audio_file,
                args.output_dir,
                model_name=args.noise_model
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
            check_subs_first=check_subs_first,
            cleanup_no_subs=cleanup_no_subs,
            cleanup_temp=cleanup_temp,
            remove_noise=args.remove_noise,
            noise_model=args.noise_model
        )
        
        print("\n" + "="*50)
        print("Download Summary:")
        print(f"  Total:   {results['total']}")
        print(f"  Success: {results['success']}")
        print(f"  Failed:  {results['failed']}")
        if check_subs_first and download_subtitles:
            print(f"  Skipped (no subs): {results['skipped_no_subs']}")
        if args.remove_noise:
            print(f"  Noise removed: {results['noise_removed']}")
        print("="*50)
        
        sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
