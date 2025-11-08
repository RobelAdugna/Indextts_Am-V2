#!/usr/bin/env python3
"""
Fix Subtitle Pairing for Vocal-Separated Audio Files

After music removal, vocal files have different names (e.g., file_(Vocals)_UVR_MDXNET.wav)
but subtitle files still have original names. This script copies subtitles to match
the separated vocal filenames.

Usage:
    python tools/fix_vocal_subtitles.py --vocal-dir amharic_vocals --original-dir amharic_downloads
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


def find_subtitle_file(audio_stem: str, search_dir: Path) -> Optional[Path]:
    """Find subtitle file for given audio filename
    
    Args:
        audio_stem: Audio filename without extension
        search_dir: Directory to search for subtitles
    
    Returns:
        Path to subtitle file or None
    """
    # Try various subtitle extensions and language codes
    subtitle_exts = ['.srt', '.vtt', '.webvtt']
    lang_codes = ['am', 'amh', 'en', 'en-US']
    
    # Try exact match first
    for ext in subtitle_exts:
        candidate = search_dir / f"{audio_stem}{ext}"
        if candidate.exists():
            return candidate
    
    # Try with language codes
    for lang_code in lang_codes:
        for ext in subtitle_exts:
            candidate = search_dir / f"{audio_stem}.{lang_code}{ext}"
            if candidate.exists():
                return candidate
    
    return None


def extract_original_name(vocal_filename: str) -> str:
    """Extract original filename from vocal-separated filename
    
    Examples:
        'video_(Vocals)_UVR_MDXNET_KARA_2.wav' -> 'video'
        'video_(Instrumental)_UVR_MDXNET.wav' -> 'video'
    
    Args:
        vocal_filename: Separated audio filename
    
    Returns:
        Original audio filename (stem)
    """
    # Remove extension
    stem = Path(vocal_filename).stem
    
    # Remove UVR suffixes
    # Pattern: _(Vocals|Instrumental)_UVR_*
    pattern = r'_(Vocals|Instrumental)_UVR.*$'
    original = re.sub(pattern, '', stem, flags=re.IGNORECASE)
    
    return original


def copy_subtitle_for_vocal(
    vocal_file: Path,
    original_dir: Path,
    output_dir: Path = None,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """Copy subtitle file to match vocal filename
    
    Args:
        vocal_file: Path to vocal audio file
        original_dir: Directory containing original subtitle files
        output_dir: Output directory (default: same as vocal_file)
        dry_run: If True, only print actions without executing
    
    Returns:
        (success, message) tuple
    """
    if output_dir is None:
        output_dir = vocal_file.parent
    
    # Extract original name
    original_stem = extract_original_name(vocal_file.name)
    
    # Find subtitle in original directory
    subtitle_file = find_subtitle_file(original_stem, original_dir)
    
    if not subtitle_file:
        return False, f"No subtitle found for '{original_stem}' in {original_dir}"
    
    # Create new subtitle filename matching vocal file
    new_subtitle_name = vocal_file.stem + subtitle_file.suffix
    new_subtitle_path = output_dir / new_subtitle_name
    
    # Check if already exists
    if new_subtitle_path.exists():
        return True, f"Subtitle already exists: {new_subtitle_name}"
    
    if dry_run:
        return True, f"Would copy: {subtitle_file.name} -> {new_subtitle_name}"
    
    # Copy subtitle
    try:
        shutil.copy2(subtitle_file, new_subtitle_path)
        return True, f"Copied: {subtitle_file.name} -> {new_subtitle_name}"
    except Exception as e:
        return False, f"Error copying {subtitle_file.name}: {e}"


def process_directory(
    vocal_dir: Path,
    original_dir: Path,
    output_dir: Path = None,
    dry_run: bool = False
) -> dict:
    """Process all vocal files in directory
    
    Args:
        vocal_dir: Directory containing vocal-separated audio files
        original_dir: Directory containing original subtitle files
        output_dir: Output directory for new subtitles (default: same as vocal_dir)
        dry_run: If True, only print actions
    
    Returns:
        Statistics dictionary
    """
    if output_dir is None:
        output_dir = vocal_dir
    
    stats = {
        'total_vocals': 0,
        'subtitles_copied': 0,
        'subtitles_existed': 0,
        'subtitles_not_found': 0,
        'errors': 0
    }
    
    # Find all vocal files
    vocal_pattern = r'_(Vocals|Instrumental)_UVR.*\.(wav|mp3|flac|m4a)$'
    vocal_files = []
    
    for ext in ['.wav', '.mp3', '.flac', '.m4a']:
        for f in vocal_dir.glob(f"*{ext}"):
            if re.search(vocal_pattern, f.name, re.IGNORECASE):
                vocal_files.append(f)
    
    stats['total_vocals'] = len(vocal_files)
    
    if not vocal_files:
        print(f"⚠️  No vocal-separated files found in {vocal_dir}")
        return stats
    
    print(f"Found {len(vocal_files)} vocal-separated files")
    if dry_run:
        print("\n🔍 DRY RUN MODE - No files will be modified\n")
    
    # Process each vocal file
    for vocal_file in vocal_files:
        success, message = copy_subtitle_for_vocal(
            vocal_file, original_dir, output_dir, dry_run
        )
        
        if success:
            if "already exists" in message:
                stats['subtitles_existed'] += 1
                print(f"  ✓ {message}")
            else:
                stats['subtitles_copied'] += 1
                print(f"  ✅ {message}")
        else:
            if "No subtitle found" in message:
                stats['subtitles_not_found'] += 1
                print(f"  ⚠️  {message}")
            else:
                stats['errors'] += 1
                print(f"  ❌ {message}")
    
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix subtitle pairing for vocal-separated audio files"
    )
    
    parser.add_argument(
        "--vocal-dir",
        type=Path,
        required=True,
        help="Directory containing vocal-separated audio files"
    )
    
    parser.add_argument(
        "--original-dir",
        type=Path,
        required=True,
        help="Directory containing original subtitle files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for subtitle copies (default: same as vocal-dir)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing them"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.vocal_dir.exists():
        print(f"❌ Error: Vocal directory not found: {args.vocal_dir}")
        return 1
    
    if not args.original_dir.exists():
        print(f"❌ Error: Original directory not found: {args.original_dir}")
        return 1
    
    print(f"\n📂 Processing vocal files...")
    print(f"  Vocal dir: {args.vocal_dir}")
    print(f"  Original dir: {args.original_dir}")
    print(f"  Output dir: {args.output_dir or args.vocal_dir}")
    print()
    
    stats = process_directory(
        args.vocal_dir,
        args.original_dir,
        args.output_dir,
        args.dry_run
    )
    
    # Print summary
    print(f"\n{'📊' if not args.dry_run else '🔍'} Summary:")
    print(f"  Total vocal files: {stats['total_vocals']}")
    print(f"  Subtitles copied: {stats['subtitles_copied']}")
    print(f"  Already existed: {stats['subtitles_existed']}")
    print(f"  Not found: {stats['subtitles_not_found']}")
    print(f"  Errors: {stats['errors']}")
    
    if args.dry_run:
        print("\n💡 Run without --dry-run to actually copy files")
    
    return 0


if __name__ == "__main__":
    exit(main())
