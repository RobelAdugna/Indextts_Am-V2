#!/usr/bin/env python3
"""
Universal Media Downloader for Amharic Dataset Creation

Supports:
- Direct media URLs (MP4, WAV, MP3, etc.)
- Auto-pairing with SRT files from different sources
- Local files + remote URLs (hybrid mode)
- Batch processing from manifest files
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm


class MediaDownloader:
    """Universal media and subtitle downloader"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, output_path: Path, description: str = "Downloading") -> bool:
        """Download file from URL with progress bar
        
        Args:
            url: URL to download from
            output_path: Where to save the file
            description: Progress bar description
        
        Returns:
            True if successful
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
        
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def get_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        parsed = urlparse(url)
        filename = Path(parsed.path).name
        if not filename:
            filename = "downloaded_file"
        return filename
    
    def download_media_srt_pair(
        self,
        media_url: str,
        srt_url: Optional[str] = None,
        media_local: Optional[Path] = None,
        srt_local: Optional[Path] = None,
        base_name: Optional[str] = None
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Download or use local media and SRT pair
        
        Args:
            media_url: URL to media file (or None if using local)
            srt_url: URL to SRT file (or None if using local)
            media_local: Path to local media file
            srt_local: Path to local SRT file
            base_name: Base name for saved files (auto-generated if None)
        
        Returns:
            Tuple of (media_path, srt_path) or (None, None) on failure
        """
        # Determine base name
        if base_name is None:
            if media_url:
                base_name = Path(self.get_filename_from_url(media_url)).stem
            elif media_local:
                base_name = media_local.stem
            else:
                base_name = "media"
        
        media_path = None
        srt_path = None
        
        # Handle media
        if media_local:
            # Use local file
            if not media_local.exists():
                print(f"Error: Local media file not found: {media_local}")
                return None, None
            media_path = media_local
            print(f"Using local media: {media_local}")
        
        elif media_url:
            # Download from URL
            media_filename = self.get_filename_from_url(media_url)
            media_ext = Path(media_filename).suffix or ".mp4"
            media_path = self.output_dir / f"{base_name}{media_ext}"
            
            print(f"Downloading media: {media_url}")
            if not self.download_file(media_url, media_path, f"Media: {base_name}"):
                return None, None
        
        # Handle SRT
        if srt_local:
            # Use local file
            if not srt_local.exists():
                print(f"Warning: Local SRT file not found: {srt_local}")
                srt_path = None
            else:
                srt_path = srt_local
                print(f"Using local SRT: {srt_local}")
        
        elif srt_url:
            # Download from URL
            srt_path = self.output_dir / f"{base_name}.srt"
            
            print(f"Downloading SRT: {srt_url}")
            if not self.download_file(srt_url, srt_path, f"SRT: {base_name}"):
                srt_path = None
        
        return media_path, srt_path


def process_manifest(
    manifest_path: Path,
    output_dir: Path
) -> List[Tuple[Path, Optional[Path]]]:
    """Process manifest file with media/SRT pairs
    
    Manifest format (JSONL):
    {
        "media_url": "https://example.com/media.mp4",  # Optional
        "srt_url": "https://example.com/media.srt",    # Optional
        "media_local": "path/to/local/media.mp4",      # Optional
        "srt_local": "path/to/local/media.srt",        # Optional
        "base_name": "custom_name"                     # Optional
    }
    
    Args:
        manifest_path: Path to manifest file
        output_dir: Output directory
    
    Returns:
        List of (media_path, srt_path) tuples
    """
    downloader = MediaDownloader(output_dir)
    pairs = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            
            # Extract parameters
            media_url = entry.get('media_url')
            srt_url = entry.get('srt_url')
            media_local = Path(entry['media_local']) if entry.get('media_local') else None
            srt_local = Path(entry['srt_local']) if entry.get('srt_local') else None
            base_name = entry.get('base_name')
            
            # Validate
            if not media_url and not media_local:
                print(f"Line {line_num}: Missing both media_url and media_local")
                continue
            
            # Download/use files
            media_path, srt_path = downloader.download_media_srt_pair(
                media_url=media_url,
                srt_url=srt_url,
                media_local=media_local,
                srt_local=srt_local,
                base_name=base_name
            )
            
            if media_path:
                pairs.append((media_path, srt_path))
    
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal media downloader with auto SRT pairing"
    )
    
    # Single pair mode
    parser.add_argument(
        "--media-url",
        type=str,
        help="URL to media file"
    )
    
    parser.add_argument(
        "--srt-url",
        type=str,
        help="URL to SRT file"
    )
    
    parser.add_argument(
        "--media-local",
        type=Path,
        help="Path to local media file"
    )
    
    parser.add_argument(
        "--srt-local",
        type=Path,
        help="Path to local SRT file"
    )
    
    parser.add_argument(
        "--base-name",
        type=str,
        help="Base name for output files"
    )
    
    # Batch mode
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSONL manifest file with multiple media/SRT pairs"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloaded_media"),
        help="Output directory"
    )
    
    # Processing
    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Automatically create dataset from downloaded pairs"
    )
    
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("amharic_dataset"),
        help="Dataset output directory (if --create-dataset)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    pairs = []
    
    if args.manifest:
        # Batch mode
        print(f"Processing manifest: {args.manifest}")
        pairs = process_manifest(args.manifest, args.output_dir)
    
    else:
        # Single pair mode
        if not args.media_url and not args.media_local:
            print("Error: Must provide either --media-url or --media-local")
            sys.exit(1)
        
        downloader = MediaDownloader(args.output_dir)
        media_path, srt_path = downloader.download_media_srt_pair(
            media_url=args.media_url,
            srt_url=args.srt_url,
            media_local=args.media_local,
            srt_local=args.srt_local,
            base_name=args.base_name
        )
        
        if media_path:
            pairs.append((media_path, srt_path))
    
    # Summary
    print("\n" + "="*50)
    print("Download Summary")
    print("="*50)
    print(f"Total pairs: {len(pairs)}")
    print(f"With subtitles: {sum(1 for _, srt in pairs if srt is not None)}")
    print(f"Output directory: {args.output_dir}")
    
    # Optionally create dataset
    if args.create_dataset and pairs:
        print("\nCreating dataset...")
        try:
            from tools.create_amharic_dataset import segment_audio
            from indextts.utils.front import TextNormalizer
            
            normalizer = TextNormalizer(preferred_language="am")
            all_entries = []
            
            for media_path, srt_path in pairs:
                if srt_path and srt_path.exists():
                    print(f"Processing: {media_path.name}")
                    entries = segment_audio(
                        media_path,
                        srt_path,
                        args.dataset_output,
                        normalizer
                    )
                    all_entries.extend(entries)
            
            # Write manifest
            manifest_path = args.dataset_output / "manifest.jsonl"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                for entry in all_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"\nDataset created: {len(all_entries)} segments")
            print(f"Manifest: {manifest_path}")
        
        except ImportError as e:
            print(f"Error: Cannot create dataset. Missing dependencies: {e}")
            print("Run manually: python tools/create_amharic_dataset.py")


if __name__ == "__main__":
    main()
