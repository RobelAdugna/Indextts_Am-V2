#!/usr/bin/env python3
"""
Dataset Segment Noise Removal Tool

Processes existing dataset audio segments to remove background music/noise.
Maintains original filenames and ensures all segments are processed.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from audio_separator.separator import Separator
    from tqdm import tqdm
    SEPARATOR_AVAILABLE = True
except ImportError:
    SEPARATOR_AVAILABLE = False
    print("Warning: audio-separator not installed")
    print("Install with: pip install audio-separator")


def check_audio_separator() -> bool:
    """Check if audio-separator is available"""
    return SEPARATOR_AVAILABLE


def find_audio_segments(
    audio_dir: Path,
    audio_exts: Optional[List[str]] = None
) -> List[Path]:
    """Find all audio segment files in directory
    
    Args:
        audio_dir: Directory containing audio segments
        audio_exts: List of audio file extensions to search for
    
    Returns:
        List of audio file paths
    """
    if audio_exts is None:
        audio_exts = ['.wav', '.mp3', '.flac', '.m4a']
    
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
    
    # Sort for consistent processing order
    return sorted(audio_files)


def process_segment(
    audio_file: Path,
    separator: Separator,
    temp_dir: Path,
    keep_backup: bool = False
) -> Tuple[bool, Optional[str]]:
    """Process a single audio segment to remove background noise
    
    Args:
        audio_file: Path to audio segment
        separator: Initialized Separator instance
        temp_dir: Temporary directory for processing
        keep_backup: If True, keep backup of original file
    
    Returns:
        (success, error_message) tuple
    """
    try:
        # Separate vocals in temp directory
        output_files = separator.separate(str(audio_file))
        
        # Find the vocal file
        vocal_file = None
        for output_file in output_files:
            if "Vocals" in output_file or "vocals" in output_file:
                vocal_file = Path(output_file)
                break
        
        if not vocal_file or not vocal_file.exists():
            return False, "Vocal file not found in separator output"
        
        # Backup original if requested
        if keep_backup:
            backup_path = audio_file.with_suffix(audio_file.suffix + '.backup')
            shutil.copy2(audio_file, backup_path)
        
        # Replace original with vocal version (maintains filename)
        shutil.move(str(vocal_file), str(audio_file))
        
        # Clean up other separation outputs (instrumentals, etc.)
        for output_file in output_files:
            output_path = Path(output_file)
            if output_path.exists() and output_path != vocal_file:
                output_path.unlink()
        
        return True, None
    
    except Exception as e:
        return False, str(e)


def process_dataset(
    audio_dir: Path,
    model_name: str = "UVR-MDX-NET-Inst_HQ_3",
    keep_backup: bool = False,
    resume: bool = True,
    progress_file: Optional[Path] = None,
    batch_size: int = 10
) -> Dict[str, any]:
    """Process entire dataset directory
    
    Args:
        audio_dir: Directory containing audio segments
        model_name: audio-separator model to use
        keep_backup: If True, keep backup of original files
        resume: If True, resume from previous run
        progress_file: Path to save/load progress
        batch_size: Number of files to process before saving progress
    
    Returns:
        Statistics dictionary
    """
    if not check_audio_separator():
        raise ImportError("audio-separator is required. Install with: pip install audio-separator")
    
    # Find all audio segments
    audio_files = find_audio_segments(audio_dir)
    
    if not audio_files:
        return {
            'total': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
    
    # Load progress if resuming
    processed_files = set()
    if resume and progress_file and progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                processed_files = set(progress_data.get('processed', []))
            print(f"Resuming: {len(processed_files)} files already processed")
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")
    
    # Create temporary directory for separator output
    temp_dir = Path(tempfile.mkdtemp(prefix="segment_processing_"))
    
    try:
        # Initialize separator
        print("Initializing audio separator...")
        separator = Separator(
            output_dir=str(temp_dir),
            output_format='WAV'
        )
        separator.load_model(model_name)
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ Using GPU acceleration (CUDA {torch.version.cuda})")
            else:
                print("⚠ Using CPU (slower)")
        except:
            print("⚠ Using CPU (slower)")
        
        # Process files
        stats = {
            'total': len(audio_files),
            'processed': 0,
            'skipped': len(processed_files),
            'failed': 0,
            'errors': []
        }
        
        files_to_process = [
            f for f in audio_files 
            if str(f.relative_to(audio_dir)) not in processed_files
        ]
        
        print(f"\nProcessing {len(files_to_process)} segments...")
        
        for i, audio_file in enumerate(tqdm(files_to_process, desc="Processing segments")):
            success, error = process_segment(
                audio_file,
                separator,
                temp_dir,
                keep_backup=keep_backup
            )
            
            if success:
                stats['processed'] += 1
                processed_files.add(str(audio_file.relative_to(audio_dir)))
            else:
                stats['failed'] += 1
                stats['errors'].append({
                    'file': str(audio_file.name),
                    'error': error
                })
            
            # Save progress periodically
            if progress_file and (i + 1) % batch_size == 0:
                try:
                    with open(progress_file, 'w') as f:
                        json.dump({
                            'processed': list(processed_files),
                            'stats': stats
                        }, f, indent=2)
                except Exception as e:
                    print(f"Warning: Could not save progress: {e}")
        
        # Save final progress
        if progress_file:
            try:
                with open(progress_file, 'w') as f:
                    json.dump({
                        'processed': list(processed_files),
                        'stats': stats
                    }, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save final progress: {e}")
        
        return stats
    
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temp directory: {e}")


def process_from_manifest(
    manifest_path: Path,
    model_name: str = "UVR-MDX-NET-Inst_HQ_3",
    keep_backup: bool = False,
    resume: bool = True,
    batch_size: int = 10
) -> Dict[str, any]:
    """Process segments listed in manifest.jsonl
    
    Args:
        manifest_path: Path to manifest.jsonl file
        model_name: audio-separator model to use
        keep_backup: If True, keep backup of original files
        resume: If True, resume from previous run
        batch_size: Number of files to process before saving progress
    
    Returns:
        Statistics dictionary
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    # Read manifest to get audio file paths
    segments = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                segments.append(json.loads(line))
    
    if not segments:
        return {
            'total': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
    
    # Get dataset directory from manifest path
    dataset_dir = manifest_path.parent
    
    # Process using audio directory
    audio_dir = dataset_dir / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    # Use progress file in dataset directory
    progress_file = dataset_dir / ".noise_removal_progress.json"
    
    return process_dataset(
        audio_dir,
        model_name=model_name,
        keep_backup=keep_backup,
        resume=resume,
        progress_file=progress_file,
        batch_size=batch_size
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove background music/noise from dataset audio segments"
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--audio-dir",
        type=Path,
        help="Directory containing audio segments (e.g., dataset/audio)"
    )
    input_group.add_argument(
        "--manifest",
        type=Path,
        help="Path to manifest.jsonl (will process dataset/audio)"
    )
    
    # Processing options
    parser.add_argument(
        "--model",
        default="UVR-MDX-NET-Inst_HQ_3",
        help="audio-separator model to use (default: UVR-MDX-NET-Inst_HQ_3)"
    )
    
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="Keep backup of original files (.backup extension)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous run (start fresh)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Save progress after this many files (default: 10)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check dependencies
    if not check_audio_separator():
        print("Error: audio-separator is required")
        print("Install with: pip install audio-separator")
        sys.exit(1)
    
    print("="*60)
    print("Dataset Segment Noise Removal")
    print("="*60)
    
    try:
        if args.manifest:
            print(f"Processing from manifest: {args.manifest}")
            stats = process_from_manifest(
                args.manifest,
                model_name=args.model,
                keep_backup=args.keep_backup,
                resume=not args.no_resume,
                batch_size=args.batch_size
            )
        else:
            print(f"Processing audio directory: {args.audio_dir}")
            progress_file = args.audio_dir / ".noise_removal_progress.json"
            stats = process_dataset(
                args.audio_dir,
                model_name=args.model,
                keep_backup=args.keep_backup,
                resume=not args.no_resume,
                progress_file=progress_file,
                batch_size=args.batch_size
            )
        
        # Print results
        print("\n" + "="*60)
        print("Processing Complete")
        print("="*60)
        print(f"Total segments: {stats['total']}")
        print(f"Processed: {stats['processed']}")
        print(f"Skipped (already done): {stats['skipped']}")
        print(f"Failed: {stats['failed']}")
        
        if stats['errors']:
            print("\nErrors:")
            for error in stats['errors'][:10]:  # Show first 10 errors
                print(f"  - {error['file']}: {error['error']}")
            if len(stats['errors']) > 10:
                print(f"  ... and {len(stats['errors']) - 10} more")
        
        # Exit code
        if stats['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
