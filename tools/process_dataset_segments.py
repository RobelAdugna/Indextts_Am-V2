#!/usr/bin/env python3
"""
Dataset Segment Noise Removal Tool

Processes existing dataset audio segments to remove background music/noise.
Maintains original filenames and ensures all segments are processed.
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import traceback
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


def find_vocal_file(
    output_files: List[str],
    temp_dir: Path,
    audio_file: Path
) -> Optional[Path]:
    """Find the vocal file from separator output
    
    Args:
        output_files: List of output file paths from separator
        temp_dir: Temporary directory where files are stored
        audio_file: Original audio file being processed
    
    Returns:
        Path to vocal file if found, None otherwise
    """
    vocal_file = None
    
    # Log temp directory contents if no output files returned
    if not output_files:
        temp_files = list(temp_dir.glob("*"))
        logging.debug(f"No output_files returned. Temp dir contents: {[f.name for f in temp_files]}")
    
    # Pattern 1: Check returned output_files list
    if output_files:
        for output_file in output_files:
            output_str = str(output_file).lower()
            if "vocal" in output_str:
                vocal_file = Path(output_file)
                break
    
    # Pattern 2: Search temp directory directly for vocal files
    if not vocal_file or not vocal_file.exists():
        vocal_patterns = ["*Vocal*.wav", "*vocal*.wav", "*Vocal*.WAV", "*vocal*.WAV"]
        for pattern in vocal_patterns:
            matches = list(temp_dir.glob(pattern))
            if matches:
                vocal_file = matches[0]
                break
    
    # Pattern 3: Look for files with the original name + vocal suffix
    if not vocal_file or not vocal_file.exists():
        base_name = audio_file.stem
        vocal_patterns = [
            f"{base_name}_(Vocals)*.wav",
            f"{base_name}_Vocals*.wav",
            f"{base_name}*vocal*.wav"
        ]
        for pattern in vocal_patterns:
            matches = list(temp_dir.glob(pattern))
            if matches:
                vocal_file = matches[0]
                break
    
    return vocal_file if vocal_file and vocal_file.exists() else None


def process_batch(
    audio_files: List[Path],
    separator: Separator,
    temp_dir: Path,
    keep_backup: bool = False
) -> List[Tuple[Path, bool, Optional[str]]]:
    """Process a chunk of audio segments sequentially (not parallel)
    
    Note: Despite the name, this processes files sequentially to avoid
    memory issues. The 'batch' refers to grouping files for progress tracking.
    
    Args:
        audio_files: List of audio file paths to process
        separator: Initialized Separator instance
        temp_dir: Temporary directory for processing
        keep_backup: If True, keep backup of original files
    
    Returns:
        List of (file_path, success, error_message) tuples
    """
    results = []
    
    for audio_file in audio_files:
        try:
            # Separate vocals in temp directory
            output_files = separator.separate(str(audio_file))
            
            # Find the vocal file using shared helper
            vocal_file = find_vocal_file(output_files, temp_dir, audio_file)
            
            if not vocal_file:
                # List what we actually got for debugging
                all_temp_files = list(temp_dir.glob("*"))
                error_msg = f"Vocal file not found. Output files: {output_files}. Temp dir: {[f.name for f in all_temp_files]}"
                results.append((audio_file, False, error_msg))
                continue
            
            # Backup original if requested
            if keep_backup:
                backup_path = audio_file.with_suffix(audio_file.suffix + '.backup')
                shutil.copy2(audio_file, backup_path)
            
            # Replace original with vocal version (maintains filename)
            shutil.move(str(vocal_file), str(audio_file))
            
            # Clean up other separation outputs (instrumentals, etc.)
            for temp_file in temp_dir.glob("*"):
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except (OSError, PermissionError):
                        pass
            
            results.append((audio_file, True, None))
        
        except Exception as e:
            results.append((audio_file, False, f"{str(e)}\n{traceback.format_exc()}"))
    
    return results


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
        
        # Find the vocal file using shared helper
        vocal_file = find_vocal_file(output_files, temp_dir, audio_file)
        
        if not vocal_file:
            # List what we actually got for debugging
            all_temp_files = list(temp_dir.glob("*"))
            error_msg = f"Vocal file not found. Output files: {output_files}. Temp dir: {[f.name for f in all_temp_files]}"
            return False, error_msg
        
        # Backup original if requested
        if keep_backup:
            backup_path = audio_file.with_suffix(audio_file.suffix + '.backup')
            shutil.copy2(audio_file, backup_path)
        
        # Replace original with vocal version (maintains filename)
        shutil.move(str(vocal_file), str(audio_file))
        
        # Clean up other separation outputs (instrumentals, etc.)
        for temp_file in temp_dir.glob("*"):
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except (OSError, PermissionError):
                    pass
        
        return True, None
    
    except Exception as e:
        return False, f"{str(e)}\n{traceback.format_exc()}"


def process_dataset(
    audio_dir: Path,
    model_name: str = "UVR-MDX-NET-Inst_HQ_4.onnx",
    keep_backup: bool = False,
    resume: bool = True,
    progress_file: Optional[Path] = None,
    batch_size: int = 10,
    chunk_size: int = 1,
    use_autocast: bool = True,
    mdx_batch_size: int = 4,
    normalization: float = 0.9
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
        # Initialize separator with GPU optimizations
        print("Initializing audio separator with GPU optimizations...")
        separator = Separator(
            output_dir=str(temp_dir),
            output_format='WAV',
            normalization_threshold=normalization
        )
        
        # Load model with GPU-specific parameters
        model_params = {}
        if use_autocast:
            model_params['use_autocast'] = True
        if 'mdx' in model_name.lower():
            model_params['mdx_batch_size'] = mdx_batch_size
        
        separator.load_model(model_filename=model_name, **model_params)
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                logging.info(f"✓ Using GPU acceleration (CUDA {torch.version.cuda})")
                logging.info(f"  GPU: {torch.cuda.get_device_name(0)}")
                logging.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                logging.warning("⚠ Using CPU (slower)")
        except Exception as e:
            logging.warning(f"⚠ Using CPU (slower) - Could not detect GPU: {e}")
        
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
        
        logging.info(f"\nProcessing {len(files_to_process)} segments in chunks of {chunk_size}...")
        logging.info(f"GPU optimizations: autocast={use_autocast}, mdx_batch_size={mdx_batch_size}")
        
        # Process in chunks for better progress tracking
        total_processed = 0
        for batch_start in tqdm(range(0, len(files_to_process), chunk_size), desc="Processing chunks"):
            batch_end = min(batch_start + chunk_size, len(files_to_process))
            batch = files_to_process[batch_start:batch_end]
            
            # Process batch
            batch_results = process_batch(
                batch,
                separator,
                temp_dir,
                keep_backup=keep_backup
            )
            
            # Update statistics
            for audio_file, success, error in batch_results:
                total_processed += 1
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
            if progress_file and total_processed % batch_size == 0:
                try:
                    with open(progress_file, 'w') as f:
                        json.dump({
                            'processed': list(processed_files),
                            'stats': stats
                        }, f, indent=2)
                except Exception as e:
                    logging.warning(f"Could not save progress: {e}")
        
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
    model_name: str = "UVR-MDX-NET-Inst_HQ_4.onnx",
    keep_backup: bool = False,
    resume: bool = True,
    batch_size: int = 10,
    chunk_size: int = 1,
    use_autocast: bool = True,
    mdx_batch_size: int = 4,
    normalization: float = 0.9
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
        batch_size=batch_size,
        chunk_size=chunk_size,
        use_autocast=use_autocast,
        mdx_batch_size=mdx_batch_size,
        normalization=normalization
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
        default="UVR-MDX-NET-Inst_HQ_4.onnx",
        help="audio-separator model to use (default: UVR-MDX-NET-Inst_HQ_4.onnx)"
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
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Number of files to process in each chunk (default: 1). Not true parallel processing."
    )
    
    parser.add_argument(
        "--mdx-batch-size",
        type=int,
        default=4,
        help="Batch size for MDX models (default: 4, increase to 8-16 for GPUs with more VRAM)"
    )
    
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable autocast (mixed precision) for GPU processing"
    )
    
    parser.add_argument(
        "--normalization",
        type=float,
        default=0.9,
        help="Audio normalization threshold (default: 0.9)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
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
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                use_autocast=not args.no_autocast,
                mdx_batch_size=args.mdx_batch_size,
                normalization=args.normalization
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
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                use_autocast=not args.no_autocast,
                mdx_batch_size=args.mdx_batch_size,
                normalization=args.normalization
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
