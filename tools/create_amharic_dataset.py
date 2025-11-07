#!/usr/bin/env python3
"""
Amharic Dataset Creator from Media Files + Subtitles

Creates training dataset from audio files and SRT/VTT subtitles.
Supports multiple input sources and precise audio segmentation.
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from indextts.utils.front import TextNormalizer


@dataclass
class SubtitleSegment:
    """Represents a single subtitle segment"""
    start_time: float
    end_time: float
    text: str
    index: int


def parse_srt_time(time_str: str) -> float:
    """Convert SRT timestamp to seconds
    
    Args:
        time_str: Timestamp in format HH:MM:SS,mmm
    
    Returns:
        Time in seconds
    """
    # Format: HH:MM:SS,mmm
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_vtt_time(time_str: str) -> float:
    """Convert VTT timestamp to seconds
    
    Args:
        time_str: Timestamp in format HH:MM:SS.mmm or MM:SS.mmm
    
    Returns:
        Time in seconds
    """
    parts = time_str.split(':')
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid VTT timestamp: {time_str}")


def parse_srt_file(srt_path: Path) -> List[SubtitleSegment]:
    """Parse SRT subtitle file
    
    Args:
        srt_path: Path to SRT file
    
    Returns:
        List of SubtitleSegment objects
    """
    segments = []
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newline to get individual subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # First line is the index
        try:
            index = int(lines[0])
        except ValueError:
            continue
        
        # Second line is the timestamp
        timestamp_match = re.match(
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
            lines[1]
        )
        if not timestamp_match:
            continue
        
        start_time = parse_srt_time(timestamp_match.group(1))
        end_time = parse_srt_time(timestamp_match.group(2))
        
        # Rest is the text
        text = ' '.join(lines[2:]).strip()
        
        if text:
            segments.append(SubtitleSegment(
                start_time=start_time,
                end_time=end_time,
                text=text,
                index=index
            ))
    
    return segments


def parse_vtt_file(vtt_path: Path) -> List[SubtitleSegment]:
    """Parse VTT subtitle file
    
    Args:
        vtt_path: Path to VTT file
    
    Returns:
        List of SubtitleSegment objects
    """
    segments = []
    
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove WEBVTT header and metadata
    content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
    
    # Split by double newline
    blocks = re.split(r'\n\s*\n', content.strip())
    
    index = 0
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        
        # Find the timestamp line
        timestamp_line = None
        text_start_idx = 0
        
        for i, line in enumerate(lines):
            if '-->' in line:
                timestamp_line = line
                text_start_idx = i + 1
                break
        
        if not timestamp_line:
            continue
        
        # Parse timestamp
        timestamp_match = re.match(
            r'([\d:.]+)\s*-->\s*([\d:.]+)',
            timestamp_line
        )
        if not timestamp_match:
            continue
        
        start_time = parse_vtt_time(timestamp_match.group(1))
        end_time = parse_vtt_time(timestamp_match.group(2))
        
        # Get text
        text = ' '.join(lines[text_start_idx:]).strip()
        
        if text:
            index += 1
            segments.append(SubtitleSegment(
                start_time=start_time,
                end_time=end_time,
                text=text,
                index=index
            ))
    
    return segments


def detect_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_silence_len: float = 0.1
) -> List[Tuple[float, float]]:
    """Detect silent regions in audio
    
    Args:
        audio: Audio array
        sr: Sample rate
        threshold_db: Silence threshold in dB
        min_silence_len: Minimum silence length in seconds
    
    Returns:
        List of (start, end) tuples for silent regions
    """
    # Convert to dB
    audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
    
    # Find frames below threshold
    silent_frames = audio_db < threshold_db
    
    # Convert to time
    silent_regions = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silent_frames):
        time = i / sr
        
        if is_silent and not in_silence:
            silence_start = time
            in_silence = True
        elif not is_silent and in_silence:
            silence_end = time
            if silence_end - silence_start >= min_silence_len:
                silent_regions.append((silence_start, silence_end))
            in_silence = False
    
    return silent_regions


def refine_segment_boundaries(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    search_window: float = 0.3
) -> Tuple[float, float]:
    """Refine segment boundaries using silence detection
    
    Args:
        audio: Full audio array
        sr: Sample rate
        start_time: Initial start time
        end_time: Initial end time
        search_window: How far to search in seconds
    
    Returns:
        Refined (start_time, end_time)
    """
    # Extract search regions
    start_search_begin = max(0, start_time - search_window)
    start_search_end = start_time + search_window
    
    end_search_begin = end_time - search_window
    end_search_end = min(len(audio) / sr, end_time + search_window)
    
    # Get audio segments
    start_audio = audio[int(start_search_begin * sr):int(start_search_end * sr)]
    end_audio = audio[int(end_search_begin * sr):int(end_search_end * sr)]
    
    # Detect silence
    start_silences = detect_silence(start_audio, sr)
    end_silences = detect_silence(end_audio, sr)
    
    # Adjust start time
    new_start = start_time
    if start_silences:
        # Find silence closest to original start
        closest_silence = min(start_silences, key=lambda s: abs(s[1] - search_window))
        new_start = start_search_begin + closest_silence[1]
    
    # Adjust end time
    new_end = end_time
    if end_silences:
        # Find silence closest to original end
        closest_silence = min(end_silences, key=lambda s: abs(s[0] - search_window))
        new_end = end_search_begin + closest_silence[0]
    
    return new_start, new_end


def segment_audio(
    audio_path: Path,
    subtitle_path: Path,
    output_dir: Path,
    normalizer: TextNormalizer,
    refine_boundaries: bool = True,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> List[dict]:
    """Segment audio file according to subtitles
    
    Args:
        audio_path: Path to audio file
        subtitle_path: Path to subtitle file (SRT or VTT)
        output_dir: Output directory for segments
        normalizer: Text normalizer
        refine_boundaries: Use silence detection to refine boundaries
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds
    
    Returns:
        List of dataset entries
    """
    # Parse subtitles
    if subtitle_path.suffix.lower() == '.srt':
        segments = parse_srt_file(subtitle_path)
    elif subtitle_path.suffix.lower() in ['.vtt', '.webvtt']:
        segments = parse_vtt_file(subtitle_path)
    else:
        raise ValueError(f"Unsupported subtitle format: {subtitle_path.suffix}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=24000, mono=True)
    
    # Create output directory
    audio_output_dir = output_dir / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    entries = []
    base_name = audio_path.stem
    
    for seg in tqdm(segments, desc=f"Processing {audio_path.name}"):
        # Check duration
        duration = seg.end_time - seg.start_time
        if duration < min_duration or duration > max_duration:
            continue
        
        # Normalize text
        text = normalizer.normalize(seg.text, language="am")
        if not text or len(text.strip()) == 0:
            continue
        
        # Refine boundaries if requested
        start_time = seg.start_time
        end_time = seg.end_time
        
        if refine_boundaries:
            start_time, end_time = refine_segment_boundaries(
                audio, sr, start_time, end_time
            )
        
        # Extract audio segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio[start_sample:end_sample]
        
        # Generate unique ID
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        segment_id = f"{base_name}_{seg.index:04d}_{content_hash}"
        
        # Save audio
        audio_file = audio_output_dir / f"{segment_id}.wav"
        sf.write(audio_file, audio_segment, sr)
        
        # Create entry
        entry = {
            "id": segment_id,
            "text": text,
            "audio": str(audio_file.relative_to(output_dir)),
            "duration": float(end_time - start_time),
            "language": "am",
            "speaker": base_name,
        }
        entries.append(entry)
    
    return entries


def process_directory(
    input_dir: Path,
    output_dir: Path,
    normalizer: TextNormalizer,
    audio_exts: List[str] = None,
    subtitle_exts: List[str] = None,
) -> List[dict]:
    """Process all audio/subtitle pairs in a directory
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        normalizer: Text normalizer
        audio_exts: List of audio extensions to process
        subtitle_exts: List of subtitle extensions to process
    
    Returns:
        List of all dataset entries
    """
    if audio_exts is None:
        audio_exts = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    if subtitle_exts is None:
        subtitle_exts = ['.srt', '.vtt', '.webvtt']
    
    all_entries = []
    
    # Find all audio files
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(input_dir.glob(f"*{ext}"))
    
    for audio_file in audio_files:
        # Find matching subtitle
        subtitle_file = None
        
        # Try exact match first
        for ext in subtitle_exts:
            candidate = audio_file.with_suffix(ext)
            if candidate.exists():
                subtitle_file = candidate
                break
        
        # Try with language codes (e.g., video.am.srt, video.en.srt)
        if not subtitle_file:
            lang_codes = ['am', 'amh', 'en', 'en-US', 'en-GB']
            for lang_code in lang_codes:
                for ext in subtitle_exts:
                    # Try patterns like: video.am.srt, video.en.vtt
                    candidate = audio_file.parent / f"{audio_file.stem}.{lang_code}{ext}"
                    if candidate.exists():
                        subtitle_file = candidate
                        break
                if subtitle_file:
                    break
        
        if not subtitle_file:
            print(f"Warning: No subtitle found for {audio_file.name}")
            continue
        
        print(f"\nProcessing: {audio_file.name} + {subtitle_file.name}")
        
        try:
            entries = segment_audio(
                audio_file,
                subtitle_file,
                output_dir,
                normalizer,
            )
            all_entries.extend(entries)
            print(f"  Generated {len(entries)} segments")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return all_entries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Amharic dataset from audio and subtitle files"
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing audio and subtitle files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("amharic_dataset"),
        help="Output directory for dataset"
    )
    
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output manifest file (default: OUTPUT_DIR/manifest.jsonl)"
    )
    
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum segment duration in seconds"
    )
    
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum segment duration in seconds"
    )
    
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Don't refine boundaries with silence detection"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Setup output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.manifest is None:
        manifest_path = args.output_dir / "manifest.jsonl"
    else:
        manifest_path = args.manifest
    
    # Create normalizer
    normalizer = TextNormalizer(preferred_language="am")
    
    # Process all files
    print(f"Processing files in: {args.input_dir}")
    entries = process_directory(
        args.input_dir,
        args.output_dir,
        normalizer,
    )
    
    # Write manifest
    print(f"\nWriting manifest to: {manifest_path}")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nDataset created successfully!")
    print(f"  Total segments: {len(entries)}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
