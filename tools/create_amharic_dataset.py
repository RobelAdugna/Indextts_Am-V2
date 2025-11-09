#!/usr/bin/env python3
"""
Amharic Dataset Creator from Media Files + Subtitles

Creates training dataset from audio files and SRT/VTT subtitles.
Supports multiple input sources and precise audio segmentation with quality filtering.
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("Warning: webrtcvad not installed. Falling back to margin-based segmentation.")
    print("Install with: pip install webrtcvad")

from indextts.utils.front import TextNormalizer


@dataclass
class SubtitleSegment:
    """Represents a single subtitle segment"""
    start_time: float
    end_time: float
    text: str
    index: int


@dataclass
class QualityMetrics:
    """Quality metrics for a segment"""
    snr: float
    silence_ratio: float
    speech_rate: float  # characters per second
    clipping_ratio: float
    amharic_ratio: float
    passed: bool
    rejection_reasons: List[str]


# =============================================================================
# RMS and Silence Detection (from slicer2.py)
# =============================================================================

def get_rms(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Calculate RMS energy per frame (from slicer2.py)
    
    Args:
        y: Audio signal
        frame_length: Frame length
        hop_length: Hop length between frames
        pad_mode: Padding mode
    
    Returns:
        RMS values per frame
    """
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


# =============================================================================
# Text Quality Checks
# =============================================================================

def is_amharic_script(text: str) -> Tuple[bool, float]:
    """Check if text is primarily Amharic/Ethiopic script
    
    Args:
        text: Text to check
    
    Returns:
        (is_amharic, amharic_ratio) tuple
    """
    if not text:
        return False, 0.0
    
    # Ethiopic Unicode ranges
    # U+1200-U+137F: Ethiopic
    # U+1380-U+139F: Ethiopic Supplement
    # U+2D80-U+2DDF: Ethiopic Extended
    # U+AB00-U+AB2F: Ethiopic Extended-A
    ethiopic_chars = 0
    total_chars = 0
    
    for char in text:
        code = ord(char)
        total_chars += 1
        
        if ((0x1200 <= code <= 0x137F) or
            (0x1380 <= code <= 0x139F) or
            (0x2D80 <= code <= 0x2DDF) or
            (0xAB00 <= code <= 0xAB2F)):
            ethiopic_chars += 1
    
    if total_chars == 0:
        return False, 0.0
    
    ratio = ethiopic_chars / total_chars
    return ratio >= 0.5, ratio  # At least 50% Amharic


def clean_subtitle_text(text: str) -> str:
    """Remove subtitle artifacts and formatting
    
    Args:
        text: Raw subtitle text
    
    Returns:
        Cleaned text
    """
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove formatting markers
    text = re.sub(r'\{[^}]+\}', '', text)
    
    # Remove speaker labels
    text = re.sub(r'^[A-Z\s]+:\s*', '', text)
    
    # Remove sound effects markers (English and Amharic)
    text = re.sub(r'\[[^\]]+\]', '', text)  # Removes [Music], [ሙዚቃ], [Applause], etc.
    text = re.sub(r'\([^)]*[Mm]usic[^)]*\)', '', text)
    text = re.sub(r'\([^)]*[Aa]pplause[^)]*\)', '', text)
    text = re.sub(r'\([^)]*ሙዚቃ[^)]*\)', '', text)  # (ሙዚቃ) - Amharic music
    text = re.sub(r'\([^)]*ድምፅ[^)]*\)', '', text)  # (ድምፅ) - Amharic sound
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def count_words(text: str) -> int:
    """Count words in Amharic text
    
    Amharic words are typically separated by spaces or word separators (፡)
    """
    # Split by spaces and Ethiopic word separator
    words = re.split(r'[\s\u1361]+', text)
    return len([w for w in words if w.strip()])


# =============================================================================
# Audio Quality Checks
# =============================================================================

def calculate_snr(audio: np.ndarray, sr: int) -> float:
    """Calculate Signal-to-Noise Ratio
    
    Args:
        audio: Audio signal
        sr: Sample rate
    
    Returns:
        SNR in dB
    """
    # Use RMS to estimate signal and noise
    rms = get_rms(audio, hop_length=512).squeeze()
    
    if len(rms) < 10:
        return 0.0
    
    # Assume top 60% is signal, bottom 40% is noise
    sorted_rms = np.sort(rms)
    noise_threshold_idx = int(len(sorted_rms) * 0.4)
    
    noise_rms = np.mean(sorted_rms[:noise_threshold_idx])
    signal_rms = np.mean(sorted_rms[noise_threshold_idx:])
    
    if noise_rms == 0:
        return 100.0  # Very clean signal
    
    snr = 20 * np.log10(signal_rms / noise_rms)
    return float(snr)


def calculate_silence_ratio(audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> float:
    """Calculate what fraction of audio is silence
    
    Args:
        audio: Audio signal
        sr: Sample rate
        threshold_db: Silence threshold in dB
    
    Returns:
        Ratio of silent frames (0.0 to 1.0)
    """
    rms = get_rms(audio, hop_length=512).squeeze()
    
    if len(rms) == 0:
        return 1.0
    
    # Convert to dB
    rms_db = 20 * np.log10(rms + 1e-10)
    
    silent_frames = np.sum(rms_db < threshold_db)
    return silent_frames / len(rms)


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
    """Detect audio clipping
    
    Args:
        audio: Audio signal
        threshold: Amplitude threshold for clipping
    
    Returns:
        Ratio of clipped samples
    """
    clipped = np.abs(audio) >= threshold
    return np.sum(clipped) / len(audio)


def detect_speech_with_vad(
    audio: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30
) -> List[Tuple[float, float]]:
    """Use WebRTC VAD to detect speech regions
    
    Args:
        audio: Audio signal (mono)
        sr: Sample rate (must be 8000, 16000, 24000, or 48000)
        aggressiveness: VAD aggressiveness (0-3, higher = stricter)
        frame_duration_ms: Frame duration (10, 20, or 30 ms)
    
    Returns:
        List of (start_time, end_time) tuples for speech regions
    """
    if not VAD_AVAILABLE:
        return []
    
    try:
        vad = webrtcvad.Vad(aggressiveness)
    except Exception as e:
        # VAD initialization failed
        return []
    
    # Resample to 16kHz for VAD (requirement)
    if sr != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_16k * 32768).astype(np.int16)
    
    # Frame parameters
    frame_samples = int(16000 * frame_duration_ms / 1000)
    
    # Process frames
    speech_frames = []
    for i in range(0, len(audio_int16) - frame_samples, frame_samples):
        frame = audio_int16[i:i+frame_samples].tobytes()
        is_speech = vad.is_speech(frame, 16000)
        speech_frames.append((i / 16000, is_speech))
    
    # Merge consecutive speech frames into regions
    regions = []
    current_start = None
    
    for time, is_speech in speech_frames:
        if is_speech:
            if current_start is None:
                current_start = time
        else:
            if current_start is not None:
                regions.append((current_start, time))
                current_start = None
    
    # Close final region
    if current_start is not None:
        regions.append((current_start, len(audio_16k) / 16000))
    
    return regions


def find_best_boundary_with_vad(
    audio: np.ndarray,
    sr: int,
    target_time: float,
    search_window: float = 0.3,
    is_start: bool = True
) -> Optional[float]:
    """Find best boundary using VAD
    
    Args:
        audio: Full audio array
        sr: Sample rate
        target_time: Target boundary time
        search_window: How far to search
        is_start: True for start boundary, False for end
    
    Returns:
        Refined boundary time or None if VAD unavailable
    """
    # Extract search region
    search_start = max(0, target_time - search_window)
    search_end = min(len(audio) / sr, target_time + search_window)
    
    start_sample = int(search_start * sr)
    end_sample = int(search_end * sr)
    search_audio = audio[start_sample:end_sample]
    
    # Get speech regions
    regions = detect_speech_with_vad(search_audio, sr)
    
    if not regions:
        return None
    
    # Find closest speech boundary
    if is_start:
        # For start: find first speech region's start
        for region_start, region_end in regions:
            absolute_start = search_start + region_start
            if absolute_start <= target_time + search_window:
                return absolute_start
    else:
        # For end: find last speech region's end
        for region_start, region_end in reversed(regions):
            absolute_end = search_start + region_end
            if absolute_end >= target_time - search_window:
                return absolute_end
    
    return None


def assess_segment_quality(
    audio: np.ndarray,
    text: str,
    sr: int,
    config: dict
) -> QualityMetrics:
    """Assess quality of a segment
    
    Args:
        audio: Audio signal
        text: Normalized text
        sr: Sample rate
        config: Quality thresholds configuration
    
    Returns:
        QualityMetrics object
    """
    reasons = []
    
    # Audio quality
    snr = calculate_snr(audio, sr)
    silence_ratio = calculate_silence_ratio(audio, sr, config.get('silence_threshold_db', -40.0))
    clipping_ratio = detect_clipping(audio, config.get('clipping_threshold', 0.99))
    
    # Text quality
    is_amharic, amharic_ratio = is_amharic_script(text)
    word_count = count_words(text)
    duration = len(audio) / sr
    speech_rate = len(text) / duration if duration > 0 else 0
    
    # Apply thresholds
    passed = True
    
    if snr < config.get('min_snr', 15.0):
        passed = False
        reasons.append(f"Low SNR: {snr:.1f}dB")
    
    if silence_ratio > config.get('max_silence_ratio', 0.3):
        passed = False
        reasons.append(f"Too much silence: {silence_ratio:.1%}")
    
    if clipping_ratio > config.get('max_clipping_ratio', 0.01):
        passed = False
        reasons.append(f"Clipping detected: {clipping_ratio:.1%}")
    
    if not is_amharic:
        passed = False
        reasons.append(f"Not Amharic: {amharic_ratio:.1%} Ethiopic chars")
    
    if word_count < config.get('min_word_count', 3):
        passed = False
        reasons.append(f"Too few words: {word_count}")
    
    if speech_rate < config.get('min_speech_rate', 3.0):
        passed = False
        reasons.append(f"Speech too slow: {speech_rate:.1f} chars/s")
    
    if speech_rate > config.get('max_speech_rate', 25.0):
        passed = False
        reasons.append(f"Speech too fast: {speech_rate:.1f} chars/s")
    
    return QualityMetrics(
        snr=snr,
        silence_ratio=silence_ratio,
        speech_rate=speech_rate,
        clipping_ratio=clipping_ratio,
        amharic_ratio=amharic_ratio,
        passed=passed,
        rejection_reasons=reasons
    )


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


def deduplicate_subtitle_text(
    segments: List[SubtitleSegment],
    min_overlap_words: int = 3
) -> List[SubtitleSegment]:
    """Remove overlapping text from consecutive subtitle segments
    
    Many subtitle files have "rolling text" where each line repeats part
    of the previous line for viewer comprehension. This removes that overlap.
    Also handles cases where segments have overlapping timestamps with identical text.
    
    Args:
        segments: List of subtitle segments
        min_overlap_words: Minimum words for overlap detection (default: 3)
    
    Returns:
        List of segments with deduplicated text
    """
    if len(segments) <= 1:
        return segments
    
    deduplicated = []
    prev_text = ""
    
    for i, seg in enumerate(segments):
        current_text = seg.text
        
        if i == 0:
            deduplicated.append(seg)
            prev_text = current_text
            continue
        
        # Check for exact duplicate (same text in overlapping timestamps)
        if current_text == prev_text:
            # Skip this segment entirely - it's a complete duplicate
            # Don't update prev_text - keep comparing against the original
            continue
        
        # Split into words
        prev_words = prev_text.split()
        curr_words = current_text.split()
        
        if len(prev_words) < min_overlap_words or len(curr_words) < min_overlap_words:
            deduplicated.append(seg)
            prev_text = current_text
            continue
        
        # Find longest overlap: prev_suffix == curr_prefix
        max_overlap = 0
        max_check = min(len(prev_words), len(curr_words) // 2)
        
        for overlap_len in range(min_overlap_words, max_check + 1):
            prev_suffix = ' '.join(prev_words[-overlap_len:])
            curr_prefix = ' '.join(curr_words[:overlap_len])
            
            if prev_suffix == curr_prefix:
                max_overlap = overlap_len
        
        # Also check for substring matches (one text contains the other)
        # Only apply if texts are reasonably long to avoid false positives
        if max_overlap == 0 and len(current_text) > 20:
            # Check if current is a substring of previous or vice versa
            if current_text in prev_text:
                # Current is contained in previous - skip current
                continue
            elif prev_text in current_text:
                # Previous is contained in current - keep current, it's more complete
                # Update prev_text and continue to next segment
                deduplicated.append(seg)
                prev_text = current_text
                continue
        
        if max_overlap > 0:
            # Remove overlap from current
            new_text = ' '.join(curr_words[max_overlap:])
            
            # Only add if there's meaningful text left
            if new_text.strip():
                new_seg = SubtitleSegment(
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    text=new_text,
                    index=seg.index
                )
                deduplicated.append(new_seg)
            # Always update prev_text to current for next comparison
            prev_text = current_text
        else:
            deduplicated.append(seg)
            prev_text = current_text
    
    return deduplicated


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
    search_window: float = 0.2,
    threshold_db: float = -50.0,
    hop_size_ms: int = 20,
    start_margin: float = 0.15,
    end_margin: float = 0.1,
    min_silence_frames: int = 3
) -> Tuple[float, float, float]:
    """Refine segment boundaries with safety margins to prevent speech cutoff
    
    Two-stage approach:
    1. Add safety margins (expand boundaries)
    2. Optionally trim sustained silence (but never go inside original bounds)
    
    Args:
        audio: Full audio array
        sr: Sample rate  
        start_time: Initial start time from subtitle
        end_time: Initial end time from subtitle
        search_window: How far to search for silence (default: 0.2s)
        threshold_db: Silence threshold in dB (default: -50dB, very quiet)
        hop_size_ms: Hop size for RMS calculation in milliseconds
        start_margin: Safety margin before start (default: 0.15s)
        end_margin: Safety margin after end (default: 0.1s)
        min_silence_frames: Require this many consecutive silent frames
    
    Returns:
        Tuple of (refined_start, refined_end, confidence_score)
    """
    hop_size = int(sr * hop_size_ms / 1000)
    audio_duration = len(audio) / sr
    
    # Stage 1: Add safety margins (expand boundaries)
    # This prevents cutting off speech at edges
    expanded_start = max(0, start_time - start_margin)
    expanded_end = min(audio_duration, end_time + end_margin)
    
    # Stage 2: Search for sustained silence to trim excess
    # But never trim beyond original subtitle boundaries!
    start_search_begin = max(0, expanded_start)
    start_search_end = start_time  # Don't search past original start
    
    end_search_begin = end_time  # Don't search before original end  
    end_search_end = min(audio_duration, expanded_end)
    
    # Get audio segments
    start_audio = audio[int(start_search_begin * sr):int(start_search_end * sr)]
    end_audio = audio[int(end_search_begin * sr):int(end_search_end * sr)]
    
    # Find sustained silence regions for trimming
    new_start = expanded_start  # Default: use safety margin
    
    if len(start_audio) > hop_size * min_silence_frames:
        start_rms = get_rms(start_audio, hop_length=hop_size).squeeze()
        start_rms_db = 20 * np.log10(start_rms + 1e-10)
        
        # Find SUSTAINED silence (multiple consecutive frames below threshold)
        silent_mask = start_rms_db < threshold_db
        
        # Look for sustained silence from the end backward (closest to speech start)
        for i in range(len(silent_mask) - min_silence_frames, -1, -1):
            if np.all(silent_mask[i:i+min_silence_frames]):
                # Found sustained silence - trim to end of this silence region
                new_start = start_search_begin + ((i + min_silence_frames) * hop_size / sr)
                break
    
    # Same for end boundary
    new_end = expanded_end  # Default: use safety margin
    
    if len(end_audio) > hop_size * min_silence_frames:
        end_rms = get_rms(end_audio, hop_length=hop_size).squeeze()
        end_rms_db = 20 * np.log10(end_rms + 1e-10)
        
        # Find SUSTAINED silence from the beginning forward (closest to speech end)
        silent_mask = end_rms_db < threshold_db
        
        for i in range(len(silent_mask) - min_silence_frames + 1):
            if np.all(silent_mask[i:i+min_silence_frames]):
                # Found sustained silence - trim to start of this silence region
                new_end = end_search_begin + (i * hop_size / sr)
                break
    
    # Ensure boundaries make sense
    if new_end <= new_start:
        # Something went wrong, use expanded boundaries
        new_start = expanded_start
        new_end = expanded_end
    
    # Ensure we didn't somehow go inside original subtitle bounds
    new_start = min(new_start, start_time)
    new_end = max(new_end, end_time)
    
    # Calculate confidence based on how much we expanded
    expansion = (start_time - new_start) + (new_end - end_time)
    confidence = 0.9 if expansion < 0.3 else 0.7  # High confidence for reasonable margins
    
    return new_start, new_end, confidence


def refine_segment_boundaries_v2(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    prev_end_time: Optional[float] = None,
    next_start_time: Optional[float] = None,
    use_vad: bool = True,
    fallback_start_margin: float = 0.15,
    fallback_end_margin: float = 0.1
) -> Tuple[float, float, Dict[str, any]]:
    """Refine boundaries with hard constraints to prevent overlap
    
    Args:
        audio: Full audio array
        sr: Sample rate
        start_time: Subtitle start time
        end_time: Subtitle end time
        prev_end_time: Previous subtitle's end time (for boundary enforcement)
        next_start_time: Next subtitle's start time (for boundary enforcement)
        use_vad: Use VAD for speech detection
        fallback_start_margin: Safety margin if VAD unavailable
        fallback_end_margin: Safety margin if VAD unavailable
    
    Returns:
        (refined_start, refined_end, metadata_dict)
    """
    metadata = {
        'method': 'none',
        'vad_used': False,
        'constrained': False,
        'start_margin': 0.0,
        'end_margin': 0.0
    }
    
    # Calculate hard boundaries (absolute limits)
    # Never extend beyond midpoint to adjacent subtitle
    hard_start_limit = 0.0  # Can't go before audio start
    hard_end_limit = len(audio) / sr  # Can't go beyond audio end
    
    if prev_end_time is not None:
        # Midpoint between previous end and current start
        midpoint = (prev_end_time + start_time) / 2.0
        hard_start_limit = max(hard_start_limit, midpoint)
        metadata['constrained'] = True
    
    if next_start_time is not None:
        # Midpoint between current end and next start
        midpoint = (end_time + next_start_time) / 2.0
        hard_end_limit = min(hard_end_limit, midpoint)
        metadata['constrained'] = True
    
    # Try VAD first
    refined_start = start_time
    refined_end = end_time
    
    if use_vad:
        try:
            vad_start = find_best_boundary_with_vad(
                audio, sr, start_time, search_window=0.3, is_start=True
            )
            vad_end = find_best_boundary_with_vad(
                audio, sr, end_time, search_window=0.3, is_start=False
            )
            
            if vad_start is not None:
                refined_start = vad_start
                metadata['vad_used'] = True
                metadata['method'] = 'vad'
            
            if vad_end is not None:
                refined_end = vad_end
                metadata['vad_used'] = True
                metadata['method'] = 'vad'
        except:
            pass  # Fall back to margin-based
    
    # If VAD didn't work, use safety margins
    if metadata['method'] == 'none':
        # Calculate safe margins respecting hard limits
        available_before = start_time - hard_start_limit
        available_after = hard_end_limit - end_time
        
        actual_start_margin = min(fallback_start_margin, available_before)
        actual_end_margin = min(fallback_end_margin, available_after)
        
        refined_start = start_time - actual_start_margin
        refined_end = end_time + actual_end_margin
        
        metadata['method'] = 'margin'
        metadata['start_margin'] = actual_start_margin
        metadata['end_margin'] = actual_end_margin
    
    # HARD ENFORCEMENT: Never cross boundaries
    refined_start = max(hard_start_limit, min(refined_start, start_time))
    refined_end = min(hard_end_limit, max(refined_end, end_time))
    
    # Sanity check
    if refined_end <= refined_start:
        refined_start = start_time
        refined_end = end_time
        metadata['method'] = 'fallback_exact'
    
    return refined_start, refined_end, metadata


def segment_audio(
    audio_path: Path,
    subtitle_path: Path,
    output_dir: Path,
    normalizer: TextNormalizer,
    refine_boundaries: bool = True,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    quality_config: Optional[dict] = None,
    report_quality: bool = True,
    speaker_id: int = 0,
    segment_start_idx: int = 0,
    enable_text_dedup: bool = True,
    min_overlap_words: int = 3
) -> Tuple[List[dict], Dict, int]:
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
    
    # Deduplicate overlapping text (common in rolling subtitles)
    if enable_text_dedup:
        segments = deduplicate_subtitle_text(segments, min_overlap_words=min_overlap_words)
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=24000, mono=True)
    
    # Create output directory
    audio_output_dir = output_dir / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Quality config defaults
    if quality_config is None:
        quality_config = {
            'min_snr': 15.0,
            'max_silence_ratio': 0.3,
            'max_clipping_ratio': 0.01,
            'min_word_count': 3,
            'min_speech_rate': 3.0,
            'max_speech_rate': 25.0,
            'silence_threshold_db': -40.0,
            'clipping_threshold': 0.99
        }
    
    entries = []
    base_name = audio_path.stem
    quality_stats = {
        'total_segments': 0,
        'accepted': 0,
        'rejected': 0,
        'rejection_reasons': {}
    }
    
    segment_counter = segment_start_idx
    
    for idx, seg in enumerate(tqdm(segments, desc=f"Processing {audio_path.name}")):
        quality_stats['total_segments'] += 1
        
        # Check duration
        duration = seg.end_time - seg.start_time
        if duration < min_duration or duration > max_duration:
            quality_stats['rejected'] += 1
            quality_stats['rejection_reasons'].setdefault('duration', 0)
            quality_stats['rejection_reasons']['duration'] += 1
            continue
        
        # Clean and normalize text
        cleaned_text = clean_subtitle_text(seg.text)
        
        # Skip if cleaning removed all text (e.g., was only [ሙዚቃ])
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            quality_stats['rejected'] += 1
            quality_stats['rejection_reasons'].setdefault('music_or_sound_only', 0)
            quality_stats['rejection_reasons']['music_or_sound_only'] += 1
            continue
        
        text = normalizer.normalize(cleaned_text, language="am")
        if not text or len(text.strip()) == 0:
            quality_stats['rejected'] += 1
            quality_stats['rejection_reasons'].setdefault('empty_after_normalization', 0)
            quality_stats['rejection_reasons']['empty_after_normalization'] += 1
            continue
        
        # Get adjacent subtitle times for hard boundary enforcement
        prev_end_time = segments[idx - 1].end_time if idx > 0 else None
        next_start_time = segments[idx + 1].start_time if idx < len(segments) - 1 else None
        
        # Refine boundaries with hard constraints
        start_time = seg.start_time
        end_time = seg.end_time
        boundary_metadata = {}
        
        if refine_boundaries:
            start_time, end_time, boundary_metadata = refine_segment_boundaries_v2(
                audio, sr, 
                seg.start_time, seg.end_time,
                prev_end_time=prev_end_time,
                next_start_time=next_start_time,
                use_vad=True,
                fallback_start_margin=0.15,
                fallback_end_margin=0.1
            )
        
        # Extract audio segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio[start_sample:end_sample]
        
        # Quality assessment
        if report_quality:
            quality = assess_segment_quality(audio_segment, text, sr, quality_config)
            
            if not quality.passed:
                quality_stats['rejected'] += 1
                for reason in quality.rejection_reasons:
                    quality_stats['rejection_reasons'].setdefault(reason, 0)
                    quality_stats['rejection_reasons'][reason] += 1
                continue
        
        # Generate consistent segment ID
        # Format: spk{speaker_id}_{segment_number:06d}
        segment_id = f"spk{speaker_id:03d}_{segment_counter:06d}"
        segment_counter += 1
        
        # Save audio
        audio_file = audio_output_dir / f"{segment_id}.wav"
        sf.write(audio_file, audio_segment, sr)
        
        # Create entry with quality metrics
        entry = {
            "id": segment_id,
            "text": text,
            "audio": str(audio_file.relative_to(output_dir)),
            "duration": float(end_time - start_time),
            "language": "am",
            "speaker": f"spk{speaker_id:03d}",  # Consistent speaker ID
            "source_file": base_name,  # Keep original filename for reference
        }
        
        if report_quality:
            entry["quality"] = {
                "snr": float(quality.snr),
                "speech_rate": float(quality.speech_rate),
                "amharic_ratio": float(quality.amharic_ratio)
            }
        
        # Always include boundary metadata for debugging
        if boundary_metadata:
            entry["boundary_info"] = boundary_metadata
        
        entries.append(entry)
        quality_stats['accepted'] += 1
    
    return entries, quality_stats, segment_counter


def process_directory(
    input_dir: Path,
    output_dir: Path,
    normalizer: TextNormalizer,
    audio_exts: List[str] = None,
    subtitle_exts: List[str] = None,
    quality_config: Optional[dict] = None,
    report_quality: bool = True,
    single_speaker: bool = False,
    enable_text_dedup: bool = True,
    min_overlap_words: int = 3,
    start_speaker_id: int = 0,
    start_segment_number: int = 0
) -> Tuple[List[dict], Dict]:
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
    combined_stats = {
        'total_segments': 0,
        'accepted': 0,
        'rejected': 0,
        'rejection_reasons': {},
        'files_processed': 0,
        'files_failed': 0
    }
    
    # Track speaker IDs and segment numbers for consistent naming
    speaker_counter = start_speaker_id
    global_segment_counter = start_segment_number
    
    # Find all audio files
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(input_dir.glob(f"*{ext}"))
    
    for audio_file in audio_files:
        # Find matching subtitle
        subtitle_file = None
        
        # Try exact match first
        for ext in subtitle_exts:
            try:
                candidate = audio_file.with_suffix(ext)
                if candidate.exists():
                    subtitle_file = candidate
                    break
            except OSError:
                # Skip if filename too long
                continue
        
        # Try with language codes (e.g., video.am.srt, video.en.srt)
        if not subtitle_file:
            lang_codes = ['am', 'amh', 'en', 'en-US', 'en-GB']
            for lang_code in lang_codes:
                for ext in subtitle_exts:
                    try:
                        # Try patterns like: video.am.srt, video.en.vtt
                        candidate = audio_file.parent / f"{audio_file.stem}.{lang_code}{ext}"
                        if candidate.exists():
                            subtitle_file = candidate
                            break
                    except OSError:
                        # Skip if filename too long
                        continue
                if subtitle_file:
                    break
        
        if not subtitle_file:
            if len(str(audio_file)) > 250:  # Close to filesystem limit
                print(f"Warning: Filename too long (>250 chars), skipping: {audio_file.name[:80]}...")
            else:
                print(f"Warning: No subtitle found for {audio_file.name}")
            continue
        
        print(f"\nProcessing: {audio_file.name} + {subtitle_file.name} [Speaker {speaker_counter:03d}]")
        
        try:
            # Use speaker_id = 0 for single speaker, increment for multi-speaker
            current_speaker_id = 0 if single_speaker else speaker_counter
            
            entries, stats, new_segment_counter = segment_audio(
                audio_file,
                subtitle_file,
                output_dir,
                normalizer,
                quality_config=quality_config,
                report_quality=report_quality,
                speaker_id=current_speaker_id,
                segment_start_idx=global_segment_counter,
                enable_text_dedup=enable_text_dedup,
                min_overlap_words=min_overlap_words
            )
            
            # Update counters
            global_segment_counter = new_segment_counter
            if not single_speaker:
                speaker_counter += 1
            all_entries.extend(entries)
            
            # Update combined stats
            combined_stats['total_segments'] += stats['total_segments']
            combined_stats['accepted'] += stats['accepted']
            combined_stats['rejected'] += stats['rejected']
            for reason, count in stats['rejection_reasons'].items():
                combined_stats['rejection_reasons'].setdefault(reason, 0)
                combined_stats['rejection_reasons'][reason] += count
            combined_stats['files_processed'] += 1
            
            print(f"  Generated {len(entries)} segments (accepted: {stats['accepted']}, rejected: {stats['rejected']})")
        except Exception as e:
            print(f"  Error: {e}")
            combined_stats['files_failed'] += 1
            continue
    
    return all_entries, combined_stats


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
    
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Skip quality filtering"
    )
    
    parser.add_argument(
        "--min-snr",
        type=float,
        default=15.0,
        help="Minimum SNR in dB"
    )
    
    parser.add_argument(
        "--max-silence-ratio",
        type=float,
        default=0.3,
        help="Maximum silence ratio (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum word count"
    )
    
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=None,
        help="Path to save quality report JSON"
    )
    
    parser.add_argument(
        "--single-speaker",
        action="store_true",
        help="Treat all audio as single speaker (speaker ID = 0 for all)"
    )
    
    parser.add_argument(
        "--speaker-prefix",
        type=str,
        default="spk",
        help="Prefix for speaker IDs in filenames (default: 'spk')"
    )
    
    parser.add_argument(
        "--start-margin",
        type=float,
        default=0.15,
        help="Safety margin before subtitle start in seconds (default: 0.15)"
    )
    
    parser.add_argument(
        "--end-margin",
        type=float,
        default=0.1,
        help="Safety margin after subtitle end in seconds (default: 0.1)"
    )
    
    parser.add_argument(
        "--use-vad",
        action="store_true",
        default=True,
        help="Use WebRTC VAD for speech detection (recommended)"
    )
    
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD, use margin-based approach only"
    )
    
    parser.add_argument(
        "--no-text-dedup",
        action="store_true",
        help="Disable text deduplication (keep overlapping text from subtitles)"
    )
    
    parser.add_argument(
        "--min-overlap-words",
        type=int,
        default=3,
        help="Minimum words required to detect text overlap (default: 3)"
    )
    
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing dataset instead of overwriting (continues numbering)"
    )
    
    return parser.parse_args()


def get_existing_manifest_info(manifest_path: Path) -> Tuple[int, int, int]:
    """Get info from existing manifest for append mode
    
    Args:
        manifest_path: Path to existing manifest.jsonl
    
    Returns:
        (last_segment_number, last_speaker_id, total_entries)
    """
    if not manifest_path.exists():
        return 0, 0, 0
    
    last_segment = 0
    last_speaker = 0
    total = 0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    total += 1  # Only count valid entries
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
                
                # Extract segment number from ID (format: spk000_003455)
                segment_id = entry.get('id', '')
                if '_' in segment_id:
                    try:
                        num = int(segment_id.split('_')[1])
                        last_segment = max(last_segment, num)
                    except:
                        pass
                
                # Extract speaker ID
                speaker = entry.get('speaker', 'spk000')
                if speaker.startswith('spk'):
                    try:
                        spk_num = int(speaker[3:])
                        last_speaker = max(last_speaker, spk_num)
                    except:
                        pass
    
    return last_segment, last_speaker, total


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
    
    # Handle append mode
    start_segment_number = 0
    start_speaker_id = 0
    existing_entries = 0
    
    if args.append and manifest_path.exists():
        print("\n🔄 APPEND MODE: Continuing from existing dataset...")
        last_segment, last_speaker, existing_entries = get_existing_manifest_info(manifest_path)
        start_segment_number = last_segment + 1  # Start from next number
        start_speaker_id = last_speaker + (0 if args.single_speaker else 1)  # Next speaker
        
        print(f"  📊 Existing dataset info:")
        print(f"     - Total entries: {existing_entries}")
        print(f"     - Last segment: spk{last_speaker:03d}_{last_segment:06d}")
        print(f"     - Next segment will be: spk{start_speaker_id:03d}_{start_segment_number:06d}")
        print(f"  ✓ New segments will be appended to existing manifest\n")
    elif args.append:
        print("\n⚠️  Append mode requested but no existing manifest found.")
        print("  Creating new dataset...\n")
    
    # Create normalizer
    normalizer = TextNormalizer(preferred_language="am")
    
    # Build quality config (Amharic-optimized defaults)
    quality_config = {
        'min_snr': args.min_snr,
        'max_silence_ratio': args.max_silence_ratio,
        'max_clipping_ratio': 0.01,
        'min_word_count': args.min_words,
        'min_speech_rate': 5.0,  # Amharic syllabic - typically 5-20 chars/sec
        'max_speech_rate': 20.0,
        'silence_threshold_db': -40.0,
        'clipping_threshold': 0.99
    }
    
    # Process all files
    print(f"Processing files in: {args.input_dir}")
    print(f"Quality filtering: {'disabled' if args.no_quality_check else 'enabled'}")
    print(f"Speaker mode: {'single' if args.single_speaker else 'multi'}")
    print(f"Text deduplication: {'disabled' if args.no_text_dedup else 'enabled'}")
    
    entries, stats = process_directory(
        args.input_dir,
        args.output_dir,
        normalizer,
        quality_config=quality_config,
        report_quality=not args.no_quality_check,
        single_speaker=args.single_speaker,
        enable_text_dedup=not args.no_text_dedup,
        min_overlap_words=args.min_overlap_words,
        start_speaker_id=start_speaker_id,
        start_segment_number=start_segment_number
    )
    
    # Write manifest
    if args.append:
        print(f"\n📝 Appending {len(entries)} new entries to manifest: {manifest_path}")
        mode = 'a'  # Append mode
    else:
        print(f"\n📝 Writing manifest to: {manifest_path}")
        mode = 'w'  # Write mode (overwrite)
    
    with open(manifest_path, mode, encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Write quality report if requested
    if args.quality_report:
        with open(args.quality_report, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"Quality report saved to: {args.quality_report}")
    
    # Print summary
    print(f"\n{'✓' if not args.append else '✓'} Dataset {'created' if not args.append else 'updated'} successfully!")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Files failed: {stats['files_failed']}")
    print(f"  Total segments checked: {stats['total_segments']}")
    print(f"  Accepted segments: {stats['accepted']}")
    print(f"  Rejected segments: {stats['rejected']}")
    
    if args.append:
        total_entries = existing_entries + stats['accepted']
        print(f"\n  📊 Dataset totals:")
        print(f"     - Previous entries: {existing_entries}")
        print(f"     - New entries: {stats['accepted']}")
        print(f"     - Total entries now: {total_entries}")
    
    if stats['rejection_reasons']:
        print(f"\n  Rejection breakdown:")
        for reason, count in sorted(stats['rejection_reasons'].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    
    print(f"\n  Output directory: {args.output_dir}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
