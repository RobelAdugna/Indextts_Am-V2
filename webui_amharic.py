#!/usr/bin/env python3
"""
IndexTTS2 Amharic Pipeline WebUI

Comprehensive interface for the complete Amharic TTS training pipeline:
- Data Collection (YouTube downloader)
- Dataset Creation (audio segmentation)
- Corpus Collection (text aggregation)
- Tokenizer Training (BPE)
- Preprocessing (feature extraction)
- Training (GPT fine-tuning)
- Inference (TTS generation)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

try:
    from audio_separator.separator import Separator
    MUSIC_REMOVAL_AVAILABLE = True
except ImportError:
    MUSIC_REMOVAL_AVAILABLE = False

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Global state for pipeline
pipeline_state = {
    "downloads_dir": None,
    "dataset_dir": None,
    "corpus_file": None,
    "tokenizer_model": None,
    "processed_dir": None,
}


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed"""
    deps = {}
    
    # Check yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        deps["yt-dlp"] = True
    except:
        deps["yt-dlp"] = False
    
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        deps["ffmpeg"] = True
    except:
        deps["ffmpeg"] = False
    
    # Check GPU
    try:
        import torch
        deps["cuda"] = torch.cuda.is_available()
        if deps["cuda"]:
            deps["cuda_version"] = torch.version.cuda
    except:
        deps["cuda"] = False
    
    return deps


def format_status_html(status: str, success: bool = True) -> str:
    """Format status message with color"""
    color = "#4CAF50" if success else "#f44336"
    return f'<div style="padding: 10px; border-radius: 5px; background: {color}20; color: {color}; font-weight: bold; margin: 10px 0;">{status}</div>'


# ============================================================================
# Tab 1: YouTube Downloader
# ============================================================================

def analyze_dataset_stats(
    manifest_path: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Analyze dataset statistics from manifest.jsonl"""
    try:
        manifest = Path(manifest_path)
        if not manifest.exists():
            return "Manifest file not found", "error"
        
        progress(0, desc="Reading manifest...")
        
        # Parse manifest
        segments = []
        with open(manifest, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    segments.append(json.loads(line))
        
        if not segments:
            return "No data in manifest", "warning"
        
        progress(0.3, desc="Calculating statistics...")
        
        # Calculate stats
        total_segments = len(segments)
        total_duration = sum(s.get('duration', 0) for s in segments)
        total_hours = total_duration / 3600
        total_minutes = total_duration / 60
        
        # Unique sources (videos)
        unique_sources = set(s.get('source_file', 'unknown') for s in segments)
        num_videos = len(unique_sources)
        
        # Speakers
        unique_speakers = set(s.get('speaker', 'unknown') for s in segments)
        num_speakers = len(unique_speakers)
        
        # Language distribution
        lang_counts = {}
        for s in segments:
            lang = s.get('language', 'unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # Duration distribution
        durations = [s.get('duration', 0) for s in segments]
        avg_duration = total_duration / total_segments if total_segments > 0 else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # Text statistics
        total_words = 0
        total_chars = 0
        for s in segments:
            text = s.get('text', '')
            total_words += len(text.split())
            total_chars += len(text)
        
        avg_words = total_words / total_segments if total_segments > 0 else 0
        avg_chars = total_chars / total_segments if total_segments > 0 else 0
        
        # Build report
        report = f"""📊 **Dataset Statistics**

### 🎯 Overview
- **Total Segments:** {total_segments:,}
- **Total Duration:** {total_hours:.2f} hours ({total_minutes:.1f} minutes)
- **Source Videos:** {num_videos}
- **Unique Speakers:** {num_speakers}

### ⏱️ Duration Statistics
- **Average Segment:** {avg_duration:.2f}s
- **Shortest Segment:** {min_duration:.2f}s
- **Longest Segment:** {max_duration:.2f}s

### 📝 Text Statistics
- **Total Words:** {total_words:,}
- **Total Characters:** {total_chars:,}
- **Avg Words/Segment:** {avg_words:.1f}
- **Avg Chars/Segment:** {avg_chars:.1f}

### 🌍 Language Distribution
"""
        
        for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_segments) * 100
            report += f"- **{lang}:** {count:,} segments ({pct:.1f}%)\n"
        
        # Top sources
        source_counts = {}
        for s in segments:
            src = s.get('source_file', 'unknown')
            source_counts[src] = source_counts.get(src, 0) + 1
        
        report += "\n### 📹 Top Source Files\n"
        for src, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"- {src}: {count:,} segments\n"
        
        return report, "success"
    
    except Exception as e:
        return f"❌ Error: {str(e)}", "error"


def remove_background_music(
    input_dir: str,
    output_dir: str,
    model_name: str = "UVR_MDXNET_KARA_2.onnx",
    cleanup_source: bool = True,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Remove background music from audio files and copy subtitle files"""
    if not MUSIC_REMOVAL_AVAILABLE:
        return "❌ audio-separator not installed. Run: pip install 'audio-separator[cpu]'", "error"
    
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg', '*.opus']:
            audio_files.extend(list(input_path.glob(ext)))
        if not audio_files:
            return "No audio files found", "warning"
        
        logs = []
        progress(0, desc="Initializing...")
        
        # Check GPU availability
        import torch
        use_cuda = torch.cuda.is_available()
        
        # Check ONNX Runtime providers
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            has_cuda_provider = 'CUDAExecutionProvider' in providers
            logs.append(f"📋 Available ONNX providers: {', '.join(providers)}")
        except Exception as e:
            has_cuda_provider = False
            logs.append(f"⚠️ Could not check ONNX providers: {e}")
        
        # Configure ONNX session options for maximum GPU performance
        session_options = None
        if has_cuda_provider:
            try:
                import onnxruntime as ort
                session_options = ort.SessionOptions()
                # Enable all optimizations
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # Use parallel execution
                session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                # Increase thread count for better GPU utilization
                session_options.intra_op_num_threads = 4
                session_options.inter_op_num_threads = 4
                logs.append(f"⚙️ Configured ONNX with performance optimizations")
            except Exception as e:
                logs.append(f"⚠️ Could not configure session options: {e}")
        
        separator = Separator(
            output_dir=str(output_path),
            output_format='WAV',
            sample_rate=44100,  # Standard sample rate
        )
        separator.load_model(model_filename=model_name)
        
        if use_cuda and has_cuda_provider:
            logs.append(f"🚀 Using GPU acceleration (CUDA) with optimizations")
            logs.append(f"💡 Processing {len(audio_files)} files with GPU...")
        elif use_cuda and not has_cuda_provider:
            logs.append(f"⚠️ GPU available but onnxruntime-gpu not installed!")
            logs.append(f"   Install with: pip uninstall onnxruntime -y && pip install onnxruntime-gpu")
        else:
            logs.append(f"⚠️ Running on CPU (slower)")
        
        # Process files with progress tracking
        import time
        start_time = time.time()
        subtitle_stats = {'copied': 0, 'not_found': 0, 'errors': 0}
        
        renamed_count = 0
        deleted_instrumental_count = 0
        cleaned_source_count = 0
        
        for i, f in enumerate(audio_files):
            file_start = time.time()
            progress((i+1)/len(audio_files), desc=f"{i+1}/{len(audio_files)}: {f.name[:30]}...")
            try:
                # Separate vocals
                output_files = separator.separate(str(f))
                file_time = time.time() - file_start
                logs.append(f"✓ {f.name} ({file_time:.1f}s)")
                
                # Find and rename vocal file, delete instrumental
                vocal_pattern = re.compile(rf'^{re.escape(f.stem)}_\(Vocals\).*\.wav$', re.IGNORECASE)
                instrumental_pattern = re.compile(rf'^{re.escape(f.stem)}_\(Instrumental\).*\.wav$', re.IGNORECASE)
                
                for output_file in output_path.glob(f"{f.stem}*.wav"):
                    if vocal_pattern.match(output_file.name):
                        # Rename vocal file to original name
                        new_name = output_path / f.name
                        if output_file != new_name:
                            output_file.rename(new_name)
                            renamed_count += 1
                            logs.append(f"  ✏️ Renamed to: {f.name}")
                    
                    elif instrumental_pattern.match(output_file.name):
                        # Delete instrumental file
                        output_file.unlink()
                        deleted_instrumental_count += 1
                        logs.append(f"  🗑️ Deleted instrumental")
                
                # Copy subtitle file to match renamed vocal file
                subtitle_exts = ['.srt', '.vtt', '.webvtt']
                lang_codes = ['am', 'amh', 'en', 'en-US']
                
                # Find subtitle for original file
                original_subtitle = None
                for ext in subtitle_exts:
                    # Try exact match
                    candidate = f.with_suffix(ext)
                    if candidate.exists():
                        original_subtitle = candidate
                        break
                    # Try with language codes
                    for lang in lang_codes:
                        candidate = f.parent / f"{f.stem}.{lang}{ext}"
                        if candidate.exists():
                            original_subtitle = candidate
                            break
                    if original_subtitle:
                        break
                
                if original_subtitle:
                    # Copy to output with same name as audio (now renamed to original)
                    new_subtitle = output_path / f.with_suffix(original_subtitle.suffix).name
                    if not new_subtitle.exists():
                        try:
                            shutil.copy2(original_subtitle, new_subtitle)
                            subtitle_stats['copied'] += 1
                            logs.append(f"  📄 Copied subtitle: {new_subtitle.name}")
                        except Exception as e:
                            subtitle_stats['errors'] += 1
                            logs.append(f"  ⚠️ Subtitle copy failed: {e}")
                else:
                    subtitle_stats['not_found'] += 1
                    logs.append(f"  ℹ️  No subtitle found for {f.name}")
                
                # Clean up source files after successful processing
                if cleanup_source:
                    # Remove original audio from input directory
                    if f.exists():
                        f.unlink()
                        cleaned_source_count += 1
                        logs.append(f"  🗑️ Cleaned source: {f.name}")
                    
                    # Remove subtitle files
                    for ext in ['.srt', '.vtt', '.webvtt']:
                        for lang in ['', '.am', '.amh', '.en', '.en-US']:
                            subtitle = f.parent / f"{f.stem}{lang}{ext}"
                            if subtitle.exists():
                                subtitle.unlink()
                                logs.append(f"  🗑️ Cleaned subtitle: {subtitle.name}")
                    
                    # Remove .info.json
                    info_file = f.with_suffix('.info.json')
                    if info_file.exists():
                        info_file.unlink()
                        logs.append(f"  🗑️ Cleaned metadata: {info_file.name}")
                    
            except Exception as e:
                logs.append(f"✗ {f.name}: {e}")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(audio_files) if audio_files else 0
        logs.insert(0, f"⏱️ Total time: {total_time:.1f}s | Avg per file: {avg_time:.1f}s")
        
        # Add processing stats
        logs.append(f"\n✅ Files auto-renamed: {renamed_count}")
        logs.append(f"🗑️ Instrumental files deleted: {deleted_instrumental_count}")
        if cleanup_source:
            logs.append(f"🗑️ Source files cleaned: {cleaned_source_count}")
        
        # Add subtitle stats
        if subtitle_stats['copied'] > 0 or subtitle_stats['not_found'] > 0:
            logs.append(f"📄 Subtitle files: {subtitle_stats['copied']} copied, {subtitle_stats['not_found']} not found, {subtitle_stats['errors']} errors")
        
        return f"✅ Processed {len(audio_files)} files (auto-cleaned)\n" + "\n".join(logs), "success"
    except Exception as e:
        return f"❌ {e}", "error"


def download_youtube_videos(
    url_input: str,
    url_file,
    output_dir: str,
    download_subtitles: bool,
    subtitle_langs: str,
    audio_format: str,
    progress=gr.Progress()
) -> Tuple[str, str, Dict]:
    """Download YouTube videos with progress tracking"""
    
    if not output_dir:
        output_dir = "amharic_downloads"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare URLs
    urls = []
    if url_file and hasattr(url_file, 'name'):
        with open(url_file.name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)
    elif url_input:
        urls = [u.strip() for u in url_input.split('\n') if u.strip()]
    
    if not urls:
        return "No URLs provided", format_status_html("❌ Error: No URLs to download", False), pipeline_state
    
    # Build command
    script_path = Path(current_dir) / "tools" / "youtube_amharic_downloader.py"
    
    logs = []
    results = {"success": 0, "failed": 0, "total": len(urls)}
    
    for i, url in enumerate(urls):
        progress((i + 1) / len(urls), desc=f"Downloading {i+1}/{len(urls)}")
        
        cmd = [
            sys.executable,
            str(script_path),
            "--url", url,
            "--output-dir", str(output_path),
            "--audio-format", audio_format,
        ]
        
        if not download_subtitles:
            cmd.append("--no-subtitles")
        else:
            cmd.extend(["--subtitle-langs"] + subtitle_langs.split())
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout per video
            )
            
            if result.returncode == 0:
                results["success"] += 1
                logs.append(f"✓ Downloaded: {url}")
            else:
                results["failed"] += 1
                logs.append(f"✗ Failed: {url}\n  {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            results["failed"] += 1
            logs.append(f"✗ Timeout: {url} (exceeded 10 minutes)")
        except Exception as e:
            results["failed"] += 1
            logs.append(f"✗ Error: {url}\n  {str(e)}")
    
    # Update pipeline state
    pipeline_state["downloads_dir"] = str(output_path)
    
    log_text = "\n".join(logs)
    summary = f"Downloaded {results['success']}/{results['total']} videos successfully"
    
    if results['failed'] == 0:
        status_html = format_status_html(f"✅ {summary}")
    else:
        status_html = format_status_html(f"⚠️ {summary} ({results['failed']} failed)", False)
    
    return log_text, status_html, pipeline_state


# ============================================================================
# Tab 2: Dataset Creation
# ============================================================================

def create_dataset(
    input_dir: str,
    output_dir: str,
    min_duration: float,
    max_duration: float,
    refine_boundaries: bool,
    use_vad: bool,
    start_margin: float,
    end_margin: float,
    enable_text_dedup: bool,
    enable_quality_filter: bool,
    append_to_dataset: bool,
    min_snr: float,
    max_silence_ratio: float,
    min_words: int,
    min_speech_rate: float,
    max_speech_rate: float,
    single_speaker: bool,
    progress=gr.Progress()
) -> Tuple[str, str, Dict]:
    """Create dataset from audio and subtitle files"""
    
    if not input_dir:
        input_dir = pipeline_state.get("downloads_dir", "amharic_downloads")
    
    if not output_dir:
        output_dir = "amharic_dataset"
    
    input_path = Path(input_dir)
    if not input_path.exists():
        return "", format_status_html(f"❌ Error: Input directory not found: {input_dir}", False), pipeline_state
    
    output_path = Path(output_dir)
    
    progress(0.1, desc="Starting dataset creation...")
    
    # Build command
    script_path = Path(current_dir) / "tools" / "create_amharic_dataset.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir", str(input_path),
        "--output-dir", str(output_path),
        "--min-duration", str(min_duration),
        "--max-duration", str(max_duration),
        "--min-snr", str(min_snr),
        "--max-silence-ratio", str(max_silence_ratio),
        "--min-words", str(min_words),
    ]
    
    if not refine_boundaries:
        cmd.append("--no-refine")
    else:
        # Add VAD flag
        if not use_vad:
            cmd.append("--no-vad")
        
        # Add safety margins
        cmd.extend(["--start-margin", str(start_margin)])
        cmd.extend(["--end-margin", str(end_margin)])
    
    if not enable_text_dedup:
        cmd.append("--no-text-dedup")
    
    if not enable_quality_filter:
        cmd.append("--no-quality-check")
    
    if append_to_dataset:
        cmd.append("--append")
    
    if single_speaker:
        cmd.append("--single-speaker")
    
    # Add quality report
    quality_report_path = output_path / "quality_report.json"
    cmd.extend(["--quality-report", str(quality_report_path)])
    
    try:
        progress(0.3, desc="Processing audio files...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        progress(0.9, desc="Finalizing...")
        
        if result.returncode == 0:
            # Parse output for statistics
            manifest_path = output_path / "manifest.jsonl"
            segment_count = 0
            if manifest_path.exists():
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    segment_count = sum(1 for _ in f)
            
            pipeline_state["dataset_dir"] = str(output_path)
            
            # Parse quality report if available
            quality_report_path = output_path / "quality_report.json"
            quality_summary = ""
            if quality_report_path.exists():
                try:
                    with open(quality_report_path, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                    quality_summary = f" (Accepted: {stats.get('accepted', 0)}, Rejected: {stats.get('rejected', 0)})"
                except:
                    pass
            
            status_html = format_status_html(f"✅ Dataset created: {segment_count} segments{quality_summary}")
            log_text = result.stdout
        else:
            status_html = format_status_html(f"❌ Error creating dataset", False)
            log_text = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    
    except Exception as e:
        status_html = format_status_html(f"❌ Error: {str(e)}", False)
        log_text = str(e)
    
    return log_text, status_html, pipeline_state


# ============================================================================
# Tab 3: Corpus Collection
# ============================================================================

def collect_corpus(
    manifest_path: str,  # Direct path to manifest.jsonl (for remote environments like Lightning AI)
    input_files,
    output_file: str,
    min_length: int,
    show_stats: bool,
    progress=gr.Progress()
) -> Tuple[str, str, Dict]:
    """Collect and clean Amharic corpus from manifest or uploaded files"""
    
    if not output_file:
        output_file = "amharic_corpus.txt"
    
    output_path = Path(output_file)
    
    # Prepare input files - prioritize direct path input
    input_paths = []
    
    # Option 1: Direct path input (for Lightning AI/remote)
    if manifest_path and manifest_path.strip():
        manifest_file = Path(manifest_path.strip()).expanduser()  # Handle ~/path
        if manifest_file.exists():
            input_paths = [str(manifest_file)]
        else:
            return "", format_status_html(f"❌ Error: File not found: {manifest_path}", False), pipeline_state
    
    # Option 2: Uploaded files
    elif input_files:
        input_paths = [f.name for f in input_files]
    
    # Option 3: Auto-fill from dataset creation step
    elif pipeline_state.get("dataset_dir"):
        manifest = Path(pipeline_state["dataset_dir"]) / "manifest.jsonl"
        if manifest.exists():
            input_paths = [str(manifest)]
    
    if not input_paths:
        return "", format_status_html("❌ Error: No input files provided. Please enter manifest path or upload files.", False), pipeline_state
    
    progress(0.1, desc="Starting corpus collection...")
    
    # Build command
    script_path = Path(current_dir) / "tools" / "collect_amharic_corpus.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--input",
    ] + input_paths + [
        "--output", str(output_path),
        "--min-length", str(min_length),
    ]
    
    if show_stats:
        cmd.append("--stats")
    
    try:
        progress(0.3, desc="Processing texts...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        progress(0.9, desc="Finalizing...")
        
        if result.returncode == 0:
            # Count lines in output
            line_count = 0
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
            
            pipeline_state["corpus_file"] = str(output_path)
            
            status_html = format_status_html(f"✅ Corpus collected: {line_count} lines")
            log_text = result.stdout
        else:
            status_html = format_status_html("❌ Error collecting corpus", False)
            log_text = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    
    except Exception as e:
        status_html = format_status_html(f"❌ Error: {str(e)}", False)
        log_text = str(e)
    
    return log_text, status_html, pipeline_state


# ============================================================================
# Tab 4: Tokenizer Training
# ============================================================================

def train_tokenizer(
    base_model_path: str,
    manifest_path: str,
    output_model: str,
    target_size: int,
    character_coverage: float,
    test_text: str,
    progress=gr.Progress()
) -> Tuple[str, str, str, Dict]:
    """Extend base BPE tokenizer with Amharic (following video workflow at 20:05-30:00)"""
    
    # Use pipeline state defaults
    if not manifest_path and pipeline_state.get("dataset_dir"):
        manifest_path = str(Path(pipeline_state["dataset_dir"]) / "manifest.jsonl")
    
    if not base_model_path:
        base_model_path = "checkpoints/bpe.model"
    
    if not output_model:
        output_model = "tokenizers/amharic_extended_bpe.model"
    
    # Validate inputs
    if not base_model_path or not base_model_path.strip():
        return "", format_status_html("❌ Error: Base model path is empty. Please specify the base model path.", False), "", pipeline_state
    
    if not Path(base_model_path).exists():
        return "", format_status_html(f"❌ Error: Base model not found: {base_model_path}\n\nPlease run download_requirements.bat first!", False), "", pipeline_state
    
    if not manifest_path or not manifest_path.strip():
        return "", format_status_html("❌ Error: Manifest path is empty. Please complete Tab 2 (Dataset Creation) first, or manually specify the manifest path.", False), "", pipeline_state
    
    if not Path(manifest_path).exists():
        return "", format_status_html(f"❌ Error: Manifest not found: {manifest_path}\n\nPlease create a dataset in Tab 2 first.", False), "", pipeline_state
    
    progress(0.1, desc="Loading base tokenizer...")
    
    # Prepare output directory
    Path(output_model).parent.mkdir(parents=True, exist_ok=True)
    
    # Build command using extend_bpe.py (matches video!)
    script_path = Path(current_dir) / "tools" / "tokenizer" / "extend_bpe.py"
    
    if not script_path.exists():
        return "", format_status_html("❌ Error: extend_bpe.py not found", False), "", pipeline_state
    
    cmd = [
        sys.executable,
        str(script_path),
        "--base-model", base_model_path,
        "--manifests", manifest_path,
        "--output-model", output_model,
        "--target-size", str(target_size),
        "--character-coverage", str(character_coverage),
    ]
    
    try:
        progress(0.3, desc="Extending tokenizer with Amharic tokens...")
        
        # Log the command being run
        cmd_str = ' '.join(cmd)
        print(f"Running command: {cmd_str}")  # Debug log
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )
        
        progress(0.9, desc="Finalizing...")
        
        if result.returncode == 0:
            model_path = Path(output_model)
            
            if model_path.exists():
                # Update pipeline state
                pipeline_state["tokenizer_model"] = str(model_path)
                
                # Get actual vocab size
                try:
                    import sentencepiece as spm
                    sp = spm.SentencePieceProcessor()
                    sp.load(str(model_path))
                    actual_vocab = sp.get_piece_size()
                    status_html = format_status_html(
                        f"✅ Tokenizer extended: {model_path.name} (vocab: {actual_vocab})"
                    )
                except:
                    status_html = format_status_html(f"✅ Tokenizer extended: {model_path.name}")
                
                # Test tokenization
                test_result = ""
                if test_text:
                    try:
                        import sentencepiece as spm
                        sp = spm.SentencePieceProcessor()
                        sp.load(str(model_path))
                        tokens = sp.encode(test_text, out_type=str)
                        token_ids = sp.encode(test_text)
                        test_result = f"✅ Test tokenization:\nText: {test_text}\nTokens: {tokens}\nIDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}"
                    except Exception as e:
                        test_result = f"Test failed: {str(e)}"
                
                log_text = result.stdout
            else:
                status_html = format_status_html("❌ Model file not created", False)
                log_text = result.stdout
                test_result = ""
        else:
            status_html = format_status_html(f"❌ Error extending tokenizer (exit code: {result.returncode})", False)
            log_text = f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nExit code: {result.returncode}"
            test_result = ""
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        status_html = format_status_html(f"❌ Error: {str(e)}", False)
        log_text = f"Exception occurred:\n{error_details}\n\nError: {str(e)}"
        test_result = ""
    
    return log_text, status_html, test_result, pipeline_state


# ============================================================================
# Tab 5: Preprocessing
# ============================================================================

def preprocess_data(
    manifest_path: str,
    output_dir: str,
    tokenizer_path: str,
    config_path: str,
    gpt_checkpoint: str,
    language: str,
    val_ratio: float,
    progress=gr.Progress()
) -> Tuple[str, str, Dict]:
    """Preprocess data for training"""
    
    # Use pipeline state defaults
    if not manifest_path and pipeline_state.get("dataset_dir"):
        manifest_path = str(Path(pipeline_state["dataset_dir"]) / "manifest.jsonl")
    
    if not tokenizer_path and pipeline_state.get("tokenizer_model"):
        tokenizer_path = pipeline_state["tokenizer_model"]
    
    if not output_dir:
        output_dir = "processed_data"
    
    if not config_path:
        config_path = "checkpoints/config.yaml"
    
    if not gpt_checkpoint:
        gpt_checkpoint = "checkpoints/gpt.pth"
    
    # Validate inputs
    if not Path(manifest_path).exists():
        return "", format_status_html(f"❌ Error: Manifest not found: {manifest_path}", False), pipeline_state
    
    if not Path(tokenizer_path).exists():
        return "", format_status_html(f"❌ Error: Tokenizer not found: {tokenizer_path}", False), pipeline_state
    
    progress(0.1, desc="Starting preprocessing...")
    
    # Build command
    script_path = Path(current_dir) / "tools" / "preprocess_data.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--manifest", manifest_path,
        "--output-dir", output_dir,
        "--tokenizer", tokenizer_path,
        "--config", config_path,
        "--gpt-checkpoint", gpt_checkpoint,
        "--language", language,
        "--val-ratio", str(val_ratio),
        # batch_size auto-detected by preprocess_data.py
    ]
    
    try:
        progress(0.3, desc="Extracting features (this will take a while)...")
        
        # Run in subprocess with streaming output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        log_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                log_lines.append(line.strip())
                # Update progress based on output
                if "Preprocessing" in line:
                    progress(0.5, desc="Processing samples...")
        
        process.wait()
        
        progress(0.9, desc="Finalizing...")
        
        if process.returncode == 0:
            pipeline_state["processed_dir"] = output_dir
            
            # Count processed samples
            train_manifest = Path(output_dir) / "train_manifest.jsonl"
            val_manifest = Path(output_dir) / "val_manifest.jsonl"
            
            train_count = 0
            val_count = 0
            
            if train_manifest.exists():
                with open(train_manifest, 'r', encoding='utf-8') as f:
                    train_count = sum(1 for _ in f)
            
            if val_manifest.exists():
                with open(val_manifest, 'r', encoding='utf-8') as f:
                    val_count = sum(1 for _ in f)
            
            status_html = format_status_html(
                f"✅ Preprocessing complete: {train_count} train, {val_count} val samples"
            )
            log_text = "\n".join(log_lines)
        else:
            status_html = format_status_html("❌ Error during preprocessing", False)
            log_text = "\n".join(log_lines)
    
    except Exception as e:
        status_html = format_status_html(f"❌ Error: {str(e)}", False)
        log_text = str(e)
    
    return log_text, status_html, pipeline_state


# ============================================================================
# Tab 5.5: Generate GPT Pairs
# ============================================================================

def generate_gpt_pairs(
    train_manifest: str,
    val_manifest: str,
    train_output: str,
    val_output: str,
    pairs_per_target: int,
    min_text_len: int,
    progress=gr.Progress()
) -> Tuple[str, str, Dict]:
    """Generate GPT prompt-target pairs from preprocessed manifests"""
    
    # Use pipeline state defaults
    if not train_manifest and pipeline_state.get("processed_dir"):
        train_manifest = str(Path(pipeline_state["processed_dir"]) / "train_manifest.jsonl")
        val_manifest = str(Path(pipeline_state["processed_dir"]) / "val_manifest.jsonl")
    
    if not train_output:
        train_output = "processed_data/train_pairs.jsonl"
    
    if not val_output:
        val_output = "processed_data/val_pairs.jsonl"
    
    # Validate inputs
    if not Path(train_manifest).exists():
        return "", format_status_html(f"❌ Error: Train manifest not found: {train_manifest}", False), pipeline_state
    
    if not Path(val_manifest).exists():
        return "", format_status_html(f"❌ Error: Validation manifest not found: {val_manifest}", False), pipeline_state
    
    progress(0.1, desc="Starting pair generation...")
    
    # Build command for training pairs
    script_path = Path(current_dir) / "tools" / "build_gpt_prompt_pairs.py"
    
    if not script_path.exists():
        return "", format_status_html(f"❌ Error: build_gpt_prompt_pairs.py not found", False), pipeline_state
    
    logs = []
    
    try:
        # Generate training pairs
        progress(0.3, desc="Generating training pairs...")
        
        cmd_train = [
            sys.executable,
            str(script_path),
            "--manifest", train_manifest,
            "--output", train_output,
            "--pairs-per-target", str(pairs_per_target),
            "--min-text-len", str(min_text_len),
        ]
        
        result_train = subprocess.run(
            cmd_train,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result_train.returncode != 0:
            return (
                f"TRAIN:\n{result_train.stdout}\n\nERROR:\n{result_train.stderr}",
                format_status_html("❌ Error generating training pairs", False),
                pipeline_state
            )
        
        logs.append(f"✅ Training pairs: {result_train.stdout.strip()}")
        
        # Generate validation pairs
        progress(0.6, desc="Generating validation pairs...")
        
        cmd_val = [
            sys.executable,
            str(script_path),
            "--manifest", val_manifest,
            "--output", val_output,
            "--pairs-per-target", str(pairs_per_target),
            "--min-text-len", str(min_text_len),
        ]
        
        result_val = subprocess.run(
            cmd_val,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result_val.returncode != 0:
            return (
                f"TRAIN:\n{result_train.stdout}\n\nVAL ERROR:\n{result_val.stderr}",
                format_status_html("❌ Error generating validation pairs", False),
                pipeline_state
            )
        
        logs.append(f"✅ Validation pairs: {result_val.stdout.strip()}")
        
        progress(0.9, desc="Finalizing...")
        
        # Update pipeline state with pair paths
        pipeline_state["train_pairs"] = train_output
        pipeline_state["val_pairs"] = val_output
        
        # Count pairs
        train_count = 0
        val_count = 0
        
        if Path(train_output).exists():
            with open(train_output, 'r', encoding='utf-8') as f:
                train_count = sum(1 for _ in f)
        
        if Path(val_output).exists():
            with open(val_output, 'r', encoding='utf-8') as f:
                val_count = sum(1 for _ in f)
        
        status_html = format_status_html(
            f"✅ Pairs generated: {train_count} train pairs, {val_count} val pairs"
        )
        
        log_text = "\n".join(logs)
        
        return log_text, status_html, pipeline_state
    
    except Exception as e:
        return str(e), format_status_html(f"❌ Error: {str(e)}", False), pipeline_state


# ============================================================================
# Tab 6: Training
# ============================================================================

def start_training(
    train_manifest: str,
    val_manifest: str,
    output_dir: str,
    config_path: str,
    base_checkpoint: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Start GPT training"""
    
    # Use pipeline state defaults
    if not train_manifest and pipeline_state.get("processed_dir"):
        train_manifest = str(Path(pipeline_state["processed_dir"]) / "train_manifest.jsonl")
        val_manifest = str(Path(pipeline_state["processed_dir"]) / "val_manifest.jsonl")
    
    if not output_dir:
        output_dir = "training_output"
    
    if not config_path:
        config_path = "checkpoints/config.yaml"
    
    if not base_checkpoint:
        base_checkpoint = "checkpoints/gpt.pth"
    
    # Validate inputs
    if not Path(train_manifest).exists():
        return format_status_html(f"❌ Error: Train manifest not found: {train_manifest}", False), ""
    
    progress(0.1, desc="Starting training...")
    
    # Build command
    script_path = Path(current_dir) / "trainers" / "train_gpt_v2.py"
    
    # Validate training script exists
    if not script_path.exists():
        return format_status_html(f"❌ Error: Training script not found: {script_path}", False), ""
    
    cmd = [
        sys.executable,
        str(script_path),
        "--train-manifest", train_manifest,
        "--val-manifest", val_manifest,
        "--output-dir", output_dir,
        "--config", config_path,
        "--base-checkpoint", base_checkpoint,
        "--lr", str(learning_rate),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
    ]
    
    # Start training in background
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        status_html = format_status_html(
            f"✅ Training started (PID: {process.pid})\nCheck terminal for progress or use TensorBoard"
        )
        
        tensorboard_cmd = f"tensorboard --logdir {output_dir}"
        
        return status_html, f"To monitor training:\n{tensorboard_cmd}"
    
    except Exception as e:
        return format_status_html(f"❌ Error: {str(e)}", False), ""


# ============================================================================
# Main UI
# ============================================================================

def create_ui():
    """Create the main Gradio interface"""
    
    # Check dependencies
    deps = check_dependencies()
    dep_status = "**System Status:**\n"
    dep_status += f"- yt-dlp: {'✅' if deps.get('yt-dlp') else '❌'}\n"
    dep_status += f"- ffmpeg: {'✅' if deps.get('ffmpeg') else '❌'}\n"
    dep_status += f"- CUDA: {'✅ ' + deps.get('cuda_version', '') if deps.get('cuda') else '❌'}\n"
    
    with gr.Blocks(
        title="IndexTTS2 Amharic Pipeline",
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 1200px !important}"
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎙️ IndexTTS2 Amharic Training Pipeline</h1>
            <p style="font-size: 1.1em; color: #666;">Complete end-to-end pipeline from data collection to TTS generation</p>
        </div>
        """)
        
        gr.Markdown(dep_status)
        
        # Pipeline state
        state = gr.State(pipeline_state)
        
        # Overview tab
        with gr.Tab("📋 Overview"):
            gr.Markdown("""
            ## Amharic TTS Training Pipeline
            
            This interface guides you through the complete process of creating an Amharic TTS model:
            
            1. **Download** - Collect Amharic content from YouTube
            2. **Dataset** - Segment audio using subtitles
            3. **Corpus** - Clean and aggregate text (optional for extension)
            4. **Tokenizer** - EXTEND base BPE with Amharic tokens (video approach ✅)
            5. **Preprocess** - Extract features
            5.5. **Pairs** - Generate prompt-target pairs (CRITICAL!)
            6. **Train** - Fine-tune GPT model
            7. **Post-Process** - Remove noise from segments
            8. **Inference** - Generate speech
            
            ### Quick Start
            1. Prepare a text file with YouTube URLs (one per line)
            2. Go through each tab sequentially
            3. Review outputs before proceeding to next step
            
            ### Tips
            - Each step auto-fills from the previous step's output
            - You can skip steps if you already have intermediate files
            - Monitor progress in the logs section of each tab
            """)
            
            with gr.Row():
                gr.Markdown("""
                ### Current Pipeline State
                """)
            
            pipeline_status = gr.Markdown("No steps completed yet")
            
            def update_pipeline_status(state):
                status_lines = []
                if state.get("downloads_dir"):
                    status_lines.append(f"✅ **Downloads:** `{state['downloads_dir']}`")
                if state.get("dataset_dir"):
                    status_lines.append(f"✅ **Dataset:** `{state['dataset_dir']}`")
                if state.get("corpus_file"):
                    status_lines.append(f"✅ **Corpus:** `{state['corpus_file']}`")
                if state.get("tokenizer_model"):
                    status_lines.append(f"✅ **Tokenizer:** `{state['tokenizer_model']}`")
                if state.get("processed_dir"):
                    status_lines.append(f"✅ **Processed:** `{state['processed_dir']}`")
                if state.get("train_pairs") and state.get("val_pairs"):
                    status_lines.append(f"✅ **Pairs:** `{state['train_pairs']}`")
                
                return "\n".join(status_lines) if status_lines else "No steps completed yet"
        
        # Tab 1: Download
        with gr.Tab("1️⃣ Download"):
            gr.Markdown("### Download Amharic Content from YouTube")
            
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(
                        label="YouTube URLs (one per line)",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=5
                    )
                    url_file = gr.File(label="Or upload URL file")
                    
                    output_dir_download = gr.Textbox(
                        label="Output Directory",
                        value="amharic_downloads"
                    )
                    
                    with gr.Row():
                        download_subtitles = gr.Checkbox(label="Download Subtitles", value=True)
                        subtitle_langs = gr.Textbox(label="Subtitle Languages", value="am en amh")
                    
                    audio_format = gr.Dropdown(
                        label="Audio Format",
                        choices=["wav", "mp3", "flac"],
                        value="wav"
                    )
                    
                    download_btn = gr.Button("🔽 Download Videos", variant="primary", size="lg")
                
                with gr.Column():
                    download_status = gr.HTML("Ready to download")
                    download_logs = gr.Textbox(label="Logs", lines=15, max_lines=20)
            
            with gr.Accordion("🎵 Remove Background Music (Optional)", open=False):
                gr.Markdown("**Remove music before dataset creation.** Uses AI to extract vocals only.")
                
                with gr.Row():
                    music_input_dir = gr.Textbox(label="Input (with music)", placeholder="amharic_downloads")
                    music_output_dir = gr.Textbox(label="Output (vocals)", placeholder="amharic_vocals")
                
                music_model = gr.Radio(
                    label="Model",
                    choices=[
                        ("MDX-Net (Fast, 8/10 quality)", "UVR_MDXNET_KARA_2.onnx"),
                        ("Demucs (Balanced, 9/10 quality)", "htdemucs"),
                        ("Demucs FT (Slow, 9.5/10 quality)", "htdemucs_ft")
                    ],
                    value="UVR_MDXNET_KARA_2.onnx"
                )
                
                music_cleanup_source = gr.Checkbox(
                    label="Clean up source files after processing",
                    value=True,
                    info="Delete original audio+subtitles after successful vocal extraction"
                )
                
                remove_music_btn = gr.Button("🎵 Remove Music", variant="secondary")
                music_logs = gr.Textbox(label="Logs", lines=8)
                music_status = gr.Textbox(label="Status")
                
                if not MUSIC_REMOVAL_AVAILABLE:
                    gr.Markdown("⚠️ Install: `pip install audio-separator[cpu]`")
            
            download_btn.click(
                download_youtube_videos,
                inputs=[url_input, url_file, output_dir_download, download_subtitles, subtitle_langs, audio_format],
                outputs=[download_logs, download_status, state]
            ).then(
                update_pipeline_status,
                inputs=[state],
                outputs=[pipeline_status]
            )
            
            remove_music_btn.click(
                remove_background_music,
                inputs=[music_input_dir, music_output_dir, music_model, music_cleanup_source],
                outputs=[music_logs, music_status]
            )
        
        # Tab 2: Dataset Creation
        with gr.Tab("2️⃣ Dataset"):
            gr.Markdown("### Create Training Dataset from Audio + Subtitles")
            gr.Markdown("""
            **Enhanced Quality Filtering**: Automatically filters segments based on audio quality (SNR, clipping, silence) 
            and text quality (Amharic script validation, word count, speech rate). Optimized defaults for Amharic.
            """)
            
            with gr.Accordion("📊 Dataset Statistics", open=False):
                gr.Markdown("**Analyze your dataset:** Get comprehensive stats about segments, duration, speakers, and more.")
                
                stats_manifest_path = gr.Textbox(
                    label="Manifest Path",
                    placeholder="amharic_dataset/manifest.jsonl",
                    value="amharic_dataset/manifest.jsonl"
                )
                
                analyze_stats_btn = gr.Button("📊 Analyze Dataset", variant="secondary")
                stats_output = gr.Markdown(label="Statistics")
                stats_status = gr.Textbox(label="Status", visible=False)
            
            with gr.Row():
                with gr.Column():
                    input_dir_dataset = gr.Textbox(
                        label="Input Directory (audio + subtitle files)",
                        placeholder="Will auto-fill from previous step"
                    )
                    output_dir_dataset = gr.Textbox(
                        label="Output Directory",
                        value="amharic_dataset"
                    )
                    
                    gr.Markdown("#### Duration Filters")
                    with gr.Row():
                        min_duration = gr.Slider(
                            label="Min Duration (seconds)",
                            minimum=0.5,
                            maximum=5.0,
                            value=1.0,
                            step=0.1,
                            info="Minimum segment length"
                        )
                        max_duration = gr.Slider(
                            label="Max Duration (seconds)",
                            minimum=5.0,
                            maximum=60.0,
                            value=30.0,
                            step=1.0,
                            info="Maximum segment length"
                        )
                    
                    gr.Markdown("#### Boundary Refinement & Safety Margins")
                    refine_boundaries = gr.Checkbox(
                        label="Enable boundary refinement",
                        value=True,
                        info="Uses VAD or safety margins to prevent speech cutoff"
                    )
                    
                    use_vad = gr.Checkbox(
                        label="Use VAD (Voice Activity Detection)",
                        value=True,
                        info="Recommended: Detects actual speech boundaries. Requires webrtcvad package."
                    )
                    
                    with gr.Row():
                        start_margin = gr.Slider(
                            label="Start Safety Margin (seconds)",
                            minimum=0.0,
                            maximum=0.5,
                            value=0.2,
                            step=0.05,
                            info="Audio starts this much before subtitle (prevents cutoff)"
                        )
                        end_margin = gr.Slider(
                            label="End Safety Margin (seconds)",
                            minimum=0.0,
                            maximum=0.5,
                            value=0.15,
                            step=0.05,
                            info="Audio ends this much after subtitle (prevents cutoff)"
                        )
                    
                    gr.Markdown(
                        "💡 **Safety margins prevent speech cutoff.** Start margin: 0.15s (typical subtitle lag). "
                        "End margin: 0.1s (speech trails off). Increase if you hear words being cut."
                    )
                    
                    gr.Markdown("#### Text Deduplication")
                    enable_text_dedup = gr.Checkbox(
                        label="Remove overlapping text from subtitles",
                        value=True,
                        info="Subtitles often repeat text across lines. Enable to deduplicate."
                    )
                    
                    gr.Markdown(
                        "💡 **Common in audiobooks:** Each subtitle repeats part of the previous one for readability. "
                        "Deduplication removes this overlap while keeping audio intact."
                    )
                    
                    gr.Markdown("#### Naming Scheme")
                    single_speaker = gr.Checkbox(
                        label="Single Speaker Mode",
                        value=False,
                        info="All segments use speaker ID 000. Disable for multi-speaker datasets."
                    )
                    gr.Markdown(
                        "**File naming:** `spk{speaker_id}_{segment:06d}.wav` (e.g., `spk000_000001.wav`, `spk001_000042.wav`)"
                    )
                    
                    gr.Markdown("#### Quality Filtering (Amharic-Optimized)")
                    enable_quality_filter = gr.Checkbox(
                        label="Enable Quality Filtering",
                        value=False,
                        info="Recommended: Filters low-quality segments automatically"
                    )
                    
                    with gr.Accordion("Quality Filter Settings", open=False):
                        gr.Markdown("**Audio Quality**")
                        with gr.Row():
                            min_snr = gr.Slider(
                                label="Minimum SNR (dB)",
                                minimum=10.0,
                                maximum=30.0,
                                value=15.0,
                                step=1.0,
                                info="Signal-to-Noise Ratio threshold"
                            )
                            max_silence_ratio = gr.Slider(
                                label="Max Silence Ratio",
                                minimum=0.1,
                                maximum=0.5,
                                value=0.3,
                                step=0.05,
                                info="Maximum allowed silence (0-1)"
                            )
                        
                        gr.Markdown("**Text Quality (Amharic)**")
                        with gr.Row():
                            min_words = gr.Slider(
                                label="Minimum Word Count",
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                info="Minimum Amharic words per segment"
                            )
                        
                        gr.Markdown("**Speech Rate (characters/second)**")
                        with gr.Row():
                            min_speech_rate = gr.Slider(
                                label="Min Speech Rate",
                                minimum=1.0,
                                maximum=10.0,
                                value=5.0,
                                step=0.5,
                                info="Too slow = poor alignment (Amharic: 5-20 typical)"
                            )
                            max_speech_rate = gr.Slider(
                                label="Max Speech Rate",
                                minimum=15.0,
                                maximum=40.0,
                                value=20.0,
                                step=1.0,
                                info="Too fast = poor alignment (Amharic: 5-20 typical)"
                            )
                    
                    append_to_dataset = gr.Checkbox(
                        label="📝 Append to Existing Dataset",
                        value=False,
                        info="Continue numbering from existing dataset (e.g., after spk000_003455 → starts spk000_003456). Keeps all existing files."
                    )
                    
                    create_dataset_btn = gr.Button("🎵 Create Dataset", variant="primary", size="lg")
                
                with gr.Column():
                    dataset_status = gr.HTML("Ready to create dataset")
                    dataset_logs = gr.Textbox(label="Logs", lines=15, max_lines=20)
            
            # Auto-fill input directory from downloads
            state.change(
                lambda s: s.get("downloads_dir", ""),
                inputs=[state],
                outputs=[input_dir_dataset]
            )
            
            analyze_stats_btn.click(
                analyze_dataset_stats,
                inputs=[stats_manifest_path],
                outputs=[stats_output, stats_status]
            )
            
            create_dataset_btn.click(
                create_dataset,
                inputs=[
                    input_dir_dataset, 
                    output_dir_dataset, 
                    min_duration, 
                    max_duration, 
                    refine_boundaries,
                    use_vad,
                    start_margin,
                    end_margin,
                    enable_text_dedup,
                    enable_quality_filter,
                    append_to_dataset,
                    min_snr,
                    max_silence_ratio,
                    min_words,
                    min_speech_rate,
                    max_speech_rate,
                    single_speaker
                ],
                outputs=[dataset_logs, dataset_status, state]
            ).then(
                update_pipeline_status,
                inputs=[state],
                outputs=[pipeline_status]
            )
        
        # Tab 3: Corpus Collection
        with gr.Tab("3️⃣ Corpus"):
            gr.Markdown("""
            ### Collect and Clean Amharic Text Corpus
            
            **⚠️ NOTE:** When using BPE extension (Tab 4), corpus collection is OPTIONAL.
            - Extension reads text directly from manifest.jsonl (Tab 2)
            - This tab is for backwards compatibility or additional corpus sources
            
            **💡 For Lightning AI/Remote:** Use direct path input below (no upload needed)!
            
            **Example:** `/teamspace/studios/this_studio/amharic_dataset/manifest.jsonl`
            """)
            
            with gr.Row():
                with gr.Column():
                    corpus_manifest_path = gr.Textbox(
                        label="📁 Manifest Path (Direct Path - Recommended for Remote)",
                        placeholder="/teamspace/studios/this_studio/amharic_dataset/manifest.jsonl",
                        info="Full path to manifest.jsonl file"
                    )
                    
                    corpus_input_files = gr.Files(
                        label="Or Upload Files (JSONL or text files)",
                        file_types=[".jsonl", ".txt", ".json"]
                    )
                    
                    corpus_output_file = gr.Textbox(
                        label="Output Corpus File",
                        value="amharic_corpus.txt"
                    )
                    
                    min_length_corpus = gr.Slider(
                        label="Minimum Text Length",
                        minimum=1,
                        maximum=50,
                        value=5,
                        step=1
                    )
                    
                    show_stats = gr.Checkbox(label="Show character statistics", value=True)
                    
                    collect_corpus_btn = gr.Button("📝 Collect Corpus", variant="primary", size="lg")
                
                with gr.Column():
                    corpus_status = gr.HTML("Ready to collect corpus")
                    corpus_logs = gr.Textbox(label="Logs", lines=15, max_lines=20)
            
            collect_corpus_btn.click(
                collect_corpus,
                inputs=[corpus_manifest_path, corpus_input_files, corpus_output_file, min_length_corpus, show_stats],
                outputs=[corpus_logs, corpus_status, state]
            ).then(
                update_pipeline_status,
                inputs=[state],
                outputs=[pipeline_status]
            )
        
        # Tab 4: Tokenizer Training
        with gr.Tab("4️⃣ Tokenizer"):
            gr.Markdown("### Train Multilingual BPE Tokenizer")
            
            with gr.Row():
                with gr.Column():
                    tokenizer_corpus_files = gr.Files(
                        label="Corpus Files",
                        file_types=[".txt"]
                    )
                    tokenizer_model_prefix = gr.Textbox(
                        label="Model Prefix",
                        value="amharic_bpe"
                    )
                    
                    with gr.Row():
                        vocab_size = gr.Slider(
                            label="Vocabulary Size",
                            minimum=1000,
                            maximum=64000,
                            value=32000,
                            step=1000
                        )
                        character_coverage = gr.Slider(
                            label="Character Coverage",
                            minimum=0.9,
                            maximum=1.0,
                            value=0.9995,
                            step=0.0001
                        )
                    
                    test_text_tokenizer = gr.Textbox(
                        label="Test Text (optional)",
                        placeholder="ሰላም ልዑል! እንዴት ነዎት?",
                        value="ሰላም ልዑል! እንዴት ነዎት?"
                    )
                    
                    gr.Markdown("""
                    **📊 Recommended Settings (Amharic):**
                    - Target Size: 24,000 (base 12k + Amharic 12k)
                    - Character Coverage: 0.9999 (captures all Ethiopic chars)
                    - Uses: `tools/tokenizer/extend_bpe.py` (video approach ✅)
                    """)
                    
                    train_tokenizer_btn = gr.Button("🔤 Extend Tokenizer", variant="primary", size="lg")
                
                with gr.Column():
                    tokenizer_status = gr.HTML("Ready to train tokenizer")
                    tokenizer_logs = gr.Textbox(label="Logs", lines=10, max_lines=15)
                    tokenizer_test_result = gr.Textbox(label="Test Result", lines=5)
            
            train_tokenizer_btn.click(
                train_tokenizer,
                inputs=[tokenizer_corpus_files, tokenizer_model_prefix, vocab_size, character_coverage, test_text_tokenizer],
                outputs=[tokenizer_logs, tokenizer_status, tokenizer_test_result, state]
            ).then(
                update_pipeline_status,
                inputs=[state],
                outputs=[pipeline_status]
            )
        
        # Tab 5: Preprocessing
        with gr.Tab("5️⃣ Preprocess"):
            gr.Markdown("### Extract Features for Training")
            
            with gr.Row():
                with gr.Column():
                    preprocess_manifest = gr.Textbox(
                        label="Manifest Path (JSONL)",
                        placeholder="Will auto-fill from dataset step"
                    )
                    preprocess_output_dir = gr.Textbox(
                        label="Output Directory",
                        value="processed_data"
                    )
                    preprocess_tokenizer = gr.Textbox(
                        label="Tokenizer Model Path",
                        placeholder="Will auto-fill from tokenizer step"
                    )
                    
                    with gr.Row():
                        preprocess_config = gr.Textbox(
                            label="Config Path",
                            value="checkpoints/config.yaml"
                        )
                        preprocess_checkpoint = gr.Textbox(
                            label="GPT Checkpoint",
                            value="checkpoints/gpt.pth"
                        )
                    
                    with gr.Row():
                        preprocess_language = gr.Textbox(label="Language", value="am")
                        preprocess_val_ratio = gr.Slider(
                            label="Validation Ratio",
                            minimum=0.0,
                            maximum=0.2,
                            value=0.01,
                            step=0.01
                        )
                    
                    gr.Markdown("""
                    **⚡ GPU Optimization:** Batch size is auto-detected based on your GPU VRAM (L4 22GB→16, V100→16).
                    Preprocessing uses large pretrained models (12-16GB VRAM), so 30-60% GPU utilization is normal and expected.
                    """)
                    
                    preprocess_btn = gr.Button("⚙️ Preprocess Data", variant="primary", size="lg")
                
                with gr.Column():
                    preprocess_status = gr.HTML("Ready to preprocess")
                    preprocess_logs = gr.Textbox(label="Logs", lines=15, max_lines=20)
            
            preprocess_btn.click(
                preprocess_data,
                inputs=[
                    preprocess_manifest, preprocess_output_dir, preprocess_tokenizer,
                    preprocess_config, preprocess_checkpoint, preprocess_language,
                    preprocess_val_ratio
                ],
                outputs=[preprocess_logs, preprocess_status, state]
            ).then(
                update_pipeline_status,
                inputs=[state],
                outputs=[pipeline_status]
            )
        
        # Tab 5.5: Generate GPT Prompt Pairs
        with gr.Tab("5.5️⃣ Pairs"):
            gr.Markdown("### Generate GPT Prompt-Target Pairs")
            gr.Markdown("""
            **Critical Step:** Creates prompt-target pairs needed for GPT training.
            
            **Why needed?**
            - Enables voice cloning and emotion transfer
            - Pairs teach model to apply voice A to text B
            - Each target gets multiple prompts from same speaker
            
            **What it does:**
            - Takes single-sample manifest from preprocessing
            - Creates pairs: prompt (voice/emotion) + target (text/codes)
            - Same speaker requirement ensures voice consistency
            """)
            
            with gr.Row():
                with gr.Column():
                    pairs_train_manifest = gr.Textbox(
                        label="Train Manifest (from preprocessing)",
                        placeholder="Will auto-fill from preprocessing step"
                    )
                    pairs_val_manifest = gr.Textbox(
                        label="Validation Manifest (from preprocessing)",
                        placeholder="Will auto-fill from preprocessing step"
                    )
                    
                    pairs_train_output = gr.Textbox(
                        label="Output Train Pairs",
                        value="processed_data/train_pairs.jsonl"
                    )
                    pairs_val_output = gr.Textbox(
                        label="Output Validation Pairs",
                        value="processed_data/val_pairs.jsonl"
                    )
                    
                    with gr.Row():
                        pairs_per_target = gr.Slider(
                            label="Pairs per Target",
                            minimum=1,
                            maximum=5,
                            value=2,
                            step=1,
                            info="How many different prompts to pair with each target"
                        )
                        pairs_min_text_len = gr.Slider(
                            label="Min Text Length",
                            minimum=1,
                            maximum=50,
                            value=5,
                            step=1,
                            info="Skip targets shorter than this"
                        )
                    
                    gr.Markdown("""
                    **💡 Tip:** More pairs per target = more training data but longer training time.
                    Default of 2 pairs is recommended for most cases.
                    """)
                    
                    generate_pairs_btn = gr.Button("🔗 Generate Pairs", variant="primary", size="lg")
                
                with gr.Column():
                    pairs_status = gr.HTML("Ready to generate pairs")
                    pairs_logs = gr.Textbox(label="Logs", lines=15, max_lines=20)
            
            # Auto-fill from preprocessing
            state.change(
                lambda s: (
                    str(Path(s.get("processed_dir", "processed_data")) / "train_manifest.jsonl") if s.get("processed_dir") else "",
                    str(Path(s.get("processed_dir", "processed_data")) / "val_manifest.jsonl") if s.get("processed_dir") else ""
                ),
                inputs=[state],
                outputs=[pairs_train_manifest, pairs_val_manifest]
            )
            
            generate_pairs_btn.click(
                generate_gpt_pairs,
                inputs=[
                    pairs_train_manifest,
                    pairs_val_manifest,
                    pairs_train_output,
                    pairs_val_output,
                    pairs_per_target,
                    pairs_min_text_len
                ],
                outputs=[pairs_logs, pairs_status, state]
            ).then(
                update_pipeline_status,
                inputs=[state],
                outputs=[pipeline_status]
            )
        
        # Tab 6: Training
        with gr.Tab("6️⃣ Training"):
            gr.Markdown("### Fine-tune GPT Model")
            gr.Markdown("""
            **Note:** Training is a long-running process. This will start training in the background.
            Monitor progress using TensorBoard or check the terminal output.
            
            **Important:** Use the PAIRED manifests from Tab 5.5, not the single-sample manifests!
            """)
            
            with gr.Row():
                with gr.Column():
                    train_manifest_path = gr.Textbox(
                        label="Train Pairs Manifest",
                        placeholder="Will auto-fill from pairing step (use train_pairs.jsonl!)"
                    )
                    val_manifest_path = gr.Textbox(
                        label="Validation Pairs Manifest",
                        placeholder="Will auto-fill from pairing step (use val_pairs.jsonl!)"
                    )
                    train_output_dir = gr.Textbox(
                        label="Output Directory",
                        value="training_output"
                    )
                    
                    with gr.Row():
                        train_config = gr.Textbox(
                            label="Config Path",
                            value="checkpoints/config.yaml"
                        )
                        train_base_checkpoint = gr.Textbox(
                            label="Base Checkpoint",
                            value="checkpoints/gpt.pth"
                        )
                    
                    with gr.Row():
                        train_lr = gr.Number(label="Learning Rate", value=1e-5)
                        train_batch = gr.Slider(label="Batch Size", minimum=1, maximum=32, value=8, step=1)
                        train_epochs = gr.Slider(label="Epochs", minimum=1, maximum=100, value=10, step=1)
                    
                    start_training_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                
                with gr.Column():
                    training_status = gr.HTML("Ready to start training")
                    training_info = gr.Textbox(label="Training Info", lines=10)
            
            # Auto-fill from pairs
            state.change(
                lambda s: (
                    s.get("train_pairs", ""),
                    s.get("val_pairs", "")
                ),
                inputs=[state],
                outputs=[train_manifest_path, val_manifest_path]
            )
            
            start_training_btn.click(
                start_training,
                inputs=[
                    train_manifest_path, val_manifest_path, train_output_dir,
                    train_config, train_base_checkpoint, train_lr, train_batch, train_epochs
                ],
                outputs=[training_status, training_info]
            )
        
        # Tab 7: Dataset Segment Processing
        with gr.Tab("7️⃣ Post-Process"):
            gr.Markdown("### Remove Noise from Existing Dataset Segments")
            gr.Markdown("""
            **Post-processing for existing datasets:** Remove background music/noise from already-created 
            dataset segments while maintaining all filenames and manifest structure.
            
            **Features:**
            - Processes all audio segments in dataset/audio directory
            - Maintains exact filenames (in-place replacement)
            - Supports resume from interruption
            - Progress tracking with periodic saves
            - Optional backup of original files
            """)
            
            with gr.Row():
                with gr.Column():
                    segment_input_source = gr.Radio(
                        label="Input Source",
                        choices=[
                            ("From Manifest (Recommended)", "manifest"),
                            ("Audio Directory", "directory")
                        ],
                        value="manifest"
                    )
                    
                    segment_manifest_path = gr.Textbox(
                        label="Manifest Path",
                        placeholder="amharic_dataset/manifest.jsonl",
                        visible=True
                    )
                    
                    segment_audio_dir = gr.Textbox(
                        label="Audio Directory",
                        placeholder="amharic_dataset/audio",
                        visible=False
                    )
                    
                    segment_model = gr.Radio(
                        label="Noise Removal Model",
                        choices=[
                            ("UVR-MDX-NET (Fast, High Quality)", "UVR-MDX-NET-Inst_HQ_3"),
                            ("UVR-MDX-NET KARA", "UVR_MDXNET_KARA_2.onnx"),
                            ("Demucs (Slower, Best Quality)", "htdemucs")
                        ],
                        value="UVR-MDX-NET-Inst_HQ_3",
                        info="MDX-NET recommended for speed/quality balance"
                    )
                    
                    with gr.Row():
                        segment_keep_backup = gr.Checkbox(
                            label="Keep Backup Files",
                            value=False,
                            info="Save originals with .backup extension"
                        )
                        segment_resume = gr.Checkbox(
                            label="Resume from Previous Run",
                            value=True,
                            info="Skip already-processed files"
                        )
                    
                    segment_batch_size = gr.Slider(
                        label="Progress Save Interval",
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5,
                        info="Save progress after this many files"
                    )
                    
                    with gr.Accordion("🚀 GPU Optimization Settings", open=False):
                        gr.Markdown("""
                        **GPU Performance Tuning:**
                        - Chunk Size: Number of files to process in each group (not parallel)
                        - MDX Batch Size: Higher values use more VRAM but process faster
                        - Autocast: Mixed precision for faster processing on modern GPUs
                        - Normalization: Audio level normalization (0.0-1.0)
                        """)
                        
                        segment_chunk_size = gr.Slider(
                            label="Chunk Size",
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=1,
                            info="Files per chunk (for progress tracking)"
                        )
                        
                        segment_mdx_batch = gr.Slider(
                            label="MDX Batch Size",
                            minimum=1,
                            maximum=16,
                            value=4,
                            step=1,
                            info="GPU batch size (8-16 for 16GB+ VRAM, 4 for 8GB)"
                        )
                        
                        segment_autocast = gr.Checkbox(
                            label="Enable Autocast (Mixed Precision)",
                            value=True,
                            info="Faster on modern GPUs, disable if issues occur"
                        )
                        
                        segment_normalization = gr.Slider(
                            label="Audio Normalization",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.1,
                            info="Normalize audio levels (0.9 recommended)"
                        )
                    
                    process_segments_btn = gr.Button(
                        "🎵 Process Dataset Segments",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column():
                    segment_status = gr.HTML("Ready to process segments")
                    segment_logs = gr.Textbox(label="Logs", lines=15, max_lines=20)
                    
                    gr.Markdown("""
                    **⚠️ Important Notes:**
                    - This replaces original audio files with noise-removed versions
                    - Enable "Keep Backup Files" if you want to preserve originals
                    - Process can be interrupted and resumed later
                    - Manifest.jsonl is not modified (only audio files change)
                    """)
            
            # Toggle visibility based on input source
            def toggle_input_source(source):
                return (
                    gr.update(visible=source == "manifest"),
                    gr.update(visible=source == "directory")
                )
            
            segment_input_source.change(
                toggle_input_source,
                inputs=[segment_input_source],
                outputs=[segment_manifest_path, segment_audio_dir]
            )
            
            # Auto-fill from pipeline state
            state.change(
                lambda s: str(Path(s.get("dataset_dir", "amharic_dataset")) / "manifest.jsonl") if s.get("dataset_dir") else "amharic_dataset/manifest.jsonl",
                inputs=[state],
                outputs=[segment_manifest_path]
            )
            
            def process_dataset_segments(
                input_source: str,
                manifest_path: str,
                audio_dir: str,
                model_name: str,
                keep_backup: bool,
                resume: bool,
                batch_size: int,
                chunk_size: int,
                mdx_batch_size: int,
                use_autocast: bool,
                normalization: float,
                progress=gr.Progress()
            ) -> Tuple[str, str]:
                """Process dataset segments to remove noise"""
                try:
                    script_path = Path(current_dir) / "tools" / "process_dataset_segments.py"
                    
                    if not script_path.exists():
                        return format_status_html("❌ Error: Processing script not found", False), ""
                    
                    # Build command
                    cmd = [sys.executable, str(script_path)]
                    
                    if input_source == "manifest":
                        if not manifest_path:
                            return format_status_html("❌ Error: Manifest path required", False), ""
                        cmd.extend(["--manifest", manifest_path])
                    else:
                        if not audio_dir:
                            return format_status_html("❌ Error: Audio directory required", False), ""
                        cmd.extend(["--audio-dir", audio_dir])
                    
                    cmd.extend(["--model", model_name])
                    cmd.extend(["--batch-size", str(int(batch_size))])
                    cmd.extend(["--chunk-size", str(int(chunk_size))])
                    cmd.extend(["--mdx-batch-size", str(int(mdx_batch_size))])
                    cmd.extend(["--normalization", str(normalization)])
                    
                    if keep_backup:
                        cmd.append("--keep-backup")
                    
                    if not resume:
                        cmd.append("--no-resume")
                    
                    if not use_autocast:
                        cmd.append("--no-autocast")
                    
                    progress(0.1, desc="Starting processing...")
                    
                    # Run processing
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    log_lines = []
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            log_lines.append(line.strip())
                            # Update progress based on output
                            if "Processing" in line:
                                progress(0.5, desc="Processing segments...")
                    
                    process.wait()
                    
                    progress(0.9, desc="Finalizing...")
                    
                    log_text = "\n".join(log_lines)
                    
                    if process.returncode == 0:
                        # Parse stats from output
                        total = processed = 0
                        for line in log_lines:
                            if "Total segments:" in line:
                                total = int(line.split(":")[-1].strip())
                            elif "Processed:" in line:
                                processed = int(line.split(":")[-1].strip())
                        
                        status_html = format_status_html(
                            f"✅ Processing complete: {processed}/{total} segments processed"
                        )
                    else:
                        status_html = format_status_html("❌ Processing failed", False)
                    
                    return status_html, log_text
                
                except Exception as e:
                    return format_status_html(f"❌ Error: {str(e)}", False), str(e)
            
            process_segments_btn.click(
                process_dataset_segments,
                inputs=[
                    segment_input_source,
                    segment_manifest_path,
                    segment_audio_dir,
                    segment_model,
                    segment_keep_backup,
                    segment_resume,
                    segment_batch_size,
                    segment_chunk_size,
                    segment_mdx_batch,
                    segment_autocast,
                    segment_normalization
                ],
                outputs=[segment_status, segment_logs]
            )
        
        # Tab 8: Inference (link to existing webui)
        with gr.Tab("8️⃣ Inference"):
            gr.Markdown("### Generate Speech")
            gr.Markdown("""
            For inference, use the main WebUI:
            
            ```bash
            python webui.py --model_dir checkpoints
            ```
            
            Or for batch generation:
            
            ```bash
            python webui_parallel.py --model_dir checkpoints
            ```
            
            Make sure to specify your trained checkpoint directory!
            """)
    
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="IndexTTS2 Amharic Pipeline WebUI")
    parser.add_argument("--port", type=int, default=7863, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
