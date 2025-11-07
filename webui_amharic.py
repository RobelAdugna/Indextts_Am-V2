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
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

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
    input_files,
    output_file: str,
    min_length: int,
    show_stats: bool,
    progress=gr.Progress()
) -> Tuple[str, str, Dict]:
    """Collect and clean Amharic corpus"""
    
    if not output_file:
        output_file = "amharic_corpus.txt"
    
    output_path = Path(output_file)
    
    # Prepare input files
    input_paths = []
    if input_files:
        input_paths = [f.name for f in input_files]
    elif pipeline_state.get("dataset_dir"):
        # Use manifest from dataset creation
        manifest = Path(pipeline_state["dataset_dir"]) / "manifest.jsonl"
        if manifest.exists():
            input_paths = [str(manifest)]
    
    if not input_paths:
        return "", format_status_html("❌ Error: No input files provided", False), pipeline_state
    
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
    corpus_files,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float,
    test_text: str,
    progress=gr.Progress()
) -> Tuple[str, str, str, Dict]:
    """Train multilingual BPE tokenizer"""
    
    if not model_prefix:
        model_prefix = "amharic_bpe"
    
    # Prepare corpus files
    corpus_paths = []
    if corpus_files:
        corpus_paths = [f.name for f in corpus_files]
    elif pipeline_state.get("corpus_file"):
        corpus_paths = [pipeline_state["corpus_file"]]
    
    if not corpus_paths:
        return "", format_status_html("❌ Error: No corpus files provided", False), "", pipeline_state
    
    progress(0.1, desc="Starting tokenizer training...")
    
    # Build command
    script_path = Path(current_dir) / "tools" / "train_multilingual_bpe.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--corpus",
    ] + corpus_paths + [
        "--model-prefix", model_prefix,
        "--vocab-size", str(vocab_size),
        "--character-coverage", str(character_coverage),
    ]
    
    try:
        progress(0.3, desc="Training tokenizer (this may take a while)...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )
        
        progress(0.9, desc="Finalizing...")
        
        if result.returncode == 0:
            model_path = Path(f"{model_prefix}.model")
            
            if model_path.exists():
                pipeline_state["tokenizer_model"] = str(model_path)
                status_html = format_status_html(f"✅ Tokenizer trained: {model_path.name}")
                
                # Test tokenization
                test_result = ""
                if test_text:
                    try:
                        import sentencepiece as spm
                        sp = spm.SentencePieceProcessor()
                        sp.load(str(model_path))
                        tokens = sp.encode(test_text, out_type=str)
                        test_result = f"Test tokenization:\nText: {test_text}\nTokens: {tokens}"
                    except Exception as e:
                        test_result = f"Test failed: {str(e)}"
                
                log_text = result.stdout
            else:
                status_html = format_status_html("❌ Model file not created", False)
                log_text = result.stdout
                test_result = ""
        else:
            status_html = format_status_html("❌ Error training tokenizer", False)
            log_text = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            test_result = ""
    
    except Exception as e:
        status_html = format_status_html(f"❌ Error: {str(e)}", False)
        log_text = str(e)
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
    batch_size: int,
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
        "--batch-size", str(batch_size),
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
            3. **Corpus** - Clean and aggregate text
            4. **Tokenizer** - Train BPE model
            5. **Preprocess** - Extract features
            6. **Train** - Fine-tune GPT model
            7. **Inference** - Generate speech
            
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
            
            download_btn.click(
                download_youtube_videos,
                inputs=[url_input, url_file, output_dir_download, download_subtitles, subtitle_langs, audio_format],
                outputs=[download_logs, download_status, state]
            ).then(
                update_pipeline_status,
                inputs=[state],
                outputs=[pipeline_status]
            )
        
        # Tab 2: Dataset Creation
        with gr.Tab("2️⃣ Dataset"):
            gr.Markdown("### Create Training Dataset from Audio + Subtitles")
            gr.Markdown("""
            **Enhanced Quality Filtering**: Automatically filters segments based on audio quality (SNR, clipping, silence) 
            and text quality (Amharic script validation, word count, speech rate). Optimized defaults for Amharic.
            """)
            
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
                            value=0.15,
                            step=0.05,
                            info="Audio starts this much before subtitle (prevents cutoff)"
                        )
                        end_margin = gr.Slider(
                            label="End Safety Margin (seconds)",
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
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
                        value=True,
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
            gr.Markdown("### Collect and Clean Amharic Text Corpus")
            
            with gr.Row():
                with gr.Column():
                    corpus_input_files = gr.Files(
                        label="Input Files (JSONL or text files)",
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
                inputs=[corpus_input_files, corpus_output_file, min_length_corpus, show_stats],
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
                    
                    train_tokenizer_btn = gr.Button("🔤 Train Tokenizer", variant="primary", size="lg")
                
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
                        preprocess_batch_size = gr.Slider(
                            label="Batch Size",
                            minimum=1,
                            maximum=16,
                            value=4,
                            step=1
                        )
                    
                    preprocess_btn = gr.Button("⚙️ Preprocess Data", variant="primary", size="lg")
                
                with gr.Column():
                    preprocess_status = gr.HTML("Ready to preprocess")
                    preprocess_logs = gr.Textbox(label="Logs", lines=15, max_lines=20)
            
            preprocess_btn.click(
                preprocess_data,
                inputs=[
                    preprocess_manifest, preprocess_output_dir, preprocess_tokenizer,
                    preprocess_config, preprocess_checkpoint, preprocess_language,
                    preprocess_val_ratio, preprocess_batch_size
                ],
                outputs=[preprocess_logs, preprocess_status, state]
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
            """)
            
            with gr.Row():
                with gr.Column():
                    train_manifest_path = gr.Textbox(
                        label="Train Manifest",
                        placeholder="Will auto-fill from preprocessing step"
                    )
                    val_manifest_path = gr.Textbox(
                        label="Validation Manifest",
                        placeholder="Will auto-fill from preprocessing step"
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
            
            start_training_btn.click(
                start_training,
                inputs=[
                    train_manifest_path, val_manifest_path, train_output_dir,
                    train_config, train_base_checkpoint, train_lr, train_batch, train_epochs
                ],
                outputs=[training_status, training_info]
            )
        
        # Tab 7: Inference (link to existing webui)
        with gr.Tab("7️⃣ Inference"):
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
