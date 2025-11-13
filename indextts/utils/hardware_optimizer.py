#!/usr/bin/env python3
"""
Hardware detection and auto-optimization for IndexTTS2 training/preprocessing.
Automatically detects GPU, CPU, and RAM to suggest optimal training parameters.
"""

from __future__ import annotations

import os
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class HardwareConfig:
    """Detected hardware configuration and optimal settings"""
    # Hardware detection
    has_cuda: bool
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    cpu_count: int
    
    # Optimal training settings
    batch_size: int
    grad_accumulation: int
    num_workers: int
    use_amp: bool
    amp_dtype: str  # "bfloat16", "float16", or "float32"
    use_tf32: bool
    
    # Recommendations
    recommendations: list[str]


def detect_hardware() -> HardwareConfig:
    """
    Detect hardware and return optimal training configuration.
    
    Returns:
        HardwareConfig with auto-tuned parameters
    """
    has_cuda = torch.cuda.is_available()
    gpu_name = None
    gpu_vram_gb = None
    recommendations = []
    
    # Detect GPU
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        recommendations.append(f"[OK] Detected: {gpu_name} ({gpu_vram_gb:.1f}GB VRAM)")
    else:
        recommendations.append("[WARN] No CUDA GPU detected - training will be VERY slow on CPU")
    
    # Detect CPU count
    cpu_count = multiprocessing.cpu_count()
    recommendations.append(f"[OK] Detected: {cpu_count} CPU cores")
    
    # Auto-tune settings based on hardware
    config = _auto_tune_config(has_cuda, gpu_name, gpu_vram_gb, cpu_count)
    config.recommendations.extend(recommendations)
    
    return config


def _auto_tune_config(
    has_cuda: bool,
    gpu_name: Optional[str],
    gpu_vram_gb: Optional[float],
    cpu_count: int
) -> HardwareConfig:
    """
    Auto-tune training parameters based on detected hardware.
    """
    recommendations = []
    
    # Default CPU-only settings
    batch_size = 1
    grad_accumulation = 32
    num_workers = min(4, max(1, cpu_count // 2))
    use_amp = False
    amp_dtype = "float32"
    use_tf32 = False
    
    if has_cuda and gpu_vram_gb:
        # Enable AMP on all CUDA GPUs
        use_amp = True
        
        # Detect bfloat16 support (Ampere/Ada/Hopper: A100, L4, H100, RTX 30xx/40xx)
        supports_bf16 = torch.cuda.is_bf16_supported()
        if supports_bf16:
            amp_dtype = "bfloat16"
            recommendations.append("[PERF] Using bfloat16 AMP (native GPU support, better stability)")
        else:
            amp_dtype = "float16"
            recommendations.append("[PERF] Using float16 AMP (with gradient scaling)")
        
        # Enable TF32 on Ampere+ GPUs (massive speedup for matmul)
        if supports_bf16:  # bfloat16 support implies Ampere+ architecture
            use_tf32 = True
            recommendations.append("[PERF] TF32 enabled (3-8x faster matmul)")
        
        # Auto-tune batch size based on VRAM
        if gpu_vram_gb >= 80:  # A100 80GB
            batch_size = 64
            grad_accumulation = 1
            recommendations.append("[TUNED] A100 80GB: batch=64, grad_accum=1 (effective=64)")
            recommendations.append("[PERF] Maximum VRAM utilization for A100 80GB (50-60GB target)")
        elif gpu_vram_gb >= 40:  # A100 40GB, H100
            batch_size = 16
            grad_accumulation = 2
            recommendations.append("[TUNED] Large GPU: batch=16, grad_accum=2 (effective=32)")
        elif gpu_vram_gb >= 24:  # L4, RTX 3090/4090, A10
            batch_size = 8
            grad_accumulation = 4
            recommendations.append("[TUNED] Medium GPU: batch=8, grad_accum=4 (effective=32)")
        elif gpu_vram_gb >= 16:  # V100 16GB, RTX 4080
            batch_size = 6
            grad_accumulation = 6
            recommendations.append("[TUNED] Medium GPU: batch=6, grad_accum=6 (effective=36)")
        elif gpu_vram_gb >= 12:  # RTX 3060, T4
            batch_size = 4
            grad_accumulation = 8
            recommendations.append("[WARN] Small GPU: batch=4, grad_accum=8 (effective=32)")
        elif gpu_vram_gb >= 8:  # RTX 3050, older GPUs
            batch_size = 2
            grad_accumulation = 16
            recommendations.append("[WARN] Small GPU: batch=2, grad_accum=16 (effective=32)")
            recommendations.append("[TIP] Consider using --grad-checkpointing to save VRAM")
        else:  # < 8GB VRAM
            batch_size = 1
            grad_accumulation = 32
            recommendations.append("[WARN] Very small GPU: batch=1, grad_accum=32 (effective=32)")
            recommendations.append("[TIP] Strongly recommend --grad-checkpointing")
        
        # Auto-tune num_workers based on CPU count
        # Rule: Use 1-2 workers per CPU core, cap at 16 (24 for A100 80GB)
        if gpu_vram_gb >= 80:  # A100 80GB can handle more workers
            num_workers = min(24, cpu_count)
        else:
            num_workers = min(16, max(4, cpu_count))
        recommendations.append(f"[PERF] Using {num_workers} data workers (optimal for {cpu_count} CPUs)")
    
    else:
        recommendations.append("[WARN] CPU-only mode: Training will be 50-100x slower than GPU")
        recommendations.append("[TIP] Consider using Google Colab or cloud GPU for faster training")
    
    # Validate effective batch size
    effective_batch = batch_size * grad_accumulation
    if effective_batch < 16:
        recommendations.append("[WARN] Warning: Effective batch <16 may hurt convergence quality")
        recommendations.append("[TIP] Consider increasing grad_accumulation if possible")
    
    return HardwareConfig(
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        cpu_count=cpu_count,
        batch_size=batch_size,
        grad_accumulation=grad_accumulation,
        num_workers=num_workers,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_tf32=use_tf32,
        recommendations=recommendations,
    )


def get_optimal_preprocessing_workers() -> int:
    """
    Get optimal number of workers for preprocessing based on CPU count.
    
    Returns:
        Optimal number of workers (8-16 for most systems)
    """
    cpu_count = multiprocessing.cpu_count()
    
    # For preprocessing, use more workers since it's I/O bound
    # Rule: Use 1-2 workers per core, cap at 16
    workers = min(16, max(8, cpu_count))
    
    return workers


def get_optimal_mdx_batch_size() -> int:
    """
    Auto-detect optimal MDX batch size for audio-separator based on GPU VRAM.
    
    Returns:
        Optimal batch size (1-48 depending on hardware)
    """
    try:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 80:  # A100 80GB
                return 48
            elif vram_gb >= 40:  # A100 40GB, H100
                return 24
            elif vram_gb >= 24:  # L4, RTX 3090/4090, A10
                return 16
            elif vram_gb >= 16:  # V100, RTX 4080
                return 12
            elif vram_gb >= 12:  # T4, RTX 3060
                return 8
            elif vram_gb >= 8:  # RTX 3050
                return 6
            else:
                return 4
    except (ImportError, RuntimeError):
        pass
    
    # CPU fallback
    return 1


def print_hardware_summary(config: HardwareConfig) -> None:
    """
    Print a formatted summary of detected hardware and recommendations.
    """
    print("="*70)
    print("[Hardware Detection & Auto-Optimization]")
    print("="*70)
    
    if config.has_cuda:
        print(f"\n[GPU] {config.gpu_name}")
        print(f"   VRAM: {config.gpu_vram_gb:.1f} GB")
        print(f"   AMP dtype: {config.amp_dtype}")
        if config.use_tf32:
            print("   TF32: Enabled (3-8x matmul speedup)")
    else:
        print("\n[CPU] CPU-only mode (No GPU detected)")
    
    print(f"\n[CPU] {config.cpu_count} cores")
    
    print("\n[Optimized Training Settings]")
    print(f"   • Batch size: {config.batch_size}")
    print(f"   • Gradient accumulation: {config.grad_accumulation}")
    print(f"   • Effective batch: {config.batch_size * config.grad_accumulation}")
    print(f"   • Data workers: {config.num_workers}")
    print(f"   • Mixed precision: {'Enabled (' + config.amp_dtype + ')' if config.use_amp else 'Disabled'}")
    
    if config.recommendations:
        print("\n[Recommendations]")
        for rec in config.recommendations:
            print(f"   {rec}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Test hardware detection
    config = detect_hardware()
    print_hardware_summary(config)
    
    print("\n[Suggested Training Command]")
    print(f"\npython trainers/train_gpt_v2.py \\")
    print(f"  --train-manifest preprocessed/train_manifest.jsonl \\")
    print(f"  --val-manifest preprocessed/val_manifest.jsonl \\")
    print(f"  --tokenizer amharic_bpe.model \\")
    print(f"  --batch-size {config.batch_size} \\")
    print(f"  --grad-accumulation {config.grad_accumulation} \\")
    print(f"  --num-workers {config.num_workers} \\")
    if config.use_amp:
        print(f"  --amp \\")
    if config.gpu_vram_gb and config.gpu_vram_gb < 12:
        print(f"  --grad-checkpointing \\")
    print(f"  --epochs 10")
