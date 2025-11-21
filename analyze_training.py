#!/usr/bin/env python3
"""Extract training metrics from TensorBoard event files."""

import sys
from pathlib import Path
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    sys.exit(1)

def extract_metrics(log_dir: Path):
    """Extract training and validation metrics from TensorBoard logs."""
    # Find most recent run directory
    log_root = log_dir / "logs"
    if not log_root.exists():
        print(f"No logs directory found at {log_root}")
        return
    
    run_dirs = sorted([d for d in log_root.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        print(f"No run directories found in {log_root}")
        return
    
    latest_run = run_dirs[-1]
    print(f"Analyzing run: {latest_run.name}\n")
    
    # Load events
    ea = event_accumulator.EventAccumulator(str(latest_run))
    ea.Reload()
    
    # Get available metrics
    scalars = ea.Tags().get('scalars', [])
    
    if not scalars:
        print("No scalar metrics found in logs")
        return
    
    print("Available metrics:", scalars)
    print("\n" + "="*80)
    
    # Extract training metrics
    train_metrics = {}
    val_metrics = {}
    
    for tag in scalars:
        events = ea.Scalars(tag)
        if tag.startswith('train/'):
            metric_name = tag.replace('train/', '')
            train_metrics[metric_name] = [(e.step, e.value) for e in events]
        elif tag.startswith('val/'):
            metric_name = tag.replace('val/', '')
            val_metrics[metric_name] = [(e.step, e.value) for e in events]
    
    # Print summary
    if train_metrics:
        print("\nTRAINING PROGRESS (last 20 steps):\n")
        for metric in ['text_loss', 'mel_loss', 'mel_top1', 'lr']:
            if metric in train_metrics:
                data = train_metrics[metric][-20:]  # Last 20 points
                print(f"{metric}:")
                for step, value in data:
                    print(f"  Step {step:6d}: {value:.6f}")
                print()
    
    if val_metrics:
        print("\nVALIDATION PROGRESS (all checkpoints):\n")
        for metric in ['text_loss', 'mel_loss', 'mel_top1']:
            if metric in val_metrics:
                data = val_metrics[metric]
                print(f"{metric}:")
                for step, value in data:
                    print(f"  Step {step:6d}: {value:.6f}")
                print()
    
    # Analysis
    print("="*80)
    print("\nANALYSIS:\n")
    
    if 'text_loss' in train_metrics and len(train_metrics['text_loss']) >= 2:
        first_loss = train_metrics['text_loss'][0][1]
        last_loss = train_metrics['text_loss'][-1][1]
        improvement = first_loss - last_loss
        pct_improvement = (improvement / first_loss) * 100
        
        print(f"Text Loss:")
        print(f"  Initial: {first_loss:.4f}")
        print(f"  Current: {last_loss:.4f}")
        print(f"  Improvement: {improvement:.4f} ({pct_improvement:.1f}%)")
        
        if last_loss < 3.0:
            print("  ✅ EXCELLENT - Model is learning Amharic!")
        elif last_loss < 5.0:
            print("  ✅ GOOD - On track, continue training")
        elif last_loss < 7.0:
            print("  ⚠️  OK - Slower than expected, but progressing")
        else:
            print("  ❌ CONCERN - Loss not dropping enough")
        print()
    
    if 'mel_loss' in train_metrics and len(train_metrics['mel_loss']) >= 2:
        first_loss = train_metrics['mel_loss'][0][1]
        last_loss = train_metrics['mel_loss'][-1][1]
        improvement = first_loss - last_loss
        pct_improvement = (improvement / first_loss) * 100
        
        print(f"Mel Loss:")
        print(f"  Initial: {first_loss:.4f}")
        print(f"  Current: {last_loss:.4f}")
        print(f"  Improvement: {improvement:.4f} ({pct_improvement:.1f}%)")
        
        if last_loss < 2.5:
            print("  ✅ EXCELLENT - High quality synthesis expected!")
        elif last_loss < 3.5:
            print("  ✅ GOOD - On track for quality results")
        elif last_loss < 4.5:
            print("  ⚠️  OK - Slower convergence, continue training")
        else:
            print("  ❌ CONCERN - Loss plateau may indicate issues")
        print()
    
    if 'lr' in train_metrics and len(train_metrics['lr']) >= 2:
        first_lr = train_metrics['lr'][0][1]
        last_lr = train_metrics['lr'][-1][1]
        peak_lr = max(v for _, v in train_metrics['lr'])
        
        print(f"Learning Rate:")
        print(f"  Initial: {first_lr:.2e}")
        print(f"  Current: {last_lr:.2e}")
        print(f"  Peak: {peak_lr:.2e}")
        
        if first_lr < 1e-5 and last_lr > first_lr:
            print("  ✅ Warmup phase detected - correct!")
        elif abs(last_lr - peak_lr) < 1e-6:
            print("  ✅ At peak LR - maximum learning rate")
        else:
            print("  ✅ Decay phase - normal for later training")
        print()

if __name__ == "__main__":
    output_dir = Path("training_output")
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)
    
    extract_metrics(output_dir)
