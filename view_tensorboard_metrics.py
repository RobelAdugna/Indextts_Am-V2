#!/usr/bin/env python3
"""Extract recent training metrics from TensorBoard logs."""

import sys
from pathlib import Path
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("TensorBoard not installed. Install with: pip install tensorboard")
    sys.exit(1)

def get_recent_metrics(log_dir, n=50):
    """Get the most recent n scalar values from TensorBoard logs."""
    log_path = Path(log_dir)
    event_files = list(log_path.glob('events.out.tfevents.*'))
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return
    
    # Use the most recent event file
    event_file = sorted(event_files)[-1]
    print(f"Reading: {event_file.name}\n")
    
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    
    # Get available scalar tags
    tags = ea.Tags().get('scalars', [])
    
    if not tags:
        print("No scalar metrics found in TensorBoard logs")
        return
    
    print("📊 Recent Training Metrics (last 50 steps):\n")
    
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        if not events:
            continue
            
        # Get last n events
        recent = events[-n:] if len(events) > n else events
        
        if len(recent) >= 3:
            latest = recent[-1]
            prev_5 = recent[-6:-1] if len(recent) > 5 else recent[:-1]
            avg_5 = sum(e.value for e in prev_5) / len(prev_5) if prev_5 else latest.value
            
            print(f"{tag}:")
            print(f"  Latest (step {latest.step}): {latest.value:.4f}")
            print(f"  Avg last 5: {avg_5:.4f}")
            
            # Show trend
            if len(recent) >= 10:
                first_half = recent[:len(recent)//2]
                second_half = recent[len(recent)//2:]
                avg_first = sum(e.value for e in first_half) / len(first_half)
                avg_second = sum(e.value for e in second_half) / len(second_half)
                
                if avg_second < avg_first:
                    trend = "📉 Decreasing (good!)"
                elif avg_second > avg_first:
                    trend = "📈 Increasing (check if expected)"
                else:
                    trend = "➡️ Stable"
                print(f"  Trend: {trend}")
            print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='training_output/logs', help='TensorBoard log directory')
    parser.add_argument('--n', type=int, default=50, help='Number of recent steps to analyze')
    args = parser.parse_args()
    
    get_recent_metrics(args.log_dir, args.n)
