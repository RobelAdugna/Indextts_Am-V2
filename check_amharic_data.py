#!/usr/bin/env python3
"""
Quick verification script for Amharic dataset on Lightning.ai.
Run this to check if your data is properly set up before training.
"""

import json
import re
from pathlib import Path
from collections import Counter
import argparse


def check_amharic_script(text: str) -> bool:
    """Check if text contains Amharic/Ethiopic script"""
    return bool(re.search(r'[\u1200-\u137f\u1380-\u139f\u2d80-\u2ddf\uab00-\uab2f]', text))


def analyze_manifest(manifest_path: Path) -> dict:
    """Analyze a manifest file for Amharic content"""
    total = 0
    amharic_texts = 0
    speakers = Counter()
    durations = []
    text_lengths = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            
            text = record.get('text', '')
            if check_amharic_script(text):
                amharic_texts += 1
            
            speakers[record.get('speaker', 'unknown')] += 1
            
            if 'duration' in record:
                durations.append(record['duration'])
            
            text_lengths.append(len(text))
    
    return {
        'total': total,
        'amharic_texts': amharic_texts,
        'amharic_pct': 100 * amharic_texts / total if total > 0 else 0,
        'unique_speakers': len(speakers),
        'avg_duration': sum(durations) / len(durations) if durations else 0,
        'min_duration': min(durations) if durations else 0,
        'max_duration': max(durations) if durations else 0,
        'avg_text_len': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
    }


def analyze_pairs(pairs_path: Path, limit: int = 5) -> None:
    """Analyze prompt-target pairs for language consistency"""
    print(f"\n{'='*80}")
    print(f"PROMPT-TARGET PAIRS ANALYSIS: {pairs_path.name}")
    print(f"{'='*80}")
    
    if not pairs_path.exists():
        print(f"❌ File not found: {pairs_path}")
        return
    
    prompt_amharic_count = 0
    target_amharic_count = 0
    total_pairs = 0
    
    print(f"\nFirst {limit} pairs:")
    with open(pairs_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            
            record = json.loads(line)
            total_pairs += 1
            
            # Check target text
            target_text = record.get('target_text', '')
            if check_amharic_script(target_text):
                target_amharic_count += 1
            
            # For prompt, we need to check the audio path or assume same speaker = same language
            # Since prompts are from same speaker as targets, if target is Amharic, prompt should be too
            prompt_id = record.get('prompt_id', '')
            if check_amharic_script(target_text):  # Assume same speaker = same language
                prompt_amharic_count += 1
            
            if idx < limit:
                print(f"\n  Pair {idx+1}:")
                print(f"    ID: {record.get('id', 'N/A')}")
                print(f"    Speaker: {record.get('speaker', 'N/A')}")
                print(f"    Prompt ID: {prompt_id}")
                print(f"    Target Text: {target_text[:50]}..." if len(target_text) > 50 else f"    Target Text: {target_text}")
                print(f"    Target is Amharic: {'✅' if check_amharic_script(target_text) else '❌'}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total pairs: {total_pairs:,}")
    print(f"  Targets with Amharic: {target_amharic_count:,} ({100*target_amharic_count/total_pairs:.1f}%)")
    
    if target_amharic_count < total_pairs * 0.9:
        print(f"\n⚠️  WARNING: Less than 90% of targets are Amharic!")
        print(f"   This could cause training issues. Check your preprocessing.")
    else:
        print(f"\n✅ Good! Most targets are Amharic.")
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Amharic dataset quality on Lightning.ai"
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="preprocessed_amharic/train_manifest.jsonl",
        help="Path to train manifest"
    )
    parser.add_argument(
        "--train-pairs",
        type=str,
        default="preprocessed_amharic/train_pairs.jsonl",
        help="Path to train pairs manifest"
    )
    parser.add_argument(
        "--val-pairs",
        type=str,
        default="preprocessed_amharic/val_pairs.jsonl",
        help="Path to validation pairs manifest"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("AMHARIC DATASET QUALITY VERIFICATION")
    print("="*80)
    
    # Check train manifest
    train_manifest_path = Path(args.train_manifest)
    if train_manifest_path.exists():
        print(f"\n[1/3] Analyzing: {train_manifest_path}")
        stats = analyze_manifest(train_manifest_path)
        print(f"\n  Total samples: {stats['total']:,}")
        print(f"  Amharic samples: {stats['amharic_texts']:,} ({stats['amharic_pct']:.1f}%)")
        print(f"  Unique speakers: {stats['unique_speakers']}")
        if stats['avg_duration'] > 0:
            print(f"  Avg duration: {stats['avg_duration']:.2f}s")
            print(f"  Duration range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s")
        print(f"  Avg text length: {stats['avg_text_len']:.1f} characters")
        
        if stats['amharic_pct'] < 90:
            print(f"\n  ⚠️  WARNING: Only {stats['amharic_pct']:.1f}% Amharic! Expected >90%")
        else:
            print(f"\n  ✅ Dataset is primarily Amharic")
        
        if stats['unique_speakers'] < 10:
            print(f"\n  ⚠️  WARNING: Only {stats['unique_speakers']} speakers! Recommend >50 for robustness")
        elif stats['unique_speakers'] < 50:
            print(f"\n  ⚠️  Low speaker diversity ({stats['unique_speakers']}). Recommend >50")
        else:
            print(f"\n  ✅ Good speaker diversity ({stats['unique_speakers']} speakers)")
    else:
        print(f"\n❌ Train manifest not found: {train_manifest_path}")
    
    # Check train pairs
    analyze_pairs(Path(args.train_pairs), limit=5)
    
    # Check val pairs
    val_pairs_path = Path(args.val_pairs)
    if val_pairs_path.exists():
        with open(val_pairs_path, 'r', encoding='utf-8') as f:
            val_count = sum(1 for line in f if line.strip())
        print(f"\n[3/3] Validation pairs: {val_count:,}")
    
    print(f"\n{'='*80}")
    print("FINAL CHECKLIST")
    print(f"{'='*80}")
    print("\nBefore training, ensure:")
    print("  ☐ Train manifest has >90% Amharic content")
    print("  ☐ Train pairs show Amharic in target_text")
    print("  ☐ At least 50+ unique speakers (100+ ideal)")
    print("  ☐ Duration range 2-15 seconds (avoid extremes)")
    print("  ☐ Tokenizer at tokenizers/amharic_extended_bpe.model exists")
    print("  ☐ Ready to run: python verify_amharic_training.py")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
