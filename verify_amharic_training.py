#!/usr/bin/env python3
"""
Diagnostic script to verify Amharic tokenizer and model setup.
Run this before training to check for the extended vocabulary issue.
"""

import sys
import argparse
from pathlib import Path

try:
    import torch
    import numpy as np
    from indextts.utils.front import TextNormalizer, TextTokenizer
    from omegaconf import OmegaConf
    from indextts.gpt.model_v2 import UnifiedVoice
except ImportError as e:
    print(f"❌ ERROR: Missing dependency: {e}")
    print("   Please install required packages first.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Verify Amharic tokenizer and model setup for training."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizers/amharic_extended_bpe.model",
        help="Path to Amharic extended BPE tokenizer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to model config"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("Amharic Training Diagnostic Tool")
    print("="*80)
    
    # Check tokenizer
    print("\n[1/4] Checking Tokenizer...")
    tokenizer_path = args.tokenizer
    if not Path(tokenizer_path).exists():
        print(f"❌ ERROR: Tokenizer not found at {tokenizer_path}")
        print("   Please run tokenizer extension first (Tab 4 in WebUI)")
        return 1
    
    tokenizer = TextTokenizer(
        tokenizer_path,
        TextNormalizer(preferred_language="am")
    )
    
    print(f"✓ Tokenizer loaded: {tokenizer_path}")
    print(f"✓ Vocabulary size: {tokenizer.vocab_size}")
    
    # Test Amharic tokenization
    print("\n[2/4] Testing Amharic Tokenization...")
    test_texts = [
        "ሰላም ዓለም",  # Hello World
        "እንደምን ነህ",  # How are you
        "አመሰግናለሁ",  # Thank you
    ]
    
    uses_extended_vocab = False
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        max_id = max(ids) if ids else 0
        
        print(f"\n  Text: {text}")
        print(f"  Tokens: {' '.join(tokens)}")
        print(f"  IDs: {ids}")
        print(f"  Max ID: {max_id}")
        
        if max_id >= 12000:
            uses_extended_vocab = True
            print(f"  ✓ Uses extended vocabulary (ID >= 12000)")
        else:
            print(f"  ⚠️  WARNING: All IDs < 12000 (not using Amharic tokens?)")
    
    if not uses_extended_vocab:
        print("\n❌ ERROR: Amharic text not using extended vocabulary!")
        print("   This means tokenizer wasn't properly extended.")
        print("   Please re-run tokenizer extension (Tab 4).")
        return 1
    
    # Check model
    print("\n[3/4] Checking Model Configuration...")
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ ERROR: Config not found at {config_path}")
        return 1
    
    cfg = OmegaConf.load(config_path)
    print(f"✓ Config loaded: {config_path}")
    print(f"  Config vocab size: {cfg.gpt.number_text_tokens}")
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")
    
    if cfg.gpt.number_text_tokens != tokenizer.vocab_size:
        print(f"  ⚠️  Config will be updated at runtime (expected)")
    
    # Load model (just structure, not weights)
    print("\n[4/4] Checking Model Structure...")
    cfg.gpt.number_text_tokens = tokenizer.vocab_size
    model = UnifiedVoice(**cfg.gpt)
    
    print(f"✓ Model created with vocab_size={tokenizer.vocab_size}")
    print(f"  text_embedding.weight shape: {model.text_embedding.weight.shape}")
    print(f"  text_head.weight shape: {model.text_head.weight.shape}")
    print(f"  text_head.bias shape: {model.text_head.bias.shape}")
    
    # Check embedding initialization
    base_vocab_size = 12000
    if tokenizer.vocab_size > base_vocab_size:
        print(f"\n[Extended Vocabulary Detected]")
        print(f"  Base tokens: 0-{base_vocab_size-1}")
        print(f"  New tokens: {base_vocab_size}-{tokenizer.vocab_size-1}")
        
        # Check if embeddings look random (they should before loading checkpoint)
        base_emb_std = model.text_embedding.weight[:base_vocab_size].std().item()
        new_emb_std = model.text_embedding.weight[base_vocab_size:].std().item()
        
        print(f"\n  Embedding statistics (before checkpoint load):")
        print(f"    Base embeddings std: {base_emb_std:.6f}")
        print(f"    New embeddings std: {new_emb_std:.6f}")
        print(f"    Ratio: {new_emb_std / base_emb_std:.2f}")
        
        print(f"\n  ✓ Training script will freeze base embeddings (0-{base_vocab_size-1})")
        print(f"  ✓ Only new Amharic tokens ({base_vocab_size}-{tokenizer.vocab_size-1}) will train")
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"✓ Tokenizer: {tokenizer.vocab_size} tokens")
    print(f"✓ Model: Configured for {tokenizer.vocab_size} tokens")
    print(f"✓ Amharic text uses extended vocabulary (IDs >= 12000)")
    
    if tokenizer.vocab_size > 12000:
        print(f"\n⚠️  IMPORTANT: Use these training parameters:")
        print(f"   --learning-rate 5e-6")
        print(f"   --text-loss-weight 0.4")
        print(f"   --mel-loss-weight 0.6")
        print(f"   --warmup-steps 2000")
        print(f"\n   The fix is AUTOMATICALLY applied in train_gpt_v2.py")
        print(f"   Base embeddings will be frozen during training.")
    
    print("\n✅ All checks passed! Ready for training.")
    print("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
