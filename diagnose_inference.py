#!/usr/bin/env python3
"""Diagnostic script to check tokenizer vs checkpoint vocab mismatch."""

import sys
import torch
import sentencepiece as spm
from pathlib import Path

def main():
    checkpoint_path = "training_output/model_step23000.pth"
    tokenizer_path = "tokenizers/amharic_extended_bpe.model"
    
    print("=" * 80)
    print("DIAGNOSTIC: Tokenizer vs Checkpoint Compatibility Check")
    print("=" * 80)
    
    # Check if files exist
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    if not Path(tokenizer_path).exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return 1
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer: {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    tokenizer_vocab_size = tokenizer.vocab_size()
    print(f"   ✅ Tokenizer vocab size: {tokenizer_vocab_size}")
    
    # Load checkpoint
    print(f"\n2. Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Check model vocab size
    if "model" in checkpoint:
        model_state = checkpoint["model"]
        
        # Find text embedding size
        for key in model_state.keys():
            if "text_embedding.weight" in key:
                embedding_shape = model_state[key].shape
                checkpoint_vocab_size = embedding_shape[0]
                print(f"   ✅ Checkpoint vocab size (from {key}): {checkpoint_vocab_size}")
                break
        else:
            print("   ❌ No text_embedding.weight found in checkpoint")
            return 1
    else:
        print("   ❌ No 'model' key in checkpoint")
        return 1
    
    # Compare
    print(f"\n3. Compatibility Check:")
    if tokenizer_vocab_size == checkpoint_vocab_size:
        print(f"   ✅ MATCH: Both have {tokenizer_vocab_size} tokens")
        print(f"   This is CORRECT for inference!")
    else:
        print(f"   ❌ MISMATCH DETECTED!")
        print(f"   Tokenizer: {tokenizer_vocab_size}")
        print(f"   Checkpoint: {checkpoint_vocab_size}")
        print(f"   Difference: {abs(tokenizer_vocab_size - checkpoint_vocab_size)} tokens")
        
        if checkpoint_vocab_size < tokenizer_vocab_size:
            print(f"\n   ⚠️  CRITICAL ISSUE:")
            print(f"   Checkpoint was trained with FEWER tokens than current tokenizer!")
            print(f"   This means:")
            print(f"   - Training used {checkpoint_vocab_size} tokens")
            print(f"   - But you're trying to inference with {tokenizer_vocab_size} tokens")
            print(f"   - Missing tokens ({tokenizer_vocab_size - checkpoint_vocab_size}) were NEVER trained!")
        else:
            print(f"\n   ⚠️  Tokenizer has fewer tokens than checkpoint (unusual)")
    
    # Test tokenization
    print(f"\n4. Testing Amharic tokenization:")
    test_text = "ሰላም ልጆች እንዴት ናችሁ"
    tokens = tokenizer.encode(test_text)
    print(f"   Input: '{test_text}'")
    print(f"   Tokens: {tokens}")
    print(f"   Token range: min={min(tokens)}, max={max(tokens)}")
    
    if max(tokens) >= checkpoint_vocab_size:
        print(f"   ❌ PROBLEM: Some tokens ({max(tokens)}) >= checkpoint vocab ({checkpoint_vocab_size})!")
        print(f"   These tokens have RANDOM/UNTRAINED embeddings in the model!")
    else:
        print(f"   ✅ All tokens are within checkpoint vocab range")
    
    print("\n" + "=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
