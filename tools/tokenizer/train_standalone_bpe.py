#!/usr/bin/env python3
"""
Train a standalone BPE tokenizer from scratch (NOT extending base model).
Use this for languages with unique scripts where you want exactly 12,000 tokens.

Example:
    python tools/tokenizer/train_standalone_bpe.py \
        --corpus amharic_corpus.txt \
        --output tokenizers/amharic_standalone_bpe.model \
        --vocab-size 12000 \
        --character-coverage 0.9999
"""

import argparse
import os
import sys
import sentencepiece as spm

def train_standalone_tokenizer(
    corpus_path: str,
    output_model: str,
    vocab_size: int = 12000,
    character_coverage: float = 0.9999,
    user_defined_symbols: str = None
):
    """
    Train a BPE tokenizer from scratch.
    
    Args:
        corpus_path: Path to text corpus file
        output_model: Output model path
        vocab_size: Target vocabulary size (default: 12000, same as base model)
        character_coverage: Character coverage (0.9999 for scripts like Ethiopic)
        user_defined_symbols: Comma-separated custom symbols
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_model) or ".", exist_ok=True)
    
    # Build training arguments
    train_args = [
        f"--input={corpus_path}",
        f"--model_prefix={output_model.replace('.model', '')}",
        f"--vocab_size={vocab_size}",
        f"--character_coverage={character_coverage}",
        "--model_type=bpe",
        "--pad_id=0",
        "--unk_id=1",
        "--bos_id=2",
        "--eos_id=3",
        "--normalization_rule_name=nfkc",
    ]
    
    if user_defined_symbols:
        train_args.append(f"--user_defined_symbols={user_defined_symbols}")
    
    print("[Train Standalone BPE]")
    print(f"  Corpus: {corpus_path}")
    print(f"  Output: {output_model}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Character coverage: {character_coverage}")
    if user_defined_symbols:
        print(f"  User symbols: {user_defined_symbols}")
    
    # Train the model
    print("\n[Training...]")
    spm.SentencePieceTrainer.Train(" ".join(train_args))
    
    # Verify output
    if os.path.exists(output_model):
        sp = spm.SentencePieceProcessor()
        sp.load(output_model)
        actual_size = sp.vocab_size()
        print(f"\n[Success] Trained tokenizer with {actual_size} tokens")
        print(f"  Saved to: {output_model}")
        
        if actual_size != vocab_size:
            print(f"\n⚠️  Warning: Final vocab ({actual_size}) differs from target ({vocab_size})")
            print(f"  This is normal - SentencePiece may adjust based on corpus.")
    else:
        raise RuntimeError(f"Failed to create model: {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train standalone BPE tokenizer (not extending base model)"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to text corpus file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output model path (e.g., tokenizers/amharic_standalone_bpe.model)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=12000,
        help="Target vocabulary size (default: 12000, same as base model)"
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9999,
        help="Character coverage (0.9999 for Ethiopic/Arabic/etc., 0.9995 for Latin)"
    )
    parser.add_argument(
        "--user-defined-symbols",
        type=str,
        default=None,
        help="Comma-separated custom symbols (e.g., '።,፣,፤,፥,፧')"
    )
    
    args = parser.parse_args()
    
    train_standalone_tokenizer(
        corpus_path=args.corpus,
        output_model=args.output,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        user_defined_symbols=args.user_defined_symbols
    )
