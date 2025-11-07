#!/usr/bin/env python3
"""
Multilingual BPE Tokenizer Training

Trains SentencePiece BPE model with support for multiple languages including Amharic.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import sentencepiece as spm


def train_tokenizer(
    corpus_files: List[Path],
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    language_weights: Optional[Dict[str, float]] = None,
) -> None:
    """Train multilingual SentencePiece tokenizer
    
    Args:
        corpus_files: List of corpus file paths
        model_prefix: Output model prefix (will create .model and .vocab)
        vocab_size: Vocabulary size
        model_type: Model type (bpe or unigram)
        character_coverage: Character coverage ratio
        language_weights: Optional dict of language -> weight
    """
    # Prepare input
    input_files = []
    for corpus_file in corpus_files:
        if not corpus_file.exists():
            print(f"Warning: Corpus file not found: {corpus_file}")
            continue
        input_files.append(str(corpus_file))
    
    if not input_files:
        raise ValueError("No valid corpus files provided")
    
    # Build command
    input_str = ",".join(input_files)
    
    # User defined symbols (for Amharic punctuation)
    user_defined_symbols = [
        "።",  # Amharic full stop
        "፣",  # Amharic comma
        "፤",  # Amharic semicolon
        "፥",  # Amharic colon
        "፦",  # Amharic preface colon
        "፧",  # Amharic question mark
        "፨",  # Amharic paragraph separator
    ]
    
    print("Training SentencePiece model...")
    print(f"  Model type: {model_type}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Character coverage: {character_coverage}")
    print(f"  Input files: {len(input_files)}")
    
    spm.SentencePieceTrainer.train(
        input=input_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        user_defined_symbols=user_defined_symbols,
        normalization_rule_name="nfkc",  # Normalize Unicode
        remove_extra_whitespaces=True,
        split_by_unicode_script=True,  # Important for mixed scripts
        split_by_whitespace=True,
        split_by_number=True,
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=False,
    )
    
    print(f"\nModel saved to: {model_prefix}.model")
    print(f"Vocabulary saved to: {model_prefix}.vocab")
    
    # Test the model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    print(f"\nModel loaded successfully!")
    print(f"  Vocabulary size: {sp.get_piece_size()}")
    print(f"  BOS ID: {sp.bos_id()}")
    print(f"  EOS ID: {sp.eos_id()}")
    print(f"  UNK ID: {sp.unk_id()}")
    print(f"  PAD ID: {sp.pad_id()}")
    
    # Test encoding
    test_texts = [
        "Hello, world!",
        "你好世界",
        "こんにちは世界",
        "ሰላም ልዑል!"  # Hello world in Amharic
    ]
    
    print("\nTest encodings:")
    for text in test_texts:
        tokens = sp.encode(text, out_type=str)
        ids = sp.encode(text, out_type=int)
        print(f"  {text}")
        print(f"    Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"    Tokens: {tokens}")
        print(f"    IDs: {ids[:10]}..." if len(ids) > 10 else f"    IDs: {ids}")


def analyze_coverage(
    model_path: Path,
    test_files: List[Path],
) -> None:
    """Analyze tokenizer coverage on test files
    
    Args:
        model_path: Path to .model file
        test_files: List of test corpus files
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    
    print("\nAnalyzing coverage...")
    
    for test_file in test_files:
        if not test_file.exists():
            continue
        
        unk_count = 0
        total_count = 0
        line_count = 0
        
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                line_count += 1
                ids = sp.encode(line, out_type=int)
                
                for id in ids:
                    total_count += 1
                    if id == sp.unk_id():
                        unk_count += 1
        
        unk_rate = (unk_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n  File: {test_file.name}")
        print(f"    Lines: {line_count:,}")
        print(f"    Tokens: {total_count:,}")
        print(f"    UNK tokens: {unk_count:,} ({unk_rate:.2f}%)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multilingual BPE tokenizer with Amharic support"
    )
    
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        required=True,
        help="Corpus file(s) for training"
    )
    
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="multilingual_bpe",
        help="Output model prefix (default: multilingual_bpe)"
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)"
    )
    
    parser.add_argument(
        "--model-type",
        choices=["bpe", "unigram"],
        default="bpe",
        help="Model type (default: bpe)"
    )
    
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage (default: 0.9995)"
    )
    
    parser.add_argument(
        "--test-files",
        type=Path,
        nargs="*",
        help="Test files for coverage analysis"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Train tokenizer
    train_tokenizer(
        corpus_files=args.corpus,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
    )
    
    # Analyze coverage if test files provided
    if args.test_files:
        model_path = Path(f"{args.model_prefix}.model")
        analyze_coverage(model_path, args.test_files)


if __name__ == "__main__":
    main()
