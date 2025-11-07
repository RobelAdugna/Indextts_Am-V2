#!/usr/bin/env python3
"""
Amharic Corpus Collector

Collects and cleans Amharic text from multiple sources for tokenizer training.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List

from tqdm import tqdm

from indextts.utils.front import TextNormalizer


class AmharicCorpusCollector:
    """Collect and process Amharic text corpus"""
    
    def __init__(self, normalizer: TextNormalizer = None):
        self.normalizer = normalizer or TextNormalizer(preferred_language="am")
        self.texts: List[str] = []
        self.stats = {
            'total_lines': 0,
            'valid_lines': 0,
            'duplicate_lines': 0,
            'short_lines': 0,
            'non_amharic_lines': 0,
        }
    
    def is_valid_amharic(self, text: str, min_amharic_ratio: float = 0.5) -> bool:
        """Check if text is valid Amharic
        
        Args:
            text: Text to check
            min_amharic_ratio: Minimum ratio of Amharic characters
        
        Returns:
            True if valid Amharic text
        """
        # Count Amharic characters
        amharic_chars = re.findall(r'[\u1200-\u137f\u1380-\u139f\u2d80-\u2ddf\uab00-\uab2f]', text)
        total_chars = re.findall(r'[\w]', text)
        
        if not total_chars:
            return False
        
        ratio = len(amharic_chars) / len(total_chars)
        return ratio >= min_amharic_ratio
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Normalize
        text = self.normalizer.normalize(text, language="am")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove repeated punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        
        return text.strip()
    
    def add_from_jsonl(self, jsonl_path: Path, text_field: str = "text"):
        """Add texts from JSONL file
        
        Args:
            jsonl_path: Path to JSONL file
            text_field: Field name containing text
        """
        print(f"Reading from {jsonl_path}...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing JSONL"):
                self.stats['total_lines'] += 1
                
                try:
                    data = json.loads(line)
                    text = data.get(text_field, "")
                    
                    if text:
                        self.add_text(text)
                except json.JSONDecodeError:
                    continue
    
    def add_from_text_file(self, text_path: Path):
        """Add texts from plain text file
        
        Args:
            text_path: Path to text file (one sentence per line)
        """
        print(f"Reading from {text_path}...")
        
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing text file"):
                self.stats['total_lines'] += 1
                line = line.strip()
                
                if line:
                    self.add_text(line)
    
    def add_text(self, text: str, min_length: int = 5):
        """Add a single text to corpus
        
        Args:
            text: Text to add
            min_length: Minimum text length
        """
        # Clean
        text = self.clean_text(text)
        
        # Check length
        if len(text) < min_length:
            self.stats['short_lines'] += 1
            return
        
        # Check if Amharic
        if not self.is_valid_amharic(text):
            self.stats['non_amharic_lines'] += 1
            return
        
        self.texts.append(text)
        self.stats['valid_lines'] += 1
    
    def remove_duplicates(self):
        """Remove duplicate texts"""
        original_count = len(self.texts)
        self.texts = list(dict.fromkeys(self.texts))  # Preserves order
        self.stats['duplicate_lines'] = original_count - len(self.texts)
    
    def get_character_stats(self) -> dict:
        """Get character frequency statistics
        
        Returns:
            Dict with character statistics
        """
        all_text = ' '.join(self.texts)
        char_counts = Counter(all_text)
        
        # Filter for Amharic characters
        amharic_chars = {}
        for char, count in char_counts.items():
            if re.match(r'[\u1200-\u137f\u1380-\u139f\u2d80-\u2ddf\uab00-\uab2f]', char):
                amharic_chars[char] = count
        
        return {
            'total_characters': len(all_text),
            'unique_characters': len(char_counts),
            'amharic_characters': len(amharic_chars),
            'most_common_amharic': char_counts.most_common(50),
        }
    
    def save_corpus(self, output_path: Path, shuffle: bool = True):
        """Save corpus to file
        
        Args:
            output_path: Output file path
            shuffle: Whether to shuffle lines
        """
        import random
        
        texts = self.texts.copy()
        
        if shuffle:
            random.shuffle(texts)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        print(f"\nCorpus saved to: {output_path}")
        print(f"Total lines: {len(texts)}")
    
    def print_stats(self):
        """Print collection statistics"""
        print("\n" + "="*50)
        print("Corpus Collection Statistics")
        print("="*50)
        print(f"Total lines processed:    {self.stats['total_lines']:,}")
        print(f"Valid lines:              {self.stats['valid_lines']:,}")
        print(f"Duplicate lines removed:  {self.stats['duplicate_lines']:,}")
        print(f"Short lines skipped:      {self.stats['short_lines']:,}")
        print(f"Non-Amharic lines:        {self.stats['non_amharic_lines']:,}")
        print(f"Final corpus size:        {len(self.texts):,} lines")
        print("="*50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect and clean Amharic text corpus for tokenizer training"
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Input file(s) - can be JSONL or text files"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output corpus file"
    )
    
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field name for text in JSONL files (default: text)"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum text length (default: 5)"
    )
    
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle output corpus"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show character statistics"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create collector
    collector = AmharicCorpusCollector()
    
    # Process input files
    for input_file in args.input:
        if not input_file.exists():
            print(f"Warning: File not found: {input_file}")
            continue
        
        if input_file.suffix in ['.jsonl', '.json']:
            collector.add_from_jsonl(input_file, text_field=args.text_field)
        else:
            collector.add_from_text_file(input_file)
    
    # Remove duplicates
    print("\nRemoving duplicates...")
    collector.remove_duplicates()
    
    # Print stats
    collector.print_stats()
    
    if args.stats:
        print("\nComputing character statistics...")
        char_stats = collector.get_character_stats()
        print(f"\nTotal characters: {char_stats['total_characters']:,}")
        print(f"Unique characters: {char_stats['unique_characters']:,}")
        print(f"Amharic characters: {char_stats['amharic_characters']:,}")
        print("\nMost common Amharic characters:")
        for char, count in char_stats['most_common_amharic'][:20]:
            print(f"  {char}: {count:,}")
    
    # Save corpus
    collector.save_corpus(args.output, shuffle=not args.no_shuffle)


if __name__ == "__main__":
    main()
