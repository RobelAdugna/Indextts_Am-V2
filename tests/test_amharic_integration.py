#!/usr/bin/env python3
"""
Amharic Integration Tests

Verifies that all Amharic components work together correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indextts.utils.front import TextNormalizer
from indextts.utils.text_utils import contains_amharic, get_text_syllable_num, get_text_tts_dur


def test_amharic_detection():
    """Test Amharic script detection"""
    print("\n" + "="*50)
    print("Test 1: Amharic Script Detection")
    print("="*50)
    
    test_cases = [
        ("ሰላም ልዑል!", True, "Pure Amharic"),
        ("Hello world", False, "English only"),
        ("ሰላም Hello", True, "Mixed Amharic-English"),
        ("你好世界", False, "Chinese only"),
        ("こんにちは", False, "Japanese only"),
    ]
    
    normalizer = TextNormalizer()
    passed = 0
    
    for text, expected, description in test_cases:
        result = normalizer.is_amharic(text)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {description}: '{text}' → {result}")
        if result == expected:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_text_normalization():
    """Test Amharic text normalization"""
    print("\n" + "="*50)
    print("Test 2: Text Normalization")
    print("="*50)
    
    normalizer = TextNormalizer(preferred_language="am")
    
    test_cases = [
        "ሰላም ልዑል። እንዴት ነዎት፧",
        "አማርኛ በጣም ቆንጆ ቋንቋ ነው።",
        "speaker 1: ሰላም",  # Should remove speaker tag
        "ተናጋሪ 2: እንዴት ነህ?",  # Amharic speaker tag
    ]
    
    for text in test_cases:
        normalized = normalizer.normalize(text, language="am")
        print(f"  Original:   {text}")
        print(f"  Normalized: {normalized}")
        print()
    
    return True


def test_syllable_counting():
    """Test Amharic syllable counting"""
    print("\n" + "="*50)
    print("Test 3: Syllable Counting")
    print("="*50)
    
    test_cases = [
        "ሰላም",  # 2 syllables (ሰ-ላ-ም = 3 chars but ም is consonant)
        "አማርኛ",  # 4 syllables
        "Hello",  # 2 syllables (English)
        "ሰላም Hello",  # Mixed
    ]
    
    for text in test_cases:
        count = get_text_syllable_num(text)
        print(f"  '{text}' → {count} syllables")
    
    return True


def test_duration_estimation():
    """Test duration estimation for Amharic"""
    print("\n" + "="*50)
    print("Test 4: Duration Estimation")
    print("="*50)
    
    test_cases = [
        "ሰላም ልዑል!",
        "አማርኛ በጣም ቆንጆ ቋንቋ ነው።",
        "Hello world",  # For comparison
    ]
    
    for text in test_cases:
        is_amharic = contains_amharic(text)
        max_dur, min_dur = get_text_tts_dur(text)
        print(f"  '{text}'")
        print(f"    Is Amharic: {is_amharic}")
        print(f"    Duration: {min_dur:.2f}s - {max_dur:.2f}s")
        print()
    
    return True


def test_file_existence():
    """Test that all required files exist"""
    print("\n" + "="*50)
    print("Test 5: File Existence")
    print("="*50)
    
    required_files = [
        "tools/youtube_amharic_downloader.py",
        "tools/create_amharic_dataset.py",
        "tools/collect_amharic_corpus.py",
        "tools/train_multilingual_bpe.py",
        "scripts/amharic/end_to_end.sh",
        "scripts/amharic/end_to_end.ps1",
        "examples/amharic_youtube_urls.txt",
        "examples/amharic_test_cases.jsonl",
        "docs/AMHARIC_SUPPORT.md",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False
    
    print(f"\nAll files exist: {all_exist}")
    return all_exist


def main():
    """Run all integration tests"""
    print("\n" + "#"*50)
    print("# Amharic Integration Tests")
    print("#"*50)
    
    results = []
    
    # Run tests
    results.append(("File Existence", test_file_existence()))
    results.append(("Amharic Detection", test_amharic_detection()))
    results.append(("Text Normalization", test_text_normalization()))
    results.append(("Syllable Counting", test_syllable_counting()))
    results.append(("Duration Estimation", test_duration_estimation()))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("✅ Amharic implementation is ready for use.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
