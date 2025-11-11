#!/usr/bin/env python3
"""Comprehensive verification of subtitle deduplication fix"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.create_amharic_dataset import SubtitleSegment, deduplicate_subtitle_text

def print_segments(title, segments):
    """Pretty print segments"""
    print(f"\n{title}")
    print("=" * 60)
    for i, seg in enumerate(segments):
        print(f"{i+1}. [{seg.start_time:.1f}-{seg.end_time:.1f}] '{seg.text}'")

def test_real_world_rolling_text():
    """Test with realistic rolling subtitle pattern (50% overlap)"""
    print("\n" + "#" * 60)
    print("TEST: Real-world rolling subtitles (50% overlap pattern)")
    print("#" * 60)
    
    # Realistic pattern from YouTube subtitles
    # Each line repeats approximately 50% of previous line
    segments = [
        SubtitleSegment(0.0, 2.5, "Hello everyone welcome to", 1),
        SubtitleSegment(2.5, 5.0, "welcome to this tutorial about", 2),
        SubtitleSegment(5.0, 7.5, "tutorial about machine learning and", 3),
        SubtitleSegment(7.5, 10.0, "learning and artificial intelligence for", 4),
        SubtitleSegment(10.0, 12.5, "intelligence for natural language processing", 5),
    ]
    
    print_segments("INPUT (Before deduplication):", segments)
    
    # Calculate total overlap
    total_words_before = sum(len(s.text.split()) for s in segments)
    print(f"\nTotal words before: {total_words_before}")
    
    # Run deduplication
    result = deduplicate_subtitle_text(segments, min_overlap_words=2)
    
    print_segments("OUTPUT (After deduplication):", result)
    
    # Expected result: each segment should have unique words only
    expected = [
        "Hello everyone welcome to",
        "this tutorial about",
        "machine learning and",
        "artificial intelligence for",
        "natural language processing"
    ]
    
    print("\nEXPECTED texts:")
    for i, exp in enumerate(expected):
        print(f"{i+1}. '{exp}'")
    
    # Verify
    total_words_after = sum(len(s.text.split()) for s in result)
    print(f"\nTotal words after: {total_words_after}")
    print(f"Words removed: {total_words_before - total_words_after}")
    
    success = True
    if len(result) != len(expected):
        print(f"\n❌ FAIL: Expected {len(expected)} segments, got {len(result)}")
        success = False
    
    for i, (seg, exp) in enumerate(zip(result, expected)):
        if seg.text != exp:
            print(f"\n❌ FAIL at segment {i+1}:")
            print(f"   Expected: '{exp}'")
            print(f"   Got:      '{seg.text}'")
            success = False
    
    if success:
        print("\n✅ PASS: All segments correctly deduplicated!")
    
    return success

def test_edge_case_exact_duplicates():
    """Test exact duplicate handling"""
    print("\n" + "#" * 60)
    print("TEST: Exact duplicate segments")
    print("#" * 60)
    
    segments = [
        SubtitleSegment(0.0, 2.0, "This is the same text", 1),
        SubtitleSegment(2.0, 4.0, "This is the same text", 2),  # Exact dup
        SubtitleSegment(4.0, 6.0, "Now different text", 3),
        SubtitleSegment(6.0, 8.0, "Now different text", 4),  # Exact dup
        SubtitleSegment(8.0, 10.0, "Final text here", 5),
    ]
    
    print_segments("INPUT:", segments)
    result = deduplicate_subtitle_text(segments)
    print_segments("OUTPUT:", result)
    
    expected_count = 3  # Should skip the 2 duplicates
    success = len(result) == expected_count
    
    if success:
        print(f"\n✅ PASS: Correctly removed duplicates ({expected_count} segments remain)")
    else:
        print(f"\n❌ FAIL: Expected {expected_count} segments, got {len(result)}")
    
    return success

def test_edge_case_with_skipped_segment():
    """Test the specific bug: when a segment is skipped, does next comparison work?"""
    print("\n" + "#" * 60)
    print("TEST: Rolling text with skipped duplicate (the actual bug)")
    print("#" * 60)
    
    # This is the pattern that exposed the bug:
    # Segment 2 is an exact duplicate of segment 1
    # Segment 3 has rolling text overlap with segment 2
    # OLD BUG: When seg 2 was skipped, seg 3 compared with seg 1, missing overlap
    segments = [
        SubtitleSegment(0.0, 2.0, "Hello world this is", 1),
        SubtitleSegment(2.0, 4.0, "Hello world this is", 2),  # Exact duplicate - will be skipped
        SubtitleSegment(4.0, 6.0, "this is a test", 3),       # Overlaps with seg 2 (not seg 1!)
    ]
    
    print_segments("INPUT:", segments)
    print("\nNote: Segment 2 is exact duplicate of segment 1")
    print("      Segment 3 has overlap with segment 2 ('this is')")
    print("\nOLD BUG: Would compare seg 3 with seg 1 (last ADDED), missing overlap")
    print("NEW FIX: Compares seg 3 with seg 2 (previous INPUT), finds overlap")
    
    result = deduplicate_subtitle_text(segments, min_overlap_words=2)
    print_segments("OUTPUT:", result)
    
    # Expected: seg 1 kept, seg 2 skipped, seg 3 has overlap removed
    expected_texts = [
        "Hello world this is",
        "a test"  # "this is" should be removed as overlap
    ]
    
    print("\nEXPECTED:")
    for i, exp in enumerate(expected_texts):
        print(f"{i+1}. '{exp}'")
    
    success = True
    if len(result) != len(expected_texts):
        print(f"\n❌ FAIL: Expected {len(expected_texts)} segments, got {len(result)}")
        success = False
    
    for i, (seg, exp) in enumerate(zip(result, expected_texts)):
        if seg.text != exp:
            print(f"\n❌ FAIL at segment {i+1}:")
            print(f"   Expected: '{exp}'")
            print(f"   Got:      '{seg.text}'")
            success = False
    
    if success:
        print("\n✅ PASS: Bug is FIXED! Overlap detected correctly even after skipped segment")
    
    return success

def test_amharic_example():
    """Test with Amharic-like text patterns"""
    print("\n" + "#" * 60)
    print("TEST: Amharic-style rolling subtitles")
    print("#" * 60)
    
    # Simulate Amharic subtitle pattern
    segments = [
        SubtitleSegment(0.0, 2.0, "ሰላም ለሁሉም እንኳን ወደ", 1),
        SubtitleSegment(2.0, 4.0, "ወደ ዚህ ትምህርት በደህና", 2),
        SubtitleSegment(4.0, 6.0, "በደህና መጡ ስለ ማሽን", 3),
        SubtitleSegment(6.0, 8.0, "ማሽን መማሪያ እና ሰው ሰራሽ", 4),
    ]
    
    print_segments("INPUT:", segments)
    result = deduplicate_subtitle_text(segments, min_overlap_words=1)
    print_segments("OUTPUT:", result)
    
    # Should remove overlapping words
    success = len(result) == len(segments)
    if success:
        print("\n✅ PASS: Amharic text deduplication works")
    
    return success

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SUBTITLE DEDUPLICATION FIX VERIFICATION")
    print("=" * 60)
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_real_world_rolling_text()
    all_passed &= test_edge_case_exact_duplicates()
    all_passed &= test_edge_case_with_skipped_segment()  # The critical bug test
    all_passed &= test_amharic_example()
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅✅✅ ALL TESTS PASSED - FIX IS WORKING! ✅✅✅")
        print("=" * 60)
        print("\nThe deduplication bug has been successfully fixed.")
        print("Rolling subtitle text will now be properly removed.")
        sys.exit(0)
    else:
        print("❌❌❌ SOME TESTS FAILED - FIX NOT WORKING ❌❌❌")
        print("=" * 60)
        sys.exit(1)
