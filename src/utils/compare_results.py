#!/usr/bin/env python3
"""
Compare two webqsp result files and categorize indices based on correctness.

Usage:
    python compare_results.py <file1.jsonl> <file2.jsonl>
"""

import sys
import argparse
from src.utils.evaluate import list_indices


def main():
    parser = argparse.ArgumentParser(
        description="Compare two webqsp result files and categorize indices"
    )
    parser.add_argument(
        "file1",
        type=str,
        help="Path to the first JSONL results file"
    )
    parser.add_argument(
        "file2",
        type=str,
        help="Path to the second JSONL results file"
    )
    
    args = parser.parse_args()
    
    # Get correct and incorrect indices for both files
    correct1, incorrect1 = list_indices(args.file1)
    correct2, incorrect2 = list_indices(args.file2)
    
    # Assert that both files have the same number of elements
    total1 = len(correct1) + len(incorrect1)
    total2 = len(correct2) + len(incorrect2)
    
    if total1 != total2:
        print(f"Error: Files have different number of elements!")
        print(f"  File 1: {total1} elements")
        print(f"  File 2: {total2} elements")
        sys.exit(1)
    
    # Convert to sets for easier operations
    correct1_set = set(correct1)
    incorrect1_set = set(incorrect1)
    correct2_set = set(correct2)
    incorrect2_set = set(incorrect2)
    
    # Categorize indices
    # 1) Correct in both
    correct_both = sorted(list(correct1_set & correct2_set))
    
    # 2) Incorrect in both
    incorrect_both = sorted(list(incorrect1_set & incorrect2_set))
    
    # 3) Correct in first but incorrect in second
    correct_first_incorrect_second = sorted(list(correct1_set & incorrect2_set))
    
    # 4) Correct in second but incorrect in first
    correct_second_incorrect_first = sorted(list(correct2_set & incorrect1_set))
    
    # Print results
    print(f"\nTotal elements: {total1}")
    print(f"\n1) Correct in both: {len(correct_both)} indices")
    if correct_both:
        print(f"   Indices: {correct_both}")
    
    print(f"\n2) Incorrect in both: {len(incorrect_both)} indices")
    if incorrect_both:
        print(f"   Indices: {incorrect_both}")
    
    print(f"\n3) Correct in first but incorrect in second: {len(correct_first_incorrect_second)} indices")
    if correct_first_incorrect_second:
        print(f"   Indices: {correct_first_incorrect_second}")
    
    print(f"\n4) Correct in second but incorrect in first: {len(correct_second_incorrect_first)} indices")
    if correct_second_incorrect_first:
        print(f"   Indices: {correct_second_incorrect_first}")
    
    # Verify all indices are accounted for
    all_indices = set(range(total1))
    accounted = (set(correct_both) | set(incorrect_both) | 
                 set(correct_first_incorrect_second) | set(correct_second_incorrect_first))
    
    if all_indices != accounted:
        print(f"\nWarning: Some indices are missing from the categorization!")
        missing = sorted(list(all_indices - accounted))
        print(f"  Missing indices: {missing}")


if __name__ == "__main__":
    main()

