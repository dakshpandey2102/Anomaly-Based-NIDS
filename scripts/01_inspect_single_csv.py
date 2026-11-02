"""
AutoNIDS - Phase 1: Task 1.1 Single-CSV Dataset Inspection (Standard Library)
=============================================================================
Target File: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

This script uses ONLY Python's built-in standard library:
- csv: To read and parse comma-separated values row by row.
- math: To verify IEEE floating-point conditions (isinf, isnan).
- collections: Counter and defaultdict for tallying frequencies without external dependencies.
- pathlib: For safe, platform-independent file path resolution.

Memory Concept:
Rather than loading all ~74 MB of raw CSV into memory at once (like pandas read_csv),
this script streams the file line by line using Python's file iterator protocol.
At any given moment, only a single row string resides in RAM.
This guarantees O(1) auxiliary memory consumption regardless of file size.
"""

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

# Define file paths using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = (
    BASE_DIR
    / "data_raw"
    / "MachineLearningCVE"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


def inspect_dataset(file_path: Path):
    print("=" * 75)
    print("AUTONIDS - PHASE 1: DATASET INSPECTION (STANDARD LIBRARY)")
    print(f"Target File: {file_path.name}")
    print(f"Full Path:   {file_path}")
    print("=" * 75)

    if not file_path.exists():
        print(f"\n[Error] The file does not exist at: {file_path}")
        return

    # Check file size on disk using built-in stat()
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"\n[File Info] File size on disk: {file_size_mb:.2f} MB")

    with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)

        # 1. Read and inspect header row
        try:
            raw_headers = next(reader)
        except StopIteration:
            print("\n[Error] The CSV file is empty.")
            return

        num_cols = len(raw_headers)
        print(f"\n[Header Inspection]")
        print(f"  Total Columns Detected: {num_cols}")

        # Check for whitespace issues in column names
        headers_with_whitespace = [h for h in raw_headers if h != h.strip()]
        print(f"  Columns with leading/trailing spaces: {len(headers_with_whitespace)} / {num_cols}")

        # Clean headers for internal mapping (strip spaces)
        clean_headers = [h.strip() for h in raw_headers]

        # Print sample column names to verify layout
        print("  First 5 columns :", clean_headers[:5])
        print("  Last 5 columns  :", clean_headers[-5:])

        # 2. Identify the target label column
        label_col_index = None
        actual_label_name = None
        for idx, col in enumerate(clean_headers):
            if col.lower() == "label":
                label_col_index = idx
                actual_label_name = raw_headers[idx]
                break

        if label_col_index is None:
            print("\n[Warning] Could not find a column named 'Label'!")
        else:
            print(f"  Found Label Column: '{actual_label_name}' at column index {label_col_index}")

        # 3. Stream through rows one by one
        print("\n[Streaming Rows] Processing dataset line-by-line (O(1) memory)...")

        total_rows = 0
        label_counts = Counter()
        blank_counts = defaultdict(int)
        inf_counts = defaultdict(int)
        nan_counts = defaultdict(int)

        for row in reader:
            total_rows += 1

            # Verify row length matches header length
            if len(row) != num_cols:
                continue

            # Tally label if present
            if label_col_index is not None:
                label_val = row[label_col_index].strip()
                label_counts[label_val] += 1

            # Check each field for blank, NaN, or Infinity
            for col_idx, val in enumerate(row):
                val_str = val.strip()

                # Check for blank / empty fields
                if val_str == "":
                    blank_counts[clean_headers[col_idx]] += 1
                    continue

                # Check for textual representation of 'nan' or 'inf'
                val_lower = val_str.lower()
                if val_lower == "nan":
                    nan_counts[clean_headers[col_idx]] += 1
                    continue
                if "inf" in val_lower:
                    inf_counts[clean_headers[col_idx]] += 1
                    continue

                # Numerical check for IEEE float nan / inf
                try:
                    num_val = float(val_str)
                    if math.isnan(num_val):
                        nan_counts[clean_headers[col_idx]] += 1
                    elif math.isinf(num_val):
                        inf_counts[clean_headers[col_idx]] += 1
                except ValueError:
                    # Non-numeric text values (like labels) are safely skipped
                    pass

    # 4. Display findings
    print("\n" + "-" * 75)
    print("SUMMARY OF VERIFIED DATASET FINDINGS")
    print("-" * 75)
    print(f"Total Data Rows Processed: {total_rows:,}")

    print(f"\nLabel Frequencies (Target Classes):")
    for label, count in label_counts.most_common():
        pct = (count / total_rows) * 100 if total_rows > 0 else 0
        print(f"  - {label:<20}: {count:>8,} records ({pct:6.2f}%)")

    print(f"\nData Quality - Blank Fields:")
    if blank_counts:
        for col, count in blank_counts.items():
            print(f"  - Column '{col}': {count:,} blank cells")
    else:
        print("  - None detected.")

    print(f"\nData Quality - NaN (Not A Number) Values:")
    if nan_counts:
        for col, count in nan_counts.items():
            print(f"  - Column '{col}': {count:,} NaN cells")
    else:
        print("  - None detected.")

    print(f"\nData Quality - Infinity (+Inf / -Inf) Values:")
    if inf_counts:
        for col, count in inf_counts.items():
            print(f"  - Column '{col}': {count:,} infinite cells")
    else:
        print("  - None detected.")

    print("\n" + "=" * 75)
    print("INSPECTION SCRIPT COMPLETED SUCCESSFULLY.")
    print("=" * 75)


if __name__ == "__main__":
    inspect_dataset(CSV_PATH)
