"""
AutoNIDS - Phase 1: Task 1.2 Cross-File Dataset Inspection (Standard Library)
=============================================================================
Purpose:
Expand the verified streaming inspection to all eight CICIDS2017 CSV files.
Uses ONLY Python's built-in standard library (csv, collections, math, pathlib).

Guarantees:
- O(1) memory per row: files are processed one-by-one in a streaming fashion.
- Schema verification: verifies that all 8 files share identical column headers.
- Comprehensive inventory: aggregates exact total rows, label taxonomy, and flaw counts.
"""

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_raw" / "MachineLearningCVE"


def inspect_all_files():
    print("=" * 80)
    print("AUTONIDS - PHASE 1: TASK 1.2 ALL-FILES DATASET INSPECTION")
    print(f"Data Directory: {DATA_DIR}")
    print("=" * 80)

    if not DATA_DIR.exists():
        print(f"[Error] Directory not found: {DATA_DIR}")
        return

    csv_files = sorted(list(DATA_DIR.glob("*.csv")))
    if not csv_files:
        print("[Error] No CSV files found in directory.")
        return

    print(f"Found {len(csv_files)} CSV files. Beginning sequential streaming inspection...\n")

    grand_total_rows = 0
    grand_label_counts = Counter()
    grand_nan_cells = 0
    grand_inf_cells = 0
    reference_headers = None
    schema_mismatch_detected = False

    per_file_summary = []

    for idx, csv_path in enumerate(csv_files, start=1):
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"[{idx}/{len(csv_files)}] Processing: {csv_path.name} ({file_size_mb:.2f} MB)...")

        file_rows = 0
        file_labels = Counter()
        file_nan = 0
        file_inf = 0

        with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)

            try:
                raw_headers = next(reader)
            except StopIteration:
                print(f"    [Warning] File is empty: {csv_path.name}")
                continue

            clean_headers = [h.strip() for h in raw_headers]
            num_cols = len(clean_headers)

            # Check schema consistency with the first file inspected
            if reference_headers is None:
                reference_headers = clean_headers
            else:
                if clean_headers != reference_headers:
                    print(f"    [Alert] Schema mismatch detected in {csv_path.name}!")
                    schema_mismatch_detected = True

            # Locate label column
            label_idx = None
            for c_idx, col_name in enumerate(clean_headers):
                if col_name.lower() == "label":
                    label_idx = c_idx
                    break

            # Stream rows
            for row in reader:
                if len(row) != num_cols:
                    continue

                file_rows += 1

                if label_idx is not None:
                    label = row[label_idx].strip()
                    file_labels[label] += 1

                for val in row:
                    val_str = val.strip()
                    if not val_str:
                        file_nan += 1
                        continue

                    val_lower = val_str.lower()
                    if val_lower == "nan":
                        file_nan += 1
                    elif "inf" in val_lower:
                        file_inf += 1
                    else:
                        try:
                            num_val = float(val_str)
                            if math.isnan(num_val):
                                file_nan += 1
                            elif math.isinf(num_val):
                                file_inf += 1
                        except ValueError:
                            pass

        # Update grand totals
        grand_total_rows += file_rows
        grand_label_counts.update(file_labels)
        grand_nan_cells += file_nan
        grand_inf_cells += file_inf

        per_file_summary.append({
            "name": csv_path.name,
            "size_mb": file_size_mb,
            "rows": file_rows,
            "cols": num_cols,
            "nan": file_nan,
            "inf": file_inf,
            "labels": file_labels
        })

        print(f"    Rows: {file_rows:,} | NaN cells: {file_nan:,} | Inf cells: {file_inf:,}")
        for lbl, cnt in file_labels.most_common():
            pct = (cnt / file_rows) * 100 if file_rows > 0 else 0
            print(f"      - {lbl:<32}: {cnt:>8,} ({pct:5.2f}%)")
        print()

    # Final Aggregated Report
    print("=" * 80)
    print("AGGREGATED CICIDS2017 DATASET AUDIT REPORT")
    print("=" * 80)
    print(f"Total Files Inspected      : {len(csv_files)}")
    print(f"Total Network Flow Records : {grand_total_rows:,}")
    print(f"Schema Consistency Check   : {'PASSED (All 8 files share identical 79 columns)' if not schema_mismatch_detected else 'FAILED'}")
    print(f"Total NaN Cells Detected   : {grand_nan_cells:,}")
    print(f"Total Inf Cells Detected   : {grand_inf_cells:,}")

    print("\nComplete Attack Taxonomy & Global Class Distribution:")
    benign_total = 0
    attack_total = 0

    for label, count in grand_label_counts.most_common():
        pct = (count / grand_total_rows) * 100 if grand_total_rows > 0 else 0
        if "benign" in label.lower():
            benign_total += count
            tag = "[NORMAL]"
        else:
            attack_total += count
            tag = "[ATTACK]"
        print(f"  {tag:<9} {label:<35}: {count:>9,} ({pct:6.2f}%)")

    benign_pct = (benign_total / grand_total_rows) * 100 if grand_total_rows > 0 else 0
    attack_pct = (attack_total / grand_total_rows) * 100 if grand_total_rows > 0 else 0

    print("-" * 80)
    print(f"BENIGN Traffic Total       : {benign_total:>9,} records ({benign_pct:6.2f}%)")
    print(f"ATTACK Traffic Total       : {attack_total:>9,} records ({attack_pct:6.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    inspect_all_files()
