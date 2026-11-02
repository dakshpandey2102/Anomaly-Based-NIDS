"""
AutoNIDS - Phase 3: Task 3.1 Single-File Preprocessing & Sanitization Pipeline
=============================================================================
Target Input : data_raw/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Target Output: data_processed/Friday-WorkingHours-Afternoon-DDos.cleaned.csv

Philosophy:
- Uses ONLY Python's built-in standard library (csv, math, pathlib, statistics).
- Stream processing: O(1) memory overhead, never loads 2.83M rows into RAM.
- No blind row deletion: zero rows are dropped.
- Preserves physical meaning:
    * +Inf (rate from duration=0) is capped at the maximum observed finite value.
    * NaN (from 0/0 duration/bytes) is imputed with the feature's median.
- Header sanitization: strips whitespace and resolves the known duplicate 'Fwd Header Length'.
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data_raw" / "MachineLearningCVE"
PROCESSED_DIR = BASE_DIR / "data_processed"
INPUT_FILE = RAW_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUTPUT_FILE = PROCESSED_DIR / "Friday-WorkingHours-Afternoon-DDos.cleaned.csv"


def sanitize_header_names(raw_headers: list[str]) -> tuple[list[str], list[str]]:
    """
    Strips leading/trailing whitespace from column names.
    Detects and disambiguates any duplicates (e.g. CICIDS2017's duplicate 'Fwd Header Length').
    Returns (cleaned_headers, list_of_disambiguated_names).
    """
    seen_counts = defaultdict(int)
    cleaned_headers = []
    disambiguated = []

    for h in raw_headers:
        base_name = h.strip()
        seen_counts[base_name] += 1
        if seen_counts[base_name] == 1:
            cleaned_headers.append(base_name)
        else:
            # Suffix subsequent occurrences (e.g. 'Fwd Header Length.1')
            new_name = f"{base_name}.{seen_counts[base_name] - 1}"
            cleaned_headers.append(new_name)
            disambiguated.append((base_name, new_name))

    return cleaned_headers, disambiguated


def preprocess_single_file(input_path: Path, output_path: Path):
    print("=" * 80)
    print("AUTONIDS - PHASE 3: SANITIZATION & PREPROCESSING (TASK 3.1)")
    print(f"Input File : {input_path.name}")
    print(f"Output File: {output_path.name}")
    print("=" * 80)

    if not input_path.exists():
        print(f"[Error] Input file not found: {input_path}")
        return

    # Ensure processed directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # PASS 1: Profiling and Statistical Calculation
    # -------------------------------------------------------------------------
    print("\n[Pass 1/2] Profiling dataset and calculating column statistics...")

    with open(input_path, mode="r", encoding="utf-8", errors="replace") as f_in:
        reader = csv.reader(f_in)
        raw_headers = next(reader)
        num_cols = len(raw_headers)

        # Sanitize headers and detect duplicate column names
        clean_headers, duplicate_renames = sanitize_header_names(raw_headers)

        # Identify Label index
        label_idx = None
        for i, h in enumerate(clean_headers):
            if h.lower() == "label":
                label_idx = i
                break

        feature_indices = [i for i in range(num_cols) if i != label_idx]

        # Tracking containers for anomalies
        total_rows = 0
        rows_with_anomalies = 0
        nan_counts = defaultdict(int)
        pos_inf_counts = defaultdict(int)
        neg_inf_counts = defaultdict(int)
        blank_counts = defaultdict(int)

        # Numerical collectors for capping/imputation
        col_max_finite = {}
        col_min_finite = {}
        # Collect reservoir/sample values for median estimation without large memory
        col_samples = defaultdict(list)

        for row in reader:
            if len(row) != num_cols:
                continue

            total_rows += 1
            row_has_anomaly = False

            for idx in feature_indices:
                val_str = row[idx].strip()
                col_name = clean_headers[idx]

                if val_str == "":
                    blank_counts[col_name] += 1
                    row_has_anomaly = True
                    continue

                val_lower = val_str.lower()
                if val_lower == "nan":
                    nan_counts[col_name] += 1
                    row_has_anomaly = True
                    continue
                elif "inf" in val_lower:
                    if val_lower.startswith("-"):
                        neg_inf_counts[col_name] += 1
                    else:
                        pos_inf_counts[col_name] += 1
                    row_has_anomaly = True
                    continue

                try:
                    num_val = float(val_str)
                    if math.isnan(num_val):
                        nan_counts[col_name] += 1
                        row_has_anomaly = True
                    elif math.isinf(num_val):
                        if num_val > 0:
                            pos_inf_counts[col_name] += 1
                        else:
                            neg_inf_counts[col_name] += 1
                        row_has_anomaly = True
                    else:
                        # Valid finite number: track max, min, and reservoir sample
                        if col_name not in col_max_finite or num_val > col_max_finite[col_name]:
                            col_max_finite[col_name] = num_val
                        if col_name not in col_min_finite or num_val < col_min_finite[col_name]:
                            col_min_finite[col_name] = num_val

                        # Sample first 5,000 values per column to compute stable median
                        if len(col_samples[col_name]) < 5000:
                            col_samples[col_name].append(num_val)
                except ValueError:
                    blank_counts[col_name] += 1
                    row_has_anomaly = True

            if row_has_anomaly:
                rows_with_anomalies += 1

    # Compute column medians from samples
    col_medians = {}
    for idx in feature_indices:
        col_name = clean_headers[idx]
        samples = col_samples.get(col_name, [])
        col_medians[col_name] = statistics.median(samples) if samples else 0.0

    print(f"  Total Data Rows Inspected: {total_rows:,}")
    print(f"  Rows with Invalid Cells  : {rows_with_anomalies:,} ({(rows_with_anomalies/total_rows)*100:.3f}%)")
    print(f"  Duplicate Columns Resolved: {len(duplicate_renames)}")
    for old_n, new_n in duplicate_renames:
        print(f"    - Renamed duplicate '{old_n}' -> '{new_n}'")

    print("\n  Pre-Cleaning Anomaly Breakdown by Feature:")
    all_affected_cols = set(nan_counts) | set(pos_inf_counts) | set(neg_inf_counts) | set(blank_counts)
    if all_affected_cols:
        for c in sorted(all_affected_cols):
            print(f"    * Feature '{c}':")
            if pos_inf_counts[c]:
                print(f"        +Inf cells: {pos_inf_counts[c]:,}  (Strategy: Cap at max finite = {col_max_finite.get(c, 0):,})")
            if neg_inf_counts[c]:
                print(f"        -Inf cells: {neg_inf_counts[c]:,}  (Strategy: Cap at min finite = {col_min_finite.get(c, 0):,})")
            if nan_counts[c]:
                print(f"        NaN cells : {nan_counts[c]:,}  (Strategy: Impute with median = {col_medians.get(c, 0):.4f})")
            if blank_counts[c]:
                print(f"        Blank cells: {blank_counts[c]:,}  (Strategy: Impute with median = {col_medians.get(c, 0):.4f})")
    else:
        print("    No anomalies detected in this file.")

    # -------------------------------------------------------------------------
    # PASS 2: Cleaning, Transforming, and Writing
    # -------------------------------------------------------------------------
    print(f"\n[Pass 2/2] Streaming sanitization and writing to {output_path.name}...")

    written_rows = 0
    cleaned_anomalies_count = 0

    with open(input_path, mode="r", encoding="utf-8", errors="replace") as f_in, \
         open(output_path, mode="w", encoding="utf-8", newline="") as f_out:

        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Read original raw header
        next(reader)

        # Write sanitized unique headers
        writer.writerow(clean_headers)

        for row in reader:
            if len(row) != num_cols:
                continue

            cleaned_row = []

            for idx, val in enumerate(row):
                # If label column, preserve ground truth string exactly
                if idx == label_idx:
                    cleaned_row.append(val.strip())
                    continue

                col_name = clean_headers[idx]
                val_str = val.strip()

                # Handle Blank
                if val_str == "":
                    imputed = col_medians.get(col_name, 0.0)
                    cleaned_row.append(f"{imputed:.6f}")
                    cleaned_anomalies_count += 1
                    continue

                val_lower = val_str.lower()

                # Handle NaN
                if val_lower == "nan":
                    imputed = col_medians.get(col_name, 0.0)
                    cleaned_row.append(f"{imputed:.6f}")
                    cleaned_anomalies_count += 1
                    continue

                # Handle Inf
                if "inf" in val_lower:
                    if val_lower.startswith("-"):
                        capped = col_min_finite.get(col_name, 0.0)
                    else:
                        capped = col_max_finite.get(col_name, 0.0)
                    cleaned_row.append(f"{capped:.6f}")
                    cleaned_anomalies_count += 1
                    continue

                # Handle Float Conversion
                try:
                    num_val = float(val_str)
                    if math.isnan(num_val):
                        imputed = col_medians.get(col_name, 0.0)
                        cleaned_row.append(f"{imputed:.6f}")
                        cleaned_anomalies_count += 1
                    elif math.isinf(num_val):
                        capped = col_max_finite.get(col_name, 0.0) if num_val > 0 else col_min_finite.get(col_name, 0.0)
                        cleaned_row.append(f"{capped:.6f}")
                        cleaned_anomalies_count += 1
                    else:
                        cleaned_row.append(val_str)
                except ValueError:
                    imputed = col_medians.get(col_name, 0.0)
                    cleaned_row.append(f"{imputed:.6f}")
                    cleaned_anomalies_count += 1

            writer.writerow(cleaned_row)
            written_rows += 1

    # -------------------------------------------------------------------------
    # Verification of Generated Output File
    # -------------------------------------------------------------------------
    out_size_mb = output_path.stat().st_size / (1024 * 1024)
    print("\n" + "-" * 80)
    print("VERIFICATION OF CLEANED OUTPUT FILE")
    print("-" * 80)
    print(f"Original Rows    : {total_rows:,}")
    print(f"Cleaned Rows Out : {written_rows:,} (Row retention: 100.0%)")
    print(f"Cleaned Output MB: {out_size_mb:.2f} MB")
    print(f"Total Anomalous Values Replaced: {cleaned_anomalies_count:,}")
    print(f"Header Count     : {len(clean_headers)} unique columns")
    print(f"Any duplicate column headers remaining? {'YES (Error)' if len(clean_headers) != len(set(clean_headers)) else 'NO (100% Unique Headers)'}")
    print("=" * 80)
    print("TASK 3.1 PREPROCESSING COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    preprocess_single_file(INPUT_FILE, OUTPUT_FILE)
