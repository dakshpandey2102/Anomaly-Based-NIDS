"""
AutoNIDS â€“ Stage 1: Dataset Discovery and Data-Quality Audit
=============================================================
Purpose
-------
Perform a thorough, non-destructive inspection of all eight CICIDS2017
MachineLearningCSV files BEFORE any modelling decisions are made.

What this script does
---------------------
1. Inventory each CSV (path, file size, row/column counts, encoding check).
2. Detect header whitespace and column-name inconsistencies across files.
3. Show the first few rows and dtypes for manual review.
4. Count label distributions (benign vs attack) per file.
5. Detect: missing values (NaN), positive/negative infinity, duplicate rows,
   constant columns (zero variance), and columns that are entirely NaN.
6. Flag potential data-leakage shortcuts: timestamps, IPs, ports, flow IDs.
7. Write a machine-readable JSON report  â†’ reports/stage1_report.json
8. Write a human-readable text summary  â†’ reports/stage1_summary.txt

No data is modified; original CSVs are never touched.
"""

import json
import math
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT       = Path(__file__).resolve().parent.parent          # AutoNIDS/
DATA_DIR   = ROOT / "data_raw" / "MachineLearningCVE"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

JSON_REPORT  = REPORT_DIR / "stage1_report.json"
TEXT_SUMMARY = REPORT_DIR / "stage1_summary.txt"

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LABEL_COLUMN_CANDIDATES = [" Label", "Label", "label", " label"]

# Columns that are commonly identifiers / timestamps / network addresses
# â€” potential "shortcut" leakage features we must investigate.
SHORTCUT_KEYWORDS = [
    "ip", "addr", "address", "port", "src", "dst", "source", "destination",
    "timestamp", "time", "date", "id", "flow id", "flowid", "protocol",
    "pid", "uid"
]

def bytes_to_mb(n: int) -> float:
    return round(n / (1024 ** 2), 3)

def find_label_column(columns: list[str]) -> str | None:
    """Return the label column name, trying known variants."""
    for c in LABEL_COLUMN_CANDIDATES:
        if c in columns:
            return c
    # fuzzy: strip whitespace from every column and check
    for c in columns:
        if c.strip().lower() == "label":
            return c
    return None

def flag_shortcut_columns(columns: list[str]) -> list[str]:
    """Return columns whose names match known identifier/address patterns."""
    hits = []
    for c in columns:
        c_lower = c.strip().lower()
        if any(kw in c_lower for kw in SHORTCUT_KEYWORDS):
            hits.append(c)
    return hits

def inspect_file(csv_path: Path) -> dict:
    """
    Load a single CSV and return a structured audit dictionary.

    Returns
    -------
    dict with keys:
        filename, size_mb, n_rows, n_cols, columns_raw,
        columns_stripped, dtype_map, label_col,
        label_distribution, has_whitespace_in_headers,
        missing_per_col (top-20 if large), inf_count,
        duplicate_rows, constant_cols, all_nan_cols,
        shortcut_cols, sample_rows (5), warnings
    """
    audit = {
        "filename": csv_path.name,
        "size_mb": bytes_to_mb(csv_path.stat().st_size),
        "warnings": []
    }

    print(f"\n{'='*70}")
    print(f"  Reading: {csv_path.name}  ({audit['size_mb']} MB)")
    print(f"{'='*70}")

    # â”€â”€ Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        audit["warnings"].append(f"LOAD ERROR: {e}")
        return audit

    audit["n_rows"]       = len(df)
    audit["n_cols"]       = len(df.columns)
    audit["columns_raw"]  = list(df.columns)
    audit["columns_stripped"] = [c.strip() for c in df.columns]

    print(f"  Shape: {df.shape}")

    # â”€â”€ Header whitespace â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    has_ws = any(c != c.strip() for c in df.columns)
    audit["has_whitespace_in_headers"] = has_ws
    if has_ws:
        ws_cols = [c for c in df.columns if c != c.strip()]
        audit["whitespace_header_examples"] = ws_cols[:10]
        print(f"  âš   Whitespace in {len(ws_cols)} column names.")

    # â”€â”€ dtypes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    audit["dtype_map"] = {c: str(df[c].dtype) for c in df.columns}

    # â”€â”€ Label column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    label_col = find_label_column(list(df.columns))
    audit["label_col"] = label_col

    if label_col is None:
        audit["warnings"].append("Label column NOT found.")
        print("  âœ— Label column not found!")
        label_dist = {}
    else:
        label_dist = df[label_col].value_counts().to_dict()
        # convert numpy int64 â†’ int for JSON
        label_dist = {str(k): int(v) for k, v in label_dist.items()}
        print(f"  Label column: '{label_col}'")
        for lbl, cnt in label_dist.items():
            pct = 100.0 * cnt / len(df)
            marker = "  (BENIGN)" if "benign" in str(lbl).lower() else ""
            print(f"    {lbl:<40} {cnt:>8,}  ({pct:5.2f}%){marker}")

    audit["label_distribution"] = label_dist

    # â”€â”€ Separate feature matrix for numeric checks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    feature_cols = [c for c in df.columns if c != label_col]
    df_feat = df[feature_cols].copy()

    # â”€â”€ Missing values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    missing = df_feat.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_dict = {str(c): int(v) for c, v in missing.items()}
    audit["n_cols_with_missing"] = len(missing_dict)
    audit["total_missing_cells"] = int(missing.sum())
    # store top 20 to keep report readable
    audit["missing_per_col_top20"] = dict(list(missing_dict.items())[:20])
    print(f"  Missing cells: {audit['total_missing_cells']:,}  "
          f"across {audit['n_cols_with_missing']} columns")

    # â”€â”€ Infinity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    num_df = df_feat.select_dtypes(include=[np.number])
    inf_mask = np.isinf(num_df)
    n_inf_pos = int((num_df == np.inf).sum().sum())
    n_inf_neg = int((num_df == -np.inf).sum().sum())
    audit["inf_positive"] = n_inf_pos
    audit["inf_negative"] = n_inf_neg
    audit["cols_with_inf"] = [str(c) for c in num_df.columns[inf_mask.any()].tolist()]
    print(f"  +Inf cells: {n_inf_pos:,}   -Inf cells: {n_inf_neg:,}")

    # â”€â”€ Duplicates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_dup = int(df.duplicated().sum())
    audit["duplicate_rows"] = n_dup
    print(f"  Duplicate rows: {n_dup:,}")

    # â”€â”€ Constant (zero-variance) columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    const_cols = [c for c in num_df.columns if num_df[c].nunique(dropna=True) <= 1]
    audit["constant_cols"] = [str(c) for c in const_cols]
    print(f"  Constant columns: {len(const_cols)}")
    if const_cols:
        print(f"    {const_cols}")

    # â”€â”€ All-NaN columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_nan = [c for c in df_feat.columns if df_feat[c].isnull().all()]
    audit["all_nan_cols"] = [str(c) for c in all_nan]

    # â”€â”€ Shortcut / leakage-risk columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    shortcut = flag_shortcut_columns(list(df.columns))
    audit["shortcut_cols"] = shortcut
    if shortcut:
        print(f"  âš   Potential shortcut cols ({len(shortcut)}): {shortcut[:8]}")

    # â”€â”€ Basic numeric stats for quick sanity check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Only keep summary (min/max) to avoid huge JSON
    desc = num_df.agg(["min", "max"]).to_dict()
    # filter out cols with any inf for clean reporting
    clean_desc = {
        c: {
            "min": (round(v["min"], 4) if math.isfinite(v["min"]) else str(v["min"])),
            "max": (round(v["max"], 4) if math.isfinite(v["max"]) else str(v["max"]))
        }
        for c, v in desc.items()
    }
    audit["numeric_min_max"] = clean_desc

    # â”€â”€ Sample rows (5 rows, label stripped for display) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    sample = df.head(5).copy()
    # convert to safe types for JSON
    for col in sample.columns:
        sample[col] = sample[col].astype(str)
    audit["sample_rows"] = sample.to_dict(orient="records")

    return audit


def cross_file_checks(audits: list[dict]) -> dict:
    """
    Compare column sets across all files to detect schema inconsistencies.
    """
    print(f"\n{'='*70}")
    print("  Cross-file consistency check")
    print(f"{'='*70}")

    col_sets = {}
    for a in audits:
        if "columns_stripped" in a:
            col_sets[a["filename"]] = set(a["columns_stripped"])

    if not col_sets:
        return {"status": "no data loaded"}

    # union and intersection
    all_cols   = set.union(*col_sets.values())
    common_cols = set.intersection(*col_sets.values())
    only_in_some = all_cols - common_cols

    result = {
        "total_unique_columns_across_files": len(all_cols),
        "columns_common_to_all_files": len(common_cols),
        "columns_not_in_all_files": list(only_in_some)
    }

    print(f"  Total unique columns (union):      {len(all_cols)}")
    print(f"  Columns common to ALL files:       {len(common_cols)}")
    if only_in_some:
        print(f"  âš   Columns missing from â‰¥1 file:  {sorted(only_in_some)}")
    else:
        print("  âœ“  All files share identical column sets.")

    # per-file column count
    per_file_counts = {fn: len(s) for fn, s in col_sets.items()}
    result["per_file_col_counts"] = per_file_counts

    return result


def write_text_summary(audits: list[dict], cross: dict, output_path: Path):
    """Write a human-readable text summary."""
    lines = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("AutoNIDS â€“ Stage 1: Dataset Discovery & Data-Quality Audit")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    # â”€â”€ Overview table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("\n[1] FILE INVENTORY\n")
    header = f"{'File':<52} {'MB':>7} {'Rows':>10} {'Cols':>6} {'Dups':>8} {'Inf+':>7} {'Inf-':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for a in audits:
        fn   = a.get("filename", "?")[:51]
        mb   = a.get("size_mb", "?")
        rows = f"{a.get('n_rows', 0):,}"
        cols = a.get("n_cols", "?")
        dups = f"{a.get('duplicate_rows', 0):,}"
        infp = a.get("inf_positive", 0)
        infn = a.get("inf_negative", 0)
        lines.append(f"{fn:<52} {mb:>7} {rows:>10} {cols:>6} {dups:>8} {infp:>7} {infn:>7}")

    # â”€â”€ Label distributions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("\n\n[2] LABEL DISTRIBUTIONS\n")
    for a in audits:
        lines.append(f"  {a.get('filename','?')}")
        dist = a.get("label_distribution", {})
        total = sum(dist.values())
        for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            pct = 100.0 * cnt / total if total else 0
            tag = " â† BENIGN" if "benign" in str(lbl).lower() else ""
            lines.append(f"    {lbl:<42} {cnt:>8,}  ({pct:5.2f}%){tag}")
        lines.append("")

    # â”€â”€ Data quality â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("\n[3] DATA QUALITY SUMMARY\n")
    for a in audits:
        lines.append(f"  {a.get('filename','?')}")
        lines.append(f"    Missing cells    : {a.get('total_missing_cells',0):,}"
                     f"  in {a.get('n_cols_with_missing',0)} columns")
        lines.append(f"    +Inf / -Inf cells: {a.get('inf_positive',0):,} / {a.get('inf_negative',0):,}")
        lines.append(f"    Cols with Inf    : {a.get('cols_with_inf', [])}")
        lines.append(f"    Duplicate rows   : {a.get('duplicate_rows',0):,}")
        lines.append(f"    Constant cols    : {a.get('constant_cols', [])}")
        lines.append(f"    All-NaN cols     : {a.get('all_nan_cols', [])}")
        if a.get("warnings"):
            lines.append(f"    âš  Warnings       : {a['warnings']}")
        lines.append("")

    # â”€â”€ Shortcut columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("\n[4] POTENTIAL SHORTCUT / LEAKAGE COLUMNS\n")
    for a in audits:
        sc = a.get("shortcut_cols", [])
        if sc:
            lines.append(f"  {a.get('filename','?')}")
            lines.append(f"    {sc}")
            lines.append("")

    # â”€â”€ Cross-file schema â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("\n[5] CROSS-FILE SCHEMA CONSISTENCY\n")
    lines.append(f"  Total unique columns (union):   {cross.get('total_unique_columns_across_files','?')}")
    lines.append(f"  Columns in ALL files:           {cross.get('columns_common_to_all_files','?')}")
    cols_not_all = cross.get("columns_not_in_all_files", [])
    if cols_not_all:
        lines.append(f"  Columns missing from â‰¥1 file:  {cols_not_all}")
    else:
        lines.append("  All files share the same column schema. âœ“")

    per_file = cross.get("per_file_col_counts", {})
    if per_file:
        lines.append("\n  Per-file column counts:")
        for fn, cnt in per_file.items():
            lines.append(f"    {fn:<55} {cnt}")

    lines.append("\n" + sep)
    lines.append("END OF REPORT")
    lines.append(sep)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  âœ“  Text summary written â†’ {output_path}")


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        sys.exit(f"âœ— No CSV files found in {DATA_DIR}. "
                 "Check DATA_DIR path and re-run.")

    print(f"\nAutoNIDS â€“ Stage 1 Audit")
    print(f"Data directory : {DATA_DIR}")
    print(f"Files found    : {len(csv_files)}")
    print(f"Report output  : {REPORT_DIR}\n")

    audits = []
    for csv_path in csv_files:
        audit = inspect_file(csv_path)
        audits.append(audit)

    cross = cross_file_checks(audits)

    # â”€â”€ Serialise report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    report = {
        "generated_at": datetime.now().isoformat(),
        "data_directory": str(DATA_DIR),
        "files_inspected": len(audits),
        "per_file": audits,
        "cross_file": cross
    }

    JSON_REPORT.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8"
    )
    print(f"\n  âœ“  JSON report written  â†’ {JSON_REPORT}")

    write_text_summary(audits, cross, TEXT_SUMMARY)

    # â”€â”€ Quick console recap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n{'='*70}")
    print("  AUDIT COMPLETE â€” Key findings at a glance")
    print(f"{'='*70}")
    total_rows = sum(a.get("n_rows", 0) for a in audits)
    total_dups = sum(a.get("duplicate_rows", 0) for a in audits)
    total_inf  = sum(a.get("inf_positive", 0) + a.get("inf_negative", 0) for a in audits)
    print(f"  Total rows across all files : {total_rows:,}")
    print(f"  Total duplicate rows        : {total_dups:,}")
    print(f"  Total Â±Inf cells            : {total_inf:,}")
    print(f"  Schema consistent?          : "
          f"{'YES âœ“' if not cross.get('columns_not_in_all_files') else 'NO âš  â€” see report'}")
    print(f"\n  Open reports/stage1_summary.txt for the full readable report.")


if __name__ == "__main__":
    main()

