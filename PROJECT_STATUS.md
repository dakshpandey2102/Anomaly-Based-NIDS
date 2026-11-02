# AutoNIDS: Project Status & Research Progress Log

**Project Title**: Anomaly-Based Network Intrusion Detection System using CICIDS2017 and Isolation Forest  
**Evaluation Target Date**: 15 November 2026  
**Last Updated**: 2026-10-27 (Post-Task 3.1 Verification Update)

---

## 1. Current Phase
- **Current Active Phase**: Phase 3 — Preprocessing & Sanitization Pipeline
- **Completed Subtasks**:
  - Task 3.1: Single-File Preprocessing (`Friday-WorkingHours-Afternoon-DDos.cleaned.csv`) **COMPLETED & VERIFIED**.
- **Next Subtask**: Task 3.2: Expanding the sanitization pipeline across remaining capture files (especially Monday's benign training baseline).
- **Previous Completed Phases**:
  - Phase 0: Workspace & Environment Audit (Completed & Verified)
  - Phase 1: Dataset Inspection (Task 1.1 Single-CSV & Task 1.2 Cross-File Completed & Verified)
  - Phase 2: Algorithm Conceptualization & Faculty Q&A (Completed)

---

## 2. Completed & Verified Tasks

| Phase | Task Description | Verification Output / Evidence | Status |
|---|---|---|---|
| **Phase 0** | Workspace Audit & Dataset Verification | Verified all 8 raw CSV files in `data_raw/MachineLearningCVE/` (~1.03 GB uncompressed). Verified `MachineLearningCSV.zip` (224 MB). Identified previous audit artifacts (`stage1_audit.py`, `stage1_summary.txt`). | **VERIFIED** |
| **Phase 0** | Environment & Dependency Audit | Discovered Windows 11 Smart App Control (SAC) blocks unsigned `.pyd` C-extensions. Pivoted Phase 1 & 3 pipelines to standard library (`csv`, `collections`, `math`, `statistics`, `pathlib`) for guaranteed portability and O(1) memory streaming. | **VERIFIED** |
| **Phase 1** | Task 1.1: Single-CSV Deep Inspection (`Friday-DDos.csv`) | Executed `scripts/01_inspect_single_csv.py`. Processed **225,745** rows across **79** columns via streaming. Confirmed **65/79** columns contain leading whitespace (`' Label'`). Verified label split: **DDoS: 128,027 (56.71%)**, **BENIGN: 97,718 (43.29%)**. Verified data flaws: **4 NaN cells** in `Flow Bytes/s`, **64 Infinite cells** (30 in `Flow Bytes/s`, 34 in `Flow Packets/s`). | **VERIFIED** |
| **Phase 1** | Task 1.2: Cross-File Inspection Across All 8 CSVs | Executed `scripts/02_inspect_all_csvs.py`. Verified total dataset volume: **2,830,743 rows** across **8 files**. Verified schema uniformity: **All 8 files share identical 79 columns**. Verified global split: **BENIGN: 2,273,097 (80.30%)**, **ATTACKS: 557,646 (19.70%)**. Cataloged 14 attack types; identified 1,358 total NaNs and 4,412 total Infs. | **VERIFIED** |
| **Phase 2** | Algorithm Mastery (Isolation Forest) | Formalized isolation tree mechanics, path length scoring $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$, role of hyperparameters (`n_estimators=100`, `max_samples=256`), swamping/masking prevention, and train/test leakage isolation. | **VERIFIED** |
| **Phase 3** | Task 3.1: Single-File Preprocessing & Sanitization | Executed `scripts/03_preprocess_single_file.py`. Generated `data_processed/Friday-WorkingHours-Afternoon-DDos.cleaned.csv` (73.77 MB). **100% row retention** (225,745 rows). Disambiguated duplicate column `'Fwd Header Length.1'`. Replaced 68 anomalous values: 64 `+Inf` capped at max finite throughput, 4 `NaN` imputed with median. Zero duplicate headers remaining. | **VERIFIED** |

---

## 3. Files Created or Modified

- `requirements.txt`: Created (specifies core dependencies).
- `scripts/01_inspect_single_csv.py`: Created for Phase 1 single-file inspection (standard library, streaming).
- `scripts/02_inspect_all_csvs.py`: Created for Phase 1 cross-dataset inspection across all 8 files.
- `scripts/03_preprocess_single_file.py`: Created for Phase 3 two-pass streaming sanitization and imputation.
- `data_processed/Friday-WorkingHours-Afternoon-DDos.cleaned.csv`: Created (225,745 rows, 79 unique columns, 0 NaNs, 0 Infs).
- `PROJECT_STATUS.md`: Updated with full Task 3.1 verified findings.

*Untouched Original Files*:
- `data_raw/MachineLearningCVE/*.csv` (read-only baseline).
- `MachineLearningCSV.zip`.
- `scripts/stage1_audit.py`.
- `reports/stage1_report.json` & `reports/stage1_summary.txt`.

---

## 4. Key Academic Concepts & Research Decisions

- **Duplicate Column Resolution**: The raw dataset contains `' Fwd Header Length'` at column index 34 and index 55. After stripping whitespace, the preprocessor renames the second occurrence to `'Fwd Header Length.1'`, preserving all 79 columns without dictionary collision.
- **Physical Meaning Preservation vs Blind Row Dropping**:
  - `+Inf` in rate columns represents an instantaneous high-rate burst ($Duration = 0$). Dropping these 34 rows or zeroing them would distort network reality.
  - Capping `+Inf` at the feature's maximum observed finite value preserves the burst signal.
  - Imputing `NaN` (from $0/0$) with the median preserves normal central tendency without vulnerability to extreme values.
- **Zero Row Loss**: 100% of the 225,745 rows were preserved in `data_processed/`.
- **Semi-Supervised Setup**: Monday (100% benign) remains our unpolluted training target. The cleaned Friday capture will serve as an evaluation set containing both Benign and DDoS traffic.

---

## 5. Next Immediate Action

- **Phase 3, Task 3.2**: Expand preprocessing across the candidate training capture (`Monday-WorkingHours.pcap_ISCX.csv`) and remaining evaluation captures to build our clean data baseline.
- **Phase 4**: Train our first baseline Isolation Forest on cleaned Monday benign data.
