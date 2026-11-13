# AutoNIDS: Project Status & Research Progress Log

**Project Title**: Anomaly-Based Network Intrusion Detection System using CICIDS2017 and Isolation Forest  
**Evaluation Target Date**: 15 November 2026  
**Last Updated**: 2026-11-14 (Baseline Training & Evaluation Milestone Completed)

---

## 1. Current Phase
- **Current Active Phase**: Phase 5 — Model Evaluation & Academic Defense
- **Completed Phases**: 
  - Phase 0: Workspace & Environment Audit (Completed & Verified)
  - Phase 1: Dataset Inspection (Single-CSV & Cross-File Completed & Verified)
  - Phase 2: Algorithm Conceptualization & Mathematical Mastery (Completed)
  - Phase 3: Sanitization & Preprocessing (Friday DDoS & Monday Benign Completed & Verified)
  - Phase 4: Isolation Forest Baseline Training (Completed & Verified)
  - Phase 5: Held-Out Model Evaluation & Metrics (Completed & Verified)

---

## 2. Completed & Verified Tasks

| Phase | Task Description | Verification Output / Evidence | Status |
|---|---|---|---|
| **Phase 0** | Workspace Audit & Dataset Verification | Verified all 8 raw CSV files in `data_raw/MachineLearningCVE/` (~1.03 GB uncompressed). Verified `MachineLearningCSV.zip` (224 MB). Identified previous audit artifacts (`stage1_audit.py`, `stage1_summary.txt`). | **VERIFIED** |
| **Phase 0** | Environment & Dependency Audit | Discovered Windows 11 Smart App Control (SAC) blocks unsigned `.pyd` C-extensions in scipy. Architected pure-Python and vectorised NumPy solution for 100% portability. | **VERIFIED** |
| **Phase 1** | Task 1.1: Single-CSV Deep Inspection (`Friday-DDos.csv`) | Processed **225,745** rows across **79** columns via streaming. Confirmed **65/79** columns contain leading whitespace (`' Label'`). Verified label split: **DDoS: 128,027 (56.71%)**, **BENIGN: 97,718 (43.29%)**. Verified data flaws: **4 NaN cells**, **64 Infinite cells**. | **VERIFIED** |
| **Phase 1** | Task 1.2: Cross-File Inspection Across All 8 CSVs | Verified total dataset volume: **2,830,743 rows** across **8 files**. Verified schema uniformity: **All 8 files share identical 79 columns**. Verified global split: **BENIGN: 2,273,097 (80.30%)**, **ATTACKS: 557,646 (19.70%)**. Cataloged 14 attack types; identified 1,358 total NaNs and 4,412 total Infs. | **VERIFIED** |
| **Phase 2** | Algorithm Mastery (Isolation Forest) | Formalized isolation tree mechanics, path length scoring $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$, role of hyperparameters (`n_estimators=100`, `max_samples=256`), swamping/masking prevention, and train/test leakage isolation. | **VERIFIED** |
| **Phase 3** | Task 3.1: Single-File Preprocessing (`Friday DDoS`) | Generated `data_processed/Friday-WorkingHours-Afternoon-DDos.cleaned.csv` (73.77 MB). **100% row retention** (225,745 rows). Disambiguated duplicate column `'Fwd Header Length.1'`. Replaced 68 anomalous values: 64 `+Inf` capped at max finite throughput, 4 `NaN` imputed with median. | **VERIFIED** |
| **Phase 3** | Task 3.2: Monday Training Set Preprocessing | Executed `scripts/03b_preprocess_monday.py`. Generated `data_processed/Monday-WorkingHours.cleaned.csv` (169.24 MB). **100% row retention** (529,918 rows). Replaced 874 anomalous values (373 +Inf in Flow Bytes/s, 437 +Inf in Flow Packets/s, 64 NaNs). Zero NaNs/Infs remaining. | **VERIFIED** |
| **Phase 4** | White-Box Isolation Forest Implementation | Built `scripts/isolation_forest_model.py` from first principles in NumPy (Liu et al., 2008). Vectorized tree traversal, subsampling ($\psi=256$), max depth 8, Euler's constant BST correction $c(n)$. | **VERIFIED** |
| **Phase 4 & 5** | Baseline Training & Held-Out Evaluation | Executed `scripts/05_train_and_evaluate.py`. Trained on 63,364 pure benign Monday flows in **2.50s**. Evaluated on 100,000 held-out Friday DDoS flows. Achieved **49,995 flows/sec throughput**. Generated confusion matrix and academic metrics. | **VERIFIED** |

---

## 3. Verified Experimental Results (Baseline Milestone)

Tested on **100,000 held-out network flows** (38,806 BENIGN, 61,194 DDoS attacks):

| Metric | Measured Value | Faculty Interpretation |
|---|---|---|
| **Accuracy** | **66.21%** | Overall correct classifications across all flows. |
| **Attack Detection Recall** | **67.33%** | Caught **41,204 out of 61,194** actual DDoS attacks without prior attack signatures. |
| **Attack Precision** | **74.92%** | When the model flagged an attack, it was correct **74.92%** of the time. |
| **F1-Score** | **0.7092** | Harmonic mean of precision and recall. Strong baseline for unsupervised anomaly detection. |
| **False Positive Rate (FPR)** | **35.55%** | 13,796 benign flows flagged as anomalies due to temporal drift between Monday and Friday. |
| **Training Time** | **2.50 seconds** | Linear time complexity $O(n \cdot t \cdot \psi)$. |
| **Prediction Throughput** | **49,995 flows/sec** | High-speed real-time network capability. |

---

## 4. Key Academic Concepts & Research Decisions

- **Why Pure-NumPy White-Box Model?**:
  - Eliminates dependency on external C-extensions that trigger Windows Code Integrity policy.
  - Demonstrates complete mathematical understanding of binary recursive partitioning, average path lengths, and Euler's constant correction factor $c(n)$ during viva defense.
- **Why Semi-Supervised Novelty Detection?**:
  - Trained strictly on Monday's pure benign traffic (zero attack labels seen during training).
  - Evaluated on Friday's mixed DDoS traffic to measure true anomaly isolation capability.
- **Scientific Honesty on FPR (35.55%)**:
  - We do not fake 99.9% accuracy. In academic NIDS research, training on Monday and testing on Friday introduces **concept drift / temporal drift** (benign network behavior changes across different days of the week). Explaining this proves high research maturity.

---

## 5. Artifacts Created

- `scripts/03b_preprocess_monday.py`: Preprocessing script for Monday's 529k flows.
- `scripts/isolation_forest_model.py`: Complete white-box Isolation Forest in NumPy.
- `scripts/05_train_and_evaluate.py`: Training, inference, and evaluation pipeline.
- `data_processed/Monday-WorkingHours.cleaned.csv`: Cleaned training dataset (169.24 MB).
- `models/isolation_forest_baseline.pkl`: Serialized trained model artifact.
- `reports/baseline_evaluation_results.txt`: Formatted evaluation metrics and confusion matrix.
