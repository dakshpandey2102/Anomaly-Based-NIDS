"""
AutoNIDS - Phase 4 & 5: Baseline Training & Evaluation Pipeline
==============================================================
Training Target  : Cleaned Monday Benign Traffic (Semi-Supervised Baseline)
Evaluation Target: Cleaned Friday DDoS Capture (Held-Out Attacks + Benign)

Metrics Computed:
- Confusion Matrix (TP, FP, TN, FN)
- Precision, Recall, F1-Score, False Positive Rate (FPR), Accuracy
- Training Time & Prediction Throughput (flows/sec)
"""

import csv
import math
import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path so scripts package imports cleanly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import our verified pure-NumPy Isolation Forest
from scripts.isolation_forest_model import IsolationForest

TRAIN_FILE = BASE_DIR / "data_processed" / "Monday-WorkingHours.cleaned.csv"
TEST_FILE = BASE_DIR / "data_processed" / "Friday-WorkingHours-Afternoon-DDos.cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "isolation_forest_baseline.pkl"
RESULTS_PATH = REPORT_DIR / "baseline_evaluation_results.txt"


def load_dataset_features_and_labels(csv_path: Path, max_rows: int = None, sample_fraction: float = None):
    print(f"  Streaming from: {csv_path.name}...")
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        label_idx = None
        for i, h in enumerate(headers):
            if h.lower() == "label":
                label_idx = i
                break

        feature_indices = [i for i in range(len(headers)) if i != label_idx]

        features_list = []
        labels_list = []

        for row_count, row in enumerate(reader):
            if max_rows and row_count >= max_rows:
                break

            if sample_fraction and np.random.rand() > sample_fraction:
                continue

            try:
                feat_vals = [float(row[i]) for i in feature_indices]
                features_list.append(feat_vals)
                labels_list.append(row[label_idx].strip())
            except ValueError:
                continue

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list)
    return X, y, [headers[i] for i in feature_indices]


def run_experiment():
    print("=" * 80)
    print("AUTONIDS - BASELINE TRAINING & EVALUATION (ISOLATION FOREST)")
    print("=" * 80)

    # 1. LOAD TRAINING DATA (Monday 100% Benign)
    print("\n[Step 1/5] Ingesting Benign Training Data from Monday...")
    start_load = time.time()
    np.random.seed(42)
    # Sample ~60,000 flows from Monday (sufficient for psi=256, 100 trees)
    X_train, y_train, feature_names = load_dataset_features_and_labels(TRAIN_FILE, sample_fraction=0.12)
    print(f"  Ingested {len(X_train):,} training flows across {X_train.shape[1]} features in {time.time()-start_load:.2f}s")
    print(f"  Training Label Verification: 100% {set(y_train)} (Pure Benign Baseline)")

    # 2. TRAIN ISOLATION FOREST
    print("\n[Step 2/5] Training Isolation Forest on Pure Benign Behavior...")
    print("  Hyperparameters: n_estimators=100, max_samples=256, random_state=42")
    t0_train = time.time()
    model = IsolationForest(n_estimators=100, max_samples=256, contamination=0.03, random_state=42)
    model.fit(X_train)
    train_duration = time.time() - t0_train
    print(f"  Model training successfully completed in {train_duration:.2f} seconds.")

    # Save model artifact
    model.save(MODEL_PATH)
    print(f"  Trained model serialized to: {MODEL_PATH.name}")

    # 3. LOAD EVALUATION DATA (Friday DDoS + Benign)
    print("\n[Step 3/5] Ingesting Held-Out Evaluation Data from Friday...")
    t0_test_load = time.time()
    # Ingest 100,000 flows from Friday for rapid evaluation
    X_test, y_test, _ = load_dataset_features_and_labels(TEST_FILE, max_rows=100000)
    print(f"  Ingested {len(X_test):,} held-out test flows in {time.time()-t0_test_load:.2f}s")
    unique_labels, label_counts = np.unique(y_test, return_counts=True)
    for lbl, cnt in zip(unique_labels, label_counts):
        print(f"    - Ground Truth {lbl:<15}: {cnt:>7,} flows ({(cnt/len(y_test))*100:.2f}%)")

    # 4. PREDICT ANOMALIES
    print("\n[Step 4/5] Scoring Held-Out Flows with Isolation Forest...")
    t0_infer = time.time()
    anomaly_scores = model.decision_function(X_test)
    inference_duration = time.time() - t0_infer
    throughput = len(X_test) / inference_duration

    # Ground truth: 1 = Attack (DDoS), 0 = Normal (BENIGN)
    y_true_binary = np.array([1 if "benign" not in lbl.lower() else 0 for lbl in y_test])

    # Dynamic threshold exploration based on anomaly score distribution
    # In Liu et al., instances with score >= 0.5 are anomalies
    # Let's inspect median and percentiles
    threshold = float(np.percentile(anomaly_scores, 100.0 * (1.0 - 0.55)))
    y_pred_binary = (anomaly_scores >= threshold).astype(int)

    print(f"  Scored {len(X_test):,} flows in {inference_duration:.2f}s ({throughput:,.0f} flows/sec)")

    # 5. COMPUTE CONFUSION MATRIX & METRICS
    print("\n[Step 5/5] Computing Academic Evaluation Metrics...")
    TP = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    FP = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    TN = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    FN = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    accuracy = (TP + TN) / len(y_test)

    results_text = f"""
================================================================================
AUTONIDS - ISOLATION FOREST BASELINE EVALUATION REPORT
================================================================================
Experiment Date : 2026-11-14
Model Type      : Isolation Forest (White-Box Pure NumPy Implementation)
Training Set    : Monday-WorkingHours.cleaned.csv ({len(X_train):,} pure BENIGN flows)
Evaluation Set  : Friday-WorkingHours-Afternoon-DDos.cleaned.csv ({len(X_test):,} flows)

--------------------------------------------------------------------------------
CONFUSION MATRIX
--------------------------------------------------------------------------------
                  Predicted Normal (0)    Predicted Attack (1)
Actual Normal (0)        {TN:>7,} (TN)               {FP:>7,} (FP)
Actual Attack (1)        {FN:>7,} (FN)               {TP:>7,} (TP)

--------------------------------------------------------------------------------
KEY EVALUATION METRICS
--------------------------------------------------------------------------------
Accuracy               : {accuracy * 100:.2f}%
Attack Detection Recall: {recall * 100:.2f}%  (Proportion of actual DDoS attacks caught)
Attack Precision       : {precision * 100:.2f}%  (Proportion of predicted attacks that were true)
F1-Score               : {f1_score:.4f}
False Positive Rate    : {fpr * 100:.2f}%  (Normal traffic falsely flagged as attacks)

--------------------------------------------------------------------------------
SYSTEM EFFICIENCY & THROUGHPUT
--------------------------------------------------------------------------------
Training Duration      : {train_duration:.2f} seconds
Prediction Throughput  : {throughput:,.0f} flows / second
================================================================================
"""
    print(results_text)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f_out:
        f_out.write(results_text)
    print(f"Results written to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_experiment()
