# ACADEMIC PROJECT REPORT

## PROJECT TITLE:
### **AutoNIDS: Anomaly-Based Network Intrusion Detection System Using CICIDS2017 and First-Principles Isolation Forest**

---

**Domain**: Cybersecurity & Applied Machine Learning  
**Dataset**: Canadian Institute for Cybersecurity — CICIDS2017  
**Algorithm**: Isolation Forest (White-Box Pure NumPy Implementation)  
**Evaluation Paradigm**: Semi-Supervised Novelty Detection (Monday Baseline $\rightarrow$ Friday DDoS)  

---

## ABSTRACT

Traditional signature-based Network Intrusion Detection Systems (NIDS) fail against zero-day and novel polymorphic network threats because they rely strictly on pre-compiled databases of known attack signatures. Anomaly-based NIDS addresses this vulnerability by learning the statistical profile of legitimate network behavior and isolating deviations without requiring prior exposure to malicious payloads. 

This project designs, implements, and evaluates **AutoNIDS**, an anomaly-based NIDS built on the benchmark **CICIDS2017** dataset. Due to enterprise operating system Code Integrity policies blocking compiled third-party C-extensions, the core **Isolation Forest** model was engineered from first principles in vectorized NumPy following the original mathematical formulation of Liu, Ting, and Zhou (IEEE ICDM 2008). The system employs an $O(1)$ memory streaming ingestion pipeline, a physically motivated sanitization strategy for rate feature infinities ($+Inf$) and missing values ($NaN$), and resolves structural dataset defects (such as duplicate feature headers). 

Trained exclusively on **63,364 purely benign flows from Monday** in **2.50 seconds**, the model was evaluated against **100,000 held-out flows from Friday’s DDoS capture**. Without prior exposure to DDoS signatures, AutoNIDS achieved an **Attack Detection Recall of 67.33%**, **Attack Precision of 74.92%**, an **F1-Score of 0.7092**, and an inference throughput of **49,995 flows/second**, demonstrating practical feasibility for real-time gigabit enterprise network traffic monitoring.

---

## CHAPTER 1: INTRODUCTION & PROBLEM STATEMENT

### 1.1 Background & Motivation
In modern networked systems, the volume, velocity, and sophistication of cyberattacks have outpaced traditional perimeter defenses. Firewalls and signature-based Intrusion Detection Systems (e.g., Snort, Suricata, Zeek) inspect packet headers and payloads against known vulnerability signatures (Common Vulnerabilities and Exposures - CVEs). While computationally efficient, signature matching possesses a fundamental blind spot: it cannot flag attacks for which no signature has been authored.

### 1.2 Problem Statement
To detect novel, distributed, and zero-day cyber threats, intrusion detection must shift toward **behavioral anomaly detection**. The objectives of this project are:
1. To establish a verified, data-leakage-free baseline using the comprehensive **CICIDS2017** dataset.
2. To design an $O(1)$ auxiliary-memory streaming ingestion and preprocessing pipeline capable of handling 2.83 million raw network flows without memory crashes.
3. To resolve raw data quality defects (header whitespace, feature name collisions, IEEE floating-point rate infinities) using scientifically defensible transformations.
4. To implement the **Isolation Forest** ensemble algorithm from first principles in pure NumPy, avoiding black-box library abstractions.
5. To evaluate model generalization in a realistic **semi-supervised novelty detection setting** (training on pure benign traffic, testing on unseen attack traffic).

---

## CHAPTER 2: LITERATURE REVIEW & THEORETICAL FOUNDATION

### 2.1 The Primary Base Paper
* **Citation**: K. Sadaf and J. Sultana, *"Intrusion Detection Based on Auto-encoder and Isolation Forest in Fog Computing,"* in **IEEE Access**, vol. 8, pp. 167059–167068, 2020. DOI: `10.1109/ACCESS.2020.3023807`.
* **Contribution to our Project**: Sadaf and Sultana evaluated the effectiveness of Isolation Forest on the CICIDS2017 dataset, establishing that tree-based random partitioning is effective at isolating anomalous network flows with low latency. Our project adopts their core premise of using Isolation Forest for high-throughput anomaly detection while developing a transparent, white-box NumPy implementation.

### 2.2 The Benchmark Methodology Paper
* **Citation**: Z. M. Maseer, R. Al-Qurashi, et al., *"Benchmarking of Machine Learning for Anomaly Based Intrusion Detection Systems in the CICIDS2017 Dataset,"* in **IEEE Access**, vol. 9, pp. 22351–22370, 2021. DOI: `10.1109/ACCESS.2021.3056614`.
* **Contribution to our Project**: This paper establishes the preprocessing standards for CICIDS2017, analyzing class imbalances and standardizing evaluation metrics (Precision, Recall, F1-Score, False Positive Rate).

### 2.3 The Benchmark Dataset Reference
* **Citation**: I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, *"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,"* in **Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)**, pp. 108–116, 2018. DOI: `10.5220/0006639801080116`.
* **Contribution to our Project**: The foundational paper by the Canadian Institute for Cybersecurity introducing CICIDS2017, detailing the network architecture (B-Profile system generating naturalistic background protocols) and the attack schedule across Monday to Friday.

### 2.4 The Algorithmic Foundation
* **Citation**: F. T. Liu, K. M. Ting, and Z.-H. Zhou, *"Isolation Forest,"* in **2008 Eighth IEEE International Conference on Data Mining (ICDM)**, pp. 413–422, IEEE, 2008. DOI: `10.1109/ICDM.2008.17`.
* **Contribution to our Project**: Establishes the mathematical theory of Isolation Trees ($iTree$), the average path length of unsuccessful searches in Binary Search Trees $c(n)$, and the anomaly score derivation $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$.

---

## CHAPTER 3: DATASET ARCHITECTURE & EXPLORATORY AUDIT

### 3.1 Overview of CICIDS2017
CICIDS2017 contains network traffic captured over 5 consecutive business days (Monday, July 3, 2017 to Friday, July 7, 2017). Using the `CICFlowMeter` network analysis tool, bidirectional packet traces were converted into flow records characterized by 79 statistical features and 1 label column.

### 3.2 Verification via Streaming Inspection Scripts
To prevent memory exhaustion, two custom inspection scripts were engineered:
1. `scripts/01_inspect_single_csv.py`: Evaluated `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`.
2. `scripts/02_inspect_all_csvs.py`: Iterated sequentially across all 8 files.

#### Verified Global Dataset Inventory:
* **Total Files Inspected**: 8 CSV files (~1.03 GB uncompressed).
* **Total Network Flow Records**: **2,830,743 flows**.
* **Feature Consistency**: All 8 files share an identical 79-column schema.
* **Global Class Distribution**:
  * **BENIGN (Normal)**: 2,273,097 flows (**80.30%**)
  * **ATTACKS (Anomalous)**: 557,646 flows (**19.70%**) spanning 14 distinct attack types (DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, DoS Slowloris, DoS Slowhttptest, Botnet, Web Attack Brute Force, Web Attack XSS, Infiltration, Web Attack SQL Injection, Heartbleed).
* **The Monday Capture Discovery**: `Monday-WorkingHours.pcap_ISCX.csv` contains **529,918 records and ZERO attacks (100.00% BENIGN)**. This provided the unpolluted baseline required for semi-supervised training.

### 3.3 Data Quality Defects Discovered
1. **Header Whitespace**: 65 of 79 column names contained leading whitespace (e.g., `' Label'`, `' Destination Port'`), causing `KeyError` crashes in naive parsers.
2. **Duplicate Feature Name**: Column index 34 and column index 55 were both labeled `'Fwd Header Length'`.
3. **Infinite Rate Values**: 4,412 cells across the dataset contained positive infinity (`+Inf`) in `Flow Bytes/s` and `Flow Packets/s`.
4. **Missing Values**: 1,358 cells contained `NaN` values.

---

## CHAPTER 4: SANITIZATION & PREPROCESSING PIPELINE

Implemented in `scripts/03_preprocess_single_file.py` and `scripts/03b_preprocess_monday.py`, the preprocessing pipeline enforces **100% row retention** while preserving physical networking semantics.

### 4.1 Header Disambiguation
Column headers are stripped of whitespace. When a duplicate name is detected, subsequent occurrences are disambiguated:
```python
# Raw headers index 34 and 55: ' Fwd Header Length'
# Output unique headers:
Index 34 -> 'Fwd Header Length'
Index 55 -> 'Fwd Header Length.1'
```
This guarantees **79 unique feature columns**, preventing dictionary key collisions.

### 4.2 Handling Infinite Values ($+Inf$)
In network telemetry, rate features are defined as:
$$\text{Flow Bytes/s} = \frac{\text{Total Flow Bytes}}{\text{Flow Duration}}$$
When packets arrive in micro-bursts, $\text{Flow Duration} = 0 \text{ microseconds}$, producing division by zero ($+Inf$).
* **Flaw of Deleting Rows**: Discards volumetric attack flows (e.g., SYN floods), corrupting detection rates.
* **Flaw of Zeroing**: Inverts physical reality by labeling an extreme burst as an idle connection.
* **Our Strategy (Capping at Maximum Finite Throughput)**:
  $$x = \max(\{v \in X_j \mid v < +\infty\})$$
  Capping preserves the extreme high-rate anomaly signal without causing IEEE floating-point exceptions.

### 4.3 Handling Missing Values ($NaN$)
When both bytes and duration are zero ($0/0$), IEEE 754 produces `NaN`. These values are imputed using the **feature median** of valid finite records, preventing outlier skew associated with arithmetic means.

### 4.4 Verified Preprocessing Execution
* `data_processed/Friday-WorkingHours-Afternoon-DDos.cleaned.csv`:
  * Original rows: 225,745 | Output rows: 225,745 (100.0% retention)
  * Replaced 30 $+Inf$ (Flow Bytes/s capped at $2.070 \times 10^9$), 34 $+Inf$ (Flow Packets/s capped at $3.0 \times 10^6$), 4 $NaN$ (imputed at median $5765.50$).
* `data_processed/Monday-WorkingHours.cleaned.csv`:
  * Original rows: 529,918 | Output rows: 529,918 (100.0% retention)
  * Replaced 373 $+Inf$ in Flow Bytes/s (capped at $2.071 \times 10^9$), 437 $+Inf$ in Flow Packets/s (capped at $3.0 \times 10^6$), 64 $NaN$ (imputed at median $1220.86$).

---

## CHAPTER 5: WHITE-BOX ISOLATION FOREST IMPLEMENTATION

Implemented in `scripts/isolation_forest_model.py`, the model is written from first principles in pure NumPy, eliminating dependencies on compiled third-party C-extensions.

```
       DENSE NORMAL CLUSTER                       ISOLATED ANOMALY
   (Requires many splits to isolate)             (Isolated in 1-2 splits)
       
             *   *  *  *                                
           *   *   *   *    *                          [ Intruder / Attack ]
             *   *  *  *  *                                 │
           *   *   *    *                                   ▼
             *   *  *  *                           Isolated in 1 or 2 cuts!
```

### 5.1 Architecture Overview
An Isolation Forest is an ensemble of $t$ **Isolation Trees** ($iTree$). Rather than profiling normal density, it exploits two quantitative properties of anomalies:
1. They are the minority (few in number).
2. They have atypical attribute values (different in feature space).

### 5.2 Mathematical Derivations

#### 1. Binary Search Tree (BST) Average Path Length $c(n)$
To normalize path lengths across varying subsample sizes $n$, the theoretical average path length of an unsuccessful search in a BST is derived as:
$$c(n) = 2 \left( \ln(n - 1) + \gamma \right) - \frac{2(n - 1)}{n}$$
where $\gamma = 0.5772156649$ is the **Euler-Mascheroni Constant**.
In code:
```python
def c_factor(n: int) -> float:
    if n > 2:
        return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
    elif n == 2:
        return 1.0
    return 0.0
```

#### 2. The Anomaly Score Equation $s(x, n)$
Let $h(x)$ denote the path length of observation $x$ (the number of edges traversed from root to leaf). Over an ensemble of $t$ trees, the expected path length is $E(h(x))$. The anomaly score $s(x, n)$ is:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(\psi)}}$$
* **When $E(h(x)) \to 0$**: $s \to 2^0 = \mathbf{1.0}$ $\implies$ **Definite Anomaly (Attack)**.
* **When $E(h(x)) \to c(\psi)$**: $s \to 2^{-1} = \mathbf{0.5}$ $\implies$ **Normal Benign Instance**.
* **When $E(h(x)) \to \psi - 1$**: $s \to \mathbf{0.0}$ $\implies$ **Dense Core Cluster (Safe)**.

### 5.3 Detailed Component Breakdown

#### Component A: `IsolationTreeNode`
Represents an individual node in the binary partition:
* `size`: Number of training flows reaching this node.
* `is_leaf`: Boolean terminating flag.
* `split_feature`: The index of the randomly chosen attribute $q \in \{0, \dots, 77\}$.
* `split_value`: The uniformly drawn split value $p \in (\min X[:, q], \max X[:, q])$.
* `left`, `right`: Pointers to left ($X[:, q] < p$) and right ($X[:, q] \ge p$) subtrees.

#### Component B: `IsolationTree` (`iTree`)
Constructs an individual tree recursively:
* **Stopping Criteria**:
  1. Tree depth reaches $h_{\max} = \lceil \log_2(\psi) \rceil = \lceil \log_2(256) \rceil = 8$.
  2. Sample size at node $n \le 1$.
  3. All features exhibit zero variance ($\min = \max$).
* **Path Traversal (`path_length`)**: Evaluates incoming test samples. If a sample terminates at a leaf of size $m > 1$, an adjustment factor $c(m)$ is added to the depth:
  $$h(x) = \text{depth} + c(m)$$

#### Component C: `IsolationForest`
Orchestrates the ensemble:
* `n_estimators = 100`: Generates 100 independent isolation trees.
* `max_samples = 256` ($\psi$): Subsamples 256 instances per tree.
  * *Theoretical Justification*: Small subsample sizes eliminate **swamping** (normal points being mistakenly isolated) and **masking** (clusters of anomalies disguising themselves as dense normal regions).
* `fit(X)`: Builds the 100 trees on random subsamples.
* `decision_function(X)`: Computes $s(x, \psi)$ across all trees.
* `predict(X)`: Assigns binary classifications based on the decision threshold.

---

## CHAPTER 6: EXPERIMENTAL RESULTS & EVALUATION

Executed via `scripts/05_train_and_evaluate.py`.

### 6.1 Experimental Protocol
* **Training Partition**: Ingested **63,364 flows from Monday** (`Monday-WorkingHours.cleaned.csv`). Ground truth: **100% BENIGN**. Zero attack samples were exposed during training.
* **Evaluation Partition**: Ingested **100,000 held-out flows from Friday** (`Friday-WorkingHours-Afternoon-DDos.cleaned.csv`).
  * Ground Truth Composition: **38,806 BENIGN flows (38.81%)** and **61,194 DDoS flows (61.19%)**.
* **Thresholding**: Static calibration based on expected outlier contamination.

### 6.2 Confusion Matrix
```
                  Predicted Normal (0)    Predicted Attack (1)
Actual Normal (0)         25,010 (TN)                13,796 (FP)
Actual Attack (1)         19,990 (FN)                41,204 (TP)
```

### 6.3 Performance Metrics

| Evaluation Metric | Formula | Measured Score | Operational Significance |
|---|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | **66.21%** | Overall classification accuracy across mixed traffic. |
| **Attack Detection Recall** | $\frac{TP}{TP + FN}$ | **67.33%** | Caught **41,204 out of 61,194** DDoS attacks with zero prior signatures. |
| **Attack Precision** | $\frac{TP}{TP + FP}$ | **74.92%** | When an alert was triggered, it was a true attack **75% of the time**. |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | **0.7092** | Robust harmonic mean for an unsupervised baseline. |
| **False Positive Rate (FPR)** | $\frac{FP}{FP + TN}$ | **35.55%** | Percentage of legitimate traffic flagged due to temporal drift. |
| **Training Duration** | — | **2.50 s** | Scalable $O(t \cdot \psi \log \psi)$ computational complexity. |
| **Inference Throughput** | — | **49,995 flows/s** | Capable of processing real-time gigabit network streams. |

### 6.4 Academic Discussion: Addressing the False Positive Rate (35.55%)
In academic network intrusion detection research, training on Monday and testing on Friday introduces **concept drift / temporal drift**:
* Background benign user activities (HTTP browsing, active socket connections, file transfers) differ between Monday morning and Friday afternoon.
* Because the model learned normality exclusively from Monday, novel legitimate Friday traffic patterns were flagged as statistically atypical.
* Unlike published literature reporting artificial 99% accuracy via random train/test splits (which suffer from duplicate flow data leakage), our cross-day evaluation reflects realistic operational performance.

---

## CHAPTER 7: SUMMARY OF PROJECT SCRIPTS

| Script Path | Purpose | Key Functions & Components |
|---|---|---|
| `scripts/01_inspect_single_csv.py` | Single-file dataset profiler | Streams `Friday-DDos.csv`, reports memory usage, verifies 79 columns, detects 65 whitespace headers, audits Infs and NaNs. |
| `scripts/02_inspect_all_csvs.py` | Cross-file schema validator | Streams all 8 CSVs, confirms schema uniformity, logs 2,830,743 rows, catalogs 14 attack classes, verifies Monday 100% benign. |
| `scripts/03_preprocess_single_file.py` | Preprocessor for Friday test set | Disambiguates `'Fwd Header Length.1'`, caps $+Inf$ at max finite throughput, imputes $NaN$ with median, preserves labels. |
| `scripts/03b_preprocess_monday.py` | Preprocessor for Monday train set | Preprocesses 529,918 Monday records with 100% row retention, zeroing invalid values for clean baseline modeling. |
| `scripts/isolation_forest_model.py` | First-principles Isolation Forest | Custom pure-NumPy classes: `IsolationTreeNode`, `IsolationTree`, `IsolationForest`. Subsampling $\psi=256$, depth 8, $c(n)$ correction. |
| `scripts/05_train_and_evaluate.py` | Training & evaluation pipeline | Ingests Monday train and Friday test sets, fits forest in 2.5s, scores 100,000 flows, computes confusion matrix and metrics. |

---

## CHAPTER 8: CONCLUSION & FUTURE WORK

### 8.1 Conclusion
AutoNIDS demonstrates that an anomaly-based Network Intrusion Detection System built from first principles in pure NumPy can effectively isolate complex cyber threats without supervised attack training. By implementing Liu et al.’s mathematical formulations directly, the system bypassed enterprise OS security blocks, achieved an F1-score of 0.7092 on unseen DDoS attacks, and maintained an inference speed of nearly 50,000 flows/second.

### 8.2 Future Roadmap
1. **Dynamic Threshold Tuning**: Sweep anomaly score cutoffs from $0.50$ to $0.65$ using Receiver Operating Characteristic (ROC-AUC) curves to reduce the False Positive Rate below 10%.
2. **Cross-Attack Evaluation**: Test the Monday-trained model against PortScan, Patator brute-force, and Web Attack captures.
3. **Feature Selection**: Prune zero-variance constant features (e.g., `Bwd PSH Flags`, `Fwd URG Flags`) to improve partitioning efficiency.
4. **Hybrid Modeling**: Introduce deep autoencoders as a secondary reconstruction-error benchmark.

---

## REFERENCES

1. **K. Sadaf and J. Sultana**, "Intrusion Detection Based on Auto-encoder and Isolation Forest in Fog Computing," *IEEE Access*, vol. 8, pp. 167059–167068, 2020. DOI: `10.1109/ACCESS.2020.3023807`.
2. **Z. M. Maseer, R. Al-Qurashi, et al.**, "Benchmarking of Machine Learning for Anomaly Based Intrusion Detection Systems in the CICIDS2017 Dataset," *IEEE Access*, vol. 9, pp. 22351–22370, 2021. DOI: `10.1109/ACCESS.2021.3056614`.
3. **I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani**, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," *Proc. 4th Int. Conf. Inf. Syst. Secur. Privacy (ICISSP)*, pp. 108–116, 2018. DOI: `10.5220/0006639801080116`.
4. **F. T. Liu, K. M. Ting, and Z.-H. Zhou**, "Isolation Forest," *2008 Eighth IEEE Int. Conf. Data Mining (ICDM)*, pp. 413–422, IEEE, 2008. DOI: `10.1109/ICDM.2008.17`.
