"""
AutoNIDS - Phase 4: Isolation Forest from First Principles (NumPy Implementation)
=================================================================================
Reference: F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation Forest", IEEE ICDM 2008.

Why this implementation exists:
1. Enterprise code integrity policy on Windows blocks unsigned third-party C-extensions (.pyd in scipy).
2. Pure NumPy implementation bypasses all OS security blocks.
3. Provides 100% white-box academic explainability for faculty viva.

Key Concepts:
- Binary Tree (iTree) with random feature selection and uniform threshold split.
- Subsampling: psi = 256 samples per tree.
- Max Tree Depth: ceil(log2(psi)) = 8.
- Average Path Length: anomalies require fewer splits to isolate (shorter path).
- c(n): Average path length of unsuccessful search in Binary Search Tree (BST).
"""

import math
import pickle
from pathlib import Path
import numpy as np

# Euler-Mascheroni constant
EULER_MASCHERONI = 0.5772156649


def c_factor(n: int) -> float:
    """Average path length of unsuccessful searches in a Binary Search Tree (BST)."""
    if n > 2:
        return 2.0 * (math.log(n - 1) + EULER_MASCHERONI) - (2.0 * (n - 1) / n)
    elif n == 2:
        return 1.0
    else:
        return 0.0


class IsolationTreeNode:
    def __init__(self, size: int):
        self.size = size  # Number of samples reaching this node
        self.is_leaf = True
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None


class IsolationTree:
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: np.ndarray, current_depth: int = 0) -> IsolationTreeNode:
        n_samples, n_features = X.shape

        node = IsolationTreeNode(size=n_samples)

        # Base case: max depth reached, single sample, or all identical
        if current_depth >= self.max_depth or n_samples <= 1:
            return node

        # Randomly select a feature
        feature_indices = np.arange(n_features)
        np.random.shuffle(feature_indices)

        split_feature = None
        split_value = None

        for feat in feature_indices:
            col_min = X[:, feat].min()
            col_max = X[:, feat].max()
            if col_min < col_max:
                split_feature = feat
                split_value = np.random.uniform(col_min, col_max)
                break

        # If all features have zero variance (identical values)
        if split_feature is None:
            return node

        # Split data
        left_mask = X[:, split_feature] < split_value
        right_mask = ~left_mask

        # If one branch is empty, stop
        if not np.any(left_mask) or not np.any(right_mask):
            return node

        node.is_leaf = False
        node.split_feature = split_feature
        node.split_value = split_value
        node.left = self.fit(X[left_mask], current_depth + 1)
        node.right = self.fit(X[right_mask], current_depth + 1)

        return node

    def path_length(self, x: np.ndarray) -> np.ndarray:
        """Vectorized traversal for a batch of samples."""
        lengths = np.zeros(len(x), dtype=float)
        self._traverse(self.root, x, np.arange(len(x)), 0, lengths)
        return lengths

    def _traverse(self, node: IsolationTreeNode, X: np.ndarray, indices: np.ndarray, current_depth: int, lengths: np.ndarray):
        if len(indices) == 0:
            return

        if node.is_leaf:
            lengths[indices] = current_depth + c_factor(node.size)
            return

        feat = node.split_feature
        val = node.split_value

        left_mask = X[indices, feat] < val
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        if len(left_indices) > 0 and node.left is not None:
            self._traverse(node.left, X, left_indices, current_depth + 1, lengths)
        if len(right_indices) > 0 and node.right is not None:
            self._traverse(node.right, X, right_indices, current_depth + 1, lengths)


class IsolationForest:
    def __init__(self, n_estimators: int = 100, max_samples: int = 256, contamination: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.trees = []
        self.threshold = 0.5

    def fit(self, X: np.ndarray):
        """Fit forest on training feature matrix X (unsupervised, no labels)."""
        np.random.seed(self.random_state)
        n_samples = len(X)
        sample_size = min(self.max_samples, n_samples)
        self.trees = []

        for _ in range(self.n_estimators):
            subsample_idx = np.random.choice(n_samples, size=sample_size, replace=False)
            X_sub = X[subsample_idx]
            tree = IsolationTree(max_depth=int(math.ceil(math.log2(sample_size))))
            tree.root = tree.fit(X_sub, current_depth=0)
            self.trees.append(tree)

        # Compute threshold based on training contamination if desired
        scores = self.decision_function(X)
        self.threshold = float(np.percentile(scores, 100.0 * (1.0 - self.contamination)))
        return self

    def compute_average_path_length(self, X: np.ndarray) -> np.ndarray:
        all_lengths = np.zeros((self.n_estimators, len(X)), dtype=float)
        for i, tree in enumerate(self.trees):
            all_lengths[i] = tree.path_length(X)
        return np.mean(all_lengths, axis=0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Anomaly score s(x, n) = 2^(-E(h(x)) / c(psi)).
        Values close to 1 indicate anomalies (attacks).
        Values below 0.5 indicate normal benign instances.
        """
        avg_path = self.compute_average_path_length(X)
        c = c_factor(self.max_samples)
        if c == 0:
            return np.zeros(len(X))
        scores = 2.0 ** (-avg_path / c)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns:
            -1 for Anomaly (Attack)
             1 for Normal (Benign)
        """
        scores = self.decision_function(X)
        # Higher score = more anomalous
        predictions = np.where(scores >= self.threshold, -1, 1)
        return predictions

    def save(self, filepath: Path):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path):
        with open(filepath, "rb") as f:
            return pickle.load(f)
