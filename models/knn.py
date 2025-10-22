"""
Intervalal k-NN classifier (sklearn-like API) based on the distance and voting
logic from models/sonn.py.

Usage:
    from models.knn import IntervalKNN
    clf = IntervalKNN(k=5)
    clf.fit(X, y)          # X shape (n_samples, 2*m) with [min,max] pairs
    preds = clf.predict(X_test)
    acc = clf.accuracy(X_test, y_test)

This file also provides a small CLI to run the classifier on datasets/climates.csv
when executed as a script.
"""

from typing import Optional, Union
import numpy as np


class IntervalKNN:
    """Interval k-NN classifier with inverse-distance weighting.

    Parameters
    ----------
    k : int
        Number of neighbors to use.
    n_classes : Optional[int]
        If provided, the number of classes.

    Methods
    -------
    fit(X, y)
    predict(X)
    accuracy(X, y)
    """

    def __init__(self, k: Union[int, list, tuple, np.ndarray] = 3, n_classes: Optional[int] = None, **kwargs):
        # follow project naming convention: trailing underscore for learned/param attrs
        # be tolerant: k may be passed as a list (from older parameter formats);
        # pick a sensible scalar (median) and warn the user.
        if isinstance(k, (list, tuple, np.ndarray)):
            k_arr = np.asarray(k)
            if k_arr.size == 0:
                raise ValueError("k list provided is empty")
            # choose median as a robust scalar fallback
            chosen = int(np.median(k_arr))
            import warnings
            warnings.warn("IntervalKNN received list for parameter 'k'; using median value {} as k".format(chosen))
            self.k_ = chosen
        else:
            self.k_ = int(k)
        self.n_classes_ = n_classes
        self._trained = False

    @staticmethod
    def _calcular_distancia_vectorized(test_vec: np.ndarray, train_mat: np.ndarray) -> np.ndarray:
        # squared diffs per element
        diffs = (train_mat - test_vec) ** 2
        even_sum = np.sqrt(np.sum(diffs[:, ::2], axis=1))
        odd_sum = np.sqrt(np.sum(diffs[:, 1::2], axis=1))
        return even_sum + odd_sum

    def fit(self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None):
        """Fit the classifier.

        Accepts either:
          - fit(X, y) where X is (n_samples, 2*m) and y is labels
          - fit(training_list) where each row is [...features..., label] (legacy sonn style)
        """
        if y is None:
            # expect X as iterable of rows with last element the class label
            rows = list(X)
            if len(rows) == 0:
                raise ValueError("Empty training data")
            features = [np.asarray(r[:-1], dtype=float) for r in rows]
            labels = [r[-1] for r in rows]
            X = np.vstack(features)
            y = np.asarray(labels, dtype=int)
        else:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=int)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] % 2 != 0:
            raise ValueError("X must have even number of columns (min/max pairs)")

        # store with project naming convention
        self.train_X_ = X.copy()
        self.train_y_ = y.copy()
        if self.n_classes_ is None:
            # infer number of classes from labels (support non-zero based labels)
            self.n_classes_ = int(np.max(self.train_y_) + 1) if self.train_y_.size > 0 else 0
        self._trained = True
        return self

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Predict labels for X.

        X: array-like shape (n_samples, n_features)
        """
        if not self._trained:
            raise ValueError("Classifier not trained. Call fit first.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_train = self.train_X_.shape[0]
        k = min(self.k_, max(1, n_train))
        preds = []

        for i in range(X.shape[0]):
            test_vec = X[i]
            dists = self._calcular_distancia_vectorized(test_vec, self.train_X_)
            # handle exact matches
            zero_mask = (dists == 0)
            if np.any(zero_mask):
                labels, counts = np.unique(self.train_y_[zero_mask], return_counts=True)
                preds.append(int(labels[np.argmax(counts)]))
                continue
            # get k nearest
            nn_idx = np.argsort(dists)[:k]
            nn_dists = dists[nn_idx]
            # weights are inverse distance
            # guard against division by zero though we already filtered zeros
            weights = 1.0 / nn_dists
            # accumulate weighted votes per class
            votes = np.zeros(self.n_classes_ if self.n_classes_ is not None else (int(np.max(self.train_y_) + 1)), dtype=float)
            for idx, w in zip(nn_idx, weights):
                cls = int(self.train_y_[idx])
                if cls >= votes.shape[0]:
                    # expand votes array if a label larger than inferred n_classes appears
                    new_size = cls + 1
                    new_votes = np.zeros(new_size, dtype=float)
                    new_votes[:votes.shape[0]] = votes
                    votes = new_votes
                votes[cls] += w
            preds.append(int(np.argmax(votes)))

        return np.asarray(preds)

    def accuracy(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        """Return mean accuracy on the given test data and labels."""
        y = np.asarray(y)
        preds = self.predict(X)
        return float(np.mean(preds == y))

