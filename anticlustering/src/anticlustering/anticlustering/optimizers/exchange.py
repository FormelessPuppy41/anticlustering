"""
Exchange-based heuristic as a separate component (Composition over Inheritance).
"""
import numpy as np

def optimize_with_exchange(
    X: np.ndarray,
    labels: np.ndarray,
    objective: callable,
    max_iter: int = 1
) -> np.ndarray:
    """
    Iteratively swap pairs of samples across groups to increase objective.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    labels : ndarray, shape (n_samples,)
        Initial group labels.
    objective : function
        Callable(X, labels) -> float.
    max_iter : int
        Number of full passes over the data.

    Returns
    -------
    labels_opt : ndarray, shape (n_samples,)
        Improved labels.
    """
    best_labels = labels.copy()
    best_score = objective(X, best_labels)
    n_samples = X.shape[0]
    n_groups = len(np.unique(labels))

    for _ in range(max_iter):
        for i in range(n_samples):
            current_group = best_labels[i]
            for g in range(n_groups):
                if g == current_group:
                    continue
                # candidates in group g
                for j in np.where(best_labels == g)[0]:
                    candidate = best_labels.copy()
                    candidate[i], candidate[j] = g, current_group
                    score = objective(X, candidate)
                    if score > best_score:
                        best_score = score
                        best_labels = candidate
    return best_labels