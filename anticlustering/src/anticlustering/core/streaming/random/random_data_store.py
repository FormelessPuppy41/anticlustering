# anticlustering/streaming/data_manager.py

from typing import List, Optional

import logging
import numpy as np
from ....streaming.data_store import StreamingDataStore
from anticlustering.metrics.dissimilarity_matrix import get_dissimilarity_matrix

_LOG = logging.getLogger(__name__)

class RandomStreamingDataStore(StreamingDataStore):
    """
    Streaming store for raw feature vectors (option B):
      - Ingest arrays via add_features
      - Remove by ID via remove_ids
    """

    def __init__(self, feature_dim: int):
        super().__init__(feature_dim)

    def add_features(self, new_ids: List[str], X_new: np.ndarray) -> List[str]:
        if not new_ids:
            return []
        self._ids.extend(new_ids)
        if self._X.size:
            self._X = np.vstack([self._X, X_new])
        else:
            self._X = X_new.copy()
        self._D = get_dissimilarity_matrix(self._X)
        return new_ids

    def remove_ids(self, old_ids: List[str]) -> List[str]:
        if not old_ids:
            return []
        set_remove = set(old_ids)
        removed = [lid for lid in self._ids if lid in set_remove]
        if not removed:
            _LOG.warning("No IDs removed; none match %s", old_ids)
            return []
        keep_mask = [lid not in set_remove for lid in self._ids]
        self._ids = [lid for lid in self._ids if lid not in set_remove]
        self._X = self._X[keep_mask, :]
        self._D = get_dissimilarity_matrix(self._X)
        _LOG.debug("After removal, distance matrix: %s", self._D.shape)
        return removed
