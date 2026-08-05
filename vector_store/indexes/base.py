"""
Base Interface for Vector Indexing Algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class BaseVectorIndex(ABC):
    """
    Abstract Base Class for vector indexing algorithms (Flat, HNSW, IVF, etc.).
    """

    def __init__(self, dimension: int, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric.lower().strip()
        self.is_trained = True

    @abstractmethod
    def add(self, vecs: np.ndarray):
        """Adds normalized/raw vector float32 matrix to the index."""
        pass

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches nearest neighbors for query vector.
        Returns tuple of (distances_array, indices_array).
        """
        pass
