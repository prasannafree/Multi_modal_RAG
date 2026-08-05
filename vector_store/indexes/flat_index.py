"""
FAISS Flat Index (Exact Brute-Force Search).
"""

from typing import Tuple
import numpy as np
from .base import BaseVectorIndex


class FlatIndex(BaseVectorIndex):
    """
    Exact Flat Vector Index (IndexFlatIP / IndexFlatL2).
    Provides 100% recall accuracy via exact vector dot product / L2 distance calculations.
    """

    def __init__(self, dimension: int, metric: str = "cosine"):
        super().__init__(dimension, metric)
        import faiss

        if self.metric in ["cosine", "dot_product", "ip", "dot"]:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def add(self, vecs: np.ndarray):
        if vecs.shape[0] > 0:
            self.index.add(vecs)

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        fetch_k = min(self.index.ntotal, top_k)
        if fetch_k <= 0:
            return np.array([[]]), np.array([[]])
        return self.index.search(query_vec, fetch_k)
