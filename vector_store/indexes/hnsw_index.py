"""
FAISS HNSW Index (Hierarchical Navigable Small World Graph Search).
"""

from typing import Tuple
import numpy as np
from .base import BaseVectorIndex


class HNSWIndex(BaseVectorIndex):
    """
    HNSW Graph Vector Index (IndexHNSWFlat).
    Provides blazing fast O(log N) approximate nearest neighbor search via multi-layer graph navigation.
    """

    def __init__(self, dimension: int, metric: str = "cosine", m: int = 32):
        super().__init__(dimension, metric)
        import faiss

        faiss_metric = (
            faiss.METRIC_INNER_PRODUCT
            if self.metric in ["cosine", "dot_product", "ip", "dot"]
            else faiss.METRIC_L2
        )
        self.m = m
        self.index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss_metric)
        self.index.hnsw.efSearch = 64  # Depth of search exploration

    def add(self, vecs: np.ndarray):
        if vecs.shape[0] > 0:
            self.index.add(vecs)

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        fetch_k = min(self.index.ntotal, top_k)
        if fetch_k <= 0:
            return np.array([[]]), np.array([[]])
        # HNSW requires efSearch >= top_k for correct results
        if self.index.hnsw.efSearch < fetch_k:
            self.index.hnsw.efSearch = fetch_k
        return self.index.search(query_vec, fetch_k)
