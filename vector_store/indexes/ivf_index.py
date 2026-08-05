"""
FAISS IVF Index (Inverted File Partitioned Search).
"""

from typing import Tuple
import numpy as np
from .base import BaseVectorIndex


class IVFIndex(BaseVectorIndex):
    """
    IVF Clustered Vector Index (IndexIVFFlat).
    Partitions vector space into Voronoi clusters (nlist) using K-Means, then searches closest nprobe clusters.
    """

    def __init__(self, dimension: int, metric: str = "cosine", nlist: int = 4, nprobe: int = 2):
        super().__init__(dimension, metric)
        import faiss

        self.nlist = nlist
        self.nprobe = nprobe
        faiss_metric = (
            faiss.METRIC_INNER_PRODUCT
            if self.metric in ["cosine", "dot_product", "ip", "dot"]
            else faiss.METRIC_L2
        )

        if self.metric in ["cosine", "dot_product", "ip", "dot"]:
            self.quantizer = faiss.IndexFlatIP(self.dimension)
        else:
            self.quantizer = faiss.IndexFlatL2(self.dimension)

        self.index = faiss.IndexIVFFlat(self.quantizer, self.dimension, self.nlist, faiss_metric)
        self.index.nprobe = self.nprobe
        self.is_trained = False

    def add(self, vecs: np.ndarray):
        if vecs.shape[0] <= 0:
            return

        # IVF requires training the quantizer cluster centroids first
        if not self.index.is_trained:
            # Adjust nlist dynamically if total samples < nlist
            if vecs.shape[0] < self.nlist:
                import faiss

                self.nlist = max(1, vecs.shape[0])
                # Rebuild the index with the adjusted nlist since the original
                # was constructed with the old value in __init__
                faiss_metric = (
                    faiss.METRIC_INNER_PRODUCT
                    if self.metric in ["cosine", "dot_product", "ip", "dot"]
                    else faiss.METRIC_L2
                )
                if self.metric in ["cosine", "dot_product", "ip", "dot"]:
                    self.quantizer = faiss.IndexFlatIP(self.dimension)
                else:
                    self.quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    self.quantizer, self.dimension, self.nlist, faiss_metric
                )
                self.index.nprobe = self.nprobe
            self.index.train(vecs)
            self.is_trained = True

        self.index.add(vecs)

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        fetch_k = min(self.index.ntotal, top_k)
        if fetch_k <= 0 or not self.index.is_trained:
            return np.array([[]]), np.array([[]])
        return self.index.search(query_vec, fetch_k)
