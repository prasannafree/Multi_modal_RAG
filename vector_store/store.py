"""
FAISS Vector Database Store & Search Module.

Provides fast local vector indexing, metadata payload storage, metric selection 
(Cosine Similarity, Dot Product, L2 Euclidean Distance), and metadata filtering.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from chunker import Chunk
from similarity_metrics import compute_similarity, cosine_similarity


@dataclass
class SearchResult:
    """
    Result returned by vector similarity search query.

    Attributes:
        chunk_id: Unique ID of retrieved chunk.
        score: Similarity score (Higher is better for Cosine/Dot; lower is better for L2).
        text: Chunk text payload.
        metadata: Metadata associated with chunk (source, page, element_type, etc.).
    """
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }


class FAISSVectorStore:
    """
    FAISS-backed Vector Database Store with metadata payload persistence and search.
    """

    def __init__(self, dimension: int = 384, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric.lower()
        self.payloads: List[Dict[str, Any]] = []
        self._faiss = None
        self.index = None
        self._init_faiss_index()

    def _init_faiss_index(self):
        """Initializes the underlying FAISS index."""
        try:
            import faiss
            self._faiss = faiss

            if self.metric in ["cosine", "dot_product", "ip"]:
                # IndexFlatIP handles Inner Product & Cosine Similarity for normalized vectors
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                # IndexFlatL2 handles Euclidean / L2 distance
                self.index = faiss.IndexFlatL2(self.dimension)
        except ImportError:
            print("[Notice] `faiss-cpu` package not found. Falling back to Numpy Vector Store.")
            self._faiss = None
            self.index = None

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """
        Adds RAG chunks and their corresponding embedding vectors to the FAISS index.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Length of chunks and embeddings must match.")

        if not chunks:
            return

        vecs = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for Cosine Similarity if metric is cosine
        if self.metric == "cosine":
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms

        if self.index is not None:
            self.index.add(vecs)
        
        for c, vec in zip(chunks, vecs):
            payload_dict = c.to_dict()
            payload_dict["vector"] = vec.tolist()
            self.payloads.append(payload_dict)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Performs vector similarity search against indexed chunks with optional metadata filtering.

        Args:
            query_vector: High-dimensional search query embedding vector.
            top_k: Number of top nearest matches to return.
            filter_metadata: Dictionary of metadata Key-Value pairs to filter results (e.g. {"element_type": "table"}).

        Returns:
            List of SearchResult objects ordered by relevance.
        """
        if not self.payloads:
            return []

        q_vec = np.array([query_vector], dtype=np.float32)
        if self.metric == "cosine":
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm

        # 1. Search via FAISS index if available
        if self.index is not None and self.index.ntotal > 0:
            # Over-fetch if metadata filtering is requested
            fetch_k = min(self.index.ntotal, top_k * 5 if filter_metadata else top_k)
            distances, indices = self.index.search(q_vec, fetch_k)

            results: List[SearchResult] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.payloads):
                    continue

                payload = self.payloads[idx]
                meta = payload.get("metadata", {})

                # Apply metadata filter check
                if filter_metadata and not self._matches_filter(meta, filter_metadata):
                    continue

                results.append(
                    SearchResult(
                        chunk_id=payload.get("chunk_id", ""),
                        score=float(dist),
                        text=payload.get("text", ""),
                        metadata=meta,
                    )
                )
                if len(results) >= top_k:
                    break

            return results

        # 2. NumPy Fallback vector search using similarity_metrics
        return self._numpy_search(q_vec[0], top_k, filter_metadata)

    def save(self, dir_path: Union[str, Path]):
        """Persists FAISS index and metadata payload to disk."""
        folder = Path(dir_path)
        folder.mkdir(parents=True, exist_ok=True)

        if self._faiss is not None and self.index is not None:
            index_path = str(folder / "index.faiss")
            self._faiss.write_index(self.index, index_path)

        payload_path = folder / "payloads.json"
        payload_path.write_text(json.dumps(self.payloads, indent=2), encoding="utf-8")

    def load(self, dir_path: Union[str, Path]):
        """Loads persisted FAISS index and metadata payload from disk."""
        folder = Path(dir_path)

        index_path = folder / "index.faiss"
        if self._faiss is not None and index_path.exists():
            self.index = self._faiss.read_index(str(index_path))

        payload_path = folder / "payloads.json"
        if payload_path.exists():
            self.payloads = json.loads(payload_path.read_text(encoding="utf-8"))

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Helper to match metadata attributes against filter dictionary."""
        for key, val in filter_dict.items():
            if metadata.get(key) != val:
                return False
        return True

    def _numpy_search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Fallback NumPy vector search calculated via similarity_metrics engine."""
        scored_results = []
        for payload in self.payloads:
            meta = payload.get("metadata", {})
            if filter_metadata and not self._matches_filter(meta, filter_metadata):
                continue
            
            chunk_vec = payload.get("vector")
            if chunk_vec is not None:
                score = compute_similarity(query_vec, chunk_vec, metric=self.metric)
            else:
                score = 1.0

            scored_results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    score=score,
                    text=payload.get("text", ""),
                    metadata=meta,
                )
            )

        # Sort: Descending for similarity metrics (cosine, dot_product); Ascending for distance metrics (l2, l1)
        reverse_sort = self.metric in ["cosine", "dot_product", "ip", "dot"]
        scored_results.sort(key=lambda r: r.score, reverse=reverse_sort)
        return scored_results[:top_k]
