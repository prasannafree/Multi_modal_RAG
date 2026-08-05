"""
Reranking Engine for Multimodal RAG.

Provides Stage 2 Cross-Encoder reranking over Stage 1 candidate search results.
Jointly evaluates query-candidate cross-attention to compute precise relevance scores.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from vector_store import SearchResult


def _detect_device() -> str:
    """Auto-detects optimal compute device (cuda, mps, cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Reranker:
    """
    Stage 2 Cross-Encoder Reranker.

    Rescores and re-orders candidate SearchResult objects retrieved by Stage 1 vector search.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        top_n: int = 3,
    ):
        self.model_name = model_name
        self.device = device or _detect_device()
        self.top_n = top_n
        self._model = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy loads the Cross-Encoder reranking model on first use."""
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            self._initialized = True
            print(f"[Reranker] Loaded CrossEncoder model '{self.model_name}' on device '{self.device}'")
        except Exception as e:
            print(f"[Notice] Could not load CrossEncoder model '{self.model_name}': {e}")
            print("[Notice] Using heuristic term-overlap fallback reranker.")
            self._model = None
            self._initialized = True

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Reranks a list of candidate SearchResult objects for a given query string.

        Args:
            query: User search query string.
            results: List of SearchResult objects retrieved from Stage 1 FAISS search.
            top_n: Number of top reranked results to return (defaults to self.top_n).

        Returns:
            List of SearchResult objects ordered by Cross-Encoder relevance score.
        """
        if not results:
            return []

        limit = top_n if top_n is not None else self.top_n
        self._lazy_init()

        if self._model is not None:
            # Pair query with each candidate chunk text for cross-attention evaluation
            pairs = [[query, res.text] for res in results]
            raw_scores = self._model.predict(pairs, show_progress_bar=False)

            reranked: List[SearchResult] = []
            for res, score in zip(results, raw_scores):
                reranked.append(
                    SearchResult(
                        chunk_id=res.chunk_id,
                        score=float(score),
                        text=res.text,
                        metadata=res.metadata,
                    )
                )

            # Sort descending by Cross-Encoder score
            reranked.sort(key=lambda r: r.score, reverse=True)
            return reranked[:limit]

        # Offline / lightweight fallback reranking
        return self._fallback_rerank(query, results, limit)

    def _fallback_rerank(
        self,
        query: str,
        results: List[SearchResult],
        limit: int,
    ) -> List[SearchResult]:
        """Heuristic term-overlap fallback reranking."""
        query_terms = set(query.lower().split())
        scored: List[SearchResult] = []

        for res in results:
            text_terms = res.text.lower().split()
            overlap_count = sum(1 for term in query_terms if term in text_terms)
            overlap_ratio = overlap_count / max(1, len(query_terms))
            
            # Combine Stage 1 vector score with keyword overlap boost
            combined_score = (res.score * 0.7) + (overlap_ratio * 0.3)
            scored.append(
                SearchResult(
                    chunk_id=res.chunk_id,
                    score=combined_score,
                    text=res.text,
                    metadata=res.metadata,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]
