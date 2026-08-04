"""
Similarity Metrics Module Package.
"""

from .metrics import (
    MetricType,
    compute_similarity,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    manhattan_distance,
)

__all__ = [
    "MetricType",
    "compute_similarity",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
    "manhattan_distance",
]
