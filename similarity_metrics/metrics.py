"""
Similarity Metrics Engine for Multimodal RAG.

Provides mathematical implementations and unified interfaces for vector similarity & distance calculations:
- Cosine Similarity (Angular similarity)
- Dot Product / Inner Product (Magnitude + Angle)
- Euclidean Distance (L2 Distance)
- Manhattan Distance (L1 Distance)
"""

from enum import Enum
from typing import List, Union
import numpy as np


class MetricType(str, Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    IP = "ip"
    EUCLIDEAN = "l2"
    MANHATTAN = "l1"


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes Cosine Similarity between two vectors (Range: -1.0 to 1.0).
    Higher score indicates higher similarity (identical orientation).
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes Dot Product (Inner Product) between two vectors.
    Equivalent to Cosine Similarity when vectors are L2-normalized.
    """
    return float(np.dot(u, v))


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes Euclidean (L2) Distance between two vectors (Range: 0.0 to +inf).
    Lower distance indicates higher similarity (closer in spatial distance).
    """
    return float(np.linalg.norm(u - v))


def manhattan_distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes Manhattan (L1) Distance between two vectors (Range: 0.0 to +inf).
    Sum of absolute coordinate differences.
    """
    return float(np.sum(np.abs(u - v)))


def compute_similarity(
    vec1: Union[List[float], np.ndarray],
    vec2: Union[List[float], np.ndarray],
    metric: str = "cosine",
) -> float:
    """
    Unified similarity & distance computation interface.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector.
        metric: Metric choice ('cosine', 'dot_product', 'l2', 'l1').

    Returns:
        Float score or distance.
    """
    u = np.array(vec1, dtype=np.float32)
    v = np.array(vec2, dtype=np.float32)

    metric_clean = metric.lower().strip()

    if metric_clean in ["cosine"]:
        return cosine_similarity(u, v)
    elif metric_clean in ["dot_product", "ip", "dot"]:
        return dot_product(u, v)
    elif metric_clean in ["euclidean", "l2"]:
        return euclidean_distance(u, v)
    elif metric_clean in ["manhattan", "l1"]:
        return manhattan_distance(u, v)
    else:
        raise ValueError(
            f"Unsupported metric: '{metric}'. Choose from ['cosine', 'dot_product', 'l2', 'l1']."
        )
