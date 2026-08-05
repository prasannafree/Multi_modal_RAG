"""
Factory registry for instantiating vector index algorithms.
"""

from typing import Dict, Type
from .base import BaseVectorIndex
from .flat_index import FlatIndex
from .hnsw_index import HNSWIndex
from .ivf_index import IVFIndex

# Extensible Registry mapping index_type names to Index Classes
INDEX_REGISTRY: Dict[str, Type[BaseVectorIndex]] = {
    "flat": FlatIndex,
    "exact": FlatIndex,
    "hnsw": HNSWIndex,
    "graph": HNSWIndex,
    "ivf": IVFIndex,
    "cluster": IVFIndex,
}


def register_vector_index(name: str, index_cls: Type[BaseVectorIndex]):
    """
    Registers a new custom vector index algorithm into the system.

    Usage:
        register_vector_index("my_algorithm", MyCustomIndex)
    """
    INDEX_REGISTRY[name.lower().strip()] = index_cls


def create_vector_index(index_type: str, dimension: int, metric: str = "cosine") -> BaseVectorIndex:
    """
    Instantiates a vector index algorithm based on index_type string.

    Args:
        index_type: 'flat', 'hnsw', 'ivf', etc.
        dimension: Vector dimension (e.g. 512 for CLIP).
        metric: Similarity metric ('cosine', 'dot_product', 'l2').

    Returns:
        Instance of BaseVectorIndex.
    """
    name_clean = index_type.lower().strip()
    if name_clean not in INDEX_REGISTRY:
        supported = list(set(INDEX_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported index_type '{index_type}'. Choose from {supported} or register a new one."
        )

    index_cls = INDEX_REGISTRY[name_clean]
    return index_cls(dimension=dimension, metric=metric)
