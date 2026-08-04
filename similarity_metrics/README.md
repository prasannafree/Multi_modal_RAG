# Similarity Metrics Module

The `similarity_metrics` module provides vector similarity and distance calculations for the Multimodal RAG retrieval system.

---

## Supported Metrics

1. **Cosine Similarity (`cosine`)**:
   - Measures vector angular orientation. Range: $[-1.0, 1.0]$. Higher is better.
2. **Dot Product / Inner Product (`dot_product` / `ip`)**:
   - Measures vector direction + magnitude. Range: $(-\infty, +\infty)$.
3. **Euclidean Distance (`l2`)**:
   - Measures straight-line distance. Range: $[0.0, +\infty)$. Lower is better.
4. **Manhattan Distance (`l1`)**:
   - Measures grid sum of absolute differences. Range: $[0.0, +\infty)$. Lower is better.

---

## Usage Example

```python
from similarity_metrics import compute_similarity, cosine_similarity

vec1 = [0.1, 0.5, 0.8]
vec2 = [0.2, 0.4, 0.9]

# Using unified compute_similarity
score = compute_similarity(vec1, vec2, metric="cosine")
print("Cosine Score:", score)

# Using specific metric function
l2_dist = compute_similarity(vec1, vec2, metric="l2")
print("Euclidean Distance:", l2_dist)
```
