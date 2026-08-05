# Similarity Metrics Engine Documentation

The `similarity_metrics` module provides mathematical functions and unified interfaces for vector similarity and distance calculations in RAG retrieval.

---

## 1. Mathematical Algorithms & Formulas

### A. Cosine Similarity (`"cosine"`)
- **Formula**: $\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum u_i v_i}{\sqrt{\sum u_i^2} \sqrt{\sum v_i^2}}$
- **Range**: $[-1.0, 1.0]$ (1.0 = identical orientation).
- **Behavior**: Measures the angular orientation between two vectors, ignoring magnitude/length. Standard for text & multimodal embeddings.

### B. Dot Product / Inner Product (`"dot_product"`, `"ip"`)
- **Formula**: $\text{Dot Product}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i$
- **Range**: $(-\infty, +\infty)$.
- **Behavior**: Combines vector direction and magnitude. Equivalent to Cosine Similarity when vectors are $L_2$-normalized.

### C. Euclidean Distance / $L_2$ Distance (`"l2"`, `"euclidean"`)
- **Formula**: $d_{L2}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^d (u_i - v_i)^2} = \|\mathbf{u} - \mathbf{v}\|_2$
- **Range**: $[0.0, +\infty)$ (0.0 = identical spatial position).
- **Behavior**: Measures straight-line spatial distance. Lower values indicate higher similarity.

### D. Manhattan Distance / $L_1$ Distance (`"l1"`, `"manhattan"`)
- **Formula**: $d_{L1}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^d |u_i - v_i|$
- **Range**: $[0.0, +\infty)$ (0.0 = identical spatial position).
- **Behavior**: Measures sum of absolute grid/taxicab coordinate differences.

---

## 2. Functions Defined

* `cosine_similarity(u: np.ndarray, v: np.ndarray) -> float`
* `dot_product(u: np.ndarray, v: np.ndarray) -> float`
* `euclidean_distance(u: np.ndarray, v: np.ndarray) -> float`
* `manhattan_distance(u: np.ndarray, v: np.ndarray) -> float`
* `compute_similarity(vec1, vec2, metric: str = "cosine") -> float`: Unified router function supporting case-insensitive metric lookup and array conversion.
