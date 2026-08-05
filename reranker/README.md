# Reranking Engine Documentation

The `reranker` module provides Stage 2 cross-encoder relevance re-scoring over Stage 1 candidate search results.

---

## 1. Underlying Algorithms & Methods

### A. Stage 1 vs Stage 2 Architectural Comparison
- **Stage 1 (Bi-Encoder / FAISS)**: Embeds Query $E(q)$ and Document $E(d)$ independently into static vectors. Enables fast vector search ($O(\log N)$), but lacks interaction between query words and document words.
- **Stage 2 (Cross-Encoder Reranker)**: Feeds `(query, document)` pairs **jointly** into a single Transformer network. Self-attention layers evaluate cross-token relationships between every query word and document word to produce an exact scalar relevance score.

### B. Cross-Encoder Prediction Algorithm (`CrossEncoder.predict`)
1. Accepts query $q$ and candidate list $[c_1, c_2, \dots, c_N]$.
2. Constructs sequence pairs: `[[q, c1.text], [q, c2.text], ..., [q, cN.text]]`.
3. Runs PyTorch forward pass through `cross-encoder/ms-marco-MiniLM-L-6-v2`.
4. Extracts logit relevance scores and re-sorts candidates in descending order.

### C. Heuristic Term-Overlap & Vector-Score Fusion Fallback (`_fallback_rerank`)
If model loading fails or operates offline, it applies a hybrid score fusion formula:
$$\text{Score}_{\text{fallback}} = 0.7 \cdot \text{Score}_{\text{vector}} + 0.3 \cdot \left(\frac{\text{Query Terms } \cap \text{ Document Terms}}{|\text{Query Terms}|}\right)$$

---

## 2. Class Defined

### `Reranker`
Stage 2 Cross-Encoder Reranker.

* **Initialization Parameters**:
  * `model_name` (`str`, default `"cross-encoder/ms-marco-MiniLM-L-6-v2"`): HuggingFace Cross-Encoder model.
  * `device` (`Optional[str]`, default `None`): Auto-detects compute device (`"cuda"`, `"mps"`, `"cpu"`).
  * `top_n` (`int`, default `3`): Default number of top reranked results to return.
* **Methods**:
  * `rerank(query: str, results: List[SearchResult], top_n: Optional[int] = None) -> List[SearchResult]`: Rescores candidates and returns top-$N$ matches.
