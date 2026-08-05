# FAISS Vector Database Store & Index Algorithms Documentation

The `vector_store` module manages Stage 2 (**Retrieval System**) vector indexing, search algorithms (**Flat**, **HNSW**, **IVF**), metric calculations, metadata filtering, and disk persistence.

---

## 1. Vector Search Algorithms Implemented

### A. Flat Index (`"flat"`) - Exact Brute-Force Search
- **Class**: `FlatIndex` ([flat_index.py](file:///home/prasanna/Documents/my_projects/RAG/multi_modal_RAG/vector_store/indexes/flat_index.py))
- **Underlying FAISS Class**: `faiss.IndexFlatIP` / `faiss.IndexFlatL2`.
- **Complexity**: $O(N)$ search time.
- **Characteristics**: Evaluates query against every single vector in the index. Guaranteed **100% recall accuracy** (zero approximation error). Fast for datasets $< 100,000$ vectors.

### B. HNSW Index (`"hnsw"`) - Graph-Based ANN Search
- **Class**: `HNSWIndex` ([hnsw_index.py](file:///home/prasanna/Documents/my_projects/RAG/multi_modal_RAG/vector_store/indexes/hnsw_index.py))
- **Underlying FAISS Class**: `faiss.IndexHNSWFlat`.
- **Complexity**: $O(\log N)$ search time.
- **Characteristics**: Builds a multi-layer proximity graph (Hierarchical Navigable Small World). Navigates top layers with long-range skips and lower layers with fine-grained neighbor inspection (`efSearch = 64`). Blazing fast for large-scale vector search.

### C. IVF Index (`"ivf"`) - Clustered Voronoi Partition Search
- **Class**: `IVFIndex` ([ivf_index.py](file:///home/prasanna/Documents/my_projects/RAG/multi_modal_RAG/vector_store/indexes/ivf_index.py))
- **Underlying FAISS Class**: `faiss.IndexIVFFlat`.
- **Complexity**: $O(N / K)$ search time.
- **Characteristics**: Trains $K$ centroid clusters (`nlist`) using K-Means. During query, inspects vectors inside only the closest $N_{\text{probe}}$ clusters (`nprobe = 2`), bypassing the rest of the database.

---

## 2. Modular Architecture & Extensibility

```
vector_store/indexes/
├── base.py          # BaseVectorIndex Abstract Class
├── flat_index.py    # FlatIndex Implementation
├── hnsw_index.py    # HNSWIndex Implementation
├── ivf_index.py     # IVFIndex Implementation
└── factory.py       # Extensible Registry & Factory (create_vector_index)
```

### Adding New Search Algorithms
To add a new algorithm (e.g. `LSHIndex` or `PQIndex`):
1. Subclass `BaseVectorIndex` in a new file inside `vector_store/indexes/`.
2. Register it in `factory.py`: `register_vector_index("new_algo", NewAlgoClass)`.
3. Pass `index_type="new_algo"` in `FAISSVectorStore` constructor.

---

## 3. Metadata Over-Fetching & Filtering Algorithm

- **Over-fetching**: When `filter_metadata` is passed to `search()`, the vector store over-fetches candidates ($5 \times \text{top\_k}$) from the FAISS index.
- **Filtering**: Iterates candidates through `_matches_filter()`, validating metadata key-value pairs before constructing the final `SearchResult` list.
- **Disk Persistence**: Saves binary vector index to `index.faiss` and JSON metadata payload to `payloads.json`.
