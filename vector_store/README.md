# FAISS Vector Database Store Documentation

The `vector_store` module manages Stage 2 (**Retrieval System**) vector indexing, metric calculations (Cosine Similarity, Dot Product, L2 Distance), metadata filtering, and FAISS index persistence.

---

## 1. Classes Defined

### `SearchResult`
Dataclass returned by similarity search queries.

* **Attributes**:
  * `chunk_id` (`str`): ID of retrieved chunk.
  * `score` (`float`): Similarity score.
  * `text` (`str`): Chunk text payload.
  * `metadata` (`Dict[str, Any]`): Enriched chunk metadata.
* **Methods**:
  * `to_dict()`: Converts result to dictionary.

---

### `FAISSVectorStore`
FAISS vector database wrapper with metadata payload storage and metadata filtering.

* **Initialization Parameters**:
  * `dimension` (`int`, default `384`): Vector embedding dimension.
  * `metric` (`str`, default `"cosine"`): Similarity metric (`"cosine"`, `"dot_product"`, `"l2"`).
* **Methods**:
  * `add_chunks(chunks: List[Chunk], embeddings: List[List[float]])`: Adds chunks and embedding vectors to FAISS index.
  * `search(query_vector: List[float], top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]`: Performs vector similarity search with optional metadata filtering.
  * `save(dir_path: str)`: Persists FAISS index (`index.faiss`) and payload (`payloads.json`) to disk.
  * `load(dir_path: str)`: Loads persisted FAISS index and metadata payloads from disk.

---

## 2. Steps Followed in Vector Store Operations

1. **Indexing**: Converts list of vectors into NumPy `float32` arrays, applies Cosine L2-normalization if enabled, and inserts into FAISS `IndexFlatIP` or `IndexFlatL2`.
2. **Similarity Querying**: Computes nearest neighbor distances between query vector and indexed chunk vectors.
3. **Metadata Filtering**: Filters search candidate results matching key-value pairs (e.g. `{"element_type": "table"}`).
4. **Disk Persistence**: Writes binary FAISS index file and JSON payload dictionary.
