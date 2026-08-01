# Multimodal RAG Chunker & Metadata Processing Documentation

The `chunker` module handles Step 2 of the RAG Data Pipeline (**Multimodal Chunking & Metadata Processing**). It converts raw parsed `Document` items into optimized, metadata-enriched `Chunk` objects ready for embedding generation and vector database indexing.

---

## 1. Classes Defined

### `Chunk`
A dataclass representing a single RAG chunk enriched with metadata.

* **Attributes**:
  * `chunk_id` (`str`): Unique reproducible identifier (e.g. `document.pdf_p1_c001_a1b2c3d4`).
  * `text` (`str`): Chunk text payload (narrative text, Markdown table snippet, or image summary).
  * `metadata` (`Dict[str, Any]`): Enriched metadata containing `source`, `page`, `element_type`, `layout_order`, `global_chunk_index`, `char_count`, `word_count`, and `chunk_strategy`.
* **Methods**:
  * `to_dict()`: Converts the chunk object into a plain Python dictionary.

---

### `MultimodalChunker`
Main chunking engine supporting configurable boundary-based and context-based strategies.

* **Initialization Parameters**:
  * `chunk_size` (`int`, default `500`): Maximum target character size per chunk.
  * `chunk_overlap` (`int`, default `50`): Overlap character buffer between adjacent text chunks.
  * `text_strategy` (`str`, default `"recursive"`): Text strategy (`"recursive"`, `"sliding_window"`, `"fixed"`).
  * `table_strategy` (`str`, default `"atomic"`): Table strategy (`"atomic"`, `"row_wise"`).
  * `image_strategy` (`str`, default `"multimodal"`): Image strategy (`"multimodal"`).
  * `separators` (`Optional[List[str]]`, default `["\n\n", "\n", ". ", " ", ""]`): Hierarchical separators for recursive text splitting.

---

## 2. Strategies Implemented (Matching Workflow Chart)

### A. Boundary-Based Chunking
1. **Recursive Chunking ⭐⭐⭐** (`text_strategy="recursive"`):
   * Tries hierarchical separators in order (`\n\n` $\rightarrow$ `\n` $\rightarrow$ `. ` $\rightarrow$ ` `).
   * Respects natural sentence and paragraph boundaries without cutting words or sentences mid-way.
2. **Sliding Window Chunking** (`text_strategy="sliding_window"`):
   * Chunks text into fixed-size windows with sliding overlap to preserve boundary context.
3. **Fixed Chunking** (`text_strategy="fixed"`):
   * Hard character-count boundary splits.

### B. Context-Based / Element-Aware Chunking
1. **Adaptive Table Chunking** (`table_strategy="atomic"` or `"row_wise"`):
   * **Atomic**: Keeps the entire Markdown table intact as a single chunk to preserve column headers and row relationships.
   * **Row-Wise with Header Injection**: For giant tables exceeding target size, splits table row-by-row while dynamically prepending the table header row to every chunk.
2. **Multimodal Image Chunking** (`image_strategy="multimodal"`):
   * Preserves image file path, resolution, format, and visual description as an atomic multimodal reference chunk.

---

## 3. Main Functions

### `chunk_documents()`
Convenience helper function to chunk a list of documents.

* **Inputs**:
  * `documents` (`Union[List[Document], List[Dict[str, Any]]]`): Parsed document list.
  * `chunk_size` (`int`, default `500`).
  * `chunk_overlap` (`int`, default `50`).
  * `text_strategy` (`str`, default `"recursive"`).
  * `table_strategy` (`str`, default `"atomic"`).
  * `to_dict` (`bool`, default `False`).
* **Outputs**:
  * `Union[List[Chunk], List[Dict[str, Any]]]`: List of enriched RAG chunks.

---

## 4. Steps Followed in Chunking Pipeline

1. **Input Normalization**: Accepts list of `Document` objects or dictionaries.
2. **Element Routing**: Inspects `doc.metadata["element_type"]`:
   * Text / Headings $\rightarrow$ `_chunk_text()` (Recursive / Sliding Window / Fixed).
   * Tables $\rightarrow$ `_chunk_table()` (Atomic Markdown / Row-wise header injection).
   * Images $\rightarrow$ `_chunk_image()` (Atomic Multimodal Asset Chunk).
3. **Metadata Enrichment**:
   * Calculates `char_count` and `word_count`.
   * Assigns `global_chunk_index` sequence number.
   * Generates reproducible `chunk_id` hash combining filename, page, sequence number, and content snippet.
4. **Return Enriched Chunks**: Ready for embedding generation and vector database storage.
