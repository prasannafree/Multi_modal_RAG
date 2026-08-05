# Context Preparation Engine Documentation

The `context_preparation` module bridges **Retrieval (Stage 1 & Stage 2)** and **Generation (VLM/LLM)**. It structures top-ranked text chunks, Markdown tables, and extracted image file paths into a clean prompt context payload.

---

## 1. Algorithms & Methods Used

### A. Document Source Header Injection
- **Method**: Constructs metadata headers for each chunk block (`--- [Context Block #X | Source: file.pdf (Page P) | Type: TYPE | ID: chunk_id] ---`).
- **Purpose**: Informs the VLM exactly where each context snippet originated, enabling accurate in-line source attribution in the AI's final response.

### B. Multimodal Asset Path Resolution
- **Method**: Inspects metadata dictionaries for `image_path` entries and resolves them to verified `Path` objects (`Path.exists()`).
- **Purpose**: Collects raw image files (diagrams, charts, figures) for direct visual input into Vision Transformers (e.g. Gemini 1.5 Vision / LLaVA).

### C. Context Length Guardrail & Truncation
- **Method**: Calculates cumulative context string length against `max_context_length` (default `4000` chars).
- **Purpose**: Prevents prompt token overflow exceptions when passing retrieved context to LLM context windows.

---

## 2. Classes & Data Structures

### `PreparedContext` Dataclass
* **Attributes**:
  * `formatted_text` (`str`): Clean, structured context prompt text with metadata headers.
  * `image_assets` (`List[Path]`): Validated image file paths ready for VLM vision encoders.
  * `sources` (`List[Dict[str, Any]]`): Source attribution dictionaries for citation tracking.

### `ContextPreparation` Class
* **Initialization Parameters**:
  * `max_context_length` (`int`, default `4000`): Maximum context character limit.
  * `include_metadata_header` (`bool`, default `True`): Toggles metadata header injection.
* **Methods**:
  * `prepare(query: str, results: List[SearchResult]) -> PreparedContext`: Transforms search results into a `PreparedContext` payload.
