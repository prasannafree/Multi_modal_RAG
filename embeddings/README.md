# Embeddings Engine Documentation

The `embeddings` module converts text blocks, Markdown tables, and image assets into high-dimensional vector embeddings for similarity search.

---

## 1. Class Defined

### `Embedder`
Unified text and multimodal embedding generator interface.

* **Initialization Parameters**:
  * `model_name` (`str`, default `"all-MiniLM-L6-v2"`): HuggingFace SentenceTransformer model ID.
  * `device` (`str`, default `"cpu"`): `"cpu"` or `"cuda"`.
  * `dimension` (`int`, default `384`): Output vector dimension.
* **Methods**:
  * `embed_text(text: str) -> List[float]`: Embeds a single text string into a normalized floating-point vector.
  * `embed_texts(texts: List[str]) -> List[List[float]]`: Batch embeds a list of strings.
  * `embed_image(image_path: str) -> List[float]`: Embeds an image asset into the vector space.
