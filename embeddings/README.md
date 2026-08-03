# Embeddings Engine Documentation

The `embeddings` module converts text blocks, Markdown tables, and raw image assets into a joint high-dimensional vector space using **CLIP** (`clip-ViT-B-32`) for cross-modal similarity search.

---

## 1. Class Defined

### `Embedder`
True Multimodal Embedding Generator Interface.

* **Initialization Parameters**:
  * `model_name` (`str`, default `"clip-ViT-B-32"`): HuggingFace SentenceTransformer CLIP model ID (Text & Vision dual-encoder).
  * `device` (`Optional[str]`, default `None`): Auto-detects optimal compute device (`"cuda"`, `"mps"`, or `"cpu"`).
  * `dimension` (`int`, default `512`): Output joint vector space dimension.
* **Methods**:
  * `embed_text(text: str) -> List[float]`: Embeds a single text string via CLIP Text Encoder into a 512d vector.
  * `embed_texts(texts: List[str]) -> List[List[float]]`: Batch embeds a list of text strings.
  * `embed_image(image_path: Union[str, Path]) -> List[float]`: Embeds raw image pixels via CLIP Vision Transformer (ViT) into the same 512d joint vector space.
  * `embed_images(image_paths: List[Union[str, Path]]) -> List[List[float]]`: Batch embeds multiple image assets using PIL.
  * `embed_chunk(text: str, metadata: Dict[str, Any]) -> List[float]`: Auto-routes chunk embedding based on metadata (`image` vs `text`).
