# Embeddings Engine Documentation

The `embeddings` module converts text blocks, Markdown tables, and raw image assets into a joint 512-dimensional vector space using **CLIP (`clip-ViT-B-32`)** for cross-modal similarity search.

---

## 1. Underlying Model & Algorithms Used

### A. Dual-Encoder Architecture (CLIP - OpenAI / HuggingFace)
- **Model ID**: `clip-ViT-B-32` via `sentence-transformers`.
- **Text Encoder**: A Transformer Encoder that converts text strings into a 512-dimensional vector (77 BPE token context limit).
- **Vision Encoder (ViT-B/32)**: A Vision Transformer that resizes input images to $224 \times 224$ pixels, splits them into a $7 \times 7$ grid of **$32 \times 32$ pixel patches**, and passes them through self-attention layers to produce a 512-dimensional vector.

### B. Self-Supervised Contrastive Learning (InfoNCE Loss)
- Pre-trained on **400 Million (Image, Text)** web pairs.
- Maximizes the cosine similarity between matching image-text pairs while minimizing similarity for non-matching pairs over an $N \times N$ matrix.

### C. Joint Vision-Language Alignment
- Both encoders project outputs through linear projection heads into the **same $L_2$-normalized 512-dimensional hypersphere**.
- Enables direct cross-modal retrieval (e.g. text query matching raw image pixels in FAISS).

### D. Automatic Hardware Device Detection (`_detect_device`)
- Automatically detects and routes computation to PyTorch `cuda` (NVIDIA GPUs), `mps` (Apple Silicon M1/M2/M3), or `cpu`.

### E. Deterministic Fallback Generator (`_fallback_vector`)
- Uses a seeded pseudo-random generator `np.random.RandomState(abs(hash(text)))` for offline/testing environments without internet access to model weights.

---

## 2. Class Defined

### `Embedder`
True Multimodal Embedding Generator Interface.

* **Initialization Parameters**:
  * `model_name` (`str`, default `"clip-ViT-B-32"`): CLIP model ID.
  * `device` (`Optional[str]`, default `None`): Auto-detects compute device (`"cuda"`, `"mps"`, `"cpu"`).
  * `dimension` (`int`, default `512`): Joint vector space output dimension.
* **Methods**:
  * `embed_text(text: str) -> List[float]`: Embeds a text string via CLIP Text Encoder into a 512d normalized vector.
  * `embed_texts(texts: List[str]) -> List[List[float]]`: Batch embeds text strings.
  * `embed_image(image_path: Union[str, Path]) -> List[float]`: Embeds raw image pixels via CLIP Vision Transformer (ViT).
  * `embed_images(image_paths: List[Union[str, Path]]) -> List[List[float]]`: Batch embeds raw image files.
  * `embed_chunk(text: str, metadata: Dict[str, Any]) -> List[float]`: Auto-routes chunk embedding based on `element_type` metadata (`image` vs `text`).
