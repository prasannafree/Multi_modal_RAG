"""
Embeddings Engine for Multimodal RAG.

Uses CLIP (clip-ViT-B-32) to encode both text and images into a shared 512-dimensional
vector space, enabling true cross-modal retrieval (e.g., text query → image result).
Fallbacks gracefully if models are downloading or operating offline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


def _detect_device() -> str:    # check the availability of cuda or other GPU devices for embedding process
    """Auto-detects optimal compute device (cuda, mps, cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Embedder:
    """
    True Multimodal Embedder using CLIP (clip-ViT-B-32). encoder only model 

    Encodes text and images into the SAME joint 512-dimensional vector space,
    enabling cross-modal similarity search (text ↔ image retrieval).

    - Text  → CLIP Text Encoder  → 512d vector
    - Image → CLIP Vision Encoder (ViT) → 512d vector
    - Both vectors are directly comparable via cosine similarity.
    """

    """
    modularity - text , tables , images
    granularity -> text ( 50 - 60 words)
    training objective  -> self-supervised contrastive learning ( infoNCE loss)
    alignment -> cross-modal alignment  ( dual projection heads)
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",    # CLIP multimodal model (text + vision encoders)
        device: Optional[str] = None,          # default device for embedding process
        dimension: int = 512,):                # dimension of the CLIP joint embedding space

        self.model_name = model_name
        self.device = device or _detect_device()
        self.dimension = dimension
        self._model = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy loads the CLIP embedding model on first use."""
        if self._initialized:    # prevents reloading the model every time you embed resource
            return

        try:  # dynamic library import
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)   # initialize the CLIP model
            self.dimension = self._model.get_sentence_embedding_dimension()          # getting the dimension of the CLIP embeddings (512)
            self._initialized = True
            print(f"[Embedder] Loaded CLIP model '{self.model_name}' on device '{self.device}' (dim={self.dimension})")
        except Exception as e:
            print(f"[Notice] Could not load CLIP model '{self.model_name}': {e}")
            print("[Notice] Using normalized deterministic vector generator fallback.")
            self._model = None
            self._initialized = True

    # ---- Text Embedding (via CLIP Text Encoder) ----

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into the CLIP joint vector space."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of strings into normalized CLIP vector embeddings."""
        self._lazy_init()

        if not texts:
            return []

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()

        # Deterministic fallback vector generator for offline / lightweight use
        return [self._fallback_vector(t) for t in texts]

    # ---- Image Embedding (via CLIP Vision Encoder / ViT) ----

    def embed_image(self, image_path: Union[str, Path]) -> List[float]:
        """
        Embeds an image into the CLIP joint vector space using raw pixel data.

        CLIP's Vision Transformer (ViT) encodes the actual image pixels,
        placing the resulting vector in the same space as text embeddings.
        This enables true cross-modal retrieval: a text query like
        "bar chart showing revenue" can match an image of that chart.
        """
        return self.embed_images([image_path])[0]

    def embed_images(self, image_paths: List[Union[str, Path]]) -> List[List[float]]:
        """
        Batch embeds multiple images into the CLIP joint vector space.

        Each image is loaded as a PIL.Image and encoded through CLIP's
        Vision Transformer (ViT) into a normalized 512d vector.
        """
        self._lazy_init()

        if not image_paths:
            return []

        # Validate all paths exist
        resolved_paths = []
        for p in image_paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at: {p}")
            resolved_paths.append(path)

        if self._model is not None:
            from PIL import Image as PILImage

            # Load raw PIL images — CLIP's SentenceTransformer wrapper accepts PIL.Image objects
            pil_images = [PILImage.open(p).convert("RGB") for p in resolved_paths]

            # CLIP Vision Encoder: raw pixels → 512d normalized vector
            embeddings = self._model.encode(
                pil_images,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()

        # Deterministic fallback for offline / lightweight testing
        return [self._fallback_vector(f"image_{Path(p).name}") for p in resolved_paths]

    # ---- Multimodal Chunk Embedding (auto-routes text vs image) ----

    def embed_chunk(self, text: str, metadata: Dict[str, Any]) -> List[float]:
        """
        Auto-routes embedding based on chunk element_type metadata.

        - element_type == 'image' and 'image_path' in metadata → CLIP Vision Encoder (pixel embedding)
        - everything else → CLIP Text Encoder
        """
        element_type = metadata.get("element_type", "text")
        image_path = metadata.get("image_path")

        if element_type == "image" and image_path and Path(image_path).exists():
            return self.embed_image(image_path)

        return self.embed_text(text)

    # ---- Fallback ----

    def _fallback_vector(self, text: str) -> List[float]:
        """Generates a unit-normalized pseudo-embedding vector for offline testing."""
        rng = np.random.RandomState(abs(hash(text)) % (2**31 - 1))
        vec = rng.randn(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
