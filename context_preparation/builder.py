"""
Context Preparation Module for Multimodal RAG.

Formats retrieved & reranked SearchResult objects (text, tables, image paths)
into a structured prompt context and resolved image asset payload for VLM generation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from vector_store import SearchResult


@dataclass
class PreparedContext:
    """
    Prepared context payload ready for Multimodal LLM / VLM generation.

    Attributes:
        formatted_text: Structured text context string formatted with headers & metadata.
        image_assets: List of existing image file paths to pass to Vision Encoders.
        sources: List of source attribution dictionaries (filename, page, element_type).
    """
    formatted_text: str
    image_assets: List[Path] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)


class ContextPreparation:
    """
    Context Preparation Engine for structuring Multimodal RAG prompts.
    """

    def __init__(
        self,
        max_context_length: int = 4000,
        include_metadata_header: bool = True,
    ):
        self.max_context_length = max_context_length
        self.include_metadata_header = include_metadata_header

    def prepare(
        self,
        query: str,
        results: List[SearchResult],
    ) -> PreparedContext:
        """
        Structures reranked SearchResult candidates into a unified PreparedContext payload.

        Args:
            query: User search query string.
            results: Top reranked SearchResult objects from Stage 2.

        Returns:
            PreparedContext containing formatted prompt text, image asset paths, and sources.
        """
        context_blocks: List[str] = []
        image_assets: List[Path] = []
        sources: List[Dict[str, Any]] = []

        for idx, res in enumerate(results, start=1):
            meta = res.metadata or {}
            source_file = meta.get("source", "Document")
            page_num = meta.get("page", 1)
            element_type = meta.get("element_type", "text")
            chunk_id = res.chunk_id or f"chunk_{idx}"

            # Track sources for attribution
            sources.append({
                "chunk_id": chunk_id,
                "source": source_file,
                "page": page_num,
                "element_type": element_type,
                "score": res.score,
            })

            # Track image file paths for VLM visual input
            image_path_str = meta.get("image_path") or (meta.get("source") if element_type == "image" else None)
            if image_path_str:
                path_obj = Path(image_path_str)
                if path_obj.exists() and path_obj not in image_assets:
                    image_assets.append(path_obj)

            # Build formatted text block header
            if self.include_metadata_header:
                header = f"--- [Context Block #{idx} | Source: {Path(source_file).name} (Page {page_num}) | Type: {element_type.upper()} | ID: {chunk_id}] ---"
                block = f"{header}\n{res.text.strip()}"
            else:
                block = res.text.strip()

            context_blocks.append(block)

        formatted_text = "\n\n".join(context_blocks)

        # Truncate if exceeds max context length limit
        if len(formatted_text) > self.max_context_length:
            formatted_text = formatted_text[:self.max_context_length] + "\n...[Context Truncated]"

        return PreparedContext(
            formatted_text=formatted_text,
            image_assets=image_assets,
            sources=sources,
        )
