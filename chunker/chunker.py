"""
Multimodal RAG Chunker & Metadata Processing Module.

Implements boundary-based and context-based chunking strategies for:
- Text & Headings (Recursive, Sliding Window, Fixed)
- Tables (Adaptive Atomic & Row-Wise with Header Injection)
- Images (Multimodal Asset & Visual Summary Chunking)
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from parser import Document


@dataclass
class Chunk:
    """
    Standardized RAG Chunk representation enriched with search metadata.

    Attributes:
        chunk_id: Unique hash-based or indexed identifier.
        text: The text content of the chunk (narrative text, Markdown table, image summary).
        metadata: Enriched context dictionary (source, page, element_type, chunk_index, char_count, etc.).
    """
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }


class MultimodalChunker:
    """
    Element-Aware & Layout-Aware Multimodal Chunker.
    """
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        text_strategy: str = "recursive",  # "recursive" | "sliding_window" | "fixed"
        table_strategy: str = "atomic",     # "atomic" | "row_wise"
        image_strategy: str = "multimodal", # "multimodal"
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        self.text_strategy = text_strategy.lower()
        self.table_strategy = table_strategy.lower()
        self.image_strategy = image_strategy.lower()
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_documents(
        self,
        documents: Union[List[Document], List[Dict[str, Any]]],
        to_dict: bool = False,
    ) -> Union[List[Chunk], List[Dict[str, Any]]]:
        """
        Processes a list of parsed Document objects and splits them into enriched RAG Chunks.
        """
        # Convert dicts back to Document if passed as dicts
        doc_objects: List[Document] = []
        for d in documents:
            if isinstance(d, dict):
                doc_objects.append(Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {})))
            else:
                doc_objects.append(d)

        chunks: List[Chunk] = []
        global_chunk_idx = 1

        for doc in doc_objects:
            element_type = doc.metadata.get("element_type", "text")

            if element_type in ["text", "heading", "header"]:
                produced = self._chunk_text(doc)
            elif element_type == "table":
                produced = self._chunk_table(doc)
            elif element_type == "image":
                produced = self._chunk_image(doc)
            else:
                produced = self._chunk_text(doc)

            for chunk_item in produced:
                # Assign global chunk sequence & unique chunk_id
                chunk_item.metadata["global_chunk_index"] = global_chunk_idx
                chunk_item.chunk_id = self._generate_chunk_id(chunk_item, global_chunk_idx)
                chunks.append(chunk_item)
                global_chunk_idx += 1

        if to_dict:
            return [c.to_dict() for c in chunks]
        return chunks

    def _chunk_text(self, doc: Document) -> List[Chunk]:
        """Chunks narrative text or headings using configured strategy."""
        text = doc.page_content.strip()
        if not text:
            return []

        # If text is already smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [self._create_chunk(text, doc, sub_idx=1)]

        if self.text_strategy == "recursive":
            splits = self._recursive_split(text, self.chunk_size, self.chunk_overlap, self.separators)
        elif self.text_strategy == "sliding_window":
            splits = self._sliding_window_split(text, self.chunk_size, self.chunk_overlap)
        else:  # "fixed"
            splits = self._fixed_split(text, self.chunk_size)

        chunks = []
        for idx, split_text in enumerate(splits, start=1):
            if split_text.strip():
                chunks.append(self._create_chunk(split_text.strip(), doc, sub_idx=idx))

        return chunks

    def _chunk_table(self, doc: Document) -> List[Chunk]:
        """Chunks tables. Preserves atomic Markdown table or splits by rows with header injection."""
        table_md = doc.page_content.strip()
        if not table_md:
            return []

        # Atomic strategy: keep table intact unless strictly exceeding size limit
        if self.table_strategy == "atomic" or len(table_md) <= self.chunk_size:
            return [self._create_chunk(table_md, doc, sub_idx=1)]

        # Row-wise strategy with header injection for oversized tables
        lines = table_md.split("\n")
        if len(lines) <= 2:
            return [self._create_chunk(table_md, doc, sub_idx=1)]

        header_lines = lines[:2]  # Header row and markdown separator row
        header_text = "\n".join(header_lines) + "\n"
        data_rows = lines[2:]

        row_chunks = []
        current_rows = []
        current_len = len(header_text)
        sub_idx = 1

        for row in data_rows:
            row_len = len(row) + 1
            if current_len + row_len > self.chunk_size and current_rows:
                chunk_text = header_text + "\n".join(current_rows)
                row_chunks.append(self._create_chunk(chunk_text, doc, sub_idx=sub_idx))
                sub_idx += 1
                current_rows = [row]
                current_len = len(header_text) + row_len
            else:
                current_rows.append(row)
                current_len += row_len

        if current_rows:
            chunk_text = header_text + "\n".join(current_rows)
            row_chunks.append(self._create_chunk(chunk_text, doc, sub_idx=sub_idx))

        return row_chunks

    def _chunk_image(self, doc: Document) -> List[Chunk]:
        """Preserves image asset details and description as an atomic multimodal chunk."""
        return [self._create_chunk(doc.page_content, doc, sub_idx=1)]

    def _create_chunk(self, text: str, parent_doc: Document, sub_idx: int) -> Chunk:
        """Helper to construct an enriched Chunk object with metadata."""
        meta = dict(parent_doc.metadata)
        meta["char_count"] = len(text)
        meta["word_count"] = len(text.split())
        meta["sub_chunk_index"] = sub_idx
        meta["chunk_strategy"] = self.text_strategy if meta.get("element_type") == "text" else self.table_strategy

        return Chunk(
            chunk_id="",
            text=text,
            metadata=meta,
        )

    def _generate_chunk_id(self, chunk: Chunk, global_idx: int) -> str:
        """Generates a reproducible unique chunk ID."""
        source = chunk.metadata.get("source", "doc")
        file_name = Path(source).name
        page = chunk.metadata.get("page", 1)
        raw_sig = f"{source}_{page}_{global_idx}_{chunk.text[:30]}"
        short_hash = hashlib.md5(raw_sig.encode()).hexdigest()[:8]
        return f"{file_name}_p{page}_c{global_idx:03d}_{short_hash}"

    # --- Splitting Logic Implementations ---

    def _recursive_split(self, text: str, max_size: int, overlap: int, separators: List[str]) -> List[str]:
        """Hierarchical recursive splitter."""
        if len(text) <= max_size:
            return [text]

        # Find first separator that exists in text
        chosen_sep = ""
        for sep in separators:
            if sep in text:
                chosen_sep = sep
                break

        if not chosen_sep:
            return self._fixed_split(text, max_size)

        parts = text.split(chosen_sep)
        final_chunks = []
        current_chunk = []
        current_len = 0

        for part in parts:
            part_str = part + chosen_sep if chosen_sep != "" else part
            if current_len + len(part_str) > max_size and current_chunk:
                joined = "".join(current_chunk).strip()
                if joined:
                    final_chunks.append(joined)
                # Apply overlap by keeping tail of current_chunk
                current_chunk = [current_chunk[-1]] if len(current_chunk) > 1 else []
                current_len = sum(len(p) for p in current_chunk)

            current_chunk.append(part_str)
            current_len += len(part_str)

        if current_chunk:
            joined = "".join(current_chunk).strip()
            if joined:
                final_chunks.append(joined)

        return final_chunks

    def _sliding_window_split(self, text: str, window_size: int, overlap: int) -> List[str]:
        """Sliding window splitter."""
        step = max(1, window_size - overlap)
        chunks = []
        for i in range(0, len(text), step):
            chunk = text[i : i + window_size]
            chunks.append(chunk)
        return chunks

    def _fixed_split(self, text: str, chunk_size: int) -> List[str]:
        """Fixed character size splitter."""
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_documents(
    documents: Union[List[Document], List[Dict[str, Any]]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    text_strategy: str = "recursive",
    table_strategy: str = "atomic",
    to_dict: bool = False,
) -> Union[List[Chunk], List[Dict[str, Any]]]:
    """
    Convenience function to chunk documents using standard settings.
    """
    chunker = MultimodalChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        text_strategy=text_strategy,
        table_strategy=table_strategy,
    )
    return chunker.chunk_documents(documents, to_dict=to_dict)
