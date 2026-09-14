"""
ColPali Vector Store Wrapper (via Byaldi)

This module provides a wrapper around the Byaldi ColPali engine, allowing
late-interaction MaxSim search over document images.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from byaldi import RAGMultiModalModel

@dataclass
class ColPaliSearchResult:
    doc_id: int
    score: float
    page_num: int
    metadata: Dict[str, Any]
    base64_image: str


class ColPaliStore:
    def __init__(self, index_name: str = "colpali_index", model_name: str = "vidore/colpali-v1.2"):
        self.index_name = index_name
        self.model_name = model_name
        self.rag = None

    def _init_rag(self):
        if self.rag is None:
            print(f"[ColPaliStore] Loading ColPali model: {self.model_name}")
            # Suppress verbose output if possible
            self.rag = RAGMultiModalModel.from_pretrained(self.model_name)

    def build_index(self, bucket_dir: str | Path):
        """
        Indexes images and PDFs from the bucket directory.
        Ignores text/markdown files which Byaldi does not support natively.
        """
        self._init_rag()
        bucket_path = Path(bucket_dir)
        
        # Byaldi crashes if it encounters .txt or .md in the directory.
        # We create a clean staging directory with only supported files.
        staging_dir = Path(".colpali_staging")
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        
        supported_exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
        files_indexed = 0
        
        for file in bucket_path.iterdir():
            if file.is_file() and file.suffix.lower() in supported_exts:
                # Copy file to staging
                shutil.copy2(file, staging_dir / file.name)
                files_indexed += 1
                
        if files_indexed == 0:
            print("[ColPaliStore] No supported image/PDF files found in bucket.")
            return

        print(f"[ColPaliStore] Indexing {files_indexed} files using ColPali MaxSim...")
        self.rag.index(
            input_path=str(staging_dir),
            index_name=self.index_name,
            store_collection_with_index=True,
            overwrite=True
        )
        print("[ColPaliStore] Indexing complete.")
        
        # Cleanup staging
        shutil.rmtree(staging_dir)

    def load_index(self):
        """Loads an existing index from disk (.byaldi/index_name)."""
        index_path = Path(".byaldi") / self.index_name
        if index_path.exists():
            print(f"[ColPaliStore] Loading existing index '{self.index_name}'...")
            self.rag = RAGMultiModalModel.from_index(self.index_name)
            return True
        return False

    def search(self, query: str, top_k: int = 3) -> List[ColPaliSearchResult]:
        """
        Performs late-interaction search over the ColPali index.
        """
        if self.rag is None:
            if not self.load_index():
                raise ValueError("ColPali index not found. Call build_index() first.")

        print(f"[ColPaliStore] Searching for '{query}'...")
        results = self.rag.search(query, k=top_k)
        
        formatted_results = []
        for res in results:
            # Byaldi Result provides doc_id, score, page_num, base64
            metadata = res.metadata if hasattr(res, 'metadata') else {}
            
            formatted_results.append(ColPaliSearchResult(
                doc_id=res.doc_id,
                score=res.score,
                page_num=res.page_num,
                metadata=metadata,
                base64_image=res.base64
            ))
            
        return formatted_results
