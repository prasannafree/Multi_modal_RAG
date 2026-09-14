"""
RAG Sub-Agent Core Engine.

Wraps the existing Multimodal RAG pipeline (build_index, answer_query) into a
clean, reusable RAGAgent class that can be driven by the FastAPI server or
used standalone.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so all existing modules resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from chunker import chunk_documents
from context_preparation import ContextPreparation
from embeddings import Embedder
from generator import MultimodalGenerator
from parser import parse_file
from reranker import Reranker
from vector_store import FAISSVectorStore

from .config import RAGAgentConfig, load_config


@dataclass
class RAGResponse:
    """
    Structured response returned by the RAG Sub-Agent.

    Attributes:
        answer: Generated answer text from the LLM.
        sources: List of source attribution dicts (source file, page, element_type, score).
        images: List of image file paths that were used as visual context.
        query: The original query string.
    """
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "images": self.images,
            "query": self.query,
        }


class RAGAgent:
    """
    Multimodal RAG Sub-Agent.

    Encapsulates the full RAG pipeline (ingest → embed → index → search →
    rerank → generate) behind a single .query() method.

    Usage:
        agent = RAGAgent()          # loads config from rag_agent_config.xml
        agent.ingest()              # indexes documents from the bucket
        response = agent.query("How does the PDF parser handle tables?")
        print(response.answer)
    """

    def __init__(self, config: Optional[RAGAgentConfig] = None):
        self.config = config or load_config()
        self._embedder: Optional[Embedder] = None
        self._vector_store: Optional[FAISSVectorStore] = None
        self._generator: Optional[MultimodalGenerator] = None
        self._reranker: Optional[Reranker] = None
        self._indexed = False
        self._chunk_count = 0

    # ---- Ingestion ----

    def ingest(self, bucket_dir: Optional[str] = None) -> int:
        """
        Parses, chunks, embeds, and indexes all documents in the bucket directory.

        Args:
            bucket_dir: Override bucket directory path (defaults to config value).

        Returns:
            Number of chunks indexed.
        """
        # Resolve bucket directory
        if bucket_dir:
            bucket_path = Path(bucket_dir)
        else:
            bucket_path = PROJECT_ROOT / self.config.documents.bucket_dir
        bucket_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[RAG Agent] Ingesting documents from: {bucket_path}")
        supported_extensions = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}
        files_to_parse = [f for f in bucket_path.glob("*") if f.suffix.lower() in supported_extensions]

        if not files_to_parse:
            print(f"[RAG Agent] No documents found in '{bucket_path}'.")
            self._indexed = True
            self._chunk_count = 0
            return 0

        print(f"[RAG Agent] Found {len(files_to_parse)} file(s):")
        for f in files_to_parse:
            print(f"  - {f.name}")

        # Step 1: Parse
        all_documents = []
        output_img_dir = bucket_path / "extracted_images"
        for file_p in files_to_parse:
            docs = parse_file(file_p, output_image_dir=output_img_dir)
            if isinstance(docs, list):
                all_documents.extend(docs)
            else:
                all_documents.append(docs)
        print(f"[RAG Agent] Extracted {len(all_documents)} structural element(s).")

        # Step 2: Chunk
        chunks = chunk_documents(all_documents, chunk_size=400, chunk_overlap=40)
        print(f"[RAG Agent] Generated {len(chunks)} chunk(s).")

        # Step 3: Embed & Index
        cfg = self.config.retrieval
        self._embedder = Embedder()
        embeddings = [self._embedder.embed_chunk(c.text, c.metadata) for c in chunks]

        self._vector_store = FAISSVectorStore(
            dimension=self._embedder.dimension,
            metric=cfg.metric,
            index_type=cfg.index_type,
        )
        self._vector_store.add_chunks(chunks, embeddings)
        self._chunk_count = len(chunks)
        print(f"[RAG Agent] Indexed {self._chunk_count} chunk(s) into FAISS ({cfg.index_type.upper()}, {cfg.metric}).")

        # Step 4: Initialize Generator & Reranker
        self._init_generator()
        self._reranker = Reranker()

        self._indexed = True
        print("[RAG Agent] Ingestion complete. Ready to accept queries.\n")
        return self._chunk_count

    def _init_generator(self):
        """Initializes the multimodal generator based on XML config."""
        llm_cfg = self.config.llm
        provider = llm_cfg.provider

        if provider == "gemini":
            model_name = llm_cfg.gemini.model
            api_key = llm_cfg.gemini.api_key or None
        elif provider == "ollama":
            model_name = llm_cfg.ollama.model
            api_key = None
        else:
            model_name = None
            api_key = None

        self._generator = MultimodalGenerator(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            ollama_url=llm_cfg.ollama.url if provider == "ollama" else "http://localhost:11434",
        )

    # ---- Query ----

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        verbose: bool = False,
    ) -> RAGResponse:
        """
        Runs the full RAG pipeline for a single query and returns a structured response.

        Args:
            query: User search query string.
            top_k: Override Stage 2 top-N (defaults to config value).
            verbose: Print retrieval and reranking details.

        Returns:
            RAGResponse with answer, sources, and image paths.
        """
        if not self._indexed:
            return RAGResponse(
                answer="[RAG Agent Error] No documents indexed. Call /ingest first.",
                query=query,
            )

        cfg = self.config.retrieval
        stage1_pool = cfg.stage1_pool_size
        stage2_top = top_k if top_k is not None else cfg.stage2_top_n

        # Stage 1: Dual Vector Search (original + rewritten query for images)
        rewritten_query = self._generator.rewrite_query(query)
        if verbose and rewritten_query != query:
            print(f"  [Query Rewriter] '{query}' -> '{rewritten_query}'")

        query_vec_text = self._embedder.embed_text(query)
        candidates = self._vector_store.search(query_vec_text, top_k=stage1_pool)

        if rewritten_query != query:
            query_vec_img = self._embedder.embed_text(rewritten_query)
            candidates_img = self._vector_store.search(query_vec_img, top_k=stage1_pool)
            seen_ids = {c.chunk_id for c in candidates}
            for c in candidates_img:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    candidates.append(c)

        if verbose:
            print(f"  [Retrieval] {len(candidates)} candidates from vector search")

        # Stage 2: Cross-Encoder Reranking (text candidates only)
        image_candidates = [c for c in candidates if c.metadata.get("element_type") == "image"]
        text_candidates = [c for c in candidates if c.metadata.get("element_type") != "image"]

        reranked_results = self._reranker.rerank(query, text_candidates, top_n=stage2_top)

        # Inject images back in (VLMs can handle irrelevant images gracefully)
        for img in reversed(image_candidates):
            reranked_results.insert(0, img)
        reranked_results = reranked_results[:stage2_top]

        if verbose:
            print(f"  [Reranking] Top {len(reranked_results)} results selected")
            for idx, res in enumerate(reranked_results, start=1):
                meta = res.metadata or {}
                print(f"    #{idx} [Score: {res.score:.4f}] {Path(meta.get('source', 'doc')).name} | {res.text[:80]}...")

        # Context Preparation
        context_prep = ContextPreparation(max_context_length=4000, include_metadata_header=True)
        prepared_context = context_prep.prepare(query, reranked_results)

        # LLM Generation
        if hasattr(self._generator, "_generate_ollama") and self._generator.provider == "ollama":
            answer = self._generator._generate_ollama(query, prepared_context)
        else:
            answer = self._generator.generate(query, prepared_context)

        # Build structured response
        return RAGResponse(
            answer=answer,
            sources=prepared_context.sources,
            images=[str(p) for p in prepared_context.image_assets],
            query=query,
        )

    # ---- Status ----

    @property
    def is_ready(self) -> bool:
        """Whether the agent has indexed documents and is ready to accept queries."""
        return self._indexed

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently indexed."""
        return self._chunk_count
