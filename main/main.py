"""
Main Multimodal RAG Application Entrypoint.

Supports two modes:
  1. Single-shot query:  uv run python main/main.py -q "your question"
  2. Interactive chat:   uv run python main/main.py --chat

Pipeline:
1. Ingests all documents & images from the 'bucket/' repository folder.
2. Performs Multimodal Parsing & Element-Aware Chunking.
3. Generates 512d CLIP embeddings & indexes into FAISS Vector Store.
4. Stage 1 Vector Search Over-fetching (Configurable pool size).
5. Stage 2 Cross-Encoder Reranking (Configurable selected top-N).
6. Context Preparation (Formated headers & resolved image assets).
7. Multimodal LLM Generation (Switchable between Gemini API, Local Ollama VLM, and Offline Fallback).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project root directory to python import path
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
from similarity_metrics import compute_similarity
from vector_store import FAISSVectorStore


# ---- Phase 1: Ingest & Index (runs once) ----

def build_index(
    bucket_dir: Union[str, Path] = None,
    selected_metric: str = "cosine",
    selected_index_type: str = "hnsw",
):
    """
    Parses, chunks, embeds, and indexes all documents in the bucket directory.
    Returns reusable components for querying.
    """
    # Resolve document bucket directory
    if bucket_dir is None:
        bucket_dir = PROJECT_ROOT / "bucket"
    bucket_path = Path(bucket_dir)
    bucket_path.mkdir(parents=True, exist_ok=True)

    print(f"[Bucket Ingest] Scanning document repository: '{bucket_path}'")
    supported_extensions = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}
    files_to_parse = [f for f in bucket_path.glob("*") if f.suffix.lower() in supported_extensions]

    if not files_to_parse:
        print(f"[Notice] No documents found in '{bucket_path}'. Creating sample document...")
        sample_file = bucket_path / "sample_doc.md"
        sample_file.write_text("# Sample RAG Document\n\nPDF parser extracts tables as Markdown.", encoding="utf-8")
        files_to_parse = [sample_file]

    print(f"Found {len(files_to_parse)} file(s) in bucket for processing:\n" + "\n".join(f" - {f.name}" for f in files_to_parse) + "\n")

    # Step 1: Parse Documents from Bucket
    print("--- Step 1: Multimodal Document Parsing ---")
    all_documents = []
    output_img_dir = bucket_path / "extracted_images"
    for file_p in files_to_parse:
        docs = parse_file(file_p, output_image_dir=output_img_dir)
        if isinstance(docs, list):
            all_documents.extend(docs)
        else:
            all_documents.append(docs)
    print(f"Extracted {len(all_documents)} total structural element(s) across all bucket files.\n")

    # Step 2: Multimodal Chunking
    print("--- Step 2: Multimodal Chunking & Metadata Processing ---")
    chunks = chunk_documents(all_documents, chunk_size=400, chunk_overlap=40)
    print(f"Generated {len(chunks)} enriched RAG chunk(s).\n")

    # Step 3: Embeddings Generation & FAISS Indexing
    print(f"--- Step 3: Embedding Generation & FAISS Vector Indexing (Metric: '{selected_metric}', Index: '{selected_index_type}') ---")
    embedder = Embedder()
    embeddings = [embedder.embed_chunk(c.text, c.metadata) for c in chunks]

    vector_store = FAISSVectorStore(
        dimension=embedder.dimension,
        metric=selected_metric,
        index_type=selected_index_type,
    )
    vector_store.add_chunks(chunks, embeddings)
    print(f"Indexed {len(chunks)} chunk(s) into FAISS (Vector Dim: {embedder.dimension}).\n")

    return embedder, vector_store


# ---- Phase 2: Query (runs per question) ----

def answer_query(
    query: str,
    embedder: Embedder,
    vector_store: FAISSVectorStore,
    generator: MultimodalGenerator,
    reranker: Reranker,
    stage1_pool_size: int = 10,
    stage2_selected_top_n: int = 3,
    verbose: bool = False,
    chat_history: List[Dict[str, str]] = None,
) -> str:
    """
    Runs retrieval, reranking, context preparation, and generation for a single query.
    Returns the generated answer string.
    """
    # Create a context-aware search query to fix retrieval for follow-up questions
    search_query = query
    if chat_history:
        last_user_msg = next((msg["content"] for msg in reversed(chat_history) if msg["role"] == "user"), "")
        if last_user_msg:
            search_query = f"{last_user_msg} | {query}"
            if verbose:
                print(f"  [Contextual Search] Augmented Query: '{search_query}'")

    # Rewrite the query into a hypothetical image caption to dramatically improve CLIP's vector matching
    rewritten_query = generator.rewrite_query(search_query)
    if verbose and rewritten_query != search_query:
        print(f"  [Query Rewriter] '{search_query}' -> '{rewritten_query}'")

    # Stage 1: Dual Vector Search 
    # (1) Search with the original query (best for finding text documents like your .md file)
    query_vec_text = embedder.embed_text(search_query)
    candidates = vector_store.search(query_vec_text, top_k=stage1_pool_size)
    
    # (2) Search with the rewritten query (best for finding images like the car)
    if rewritten_query != search_query:
        query_vec_img = embedder.embed_text(rewritten_query)
        candidates_img = vector_store.search(query_vec_img, top_k=stage1_pool_size)
        # Merge and deduplicate
        seen_ids = {c.chunk_id for c in candidates}
        for c in candidates_img:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                candidates.append(c)
    if verbose:
        print(f"  [Retrieval] {len(candidates)} candidates from vector search")

    # Stage 2: Cross-Encoder Reranking
    # The CrossEncoder is a text-only model. If we pass image chunks to it, it will score them poorly
    # and drop them because their text is just a placeholder (e.g., "[Image File: images.jpeg]").
    # To fix this, we separate images, rerank the text, and then add highly-ranked FAISS images back in.
    image_candidates = [c for c in candidates if c.metadata.get("element_type") == "image"]
    text_candidates = [c for c in candidates if c.metadata.get("element_type") != "image"]

    reranked_results = reranker.rerank(search_query, text_candidates, top_n=stage2_selected_top_n)

    # Inject images back into the results. Since CLIP (FAISS) is sometimes bad at matching 
    # conversational questions (like "what car do i have") to images, we pass all image candidates 
    # retrieved in Stage 1 to the LLM. Vision models like Gemini are extremely smart and will 
    # simply ignore the irrelevant images (like the bedroom) and find the correct one (the car).
    top_faiss_images = image_candidates
    for img in reversed(top_faiss_images):
        reranked_results.insert(0, img)

    # Ensure we don't exceed the top_n limit after injecting images
    reranked_results = reranked_results[:stage2_selected_top_n]

    if verbose:
        print(f"  [Reranking] Top {len(reranked_results)} results selected (including images)")
        for idx, res in enumerate(reranked_results, start=1):
            meta = res.metadata or {}
            print(f"    #{idx} [Score: {res.score:.4f}] {Path(meta.get('source', 'doc')).name} | {res.text[:80]}...")

    # Context Preparation
    context_prep = ContextPreparation(max_context_length=4000, include_metadata_header=True)
    prepared_context = context_prep.prepare(search_query, reranked_results)

    # LLM Generation
    # We use a hacky patch here to pass chat history without breaking other providers for now
    if hasattr(generator, "_generate_ollama") and generator.provider == "ollama":
        answer = generator._generate_ollama(query, prepared_context, chat_history=chat_history)
    else:
        answer = generator.generate(query, prepared_context)
    return answer



# ---- Single-Shot Pipeline (original behavior) ----

def run_pipeline(
    query: str = "How does the multimodal PDF parser handle tables?",
    bucket_dir: Union[str, Path] = None,
    generator_provider: str = "fallback",  # 'gemini' | 'ollama' | 'fallback'
    gemini_api_key: Optional[str] = None,
    ollama_model: str = "llama3.1:8b",
    gemini_model: str = "gemini-2.0-flash",
    selected_metric: str = "cosine",        # 'cosine' | 'dot_product' | 'l2' | 'l1'
    selected_index_type: str = "hnsw",      # 'flat' | 'hnsw' | 'ivf'
    stage1_pool_size: int = 10,
    stage2_selected_top_n: int = 3,
):
    print("==================================================================")
    print("       MULTIMODAL RAG COMPLETE END-TO-END EXECUTION SYSTEM       ")
    print("==================================================================\n")

    embedder, vector_store = build_index(bucket_dir, selected_metric, selected_index_type)

    # Route model name based on provider
    if generator_provider == "gemini":
        model_name = gemini_model
    elif generator_provider == "ollama":
        model_name = ollama_model
    else:
        model_name = None

    generator = MultimodalGenerator(
        provider=generator_provider,
        model_name=model_name,
        api_key=gemini_api_key,
    )
    reranker = Reranker()

    final_answer = answer_query(
        query, embedder, vector_store, generator, reranker,
        stage1_pool_size, stage2_selected_top_n, verbose=True,
    )

    print("\n==================================================================")
    print("                     FINAL GROUNDED AI RESPONSE                  ")
    print("==================================================================")
    print(final_answer)
    print("==================================================================\n")
    return final_answer


# ---- Interactive Chat Mode ----

def run_chat(
    bucket_dir: Union[str, Path] = None,
    generator_provider: str = "ollama",
    gemini_api_key: Optional[str] = None,
    ollama_model: str = "llama3.1:8b",
    gemini_model: str = "gemini-2.0-flash",
    selected_metric: str = "cosine",
    selected_index_type: str = "hnsw",
    stage1_pool_size: int = 10,
    stage2_selected_top_n: int = 3,
    verbose: bool = False,
):
    """
    Interactive RAG chat loop.
    Indexes documents once, then enters a conversation where each question
    is answered by the LLM using retrieved & reranked document context.
    """
    print("==================================================================")
    print("          MULTIMODAL RAG — INTERACTIVE CHAT MODE                 ")
    print("==================================================================\n")

    # Phase 1: Build index once
    embedder, vector_store = build_index(bucket_dir, selected_metric, selected_index_type)

    # Route model name based on provider
    if generator_provider == "gemini":
        model_name = gemini_model
    elif generator_provider == "ollama":
        model_name = ollama_model
    else:
        model_name = None

    # Initialize generator and reranker once
    generator = MultimodalGenerator(
        provider=generator_provider,
        model_name=model_name,
        api_key=gemini_api_key,
    )
    reranker = Reranker()

    # Warm up models with a dummy query so first real answer is fast
    print("Warming up models...")
    _ = answer_query(
        "warmup", embedder, vector_store, generator, reranker,
        stage1_pool_size, stage2_selected_top_n, verbose=False,
    )

    print("\n==================================================================")
    print(f"  Ready! Provider: {generator_provider.upper()} | Model: {model_name or 'fallback'}")
    print("  Type your questions below. Type 'quit' or 'exit' to stop.")
    print("==================================================================\n")

    chat_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\nGoodbye!")
            break

        answer = answer_query(
            user_input, embedder, vector_store, generator, reranker,
            stage1_pool_size, stage2_selected_top_n, verbose=verbose,
            chat_history=chat_history
        )
        print(f"\nAssistant: {answer}\n")
        
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline & Generation Engine")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="How does the multimodal PDF parser handle tables?",
        help="Search query string (single-shot mode)"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        default=False,
        help="Enter interactive chat mode (indexes once, then you ask questions in a loop)"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="ollama",
        choices=["gemini", "ollama", "fallback"],
        help="LLM Generator Provider ('gemini' | 'ollama' | 'fallback')"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3.1:8b",
        help="Ollama Local Model Name (e.g., 'llama3.1:8b', 'qwen3:8b', 'gemma3:4b')"
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash",
        help="Google Gemini Model Name (e.g., 'gemini-2.0-flash', 'gemini-2.5-pro')"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "dot_product", "l2", "l1"],
        help="Similarity metric ('cosine' | 'dot_product' | 'l2' | 'l1')"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="hnsw",
        choices=["flat", "hnsw", "ivf"],
        help="FAISS Index algorithm ('flat' | 'hnsw' | 'ivf')"
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=30,
        help="Stage 1 candidate pool size"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Stage 2 selected top-N count after reranking"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show retrieval & reranking details per query"
    )

    args = parser.parse_args()

    if args.chat:
        run_chat(
            generator_provider=args.provider,
            ollama_model=args.model,
            gemini_model=args.gemini_model,
            selected_metric=args.metric,
            selected_index_type=args.index,
            stage1_pool_size=args.pool_size,
            stage2_selected_top_n=args.top_n,
            verbose=args.verbose,
        )
    else:
        run_pipeline(
            query=args.query,
            generator_provider=args.provider,
            ollama_model=args.model,
            gemini_model=args.gemini_model,
            selected_metric=args.metric,
            selected_index_type=args.index,
            stage1_pool_size=args.pool_size,
            stage2_selected_top_n=args.top_n,
        )
""" 
Multimodal RAG Application Entrypoint.

In-and-Out Execution Pipeline:
1. Ingests all documents & images from the 'bucket/' repository folder.
2. Performs Multimodal Parsing & Element-Aware Chunking.
3. Generates 512d CLIP embeddings & indexes into FAISS Vector Store.
4. Stage 1 Vector Search Over-fetching (Configurable pool size).
5. Stage 2 Cross-Encoder Reranking (Configurable selected top-N).
6. Context Preparation (Formated headers & resolved image assets).
7. Multimodal LLM Generation (Switchable between Gemini API, Local Ollama VLM, and Offline Fallback).
"""
