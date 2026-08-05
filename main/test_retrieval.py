import sys
from pathlib import Path

# Add project root directory to python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from main import build_index
from reranker import Reranker
from embeddings import Embedder

def test_retrieval():
    print("==================================================================")
    print("               PURE RETRIEVAL DIAGNOSTIC TOOL                     ")
    print("==================================================================")
    print("Building FAISS Index...")
    
    # 1. Build the index
    embedder, vector_store = build_index(
        bucket_dir=PROJECT_ROOT / "bucket", 
        selected_metric="cosine", 
        selected_index_type="hnsw"
    )
    
    # 2. Load the Reranker
    print("Loading CrossEncoder Reranker...")
    reranker = Reranker()
    
    print("\n--- System Ready ---")
    
    while True:
        query = input("\nEnter query (or 'exit' to quit): ").strip()
        if not query or query.lower() in ["exit", "quit"]:
            break
            
        print(f"\nSearching for: '{query}'")
        
        # --- Step 1: FAISS Vector Retrieval (CLIP) ---
        pool_size = 30
        query_vec = embedder.embed_text(query)
        candidates = vector_store.search(query_vec, top_k=pool_size)
        
        image_candidates = [c for c in candidates if c.metadata.get("element_type") == "image"]
        text_candidates = [c for c in candidates if c.metadata.get("element_type") != "image"]
        
        print("\n=======================================================")
        print(f" [STAGE 1: FAISS / CLIP] Retrieved {len(candidates)} total chunks")
        print("=======================================================")
        
        print(f"\n--> 🖼️ FAISS Image Results ({len(image_candidates)} found):")
        if not image_candidates:
            print("    (No images retrieved)")
        for i, c in enumerate(image_candidates):
            filename = c.metadata.get('source', 'Unknown')
            print(f"    [{i+1}] {filename}")
            
        print(f"\n--> 📄 FAISS Text Results (Top 5 of {len(text_candidates)} shown):")
        for i, c in enumerate(text_candidates[:5]):
            text_preview = c.text.strip().replace('\n', ' ')[:100]
            filename = c.metadata.get('source', 'Unknown')
            print(f"    [{i+1}] [{filename}] {text_preview}...")
            
        # --- Step 2: Cross-Encoder Reranking (Text Only) ---
        top_n = 5
        reranked_results = reranker.rerank(query, text_candidates, top_n=top_n)
        
        print("\n=======================================================")
        print(f" [STAGE 2: CROSS-ENCODER] Top {top_n} Reranked Text Chunks")
        print("=======================================================")
        for i, c in enumerate(reranked_results):
            text_preview = c.text.strip().replace('\n', ' ')[:150]
            filename = c.metadata.get('source', 'Unknown')
            print(f"    [{i+1}] [{filename}] {text_preview}...")

if __name__ == "__main__":
    test_retrieval()
