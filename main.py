from pathlib import Path
from parser import parse_file
from chunker import chunk_documents
from embeddings import Embedder
from vector_store import FAISSVectorStore


def main():
    print("=== Multimodal RAG Complete Data & Retrieval Pipeline ===")
    print("Step 1: Document Parsing")
    print("Step 2: Multimodal Chunking & Metadata Processing")
    print("Step 3: Embeddings Generation & FAISS Vector Indexing")
    print("Step 4: Vector Similarity Search & Retrieval\n")

    sample_doc = "README.md"
    if not Path(sample_doc).exists():
        print(f"No '{sample_doc}' found.")
        return

    # Step 1: Parse Document
    print(f"--- 1. Parsing Document ('{sample_doc}') ---")
    documents = parse_file(sample_doc)
    print(f"Extracted {len(documents)} structural elements.\n")

    # Step 2: Chunk Documents
    print("--- 2. Multimodal Chunking ---")
    chunks = chunk_documents(documents, chunk_size=400, chunk_overlap=40)
    print(f"Generated {len(chunks)} RAG chunks.\n")

    # Step 3: Embeddings & FAISS Indexing
    print("--- 3. Embedding Chunks & Indexing into FAISS Vector Store ---")
    embedder = Embedder()
    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(chunk_texts)

    vector_store = FAISSVectorStore(dimension=embedder.dimension, metric="cosine")
    vector_store.add_chunks(chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks into FAISS (Vector Dimension: {embedder.dimension}).\n")

    # Step 4: Vector Similarity Query
    query = "How does the multimodal PDF parser handle tables?"
    print(f"--- 4. Querying FAISS Vector Store ---")
    print(f"User Query: '{query}'")

    query_vec = embedder.embed_text(query)
    results = vector_store.search(query_vec, top_k=3)

    print(f"\nRetrieved Top {len(results)} Matches:")
    for idx, res in enumerate(results, start=1):
        print(f"\nMatch #{idx} [Score: {res.score:.4f}] | Chunk ID: {res.chunk_id}")
        print(f"  Element Type: {res.metadata.get('element_type')} | Source: {res.metadata.get('source')}")
        print(f"  Content: {res.text[:120]}...")


if __name__ == "__main__":
    main()
