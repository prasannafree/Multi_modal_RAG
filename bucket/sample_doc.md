# Multimodal RAG Pipeline Documentation

This document describes the core architecture and capabilities of the Multimodal Retrieval-Augmented Generation (RAG) system.

## Document Parsing

The parser module supports multiple file formats including PDF, DOCX, TXT, Markdown, and standalone images (PNG, JPG, WEBP). For PDF files, PyMuPDF (fitz) is used to extract three types of structural elements: text blocks, tables (converted to Markdown format), and embedded images. Each extracted element preserves its spatial bounding box coordinates and page number, enabling layout-aware processing downstream.

Tables are extracted using PyMuPDF's built-in table finder and converted to clean Markdown table syntax with proper header rows and separator lines. Text blocks that overlap with detected table regions are automatically excluded to prevent duplicate content.

## Chunking Strategy

The chunker implements element-aware splitting strategies tailored to each content type. Text and headings support recursive, sliding window, and fixed-size chunking. Tables use either atomic preservation (keeping the full table intact) or row-wise splitting with automatic header injection for oversized tables. Images are preserved as atomic multimodal chunks with their file path metadata.

Each chunk is assigned a unique hash-based identifier, character count, word count, and global sequence index for reproducible retrieval.

## Embedding Engine

The system uses CLIP (clip-ViT-B-32) from the sentence-transformers library to encode both text and images into a shared 512-dimensional vector space. This enables true cross-modal retrieval where a text query like "revenue chart" can match an image of a bar chart. The embedder auto-detects the best available compute device (CUDA, MPS, or CPU) and includes a deterministic fallback vector generator for offline testing.

## Vector Store and Indexing

FAISS powers the vector database with three configurable index algorithms: Flat (exact brute-force with 100% recall), HNSW (graph-based approximate nearest neighbor with O(log N) search), and IVF (Voronoi cluster-partitioned search). The store supports cosine similarity, dot product, L2 euclidean distance, and L1 manhattan distance metrics.

## Reranking

Stage 2 reranking uses a Cross-Encoder model (ms-marco-MiniLM-L-6-v2) that jointly evaluates query-candidate pairs through cross-attention. This produces more precise relevance scores than the bi-encoder Stage 1 search, but at higher computational cost. A heuristic term-overlap fallback is available for offline use.

## Generation

The generator supports three providers: Google Gemini API for cloud-based multimodal generation, local Ollama VLM for on-premise inference with vision models like llama3.2-vision, and an offline fallback synthesizer that structures retrieved context into a formatted summary with source citations.

## my personal info 
my name is prasanna , i am from tamilnadu 
