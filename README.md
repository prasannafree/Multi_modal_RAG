# Multi-Modal RAG Pipeline

A modular, end-to-end **Multimodal Retrieval-Augmented Generation** system that processes documents (PDF, DOCX, TXT, Markdown) and images, then answers natural language questions using retrieved context grounded in your own data.

## Architecture

The pipeline follows a 7-stage architecture:

```
Documents & Images (bucket/)
        │
        ▼
┌─────────────────────┐
│  1. PARSER           │  PyMuPDF, python-docx, PIL, Tesseract OCR
│     PDF / DOCX / TXT │  → Extracts text blocks, Markdown tables,
│     MD / Images      │    embedded images with spatial layout
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  2. CHUNKER          │  Element-aware splitting strategies:
│     Text: Recursive  │  • Recursive / Sliding Window / Fixed
│     Table: Atomic    │  • Atomic table / Row-wise with header injection
│     Image: Multimodal│  • Preserves image metadata as atomic chunks
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  3. EMBEDDER         │  CLIP (clip-ViT-B-32) — Shared 512d vector space
│     Text → CLIP Text │  enabling true cross-modal retrieval:
│     Image → CLIP ViT │  text query ↔ image result
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  4. VECTOR STORE     │  FAISS with configurable index algorithms:
│     Flat / HNSW / IVF│  • Cosine Similarity / Dot Product / L2 / L1
│     + NumPy fallback │  • Metadata filtering & persistence
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  5. RERANKER         │  Stage 2 Cross-Encoder (ms-marco-MiniLM-L-6-v2)
│     Cross-Attention  │  Joint query-candidate evaluation for precise
│     Rescoring        │  relevance scoring over Stage 1 candidates
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  6. CONTEXT PREP     │  Formats text + resolves image assets into
│     Structured Prompt│  structured LLM prompt with source attribution
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  7. GENERATOR        │  Switchable multimodal LLM providers:
│     Gemini API       │  • Google Gemini 2.0 Flash (cloud)
│     Ollama VLM       │  • Local Ollama (llama3.1, qwen3, etc.)
│     Offline Fallback │  • Zero-dependency offline synthesizer
└─────────────────────┘
```

## Key Methods & Strategies

### Parsing Strategy
- **PDF**: PyMuPDF extracts text blocks, tables (→ Markdown), and embedded images with bounding box coordinates. Text overlapping table regions is automatically excluded to prevent duplication.
- **DOCX**: python-docx walks the document XML tree in reading order, extracting paragraphs, tables, and embedded images from relationship parts.
- **Images**: PIL reads metadata (resolution, format, mode). Tesseract OCR extracts any visible text from standalone images and embedded PDF/DOCX images.
- **TXT/MD**: Split by paragraph (`\n\n`) or markdown section headers.

### Chunking Strategy
- **Text**: Recursive hierarchical splitting (default) using separator hierarchy: `\n\n` → `\n` → `. ` → ` ` → `""`. Configurable chunk size and overlap.
- **Tables**: Atomic preservation (default) keeps the full Markdown table intact. Row-wise splitting injects the header into each chunk for oversized tables.
- **Images**: Preserved as atomic multimodal chunks with metadata (path, resolution, format).
- Each chunk gets a unique hash-based ID, character/word counts, and a global sequence index.

### Embedding Strategy
- **CLIP (clip-ViT-B-32)**: Dual-encoder model that maps both text and images into the same 512-dimensional vector space using contrastive learning (InfoNCE loss).
- Text → CLIP Text Encoder → 512d vector
- Image → CLIP Vision Encoder (ViT) → 512d vector
- Enables **cross-modal retrieval**: a text query like "revenue chart" can match an image of a bar chart.

### Retrieval Strategy
- **Stage 1 — Vector Search**: FAISS over-fetches candidates (configurable pool size, default 30). Supports Flat (exact), HNSW (graph-based ANN), and IVF (cluster-partitioned) indexes.
- **Stage 2 — Cross-Encoder Reranking**: ms-marco-MiniLM-L-6-v2 jointly evaluates query-candidate pairs through cross-attention for precise relevance scoring.
- **HyDE Query Rewriting**: Conversational queries are rewritten into descriptive image captions using the LLM to improve CLIP's vector matching accuracy.
- **Dual Search**: Both the original and rewritten queries are searched independently, then results are merged and deduplicated.

### Generation Strategy
- **Gemini**: Sends structured prompt with text context + PIL images to Google Gemini 2.0 Flash API.
- **Ollama**: Sends context + base64-encoded images to local Ollama server. Supports both VLM (vision) and text-only models.
- **Fallback**: Zero-dependency offline synthesizer that structures retrieved context into a formatted summary with source citations.

## Installation

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Tesseract OCR: `sudo apt install tesseract-ocr` (Linux) or `brew install tesseract` (macOS)
- (Optional) NVIDIA GPU with CUDA for accelerated embedding/reranking

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd multi_modal_RAG

# Install dependencies
uv sync

# Set up your API key (for Gemini provider)
echo 'GEMINI_API_KEY="your-api-key-here"' > .env
```

## Usage

### 1. Add Documents to the Bucket

Place your files (PDF, DOCX, TXT, Markdown, PNG, JPG, WEBP) into the `bucket/` directory. The pipeline will automatically ingest all supported files.

### 2. Single-Shot Query

```bash
# Using offline fallback (no API keys needed)
uv run python main/main.py -q "How does the PDF parser handle tables?" -p fallback

# Using Google Gemini API
uv run python main/main.py -q "Describe the images" -p gemini

# Using local Ollama
uv run python main/main.py -q "What is in this document?" -p ollama -m llama3.1:8b

# With verbose retrieval details
uv run python main/main.py -q "Your question" -p fallback -v
```

### 3. Interactive Chat Mode

```bash
# Chat with Ollama (default)
uv run python main/main.py --chat -p ollama -m llama3.1:8b

# Chat with Gemini
uv run python main/main.py --chat -p gemini

# Chat with verbose retrieval logging
uv run python main/main.py --chat -p ollama -v
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-q`, `--query` | `"How does the multimodal PDF parser handle tables?"` | Search query (single-shot mode) |
| `--chat` | `false` | Enter interactive chat mode |
| `-p`, `--provider` | `ollama` | LLM provider: `gemini`, `ollama`, or `fallback` |
| `-m`, `--model` | `llama3.1:8b` | Ollama model name |
| `--gemini-model` | `gemini-2.0-flash` | Gemini model name |
| `--metric` | `cosine` | Similarity metric: `cosine`, `dot_product`, `l2`, `l1` |
| `--index` | `hnsw` | FAISS index: `flat`, `hnsw`, `ivf` |
| `--pool-size` | `30` | Stage 1 candidate pool size |
| `--top-n` | `3` | Stage 2 selected top-N after reranking |
| `-v`, `--verbose` | `false` | Show retrieval & reranking details |

## Project Structure

```
multi_modal_RAG/
├── bucket/                     # Document repository (your input files)
├── parser/                     # Document parsing (PDF, DOCX, TXT, MD, Images)
│   ├── parser.py               #   Multimodal parser with OCR support
│   └── __init__.py
├── chunker/                    # Element-aware chunking
│   ├── chunker.py              #   Recursive, sliding window, fixed, table, image strategies
│   └── __init__.py
├── embeddings/                 # CLIP embedding engine
│   ├── embedder.py             #   Text + Image → shared 512d vector space
│   └── __init__.py
├── vector_store/               # FAISS vector database
│   ├── store.py                #   Search, metadata filtering, persistence
│   ├── indexes/                #   Pluggable index algorithms
│   │   ├── base.py             #     Abstract base interface
│   │   ├── flat_index.py       #     Exact brute-force (IndexFlatIP/L2)
│   │   ├── hnsw_index.py       #     Graph-based ANN (IndexHNSWFlat)
│   │   ├── ivf_index.py        #     Cluster-partitioned (IndexIVFFlat)
│   │   └── factory.py          #     Registry & factory pattern
│   └── __init__.py
├── similarity_metrics/         # Cosine, Dot Product, L2, L1 implementations
│   ├── metrics.py
│   └── __init__.py
├── reranker/                   # Cross-Encoder Stage 2 reranking
│   ├── reranker.py
│   └── __init__.py
├── context_preparation/        # Prompt formatting & image asset resolution
│   ├── builder.py
│   └── __init__.py
├── generator/                  # Multimodal LLM generation (Gemini/Ollama/Fallback)
│   ├── generator.py
│   └── __init__.py
├── main/                       # Application entrypoint
│   ├── main.py                 #   Pipeline orchestration, CLI, chat mode
│   ├── test_retrieval.py       #   Retrieval diagnostic tool
│   └── __init__.py
├── pyproject.toml              # Project dependencies & build config
├── .env                        # API keys (not committed to git)
└── README.md
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `pymupdf` | PDF parsing (text, tables, images) |
| `pillow` | Image loading & processing |
| `python-docx` | Word document parsing |
| `faiss-cpu` | Vector indexing & similarity search |
| `sentence-transformers` | CLIP embeddings & Cross-Encoder reranking |
| `torch` | PyTorch backend for models |
| `numpy` | Vector math & array operations |
| `python-dotenv` | Environment variable loading |
| `google-generativeai` | Google Gemini API (legacy SDK) |
| `google-genai` | Google Gemini API (new SDK) |
| `pytesseract` | OCR text extraction from images |

## License

This project is for educational and research purposes.
