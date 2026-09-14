"""
RAG Sub-Agent — FastAPI HTTP Server.

Exposes the Multimodal RAG pipeline as a local HTTP API that any main agent
can call to retrieve grounded, source-attributed answers.

Endpoints:
    GET  /health  — Health check and index status.
    POST /query   — Submit a query, receive a structured JSON answer.
    POST /ingest  — Re-ingest documents from the bucket directory.

Start the server:
    uv run python -m rag_agent.server
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from .config import load_config
from .agent import RAGAgent


# ---- Pydantic Request / Response Models ----

class QueryRequest(BaseModel):
    """Request body for POST /query."""
    query: str = Field(..., description="The search question to answer.")
    top_k: Optional[int] = Field(None, description="Override Stage 2 top-N results count.")
    verbose: bool = Field(False, description="Print retrieval details to server console.")


class QueryResponse(BaseModel):
    """Response body for POST /query."""
    answer: str
    sources: List[Dict[str, Any]]
    images: List[str]
    query: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    indexed_chunks: int
    provider: str
    model: str


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""
    status: str
    chunks_indexed: int


# ---- Application Factory ----

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application with the RAG agent."""
    config = load_config()

    app = FastAPI(
        title="RAG Sub-Agent",
        description="Multimodal RAG Sub-Agent HTTP API. Send queries, receive grounded answers.",
        version="1.0.0",
    )

    # Allow all origins for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize the RAG Agent with XML config
    agent = RAGAgent(config=config)

    @app.on_event("startup")
    async def startup_ingest():
        """Auto-ingest documents from the bucket on server startup."""
        print("\n" + "=" * 60)
        print("   RAG SUB-AGENT — Starting Document Ingestion")
        print("=" * 60)
        agent.ingest()

    # ---- Endpoints ----

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Returns server health status and index metadata."""
        llm_cfg = config.llm
        if llm_cfg.provider == "ollama":
            model_name = llm_cfg.ollama.model
        elif llm_cfg.provider == "gemini":
            model_name = llm_cfg.gemini.model
        else:
            model_name = "fallback-synthesizer"

        return HealthResponse(
            status="ok" if agent.is_ready else "not_ready",
            indexed_chunks=agent.chunk_count,
            provider=llm_cfg.provider,
            model=model_name,
        )

    @app.post("/query", response_model=QueryResponse)
    async def handle_query(request: QueryRequest):
        """
        Main query endpoint.
        Runs the full RAG pipeline: vector search → rerank → generate.
        """
        if not agent.is_ready:
            raise HTTPException(
                status_code=503,
                detail="RAG Agent not ready. Documents have not been ingested yet.",
            )

        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        response = agent.query(
            query=request.query,
            top_k=request.top_k,
            verbose=request.verbose,
        )

        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            images=response.images,
            query=response.query,
        )

    @app.post("/ingest", response_model=IngestResponse)
    async def handle_ingest():
        """Re-ingests documents from the bucket directory."""
        chunk_count = agent.ingest()
        return IngestResponse(
            status="ok",
            chunks_indexed=chunk_count,
        )

    return app


# ---- Entrypoint ----

app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = load_config()
    print("\n" + "=" * 60)
    print("   RAG SUB-AGENT SERVER")
    print(f"   Listening on: http://{config.server.host}:{config.server.port}")
    print(f"   Provider: {config.llm.provider.upper()} | Model: {config.llm.ollama.model if config.llm.provider == 'ollama' else config.llm.gemini.model}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "rag_agent.server:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )
