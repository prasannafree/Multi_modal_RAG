"""
Allows running the RAG Sub-Agent server via:
    uv run python -m rag_agent.server
"""

from .server import app
from .config import load_config

if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(
        "rag_agent.server:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )
