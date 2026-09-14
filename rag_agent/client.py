"""
RAG Sub-Agent — Lightweight Python Client.

Provides a simple RAGClient class so any main agent can call the RAG
sub-agent server with a single method call.

Usage:
    from rag_agent.client import RAGClient

    client = RAGClient()                          # default: http://localhost:8100
    response = client.query("How does the parser handle tables?")
    print(response["answer"])
    print(response["sources"])
"""

import json
import urllib.request
from typing import Any, Dict, Optional


class RAGClient:
    """
    HTTP client for the RAG Sub-Agent server.

    Attributes:
        base_url: Base URL of the RAG sub-agent server (default: http://localhost:8100).
    """

    def __init__(self, base_url: str = "http://localhost:8100"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        """
        Checks server health and returns index status.

        Returns:
            Dict with keys: status, indexed_chunks, provider, model.
        """
        url = f"{self.base_url}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Sends a query to the RAG sub-agent and returns the structured response.

        Args:
            query: The search question to answer.
            top_k: Override Stage 2 top-N results count.
            verbose: Print retrieval details on the server console.

        Returns:
            Dict with keys: answer, sources, images, query.
        """
        url = f"{self.base_url}/query"
        payload: Dict[str, Any] = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k
        if verbose:
            payload["verbose"] = True

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def ingest(self) -> Dict[str, Any]:
        """
        Triggers re-ingestion of documents on the server.

        Returns:
            Dict with keys: status, chunks_indexed.
        """
        url = f"{self.base_url}/ingest"
        req = urllib.request.Request(
            url,
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def __repr__(self) -> str:
        return f"RAGClient(base_url='{self.base_url}')"
