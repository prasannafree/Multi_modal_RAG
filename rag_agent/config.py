"""
RAG Sub-Agent Configuration Parser.

Reads rag_agent_config.xml and exposes all settings as a typed Python dataclass.
Uses only stdlib xml.etree.ElementTree — zero external dependencies.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass
class ServerConfig:
    """FastAPI server host and port settings."""
    host: str = "0.0.0.0"
    port: int = 8100


@dataclass
class OllamaConfig:
    """Ollama local LLM settings."""
    model: str = "qwen3.6:27b"
    url: str = "http://localhost:11434"


@dataclass
class GeminiConfig:
    """Google Gemini cloud API settings."""
    model: str = "gemini-2.0-flash"
    api_key: str = ""


@dataclass
class LLMConfig:
    """LLM provider selection and provider-specific settings."""
    provider: str = "ollama"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)


@dataclass
class RetrievalConfig:
    """Vector search and reranking tuning parameters."""
    metric: str = "cosine"
    index_type: str = "hnsw"
    stage1_pool_size: int = 30
    stage2_top_n: int = 3


@dataclass
class DocumentsConfig:
    """Document bucket directory setting."""
    bucket_dir: str = "bucket"


@dataclass
class RAGAgentConfig:
    """
    Top-level configuration dataclass for the RAG Sub-Agent.

    Populated by parsing rag_agent_config.xml via load_config().
    """
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    documents: DocumentsConfig = field(default_factory=DocumentsConfig)


def _text(parent: ET.Element, tag: str, default: str = "") -> str:
    """Safely extracts text content from an XML child element."""
    el = parent.find(tag)
    if el is not None and el.text is not None:
        return el.text.strip()
    return default


def load_config(config_path: str | Path | None = None) -> RAGAgentConfig:
    """
    Parses the XML configuration file and returns a RAGAgentConfig dataclass.

    Resolution order for config file path:
      1. Explicit `config_path` argument.
      2. RAG_AGENT_CONFIG environment variable.
      3. Default: <project_root>/rag_agent_config.xml

    Returns:
        RAGAgentConfig with all settings populated from XML (or defaults).
    """
    # Resolve config file path
    if config_path is None:
        config_path = os.getenv("RAG_AGENT_CONFIG")
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "rag_agent_config.xml"

    config_path = Path(config_path)
    cfg = RAGAgentConfig()

    if not config_path.exists():
        print(f"[RAG Agent Config] Config file not found at '{config_path}'. Using defaults.")
        return cfg

    print(f"[RAG Agent Config] Loading configuration from: {config_path}")
    tree = ET.parse(config_path)
    root = tree.getroot()

    # ---- Server ----
    server_el = root.find("server")
    if server_el is not None:
        cfg.server.host = _text(server_el, "host", cfg.server.host)
        port_str = _text(server_el, "port", str(cfg.server.port))
        cfg.server.port = int(port_str)

    # ---- LLM ----
    llm_el = root.find("llm")
    if llm_el is not None:
        cfg.llm.provider = _text(llm_el, "provider", cfg.llm.provider)

        ollama_el = llm_el.find("ollama")
        if ollama_el is not None:
            cfg.llm.ollama.model = _text(ollama_el, "model", cfg.llm.ollama.model)
            cfg.llm.ollama.url = _text(ollama_el, "url", cfg.llm.ollama.url)

        gemini_el = llm_el.find("gemini")
        if gemini_el is not None:
            cfg.llm.gemini.model = _text(gemini_el, "model", cfg.llm.gemini.model)
            xml_api_key = _text(gemini_el, "api_key", "")
            # Prefer XML key, fallback to env var
            cfg.llm.gemini.api_key = xml_api_key or os.getenv("GEMINI_API_KEY", "")

    # ---- Retrieval ----
    retrieval_el = root.find("retrieval")
    if retrieval_el is not None:
        cfg.retrieval.metric = _text(retrieval_el, "metric", cfg.retrieval.metric)
        cfg.retrieval.index_type = _text(retrieval_el, "index_type", cfg.retrieval.index_type)
        pool_str = _text(retrieval_el, "stage1_pool_size", str(cfg.retrieval.stage1_pool_size))
        cfg.retrieval.stage1_pool_size = int(pool_str)
        top_n_str = _text(retrieval_el, "stage2_top_n", str(cfg.retrieval.stage2_top_n))
        cfg.retrieval.stage2_top_n = int(top_n_str)

    # ---- Documents ----
    docs_el = root.find("documents")
    if docs_el is not None:
        cfg.documents.bucket_dir = _text(docs_el, "bucket_dir", cfg.documents.bucket_dir)

    return cfg
