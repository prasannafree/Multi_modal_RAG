# Multimodal Generator Module Documentation

The `generator` module handles Step 7 (**Generative System**) of the RAG pipeline. It passes retrieved text context and raw image assets to a **Vision-Language Model (VLM)** to produce grounded, factual answers with source citations.

---

## 1. Supported Model Providers & Switchable Architecture

### A. Google Gemini API (`provider="gemini"`)
- **Library**: `google-genai` / `google.generativeai`.
- **Default Model**: `gemini-1.5-flash` or `gemini-1.5-pro`.
- **Functionality**: Passes user query, formatted text context, and raw `PIL.Image` objects to Gemini's multimodal vision API.

### B. Local Ollama VLM (`provider="ollama"`)
- **Library**: Standard `urllib` HTTP REST API (`http://localhost:11434/api/generate`).
- **Default Models**: `llama3.2-vision` or `llava`.
- **Functionality**: Encodes image files as base64 strings and submits prompt JSON to a local Ollama server running offline on your GPU/CPU.

### C. Offline Fallback Synthesizer (`provider="fallback"`)
- Zero-dependency offline answer generator. Synthesizes a structured response summarizing the top context blocks and listed sources without requiring API keys or running servers.

---

## 2. Class Defined

### `MultimodalGenerator`

* **Initialization Parameters**:
  * `provider` (`str`, default `"fallback"`): Generator backend (`"gemini"`, `"ollama"`, `"fallback"`).
  * `model_name` (`Optional[str]`, default `None`): Target model string (e.g. `"gemini-1.5-flash"` or `"llama3.2-vision"`).
  * `api_key` (`Optional[str]`, default `None`): Gemini API key (or reads `GEMINI_API_KEY` env var).
  * `ollama_url` (`str`, default `"http://localhost:11434"`): Local Ollama server endpoint.
* **Methods**:
  * `generate(query: str, prepared_context: PreparedContext) -> str`: Generates grounded answer with source citations.

---

## 3. Switchable Configuration Example

```python
from generator import MultimodalGenerator

# Switch to Google Gemini API
generator = MultimodalGenerator(provider="gemini", model_name="gemini-1.5-flash", api_key="YOUR_API_KEY")

# Switch to Local Ollama VLM (Offline)
generator_local = MultimodalGenerator(provider="ollama", model_name="llama3.2-vision")

# Generate response
answer = generator.generate(query="Explain the table handling logic", prepared_context=prepared_context)
print(answer)
```
