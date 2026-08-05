"""
Multimodal LLM / VLM Generator Module.

Supports switchable API and Local Vision-Language Models:
- "gemini"   : Google Gemini 1.5 Flash / Pro API (Google GenAI)
- "ollama"   : Local Ollama VLM (llama3.2-vision, llava)
- "fallback" : Offline zero-dependency response synthesizer
"""

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image as PILImage
from context_preparation import PreparedContext


class MultimodalGenerator:
    """
    Switchable Multimodal LLM Generator Interface.
    """

    def __init__(
        self,
        provider: str = "fallback",  # 'gemini' | 'ollama' | 'fallback'
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.provider = provider.lower().strip()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.ollama_url = ollama_url.rstrip("/")

        # Set default model names based on provider
        if model_name:
            self.model_name = model_name
        elif self.provider == "gemini":
            self.model_name = "gemini-1.5-flash"
        elif self.provider == "ollama":
            self.model_name = "llama3.1:8b"
        else:
            self.model_name = "fallback-synthesizer"

    def generate(self, query: str, prepared_context: PreparedContext) -> str:
        """
        Generates a grounded multimodal answer for the query using the configured provider.

        Args:
            query: User search question string.
            prepared_context: PreparedContext payload from Step 6 containing formatted text and images.

        Returns:
            Generated response string with citations.
        """
        if self.provider == "gemini":
            return self._generate_gemini(query, prepared_context)
        elif self.provider == "ollama":
            return self._generate_ollama(query, prepared_context)
        else:
            return self._generate_fallback(query, prepared_context)

    # ---- Provider 1: Google Gemini API ----

    def _generate_gemini(self, query: str, prepared_context: PreparedContext) -> str:
        """Generates response using Google Gemini API."""
        if not self.api_key:
            print("[Notice] GEMINI_API_KEY not found. Falling back to offline synthesizer.")
            return self._generate_fallback(query, prepared_context)

        try:
            # Try new google-genai SDK first
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                
                contents = []
                system_instruction = (
                    "You are a helpful Multimodal RAG Assistant. "
                    "Answer the user's question accurately using ONLY the provided text context and images. "
                    "Cite sources (File Name and Page Number) for your statements."
                )
                
                prompt_text = (
                    f"{system_instruction}\n\n"
                    f"USER QUESTION: {query}\n\n"
                    f"RETRIEVED CONTEXT:\n{prepared_context.formatted_text}\n\n"
                    "ANSWER:"
                )
                contents.append(prompt_text)

                # Attach PIL images if present
                from PIL import Image as PILImage
                for img_path in prepared_context.image_assets:
                    if img_path.exists():
                        contents.append(PILImage.open(img_path))

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                return response.text.strip()

            except ImportError:
                # Legacy google-generativeai SDK fallback
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model_name)

                contents = [
                    f"USER QUESTION: {query}\n\nRETRIEVED CONTEXT:\n{prepared_context.formatted_text}\n\nANSWER:"
                ]
                from PIL import Image as PILImage
                for img_path in prepared_context.image_assets:
                    if img_path.exists():
                        contents.append(PILImage.open(img_path))

                response = model.generate_content(contents)
                return response.text.strip()

        except Exception as e:
            print(f"[Notice] Gemini API call failed ({e}). Using offline fallback generator.")
            return self._generate_fallback(query, prepared_context)

    # ---- Provider 2: Local Ollama VLM ----

    def _generate_ollama(self, query: str, prepared_context: PreparedContext, chat_history: List[Dict[str, str]] = None) -> str:
        """Generates response using Local Ollama server (e.g. llama3.1:8b, qwen3:8b)."""
        target_model = self.model_name
        if ":" not in target_model and not target_model.endswith(":latest"):
            target_model = f"{target_model}:latest"

        # Detect if model supports vision (VLM) based on model name
        vlm_keywords = ["vision", "llava", "bakllava", "moondream", "minicpm-v"]
        is_vlm = any(kw in target_model.lower() for kw in vlm_keywords)

        # Combine system instructions and user prompt
        # (This avoids HTTP 500 errors on some vision models that crash on explicit 'system' roles)
        sys_instructions = (
            "You are an intelligent, conversational AI assistant. "
            "You have been provided with some retrieved context documents below. "
            "If the context contains relevant information, use it to ground your answer. "
            "If the context does not contain the answer, you MUST use your own general knowledge to answer the user's question. "
            "Always answer naturally and helpfully in a conversational tone.\n\n"
        )
        
        combined_prompt = (
            sys_instructions +
            f"CONTEXT (retrieved from documents):\n"
            f"---\n"
            f"{prepared_context.formatted_text}\n"
            f"---\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer based on the context above:"
        )

        # Only prepare image attachments for VLM models — text-only models reject them
        images_b64 = []
        if is_vlm:
            for img_path in prepared_context.image_assets[:1]:
                if img_path.exists():
                    try:
                        with PILImage.open(img_path) as img:
                            img_rgb = img.convert("RGB")
                            img_rgb.thumbnail((768, 768))
                            buffer = io.BytesIO()
                            img_rgb.save(buffer, format="JPEG", quality=80)
                            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8").strip()
                            images_b64.append(b64_str)
                    except Exception as img_err:
                        print(f"[Notice] Failed to process image '{img_path.name}': {img_err}")

        try:
            import json
            import urllib.request

            endpoint = f"{self.ollama_url}/api/chat"

            # Build messages array starting with history, then current prompt
            messages = []
            if chat_history:
                messages.extend(chat_history)
            
            # Build current user message
            user_msg: Dict[str, Any] = {"role": "user", "content": combined_prompt}
            if images_b64:
                user_msg["images"] = images_b64
            messages.append(user_msg)

            payload = {
                "model": target_model,
                "messages": messages,
                "stream": False,
            }

            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                msg = result.get("message", {})
                return msg.get("content", "").strip()

        except Exception as e:
            print(f"[Notice] Ollama /api/chat error ({e}). Trying /api/generate fallback...")
            try:
                gen_endpoint = f"{self.ollama_url}/api/generate"
                gen_payload = {
                    "model": target_model,
                    "prompt": combined_prompt,
                    "stream": False,
                }
                if images_b64:
                    gen_payload["images"] = images_b64
                req = urllib.request.Request(
                    gen_endpoint,
                    data=json.dumps(gen_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return result.get("response", "").strip()
            except Exception as inner_e:
                print(f"[Notice] Fallback generation error ({inner_e}).")

            return self._generate_fallback(query, prepared_context)

    # ---- Provider 3: Offline Fallback Synthesizer ----

    def _generate_fallback(self, query: str, prepared_context: PreparedContext) -> str:
        """Generates a structured offline synthesis of retrieved context & sources."""
        sources_str = ", ".join(
            f"{s['source']} (Page {s['page']})" for s in prepared_context.sources
        ) if prepared_context.sources else "No source attribution available"

        # Extract the most relevant content block (first block = highest reranked match)
        context_text = prepared_context.formatted_text.strip()
        content_blocks = context_text.split("\n\n") if context_text else []

        # Build a direct answer from the top content block(s)
        answer_lines = []
        for block in content_blocks:
            # Skip metadata header lines (--- [...] ---)
            lines = [line.strip() for line in block.split("\n") if line.strip() and not line.strip().startswith("--- [")]
            # Skip lines that are just markdown headers with no info (e.g. "## Generation")
            lines = [line for line in lines if not (line.startswith("#") and len(line.split()) <= 3)]
            if lines:
                answer_lines.extend(lines)

        if answer_lines:
            synthesized_answer = " ".join(answer_lines)
        else:
            synthesized_answer = context_text[:800] if context_text else "No relevant context found."

        lines = [
            f"[Offline Fallback Generator]",
            f"",
            f"Question: {query}",
            f"",
            f"Answer (synthesized from retrieved context):",
            f"  {synthesized_answer}",
            f"",
            f"Sources: {sources_str}",
            f"Image Assets: {len(prepared_context.image_assets)} image(s) available.",
        ]
        return "\n".join(lines)
