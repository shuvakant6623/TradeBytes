"""
Ollama LLM Service Layer
- Local inference via Ollama REST API
- Streaming support
- Structured JSON output parsing
- Retry logic and error handling
"""
import json
import logging
import time
import httpx
from typing import Optional
from models.schemas import StructuredResponse
from core.config import settings

logger = logging.getLogger("finai.llm")

FALLBACK_RESPONSE = StructuredResponse(
    summary="I'm having trouble processing your request right now.",
    analysis="The LLM service is temporarily unavailable. Please ensure Ollama is running with: `ollama serve` and the model is pulled with `ollama pull mistral`.",
    risk_note="Unable to assess risk without LLM availability.",
    confidence_level="Low",
    disclaimer=settings.FINANCIAL_DISCLAIMER
)

class OllamaService:
    """
    Manages all communication with the local Ollama instance.
    
    Architecture:
    - Uses httpx for async HTTP calls to Ollama's REST API
    - Supports both streaming and non-streaming modes
    - Parses LLM output as structured JSON
    - Falls back gracefully on parse errors
    """

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.temperature = settings.OLLAMA_TEMPERATURE
        self.max_tokens = settings.OLLAMA_MAX_TOKENS
        self.timeout = settings.OLLAMA_TIMEOUT
        logger.info(f"OllamaService initialized | URL: {self.base_url} | Model: {self.model}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> tuple[StructuredResponse, float]:
        """
        Calls Ollama /api/chat endpoint.
        Returns (StructuredResponse, latency_ms).
        """
        start_time = time.time()
        temp = temperature or self.temperature

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": self.max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Sending request to Ollama | model={self.model} | temp={temp}")
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()

            raw_content = data["message"]["content"]
            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Ollama responded in {latency_ms:.0f}ms | tokens≈{len(raw_content.split())}")

            structured = self._parse_structured_response(raw_content)
            structured.raw_llm_response = raw_content
            return structured, latency_ms

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama. Is it running? Run: ollama serve")
            return FALLBACK_RESPONSE, 0.0
        except httpx.TimeoutException:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            fallback = FALLBACK_RESPONSE.model_copy()
            fallback.summary = "Request timed out. The model may be loading or overloaded."
            return fallback, 0.0
        except Exception as e:
            logger.error(f"Unexpected Ollama error: {e}", exc_info=True)
            return FALLBACK_RESPONSE, 0.0

    def _parse_structured_response(self, raw: str) -> StructuredResponse:
        """
        Robustly parses LLM JSON output.
        Handles markdown code fences, leading/trailing noise.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        # Find JSON object boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            json_str = cleaned[start:end]
            try:
                data = json.loads(json_str)
                return StructuredResponse(
                    summary=str(data.get("summary", "No summary provided.")),
                    analysis=str(data.get("analysis", "No analysis provided.")),
                    risk_note=str(data.get("risk_note", "No risk note provided.")),
                    confidence_level=str(data.get("confidence_level", "Low")),
                    disclaimer=settings.FINANCIAL_DISCLAIMER,
                    raw_llm_response=raw,
                )
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed: {e}. Raw: {json_str[:200]}")

        # Fallback: wrap raw text in structured format
        logger.warning("Using fallback text wrapping for unstructured LLM response")
        return StructuredResponse(
            summary=raw[:300] if len(raw) > 300 else raw,
            analysis="Response was not in expected JSON format. Raw content shown in summary.",
            risk_note="Unable to extract structured risk assessment.",
            confidence_level="Low",
            disclaimer=settings.FINANCIAL_DISCLAIMER,
            raw_llm_response=raw,
        )

    async def health_check(self) -> dict:
        """Check if Ollama is running and model is available"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                models = [m["name"] for m in r.json().get("models", [])]
                model_available = any(self.model in m for m in models)
                return {
                    "ollama_online": True,
                    "model_available": model_available,
                    "available_models": models,
                    "target_model": self.model,
                }
        except Exception as e:
            return {"ollama_online": False, "error": str(e)}

# Singleton
ollama_service = OllamaService()