"""Thin wrapper around the Google Gemini API (google-genai SDK).

The rest of the app depends only on the small `LLMClient` protocol, so tests can
swap in a mock and never need an API key or network access.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

from backend.config import get_settings

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Raised with a user-friendly message when the AI provider call fails."""


class LLMClient(Protocol):
    def generate(self, prompt: str, system_instruction: str, json_mode: bool = False) -> str: ...


def _friendly(code: Optional[int], exc: Exception) -> str:
    name = type(exc).__name__
    tag = f" (HTTP {code})" if code else f" ({name})"
    if "timeout" in name.lower() or "timeout" in str(exc).lower():
        return "The AI service timed out. Please try again."
    if code in (401, 403):
        return "The AI service rejected the API key or the request. Check GEMINI_API_KEY."
    if code == 400 and "api key" in str(exc).lower():
        return "The Gemini API key appears to be invalid. Check GEMINI_API_KEY."
    if code == 404:
        return "The configured Gemini model was not found. Update GEMINI_MODEL in the environment or Settings."
    if code == 429:
        return "The AI service rate limit or free-tier quota was reached. Wait a minute and try again."
    if code in (500, 503, 504):
        return f"The AI service is temporarily overloaded or unavailable{tag}. Please try again in a minute, or set GEMINI_FALLBACK_MODEL / change GEMINI_MODEL."
    return f"The AI request failed ({name}). Please try again."


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_seconds: int = 60, max_retries: int = 2, fallback_model: Optional[str] = None):
        from google import genai
        from google.genai import types

        self.model = model
        self.models = [model] + ([fallback_model] if fallback_model and fallback_model != model else [])
        self.max_retries = max_retries
        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    def generate(self, prompt: str, system_instruction: str, json_mode: bool = False) -> str:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        last_exc: Optional[Exception] = None
        for model in self.models:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = self._client.models.generate_content(model=model, contents=prompt, config=cfg)
                    text = (resp.text or "").strip()
                    if not text:
                        raise AIServiceError("The AI returned an empty response (it may have been blocked by a safety filter).")
                    return text
                except AIServiceError:
                    raise
                except Exception as exc:  # SDK raises several error types
                    code = getattr(exc, "code", None)
                    last_exc = exc
                    logger.warning("Gemini call failed: model=%s %s code=%s %s", model, type(exc).__name__, code,
                                   str(exc)[:200])
                    timed_out = "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()
                    if timed_out:
                        break  # a slow model will likely be slow again: go straight to the fallback model
                    if code in (429, 500, 503, 504) and attempt < self.max_retries:
                        time.sleep(2 * (attempt + 1))
                        continue
                    if code in (404, 429, 500, 503, 504):
                        break  # try the fallback model, if any
                    raise AIServiceError(_friendly(code, exc)) from None
        raise AIServiceError(_friendly(getattr(last_exc, "code", None), last_exc or Exception("unknown")))


def get_llm_client(model_override: Optional[str] = None) -> LLMClient:
    settings = get_settings()
    if not settings.api_key:
        raise AIServiceError(
            "GEMINI_API_KEY is not configured. Add it to your .env file (local) or the Render environment settings."
        )
    return GeminiClient(settings.api_key, (model_override or settings.model).strip(), settings.timeout_seconds,
                        fallback_model=settings.fallback_model)
