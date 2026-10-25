"""Environment-based configuration. Secrets are only ever read from environment variables."""
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemini-flash-latest"


@dataclass(frozen=True)
class Settings:
    api_key: Optional[str]
    model: str
    fallback_model: Optional[str]
    timeout_seconds: int
    max_file_mb: float
    max_chars: int
    max_pdf_pages: int
    chunk_overlap: int = 1000
    max_chunks: int = 8


def _num(name: str, default, cast):
    try:
        return cast(os.getenv(name, default))
    except (TypeError, ValueError):
        return cast(default)


def get_settings() -> Settings:
    """Read settings fresh each call so tests/env changes are respected."""
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if key.lower().startswith("your_"):  # placeholder from .env.example
        key = ""
    return Settings(
        api_key=key or None,
        model=(os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip(),
        fallback_model=(os.getenv("GEMINI_FALLBACK_MODEL") or "").strip() or None,
        timeout_seconds=_num("GEMINI_TIMEOUT_SECONDS", 120, int),
        max_file_mb=_num("MAX_FILE_MB", 5, float),
        max_chars=_num("MAX_DOCUMENT_CHARS", 30000, int),
        max_pdf_pages=_num("MAX_PDF_PAGES", 30, int),
        chunk_overlap=_num("CHUNK_OVERLAP_CHARS", 1000, int),
        max_chunks=_num("MAX_CHUNKS", 8, int),
    )
