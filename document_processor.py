"""Safe upload validation and text extraction for PDF and TXT files.

Uploaded bytes are processed in memory only; nothing is written to disk and
document text is never logged.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF

from backend.config import get_settings

SUPPORTED_EXTENSIONS = {".pdf": "PDF", ".txt": "TXT"}
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_documents"


class DocumentError(Exception):
    """Raised with a user-friendly message when a document cannot be processed."""


@dataclass
class ProcessedDocument:
    filename: str
    file_type: str
    text: str
    size_bytes: int
    page_count: Optional[int]
    digest: str
    warnings: list[str] = field(default_factory=list)
    is_sample: bool = False

    @property
    def char_count(self) -> int:
        return len(self.text)


def normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_txt(data: bytes) -> tuple[str, list[str]]:
    if b"\x00" in data[:4096]:
        raise DocumentError("This file contains binary data and does not look like a plain-text document.")
    warnings: list[str] = []
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        warnings.append("The file is not valid UTF-8; it was decoded as Latin-1, so some characters may look wrong.")
    return text, warnings


def _extract_pdf(data: bytes, max_pages: int) -> tuple[str, int, list[str]]:
    if b"%PDF" not in data[:1024]:
        raise DocumentError("This file has a .pdf extension but is not a valid PDF document.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise DocumentError("The PDF could not be opened. It may be corrupted or damaged.") from None
    warnings: list[str] = []
    try:
        if doc.needs_pass:
            raise DocumentError("The PDF is password-protected. Please upload an unlocked copy.")
        page_count = doc.page_count
        if page_count == 0:
            raise DocumentError("The PDF has no pages.")
        pages = []
        for index in range(min(page_count, max_pages)):
            pages.append(doc[index].get_text("text"))
        if page_count > max_pages:
            warnings.append(f"Only the first {max_pages} of {page_count} pages were read.")
        return "\n".join(pages), page_count, warnings
    except DocumentError:
        raise
    except Exception:
        raise DocumentError("The PDF could not be read. It may be corrupted.") from None
    finally:
        doc.close()


def process_upload(filename: str, data: bytes, is_sample: bool = False) -> ProcessedDocument:
    """Validate and extract text from an uploaded file. Raises DocumentError on failure."""
    settings = get_settings()
    name = Path(filename or "").name or "uploaded_file"
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        shown = ext if ext else "(no extension)"
        raise DocumentError(f"Unsupported file type {shown}. Please upload a PDF or TXT file.")
    if not data:
        raise DocumentError("The file is empty (0 bytes).")
    max_bytes = int(settings.max_file_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise DocumentError(
            f"The file is {len(data) / 1024 / 1024:.1f} MB, which exceeds the {settings.max_file_mb:g} MB limit."
        )

    page_count: Optional[int] = None
    if ext == ".pdf":
        raw, page_count, warnings = _extract_pdf(data, settings.max_pdf_pages)
        empty_msg = (
            "No selectable text was found in this PDF. It may be a scanned or image-only document "
            "(OCR is not supported)."
        )
    else:
        raw, warnings = _extract_txt(data)
        empty_msg = "The document contains no readable text."

    text = normalise_text(raw)
    if not text:
        raise DocumentError(empty_msg)
    if len(text) < 80:
        warnings.append("The document is very short; results may be incomplete.")

    return ProcessedDocument(
        filename=name,
        file_type=SUPPORTED_EXTENSIONS[ext],
        text=text,
        size_bytes=len(data),
        page_count=page_count,
        digest=hashlib.sha256(data).hexdigest(),
        warnings=warnings,
        is_sample=is_sample,
    )


def list_samples() -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def load_sample(path: Path) -> ProcessedDocument:
    return process_upload(path.name, path.read_bytes(), is_sample=True)


@dataclass
class ChunkPlan:
    """Parts of the document that will be sent to the model, each at most `chunk_chars` long."""
    chunks: list[str]
    total_parts: int
    analysed_chars: int
    total_chars: int

    @property
    def complete(self) -> bool:
        return self.analysed_chars >= self.total_chars


def _split_points(text: str, chunk_chars: int, overlap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:  # prefer to cut at a paragraph, line, sentence or word boundary in the second half
            window_start = start + chunk_chars // 2
            for sep in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(sep, window_start, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        spans.append((start, end))
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        start = space + 1 if overlap and space != -1 else next_start
    return spans


def plan_chunks(text: str, chunk_chars: int, overlap: int = 0, max_chunks: Optional[int] = None) -> ChunkPlan:
    """Split text into overlapping parts no longer than chunk_chars, keeping at most max_chunks of them."""
    chunk_chars = max(1, int(chunk_chars))
    overlap = max(0, min(int(overlap), chunk_chars // 4))
    spans = _split_points(text, chunk_chars, overlap)
    kept = spans if not max_chunks or max_chunks < 1 else spans[:max_chunks]
    return ChunkPlan(chunks=[text[s:e] for s, e in kept], total_parts=len(spans),
                     analysed_chars=kept[-1][1] if kept else 0, total_chars=len(text))


def truncate_for_model(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True
