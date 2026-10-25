"""Deterministic helpers: date parsing, experience totals, keyword-based skill grouping."""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from models.schemas import ExperienceItem

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_PRESENT = {"present", "current", "now", "ongoing", "till date", "to date", "today"}


def parse_date(value: Optional[str], today: Optional[date] = None) -> Optional[date]:
    """Parse the date formats typically found in resumes. Returns None if unsure."""
    if not value:
        return None
    s = value.strip().lower()
    if s in _PRESENT:
        return today or date.today()
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return date(int(m.group(1)), int(m.group(2)), 1)
    m = re.fullmatch(r"(\d{1,2})[-/](\d{4})", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return date(int(m.group(2)), int(m.group(1)), 1)
    m = re.fullmatch(r"([a-z]{3,9})\.?,?\s+(\d{4})", s)
    if m and m.group(1)[:3] in _MONTHS:
        return date(int(m.group(2)), _MONTHS[m.group(1)[:3]], 1)
    full = _full_date(s)
    if full:
        return full
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return date(int(m.group(1)), 1, 1)
    return None


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _full_date(s: str) -> Optional[date]:
    """Day-level formats. Ambiguous numeric dates such as 03/04/2024 return None rather than guessing."""
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\.?,?\s+(\d{4})", s)
    if m and m.group(2)[:3] in _MONTHS:
        return _safe_date(int(m.group(3)), _MONTHS[m.group(2)[:3]], int(m.group(1)))
    m = re.fullmatch(r"([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s)
    if m and m.group(1)[:3] in _MONTHS:
        return _safe_date(int(m.group(3)), _MONTHS[m.group(1)[:3]], int(m.group(2)))
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12 >= b:
            return _safe_date(y, b, a)
        if b > 12 >= a:
            return _safe_date(y, a, b)
    return None


def experience_intervals(items: list[ExperienceItem], today: Optional[date] = None) -> list[tuple[date, date, ExperienceItem]]:
    out = []
    for item in items:
        start, end = parse_date(item.start_date, today), parse_date(item.end_date, today)
        if start and end and end >= start:
            out.append((start, end, item))
    return out


def total_experience_months(items: list[ExperienceItem], today: Optional[date] = None) -> Optional[int]:
    """Months covered by dated roles (overlaps merged). None if no role has usable dates."""
    spans = sorted((s, e) for s, e, _ in experience_intervals(items, today))
    if not spans:
        return None
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum((e.year - s.year) * 12 + (e.month - s.month) for s, e in merged)


def parse_year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    years = re.findall(r"\b(?:19|20)\d{2}\b", value)
    return int(years[-1]) if years else None


_KEYWORDS = {
    "Programming Languages": ["python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "php", "kotlin", "swift", "ruby", "r", "scala", "matlab"],
    "Web & Frameworks": ["html", "css", "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring", "express", "next.js", "streamlit", "bootstrap", "tailwind"],
    "Data & AI": ["machine learning", "deep learning", "nlp", "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "power bi", "tableau", "excel", "statistics", "data analysis", "llm"],
    "Databases": ["sql", "mysql", "postgresql", "mongodb", "sqlite", "redis", "oracle", "nosql"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "ci/cd", "jenkins", "linux", "terraform"],
    "Soft Skills": ["communication", "leadership", "teamwork", "problem solving", "problem-solving", "collaboration", "time management", "mentoring"],
}


def categorize_skills(skills: list[str]) -> dict[str, list[str]]:
    """Keyword-based grouping used when the AI did not supply categories."""
    groups: dict[str, list[str]] = {}
    for skill in skills:
        low = skill.lower().strip()
        label = "Other"
        for category, words in _KEYWORDS.items():
            if low in words or any(len(w) > 3 and w in low for w in words):
                label = category
                break
        groups.setdefault(label, []).append(skill)
    return groups


# ---------------------------------------------------------------- general-document helpers
_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "m": 1e6, "mn": 1e6, "million": 1e6,
                "crore": 1e7, "crores": 1e7, "cr": 1e7, "b": 1e9, "bn": 1e9, "billion": 1e9}
_CURRENCY = {"$": "$", "usd": "$", "€": "€", "eur": "€", "£": "£", "gbp": "£", "₹": "₹", "inr": "₹", "rs": "₹"}


def parse_metric(value: Optional[str], unit: Optional[str] = None) -> tuple[Optional[float], str]:
    """Return (number, unit group) for a figure such as '$48.2 million' or '12.5%'. (None, '') if no number."""
    text = f"{value or ''} {unit or ''}".lower()
    m = re.search(r"\d[\d,]*\.?\d*", text)
    if not m:
        return None, ""
    try:
        number = float(m.group().replace(",", ""))
    except ValueError:
        return None, ""
    tail = text[m.end():]
    word = re.match(r"\s*([a-z]+)\b", tail)
    mult = _MULTIPLIERS.get(word.group(1), 1.0) if word else 1.0
    group = ""
    if "%" in text:
        group = "%"
    else:
        for token, symbol in _CURRENCY.items():
            if (token.isalpha() and re.search(rf"\b{token}\b", text)) or (not token.isalpha() and token in text):
                group = symbol
                break
    if not group:
        leftover = re.sub(r"[\d,.\s]", " ", text).split()
        leftover = [w for w in leftover if w not in _MULTIPLIERS]
        group = leftover[0] if leftover else "count"
    return number * mult, group


def count_mentions(topic: str, text: str) -> int:
    """Occurrences of a topic in the source text; falls back to its longest word if the phrase is not verbatim."""
    low = text.lower()
    phrase = topic.lower().strip()
    if not phrase:
        return 0
    n = len(re.findall(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", low))
    if n:
        return n
    words = sorted((w for w in re.findall(r"[a-z0-9]+", phrase) if len(w) >= 5), key=len, reverse=True)
    return len(re.findall(rf"(?<![a-z0-9]){re.escape(words[0])}", low)) if words else 0
