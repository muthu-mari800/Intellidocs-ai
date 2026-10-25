"""Pydantic schemas and shared enums for IntelliDocs AI."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def _clean_str(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = str(v)
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return v


def _clean_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return []
    out = []
    for item in v:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


class EducationItem(BaseModel):
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return _clean_str(v)


class ExperienceItem(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    highlights: list[str] = Field(default_factory=list)

    @field_validator("title", "company", "start_date", "end_date", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return _clean_str(v)

    @field_validator("highlights", mode="before")
    @classmethod
    def _hl(cls, v: Any) -> list:
        return _clean_list(v)


class SkillCategory(BaseModel):
    category: str
    skills: list[str] = Field(default_factory=list)

    @field_validator("skills", mode="before")
    @classmethod
    def _sk(cls, v: Any) -> list:
        return _clean_list(v)


class ResumeData(BaseModel):
    """Validated resume extraction.

    Missing scalar values are null (None); missing collections are empty lists.
    """

    full_name: Optional[str] = None
    email: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    skill_categories: list[SkillCategory] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)

    @field_validator("full_name", "email", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return _clean_str(v)

    @field_validator("skills", mode="before")
    @classmethod
    def _skills(cls, v: Any) -> list:
        seen, out = set(), []
        for s in _clean_list(v):
            if s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    @field_validator("certifications", "projects", "quality_notes", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> list:
        return _clean_list(v)

    @field_validator("education", mode="before")
    @classmethod
    def _edu(cls, v: Any) -> list:
        if not isinstance(v, (list, tuple)):
            return []
        return [{"degree": i} if isinstance(i, str) else i for i in v if i]

    @field_validator("experience", mode="before")
    @classmethod
    def _exp(cls, v: Any) -> list:
        if not isinstance(v, (list, tuple)):
            return []
        return [{"title": i} if isinstance(i, str) else i for i in v if i]

    @field_validator("skill_categories", mode="before")
    @classmethod
    def _cats(cls, v: Any) -> list:
        if isinstance(v, dict):
            return [{"category": k, "skills": s} for k, s in v.items()]
        if not isinstance(v, (list, tuple)):
            return []
        return v


class DocType(str, Enum):
    RESUME = "resume"
    ACADEMIC_NOTES = "academic_notes"
    SYLLABUS = "syllabus"
    CERTIFICATE = "certificate"
    BUSINESS_REPORT = "business_report"
    PROJECT_REPORT = "project_report"
    RESEARCH_PAPER = "research_paper"
    GENERAL = "general"


DOC_TYPE_LABELS = {
    DocType.RESUME: "Resume / CV",
    DocType.ACADEMIC_NOTES: "Academic Notes",
    DocType.SYLLABUS: "Syllabus",
    DocType.CERTIFICATE: "Certificate",
    DocType.BUSINESS_REPORT: "Business Report",
    DocType.PROJECT_REPORT: "Project Report",
    DocType.RESEARCH_PAPER: "Research Paper",
    DocType.GENERAL: "General Document",
}


class DateItem(BaseModel):
    date: Optional[str] = None
    event: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return _clean_str(v)


class EntityItem(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return _clean_str(v)


class MetricItem(BaseModel):
    name: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return _clean_str(v)


def _items(v: Any, key: str) -> list:
    if not isinstance(v, (list, tuple)):
        return []
    return [{key: i} if isinstance(i, str) else i for i in v if i]


class GeneralData(BaseModel):
    """Validated extraction for non-resume documents. Missing values are null / empty lists."""

    title: Optional[str] = None
    author_or_issuer: Optional[str] = None
    main_topics: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    dates: list[DateItem] = Field(default_factory=list)
    entities: list[EntityItem] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    important_facts: list[str] = Field(default_factory=list)
    metrics: list[MetricItem] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)

    @field_validator("title", "author_or_issuer", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return _clean_str(v)

    @field_validator("main_topics", "key_points", "findings", "important_facts", "quality_notes", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> list:
        return _clean_list(v)

    @field_validator("dates", mode="before")
    @classmethod
    def _dates(cls, v: Any) -> list:
        return _items(v, "date")

    @field_validator("entities", "metrics", mode="before")
    @classmethod
    def _named(cls, v: Any) -> list:
        return _items(v, "name")


GENERAL_FIELDS = ["title", "author_or_issuer", "main_topics", "key_points", "dates", "entities",
                  "findings", "important_facts", "metrics"]

GENERAL_FOCUS_TOPICS = {
    "Title & Author": ["title", "author_or_issuer"],
    "Main Topics": ["main_topics"],
    "Key Points": ["key_points"],
    "Dates": ["dates"],
    "Entities": ["entities"],
    "Findings": ["findings"],
    "Important Facts": ["important_facts"],
    "Metrics & Figures": ["metrics"],
}


class AnalysisLevel(str, Enum):
    BASIC = "Basic"
    STANDARD = "Standard"
    DETAILED = "Detailed"


class SummaryStyle(str, Enum):
    BRIEF = "Brief Summary"
    DETAILED = "Detailed Summary"
    BULLETS = "Bullet-Point Summary"


# Fields that count toward the completeness score.
# Every resume field counts toward completeness; fields not checked in a run are reported as "not verified".
REQUIRED_FIELDS = ["full_name", "email", "skills", "education", "experience", "certifications", "projects"]
BASE_FIELDS = list(REQUIRED_FIELDS)

FOCUS_TOPICS = {
    "Contact Information": ["full_name", "email"],
    "Skills": ["skills"],
    "Education": ["education"],
    "Work Experience": ["experience"],
    "Certifications": ["certifications"],
    "Projects": ["projects"],
}


class AnalysisConfig(BaseModel):
    level: AnalysisLevel = AnalysisLevel.STANDARD
    summary_style: SummaryStyle = SummaryStyle.BULLETS
    focus: list[str] = Field(default_factory=list)
    document_type: Optional[DocType] = None  # None = auto-detect

    @field_validator("focus")
    @classmethod
    def _known_topics(cls, v: list[str]) -> list[str]:
        unknown = [t for t in v if t not in FOCUS_TOPICS and t not in GENERAL_FOCUS_TOPICS]
        if unknown:
            raise ValueError(f"Unknown focus topic(s): {', '.join(unknown)}")
        return v
