"""Document types: per-type field specs (by analysis level) and a fast keyword-rule classifier.

The rule classifier runs without any API call. When its confidence is low, extraction.py asks Gemini
to classify instead (see extraction.resolve_doc_type).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from models.schemas import DOC_TYPE_LABELS, DocType, REQUIRED_FIELDS, BASE_FIELDS

CONFIDENT = 0.75


@dataclass(frozen=True)
class DocTypeSpec:
    label: str
    basic: tuple
    standard: tuple
    detailed: tuple
    required: tuple       # fields that count toward the completeness score
    charts: tuple         # insight charts relevant to this type
    hint: str             # short guidance given to the model

    @property
    def universe(self) -> list[str]:
        """All fields this type can have, in canonical order."""
        seen: list[str] = []
        for f in self.detailed + self.standard + self.basic:
            if f not in seen:
                seen.append(f)
        return seen


def _spec(type_, basic, standard_extra, detailed_extra, required, charts, hint) -> DocTypeSpec:
    standard = tuple(basic) + tuple(standard_extra)
    return DocTypeSpec(DOC_TYPE_LABELS[type_], tuple(basic), standard, standard + tuple(detailed_extra),
                       tuple(required), tuple(charts), hint)


SPECS: dict[DocType, DocTypeSpec] = {
    DocType.ACADEMIC_NOTES: _spec(
        DocType.ACADEMIC_NOTES, ["title", "main_topics", "key_points"], ["important_facts"],
        ["author_or_issuer", "dates", "entities"], ["title", "main_topics", "key_points"],
        ["topics", "dates", "entities"],
        "Study or lecture notes. Capture the subject, topics covered, key concepts and definitions, formulas and facts to remember."),
    DocType.SYLLABUS: _spec(
        DocType.SYLLABUS, ["title", "main_topics", "key_points"], ["author_or_issuer", "important_facts", "dates"],
        ["entities"], ["title", "main_topics", "key_points"], ["topics", "dates"],
        "A course syllabus. Capture the course title, units/topics, learning outcomes, assessment and grading details, "
        "schedule dates, textbooks and the institution or instructor."),
    DocType.CERTIFICATE: _spec(
        DocType.CERTIFICATE, ["title", "author_or_issuer", "entities", "dates"], ["important_facts"], [],
        ["title", "author_or_issuer", "entities", "dates"], ["dates", "entities"],
        "A certificate. Capture the certificate title, issuing organisation, the recipient (as a Person entity), "
        "issue/expiry dates and identifiers such as certificate numbers."),
    DocType.BUSINESS_REPORT: _spec(
        DocType.BUSINESS_REPORT, ["title", "main_topics", "key_points", "findings"], ["metrics", "dates"],
        ["author_or_issuer", "entities", "important_facts"], ["title", "main_topics", "key_points", "findings"],
        ["metrics", "topics", "entities", "dates"],
        "A business report. Capture performance results, financial and operational figures, risks, "
        "recommendations, reporting period and responsible team or company."),
    DocType.PROJECT_REPORT: _spec(
        DocType.PROJECT_REPORT, ["title", "main_topics", "key_points", "findings"], ["dates", "important_facts"],
        ["author_or_issuer", "entities", "metrics"], ["title", "main_topics", "key_points", "findings"],
        ["topics", "metrics", "dates", "entities"],
        "A project report. Capture the objective, methodology, technologies, implementation, results, "
        "conclusions and future scope."),
    DocType.RESEARCH_PAPER: _spec(
        DocType.RESEARCH_PAPER, ["title", "author_or_issuer", "main_topics", "key_points"], ["findings", "important_facts"],
        ["dates", "entities", "metrics"], ["title", "author_or_issuer", "main_topics", "key_points", "findings"],
        ["topics", "metrics", "entities", "dates"],
        "A research paper. Capture the title, authors and affiliation, research question, method, datasets, "
        "results with their figures, conclusions and stated limitations."),
    DocType.GENERAL: _spec(
        DocType.GENERAL, ["title", "main_topics", "key_points"], ["dates", "entities", "important_facts"],
        ["findings", "author_or_issuer", "metrics"], ["title", "main_topics", "key_points"],
        ["topics", "entities", "dates", "metrics"],
        "A general document. Capture what it is about, its main topics, key points, dates, named entities and facts."),
}

SPECS[DocType.RESUME] = DocTypeSpec(
    DOC_TYPE_LABELS[DocType.RESUME],
    ("full_name", "email", "skills", "education"),
    tuple(REQUIRED_FIELDS),
    tuple(BASE_FIELDS),
    tuple(REQUIRED_FIELDS),
    ("skills", "experience", "education"),
    "A resume / CV.",
)


@dataclass
class DocTypeResult:
    doc_type: DocType
    confidence: float
    source: str          # "rules", "ai" or "user"
    reason: str = ""

    @property
    def label(self) -> str:
        return DOC_TYPE_LABELS[self.doc_type]


_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")

_KEYWORDS: dict[DocType, list[tuple[str, float]]] = {
    DocType.RESUME: [("curriculum vitae", 2), ("resume", 1.5), ("work experience", 2), ("professional experience", 2),
                     ("skills", 1.5), ("education", 1), ("objective", 0.5), ("references", 0.5), ("linkedin", 1),
                     ("employment", 1), ("internship", 1), ("projects", 0.5), ("certifications", 0.5)],
    DocType.CERTIFICATE: [("certificate of", 2.5), ("this is to certify", 3), ("certify that", 2), ("is hereby awarded", 2.5),
                          ("awarded to", 2), ("has successfully completed", 2.5), ("presented to", 2),
                          ("certificate id", 2), ("certificate no", 2), ("credential", 1), ("issued by", 1), ("issued on", 1)],
    DocType.SYLLABUS: [("syllabus", 3), ("course code", 2.5), ("credits", 1), ("prerequisite", 2), ("learning outcomes", 2),
                       ("course outcomes", 2), ("course objectives", 2), ("grading", 1.5), ("textbook", 1.5),
                       ("reference books", 1.5), ("weekly schedule", 2), ("semester", 1), ("instructor", 1), ("attendance", 1)],
    DocType.ACADEMIC_NOTES: [("lecture notes", 3), ("notes", 1), ("chapter", 1), ("definition", 1.5), ("theorem", 1.5),
                             ("for example", 0.5), ("revision", 1.5), ("formula", 1.5), ("concept", 1), ("lecture", 1.5),
                             ("unit ", 0.5), ("important for exam", 2), ("key terms", 2), ("diagram", 0.5), ("topic", 0.5)],
    DocType.BUSINESS_REPORT: [("revenue", 2), ("quarter", 1.5), ("fiscal", 1.5), ("profit", 1.5), ("market share", 2),
                              ("year-over-year", 2), ("yoy", 1.5), ("executive summary", 2), ("kpi", 1.5), ("forecast", 1),
                              ("stakeholders", 1), ("operating margin", 2), ("customers", 0.5), ("sales", 1),
                              ("churn", 1.5), ("annual report", 2), ("business report", 3), ("risks", 0.5)],
    DocType.PROJECT_REPORT: [("project report", 3), ("abstract", 1.5), ("methodology", 2), ("implementation", 1.5),
                             ("system design", 2), ("literature review", 2), ("future scope", 2.5), ("conclusion", 1),
                             ("technology stack", 1.5), ("testing", 1), ("requirements", 1), ("modules", 1),
                             ("acknowledg", 1.5), ("objectives", 1), ("problem statement", 2)],
    DocType.RESEARCH_PAPER: [("research paper", 3), ("abstract", 1.5), ("keywords", 1.5), ("related work", 2.5),
                             ("et al", 2), ("doi", 2), ("journal", 1.5), ("hypothesis", 1.5), ("proposed method", 2),
                             ("experimental results", 2), ("arxiv", 2), ("references", 1), ("dataset", 1),
                             ("limitations", 1)],
}


def classify_by_rules(text: str) -> DocTypeResult:
    """Keyword/pattern scoring. Cheap and deterministic; low confidence means 'ask the AI'."""
    low = text[:20000].lower()
    scores: dict[DocType, float] = {}
    hits: dict[DocType, list[str]] = {}
    for doc_type, words in _KEYWORDS.items():
        found = [(w, wt) for w, wt in words if w in low]
        scores[doc_type] = sum(wt for _, wt in found)
        hits[doc_type] = [w for w, _ in found]
    if _EMAIL.search(text[:20000]):
        scores[DocType.RESUME] += 1.5
    headings = re.findall(r"^\s*(skills|education|work experience|experience|summary|certifications|projects)\s*:?\s*$",
                          text[:20000], flags=re.IGNORECASE | re.MULTILINE)
    if len(set(h.lower() for h in headings)) >= 3:
        scores[DocType.RESUME] += 3
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top < 4:
        return DocTypeResult(DocType.GENERAL, 0.3, "rules", "No strong document-type signals found.")
    confidence = min(0.95, 0.4 + 0.05 * top)
    if second > 0.6 * top:
        confidence *= 0.7
    reason = "Matched: " + ", ".join(hits[best][:5])
    return DocTypeResult(best, round(confidence, 2), "rules", reason)
