"""Validation, missing-field detection and the completeness score.

Completeness rules (documented in README):
  * Every REQUIRED field of the detected document type is in the denominator.
  * Present and valid = 1.0, present but uncertain/incomplete = 0.5, missing/invalid = 0.
  * A required field that was not checked (outside the chosen level/focus, or not found in a partially
    analysed document) is "not verified": it scores 0 but is never reported as missing.
  * Fields the document type does not use are "not applicable" and never count.
  * percent = round(sum(scores) / number of required fields * 100); None when no required field was checked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

from pydantic import ValidationError

from models.schemas import BASE_FIELDS, REQUIRED_FIELDS, GeneralData, ResumeData

FIELD_LABELS = {
    "full_name": "Full name",
    "email": "Email address",
    "skills": "Skills",
    "education": "Education",
    "experience": "Work experience",
    "certifications": "Certifications",
    "projects": "Projects",
    "skill_categories": "Skill categories",
    "quality_notes": "Data quality notes",
    "title": "Title",
    "author_or_issuer": "Author / issuer",
    "main_topics": "Main topics",
    "key_points": "Key points",
    "dates": "Dates",
    "entities": "Entities",
    "findings": "Findings",
    "important_facts": "Important facts",
    "metrics": "Metrics & figures",
}

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")


class FieldStatus(str, Enum):
    PRESENT = "present"
    UNCERTAIN = "uncertain"
    MISSING = "missing"
    INVALID = "invalid"
    NOT_ANALYSED = "not analysed"
    NOT_VERIFIED = "not verified"
    NOT_APPLICABLE = "not applicable"


SCORES = {FieldStatus.PRESENT: 1.0, FieldStatus.UNCERTAIN: 0.5}


def is_valid_email(value: Optional[str]) -> bool:
    return bool(value and EMAIL_RE.match(value.strip()))


def validate_payload(raw: Any, model_cls=ResumeData) -> tuple[Optional[Any], list[str]]:
    """Validate a raw dict against a schema. Returns (data, errors); data is None on failure."""
    if not isinstance(raw, dict):
        return None, ["The AI response was not a JSON object."]
    try:
        return model_cls.model_validate(raw), []
    except ValidationError as exc:
        return None, [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]


def validate_resume_payload(raw: Any) -> tuple[Optional[ResumeData], list[str]]:
    return validate_payload(raw, ResumeData)


def evaluate_field(data: ResumeData, name: str, uncertain: Iterable[str] = ()) -> tuple[FieldStatus, str]:
    flagged = name in set(uncertain)
    if name == "full_name":
        v = data.full_name
        if not v:
            return FieldStatus.MISSING, "No name found."
        if len(v) < 2 or not any(c.isalpha() for c in v):
            return FieldStatus.INVALID, "Name does not look valid."
        if flagged:
            return FieldStatus.UNCERTAIN, "Name could not be matched exactly in the document."
        return FieldStatus.PRESENT, ""
    if name == "email":
        v = data.email
        if not v:
            return FieldStatus.MISSING, "No email found."
        if not is_valid_email(v):
            return FieldStatus.INVALID, "Email format is invalid."
        return (FieldStatus.UNCERTAIN, "Email could not be matched in the document.") if flagged else (FieldStatus.PRESENT, "")
    if name == "skills":
        if not data.skills:
            return FieldStatus.MISSING, "No skills found."
        return (FieldStatus.UNCERTAIN, "Some skills need review.") if flagged else (FieldStatus.PRESENT, "")
    if name == "education":
        items = [e for e in data.education if e.degree or e.institution]
        if not items:
            return FieldStatus.MISSING, "No education found."
        if flagged or any(not (e.degree and e.institution) for e in items):
            return FieldStatus.UNCERTAIN, "An entry is missing a degree or institution."
        return FieldStatus.PRESENT, ""
    if name == "experience":
        items = [x for x in data.experience if x.title or x.company]
        if not items:
            return FieldStatus.MISSING, "No work experience found."
        if flagged or any(not x.start_date for x in items):
            return FieldStatus.UNCERTAIN, "An entry is missing dates or could not be matched."
        return FieldStatus.PRESENT, ""
    if name in ("certifications", "projects"):
        return (FieldStatus.PRESENT, "") if getattr(data, name) else (FieldStatus.MISSING, "None found.")
    return _evaluate_general(data, name, flagged)


def _evaluate_general(data: Any, name: str, flagged: bool) -> tuple[FieldStatus, str]:
    """Rules for the non-resume document fields."""
    value = getattr(data, name, None)
    label = FIELD_LABELS.get(name, name)
    if name in ("title", "author_or_issuer"):
        if not value:
            return FieldStatus.MISSING, f"No {label.lower()} found in the document."
        return (FieldStatus.UNCERTAIN, "Could not be matched closely in the document.") if flagged else (FieldStatus.PRESENT, "")
    if name == "dates":
        usable = [d for d in value or [] if d.date]
    elif name == "entities":
        usable = [e for e in value or [] if e.name]
    elif name == "metrics":
        usable = [m for m in value or [] if m.name and m.value]
    else:
        usable = list(value or [])
    if not usable:
        return FieldStatus.MISSING, f"No {label.lower()} found in the document."
    if flagged:
        return FieldStatus.UNCERTAIN, "Some items are only loosely supported by the document text."
    if name == "metrics" and any(not (m.name and m.value) for m in value):
        return FieldStatus.UNCERTAIN, "A figure is missing its name or value."
    return FieldStatus.PRESENT, ""


@dataclass
class CompletenessReport:
    percent: Optional[int]
    statuses: dict[str, FieldStatus] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    in_scope: list[str] = field(default_factory=list)   # required fields that were actually checked
    required: list[str] = field(default_factory=list)   # every required field for the document type
    partial: bool = False                                # only part of the document was analysed

    def with_status(self, *wanted: FieldStatus) -> list[str]:
        return [f for f in self.in_scope if self.statuses[f] in wanted]

    @property
    def missing(self) -> list[str]:
        return self.with_status(FieldStatus.MISSING, FieldStatus.INVALID)

    @property
    def uncertain(self) -> list[str]:
        return self.with_status(FieldStatus.UNCERTAIN)

    @property
    def present(self) -> list[str]:
        return self.with_status(FieldStatus.PRESENT)

    @property
    def not_verified(self) -> list[str]:
        return [f for f in self.required if self.statuses.get(f) == FieldStatus.NOT_VERIFIED]

    @property
    def fully_checked(self) -> bool:
        return bool(self.required) and not self.not_verified


def compute_completeness(data: Any, active_fields: Iterable[str], uncertain: Iterable[str] = (),
                         required: Optional[Iterable[str]] = None,
                         universe: Optional[Iterable[str]] = None, partial: bool = False,
                         not_applicable: Iterable[str] = ()) -> CompletenessReport:
    """`required` = fields that are scored; `universe` = fields shown in the status table (both default to the
    resume field sets). `partial` = only part of the document was analysed, so a field that was not found is
    "not verified" rather than missing. `not_applicable` = fields the document type does not use."""
    active = set(active_fields)
    uncertain = set(uncertain)
    na = set(not_applicable)
    required = [f for f in (REQUIRED_FIELDS if required is None else required) if f not in na]
    universe = list(BASE_FIELDS if universe is None else universe)
    universe += [f for f in required if f not in universe]
    statuses: dict[str, FieldStatus] = {}
    notes: dict[str, str] = {}
    for name in universe:
        if name in na:
            statuses[name], notes[name] = FieldStatus.NOT_APPLICABLE, "Not applicable to this document type."
        elif name in active:
            statuses[name], notes[name] = evaluate_field(data, name, uncertain)
            if partial and statuses[name] == FieldStatus.MISSING:
                statuses[name], notes[name] = (FieldStatus.NOT_VERIFIED, "Not found in the analysed part of the "
                                               "document; the rest of the document was not analysed.")
        elif name in required:
            statuses[name], notes[name] = (FieldStatus.NOT_VERIFIED, "Required, but not checked in this analysis "
                                           "(outside the selected analysis level or focus fields).")
        else:
            statuses[name], notes[name] = FieldStatus.NOT_ANALYSED, "Not selected for this analysis."
    in_scope = [f for f in required if f in active]
    percent = None
    if in_scope:
        percent = round(sum(SCORES.get(statuses[f], 0.0) for f in required) / len(required) * 100)
    return CompletenessReport(percent=percent, statuses=statuses, notes=notes, in_scope=in_scope,
                              required=required, partial=partial)


def effective_uncertain(original: Any, current: Any, uncertain: Iterable[str]) -> set[str]:
    """A field the user edited counts as reviewed, so its uncertainty flag is cleared."""
    return {f for f in uncertain if getattr(original, f, None) == getattr(current, f, None)}


def apply_scope(data: Any, active_fields: Iterable[str]) -> Any:
    """Reset fields that are outside the analysis scope to their empty defaults."""
    keep = set(active_fields)
    cls = type(data)
    return cls(**{f: getattr(data, f) for f in cls.model_fields if f in keep})


def export_dict(data: Any, active_fields: Iterable[str]) -> dict:
    """JSON-ready dict containing only the analysed fields (missing ones are null / [])."""
    active = set(active_fields)
    dumped = data.model_dump()
    return {f: dumped[f] for f in type(data).model_fields if f in active}
