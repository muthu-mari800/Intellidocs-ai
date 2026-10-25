import pytest
from pydantic import ValidationError

from backend.extraction import ExtractionError, parse_model_json
from backend.insights import categorize_skills, parse_date, total_experience_months
from backend.validation import (FieldStatus, apply_scope, compute_completeness, effective_uncertain,
                                export_dict, is_valid_email, validate_resume_payload)
from models.schemas import (AnalysisConfig, EducationItem, ExperienceItem, ResumeData, SkillCategory)

ALL5 = ["full_name", "email", "skills", "education", "experience"]
ALL7 = ALL5 + ["certifications", "projects"]


def full_resume() -> ResumeData:
    return ResumeData(
        full_name="Jane Doe", email="jane@example.com", skills=["Python", "SQL"],
        education=[EducationItem(degree="B.E.", institution="Anna University", year="2020")],
        experience=[ExperienceItem(title="Dev", company="Acme", start_date="Jan 2020", end_date="Present")],
        certifications=["AWS Cloud Practitioner"], projects=["Inventory tracker"])


# ---- schema
def test_schema_defaults_are_null_or_empty():
    d = ResumeData()
    assert d.full_name is None and d.email is None and d.skills == [] and d.education == []


def test_schema_cleans_values():
    d = ResumeData(full_name="  ", skills=["Python", "python", " ", None], education=["B.Sc"], experience=None)
    assert d.full_name is None and d.skills == ["Python"] and d.education[0].degree == "B.Sc" and d.experience == []


def test_schema_accepts_numeric_dates_and_dict_categories():
    d = ResumeData(experience=[{"title": "X", "start_date": 2020}], skill_categories={"Lang": ["Python"]})
    assert d.experience[0].start_date == "2020" and d.skill_categories[0] == SkillCategory(category="Lang", skills=["Python"])


def test_schema_rejects_wrong_types():
    with pytest.raises(ValidationError):
        ResumeData(full_name=["not", "a", "string"])


def test_validate_payload_non_dict_and_bad_dict():
    assert validate_resume_payload([1, 2])[0] is None
    data, errors = validate_resume_payload({"full_name": {"a": 1}})
    assert data is None and errors


# ---- completeness
def test_completeness_full():
    r = compute_completeness(full_resume(), ALL7)
    assert r.percent == 100 and not r.missing and not r.uncertain and r.fully_checked


def test_missing_field_detection():
    d = full_resume()
    d.email, d.education = None, []
    r = compute_completeness(d, ALL7)
    assert set(r.missing) == {"email", "education"} and r.percent == 71


def test_uncertain_counts_half():
    d = full_resume()
    r = compute_completeness(d, ALL7, uncertain=["full_name"])
    assert r.statuses["full_name"] == FieldStatus.UNCERTAIN and r.percent == 93


def test_invalid_email_is_not_present():
    d = full_resume()
    d.email = "not-an-email"
    r = compute_completeness(d, ALL5)
    assert r.statuses["email"] == FieldStatus.INVALID and "email" in r.missing


def test_incomplete_entries_are_uncertain():
    d = full_resume()
    d.experience = [ExperienceItem(title="Dev", company="Acme")]  # no dates
    d.education = [EducationItem(degree="B.E.")]  # no institution
    r = compute_completeness(d, ALL5)
    assert r.statuses["experience"] == FieldStatus.UNCERTAIN and r.statuses["education"] == FieldStatus.UNCERTAIN


def test_scope_limits_score():
    r = compute_completeness(full_resume(), ["skills"])
    assert r.in_scope == ["skills"] and r.percent == 14  # never 100% / "1/1" when only one field was checked
    assert r.statuses["email"] == FieldStatus.NOT_VERIFIED and "email" not in r.missing
    assert len(r.not_verified) == 6 and not r.fully_checked


def test_no_required_fields_in_scope_gives_none():
    assert compute_completeness(full_resume(), ["quality_notes"]).percent is None


def test_empty_document_scores_zero():
    assert compute_completeness(ResumeData(), ALL5).percent == 0


def test_user_edit_clears_uncertainty():
    orig, edited = full_resume(), full_resume()
    assert effective_uncertain(orig, edited, ["full_name"]) == {"full_name"}
    edited.full_name = "Jane Q Doe"
    assert effective_uncertain(orig, edited, ["full_name"]) == set()


def test_email_validation():
    assert is_valid_email("a.b+c@example.co.in")
    assert not is_valid_email("a@b") and not is_valid_email(None) and not is_valid_email("x y@z.com")


def test_scope_and_export():
    d = apply_scope(full_resume(), ["full_name", "skills"])
    assert d.email is None and d.education == []
    assert list(export_dict(d, ["full_name", "skills"])) == ["full_name", "skills"]


# ---- malformed JSON
@pytest.mark.parametrize("raw", ["", "not json", "[1,2]", "{broken", "```json\n{oops\n```"])
def test_malformed_json_raises(raw):
    with pytest.raises(ExtractionError):
        parse_model_json(raw)


def test_json_in_fences_or_prose_is_parsed():
    assert parse_model_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_model_json('Here you go: {"a": 2} thanks') == {"a": 2}


# ---- config & insights
def test_config_rejects_unknown_focus():
    with pytest.raises(ValidationError):
        AnalysisConfig(focus=["Hobbies"])


def test_date_parsing_and_experience_months():
    assert str(parse_date("Jan 2020")) == "2020-01-01" and str(parse_date("2020-03")) == "2020-03-01"
    assert parse_date("sometime") is None and parse_date(None) is None
    from datetime import date
    items = [ExperienceItem(start_date="Jan 2020", end_date="Dec 2020"),
             ExperienceItem(start_date="Jun 2020", end_date="Jun 2021"),  # overlaps
             ExperienceItem(title="undated")]
    assert total_experience_months(items, today=date(2026, 1, 1)) == 17
    assert total_experience_months([ExperienceItem(title="x")]) is None


def test_categorize_skills():
    groups = categorize_skills(["Python", "PostgreSQL", "Docker", "Teamwork", "Origami"])
    assert "Python" in groups["Programming Languages"] and "Origami" in groups["Other"]
