"""Tests for completeness scoping, long-document chunking, upload limits and document-specific behaviour."""
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from backend import ai_service
from backend import document_processor as dp
from backend.ai_service import AIServiceError
from backend.config import get_settings
from backend.doctype import SPECS, classify_by_rules
from backend.extraction import (extract_document_chunks, generate_summary_chunks, get_active_fields,
                                merge_extractions)
from backend.validation import FieldStatus, compute_completeness
from components import charts
from models.schemas import REQUIRED_FIELDS, AnalysisConfig, AnalysisLevel, DocType, GeneralData
from tests.test_general_extraction import ACADEMIC, ACADEMIC_REPLY, BUSINESS, BUSINESS_REPLY
from tests.test_smoke_data import RESUME_REPLY

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


def cfg(level=AnalysisLevel.STANDARD, **kw):
    return AnalysisConfig(level=level, **kw)


class SeqClient:
    """Returns queued replies; an Exception in the queue is raised instead."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, system_instruction, json_mode=False):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply) if isinstance(reply, dict) else reply


# ---------------------------------------------------------------- completeness
def test_resume_requires_all_seven_fields():
    assert REQUIRED_FIELDS == ["full_name", "email", "skills", "education", "experience", "certifications", "projects"]
    assert set(SPECS[DocType.RESUME].required) == set(REQUIRED_FIELDS)
    assert set(REQUIRED_FIELDS) <= set(get_active_fields(cfg(), DocType.RESUME))


def test_basic_resume_level_reports_unchecked_fields_as_not_verified():
    from tests.test_validation import full_resume
    active = get_active_fields(cfg(AnalysisLevel.BASIC), DocType.RESUME)
    r = compute_completeness(full_resume(), active)
    assert r.percent == 57 and not r.missing
    assert set(r.not_verified) == {"experience", "certifications", "projects"}


def test_non_resume_types_never_use_resume_fields():
    for doc_type, spec in SPECS.items():
        if doc_type != DocType.RESUME:
            assert not set(spec.required) & set(REQUIRED_FIELDS), doc_type
    assert set(SPECS[DocType.RESEARCH_PAPER].required) == {"title", "author_or_issuer", "main_topics", "key_points",
                                                           "findings"}


def test_not_applicable_fields_are_not_scored():
    data = GeneralData(title="Certificate of Completion", author_or_issuer="Coursera")
    r = compute_completeness(data, ["title", "author_or_issuer"], required=["title", "author_or_issuer", "dates"],
                             universe=["title", "author_or_issuer", "dates", "metrics"], not_applicable=["metrics"])
    assert r.statuses["metrics"] == FieldStatus.NOT_APPLICABLE and "metrics" not in r.required
    assert r.statuses["dates"] == FieldStatus.NOT_VERIFIED and r.percent == 67


def test_partial_analysis_turns_missing_into_not_verified():
    data = GeneralData(title="Annual report", main_topics=["Sales"])
    req = ["title", "main_topics", "key_points"]
    full = compute_completeness(data, req, required=req, universe=req)
    part = compute_completeness(data, req, required=req, universe=req, partial=True)
    assert full.missing == ["key_points"] and part.missing == []
    assert part.statuses["key_points"] == FieldStatus.NOT_VERIFIED and part.partial and part.percent == 67


# ---------------------------------------------------------------- chunk planning
def test_short_text_is_one_complete_chunk():
    plan = dp.plan_chunks("short text", 30000, 1000, 8)
    assert plan.chunks == ["short text"] and plan.complete and plan.total_parts == 1


def test_chunks_respect_size_overlap_and_cover_everything():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(200))
    plan = dp.plan_chunks(text, 3000, 300, None)
    assert plan.complete and len(plan.chunks) > 5
    assert max(len(c) for c in plan.chunks) <= 3000
    for i in range(200):
        assert any(f"Paragraph {i}." in c for c in plan.chunks)
    assert plan.chunks[0][-100:].strip()[-20:] in plan.chunks[1]  # consecutive parts overlap


def test_chunk_cap_marks_plan_incomplete():
    plan = dp.plan_chunks("abc " * 5000, 2000, 100, 3)
    assert len(plan.chunks) == 3 and plan.total_parts > 3 and not plan.complete
    assert plan.analysed_chars < plan.total_chars


# ---------------------------------------------------------------- chunked extraction and summary
def test_chunk_results_are_merged_without_duplicates():
    half = len(BUSINESS) // 2
    chunks = [BUSINESS[: half + 200], BUSINESS[half - 200:]]
    client = SeqClient([BUSINESS_REPLY, BUSINESS_REPLY])
    res = extract_document_chunks(client, chunks, cfg())
    single = extract_document_chunks(SeqClient([BUSINESS_REPLY]), [BUSINESS], cfg())
    assert res.doc_type == DocType.BUSINESS_REPORT and res.chunks_total == 2 and res.chunks_analysed == 2
    names = [m.name for m in res.data.metrics]
    assert len(names) == len(set(names)) and set(names) <= {m.name for m in single.data.metrics}
    assert len(res.data.key_points) == len(set(res.data.key_points)) and not res.partial


def test_failed_chunk_is_reported_as_partial():
    client = SeqClient([BUSINESS_REPLY, AIServiceError("The AI service timed out. Please try again.")])
    res = extract_document_chunks(client, [BUSINESS, BUSINESS + " "], cfg())
    assert res.partial and res.chunks_analysed == 1 and "part(s) 2 of 2" in res.partial_reasons[0]
    assert "timed out" in res.partial_reasons[0]


def test_all_chunks_failing_raises():
    from backend.extraction import ExtractionError
    with pytest.raises(ExtractionError):
        extract_document_chunks(SeqClient([AIServiceError("x"), AIServiceError("x")]), [BUSINESS, BUSINESS], cfg(),
                                type_result=classify_by_rules(BUSINESS))


def test_merge_keeps_first_single_value():
    a = extract_document_chunks(SeqClient([ACADEMIC_REPLY]), [ACADEMIC], cfg())
    b = extract_document_chunks(SeqClient([{**ACADEMIC_REPLY, "title": None}]), [ACADEMIC], cfg())
    merged = merge_extractions([b, a])
    assert merged.data.title == a.data.title


def test_summary_of_long_document_combines_part_notes():
    client = SeqClient(["- note one", "- note two", "Combined summary."])
    text, reasons = generate_summary_chunks(client, ["part a", "part b"], cfg(), DocType.BUSINESS_REPORT)
    assert text == "Combined summary." and reasons == []
    assert "part 1 of 2" in client.prompts[0] and "[Part 2]" in client.prompts[2] and "note two" in client.prompts[2]


def test_summary_reports_unsummarised_parts():
    client = SeqClient(["- note one", AIServiceError("boom"), "Combined."])
    text, reasons = generate_summary_chunks(client, ["a", "b"], cfg(), DocType.GENERAL)
    assert text == "Combined." and "part(s) 2 of 2" in reasons[0]
    assert "only 1 of 2 parts" in client.prompts[-1]


# ---------------------------------------------------------------- upload limit
def test_upload_limits_are_consistent():
    toml = (ROOT / ".streamlit" / "config.toml").read_text()
    server_cap = int(re.search(r"maxUploadSize\s*=\s*(\d+)", toml).group(1))
    env_example = (ROOT / ".env.example").read_text()
    example_mb = float(re.search(r"MAX_FILE_MB=(\S+)", env_example).group(1))
    assert server_cap == example_mb == 5
    assert "max_upload_size=max(1, int(settings.max_file_mb))" in (ROOT / "app.py").read_text()


def test_backend_rejects_file_over_limit(monkeypatch):
    monkeypatch.setenv("MAX_FILE_MB", "5")
    assert get_settings().max_file_mb == 5
    with pytest.raises(dp.DocumentError, match="exceeds the 5 MB limit"):
        dp.process_upload("big.txt", b"a" * (5 * 1024 * 1024 + 1))
    assert dp.process_upload("ok.txt", b"hello world " * 20).char_count > 0


# ---------------------------------------------------------------- charts and detection
def test_counts_chart_hidden_when_nothing_extracted():
    assert charts.field_counts_chart(GeneralData(title="T"), ["title", "main_topics", "key_points"]) is None
    fig = charts.field_counts_chart(GeneralData(main_topics=["a", "b"]), ["title", "main_topics"])
    assert fig is not None and list(fig.data[0].x) == ["Main topics"]


def test_research_paper_detected_by_rules():
    text = ("Abstract. We propose a method. Keywords: retrieval. Related work: Smith et al. (2021). "
            "Experimental results on the dataset. Limitations. References. doi:10.1000/xyz journal")
    assert classify_by_rules(text).doc_type == DocType.RESEARCH_PAPER


# ---------------------------------------------------------------- UI flows (AI mocked)
class FakeClient:
    def generate(self, prompt, system_instruction, json_mode=False):
        if not json_mode:
            return "- Summary text."
        if "Document type: Academic Notes" in prompt:
            return json.dumps(ACADEMIC_REPLY)
        if "Document type: Business Report" in prompt:
            return json.dumps(BUSINESS_REPLY)
        return json.dumps(RESUME_REPLY)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(ai_service, "get_llm_client", lambda model_override=None: FakeClient())
    return AppTest.from_file(APP, default_timeout=60).run()


def _run(app):
    next(b for b in app.button if b.label == "Run Analysis").click().run()
    assert not app.exception, app.exception


def _markdown(app):
    return " ".join(m.value for m in app.markdown)


@pytest.mark.parametrize("sample,doc_type", [("sample_resume.txt", "resume"),
                                             ("sample_academic_notes.txt", "academic_notes"),
                                             ("sample_business_report.txt", "business_report")])
def test_three_document_types_end_to_end(app, sample, doc_type):
    app.button(key=f"sample_{sample}").click().run()
    _run(app)
    ex = app.session_state["extraction"]
    assert ex.doc_type.value == doc_type and app.session_state["summary"]["text"]
    if doc_type != "resume":
        assert not set(ex.required_fields) & set(REQUIRED_FIELDS)
    assert set(ex.required_fields) == set(SPECS[ex.doc_type].required)
    assert "Analysis completed" in _markdown(app) and "Partial analysis" not in _markdown(app)
    assert app.session_state["recent"][-1]["Coverage"] == "Full"


def test_long_document_is_chunked_and_fully_analysed(app, monkeypatch):
    monkeypatch.setenv("MAX_DOCUMENT_CHARS", "2000")
    monkeypatch.setenv("MAX_CHUNKS", "50")
    app.session_state["doc"] = dp.process_upload("long_report.txt", (BUSINESS + "\n\n").encode() * 4)
    app.run()
    _run(app)
    ex = app.session_state["extraction"]
    assert ex.chunks_total > 1 and not ex.partial
    assert any("overlapping parts" in i.value for i in app.info)
    assert "Analysis completed" in _markdown(app)


def test_capped_long_document_shows_partial_status(app, monkeypatch):
    monkeypatch.setenv("MAX_DOCUMENT_CHARS", "2000")
    monkeypatch.setenv("MAX_CHUNKS", "1")
    app.session_state["doc"] = dp.process_upload("long_report.txt", (BUSINESS + "\n\n").encode() * 4)
    app.run()
    _run(app)
    ex = app.session_state["extraction"]
    assert ex.partial and "MAX_CHUNKS" in ex.partial_reasons[0]
    md = _markdown(app)
    assert "Partial analysis" in md and "Analysis completed" not in md
    assert any("Partial analysis" in w.value for w in app.warning)
    assert app.session_state["recent"][-1]["Coverage"] == "Partial"


def test_sidebar_has_labels_and_hides_radio_bullets(app):
    radio = app.sidebar.radio[0]
    assert list(radio.options) == ["Overview", "AI Analyzer", "Data Extraction", "Insights Hub", "Preferences"] or \
        [o.split(" ", 1)[1] for o in radio.options] == ["Overview", "AI Analyzer", "Data Extraction", "Insights Hub",
                                                        "Preferences"]
    css = (ROOT / "components" / "ui.py").read_text()
    assert 'label[data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"])' in css
