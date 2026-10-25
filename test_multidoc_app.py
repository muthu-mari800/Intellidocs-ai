"""End-to-end UI tests with three different document types (AI mocked: no network, no key)."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from backend import ai_service
from backend import document_processor as dp
from tests.test_general_extraction import ACADEMIC_REPLY, BUSINESS_REPLY
from tests.test_smoke_data import RESUME_REPLY

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SAMPLES = {p.name: p for p in dp.list_samples()}


class FakeClient:
    """Answers extraction by the document type named in the prompt; summaries are plain text."""

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
    return AppTest.from_file(APP, default_timeout=30).run()


def analyse(app, sample, level="Standard"):
    app.button(key=f"sample_{sample}").click().run()
    app.session_state["cfg_level"] = level
    app.run()
    next(b for b in app.button if b.label == "Run Analysis").click().run()
    assert not app.exception, app.exception
    return app.session_state["extraction"]


def chart_titles(app):
    app.session_state["nav"] = "Insights Hub"
    app.run()
    assert not app.exception, app.exception
    titles = []
    for el in app.get("plotly_chart"):
        spec = json.loads(el.proto.spec)
        title = spec.get("layout", {}).get("title", {}).get("text")
        titles.append(title or "")
    return titles


def test_resume_gets_resume_charts_only(app):
    ex = analyse(app, "sample_resume.txt")
    assert ex.doc_type.value == "resume" and ex.type_source == "rules"
    titles = chart_titles(app)
    assert {"Skills by category", "Field completeness", "Work experience timeline"} <= set(titles)
    assert not any("Topic mentions" in t or "Key figures" in t for t in titles)


def test_academic_document_gets_topic_and_date_charts(app):
    ex = analyse(app, "sample_academic_notes.txt", level="Detailed")
    assert ex.doc_type.value == "academic_notes" and ex.data.title
    titles = chart_titles(app)
    assert "Topic mentions in the document" in titles and "Key dates timeline" in titles
    assert not any("Skills by category" in t or "Work experience" in t or "Education overview" in t for t in titles)


def test_business_report_gets_figures_chart(app):
    ex = analyse(app, "sample_business_report.pdf")
    assert ex.doc_type.value == "business_report" and len(ex.data.metrics) == 4  # invented figure removed
    titles = chart_titles(app)
    assert "Key figures (%)" in titles and not any("Skills by category" in t for t in titles)
    assert app.session_state["recent"][-1]["Type"] == "Business Report"


def test_extracted_page_edit_flow_for_general_document(app):
    ex = analyse(app, "sample_academic_notes.txt")
    app.session_state["nav"] = "Data Extraction"
    app.run()
    assert not app.exception
    aid = app.session_state["analysis_id"]
    labels = [m.label for m in app.metric]
    assert "Completeness" in labels and app.metric[0].value == "100%"
    app.text_input(key=f"title_{aid}").set_value("").run()
    assert not app.exception and app.metric[0].value == "67%"
    assert app.session_state["current"].title is None


def test_manual_type_override_in_ui(app):
    app.button(key="sample_sample_academic_notes.txt").click().run()
    app.selectbox(key="cfg_doctype").select("Business Report").run()
    next(b for b in app.button if b.label == "Run Analysis").click().run()
    ex = app.session_state["extraction"]
    assert ex.doc_type.value == "business_report" and ex.type_source == "user"


def test_unsupported_and_empty_uploads_still_handled():
    from backend.document_processor import DocumentError, process_upload
    with pytest.raises(DocumentError):
        process_upload("report.docx", b"x")
    with pytest.raises(DocumentError):
        process_upload("report.txt", b"")
