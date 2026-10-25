"""End-to-end UI smoke test using Streamlit's AppTest with a mocked AI client (no network, no key)."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from backend import ai_service

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Overview", "AI Analyzer", "Data Extraction", "Insights Hub", "Preferences"]

EXTRACTION = {"full_name": "Priya Raman", "email": "priya.raman@example.com",
              "skills": ["Python", "SQL", "Docker"],
              "education": [{"degree": "B.E. Computer Science", "institution": "Anna University", "year": "2020"}],
              "experience": [{"title": "Software Engineer", "company": "Nimbus Tech", "start_date": "Jun 2022", "end_date": "Present"}]}


class FakeClient:
    def generate(self, prompt, system_instruction, json_mode=False):
        return json.dumps(EXTRACTION) if json_mode else "- Priya is a backend developer."


def load(at, name="sample_resume.txt"):
    at.button(key=f"sample_{name}").click().run()


def run_analysis(at):
    next(b for b in at.button if b.label == "Run Analysis").click().run()


@pytest.fixture
def at(monkeypatch):
    monkeypatch.setattr(ai_service, "get_llm_client", lambda model_override=None: FakeClient())
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    return app


def test_full_flow(at):
    assert not at.exception
    load(at)
    assert at.session_state["doc"].filename == "sample_resume.txt"
    run_analysis(at)
    assert not at.exception and at.session_state["extraction"] is not None
    assert "backend developer" in at.session_state["summary"]["text"]
    assert at.session_state["last_run_seconds"] is not None
    for page in PAGES:
        at.session_state["nav"] = page
        at.run()
        assert not at.exception, page


def test_overview_shows_real_values_after_analysis(at):
    load(at)
    run_analysis(at)
    labels = {m.label: m.value for m in at.metric}
    assert labels["Characters"] and labels["Summary Words"] != "—" and labels["Processing Time"].endswith("s")


def test_pages_without_data_show_empty_states(at):
    for page in PAGES[1:4]:
        at.session_state["nav"] = page
        at.run()
        assert not at.exception and not at.error, page


def test_settings_survive_page_changes(at):
    at.session_state["cfg_level"] = "Detailed"
    at.run()
    at.session_state["nav"] = "Preferences"
    at.run()
    at.session_state["nav"] = "Overview"
    at.run()
    assert at.session_state["cfg_level"] == "Detailed"


def test_missing_api_key_shows_error_not_crash(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(APP, default_timeout=30).run()
    load(app)
    run_analysis(app)
    assert not app.exception and any("GEMINI_API_KEY" in e.value for e in app.error)
