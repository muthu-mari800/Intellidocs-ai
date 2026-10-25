import json

import pytest

from backend import ai_service
from backend import document_processor as dp
from backend.doctype import CONFIDENT, SPECS, classify_by_rules
from backend.extraction import ai_classify, resolve_doc_type
from models.schemas import AnalysisConfig, DocType

SAMPLES = {p.name: p for p in dp.list_samples()}


class MockClient:
    def __init__(self, replies=()):
        self.replies, self.calls = list(replies), 0

    def generate(self, prompt, system_instruction, json_mode=False):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


@pytest.mark.parametrize("name,expected", [
    ("sample_resume.txt", DocType.RESUME), ("sample_resume.pdf", DocType.RESUME),
    ("sample_resume_fresher.txt", DocType.RESUME),
    ("sample_academic_notes.txt", DocType.ACADEMIC_NOTES), ("sample_syllabus.txt", DocType.SYLLABUS),
    ("sample_certificate.txt", DocType.CERTIFICATE), ("sample_business_report.txt", DocType.BUSINESS_REPORT),
    ("sample_business_report.pdf", DocType.BUSINESS_REPORT), ("sample_project_report.txt", DocType.PROJECT_REPORT),
])
def test_rules_classify_samples(name, expected):
    result = classify_by_rules(dp.load_sample(SAMPLES[name]).text)
    assert result.doc_type == expected and result.confidence >= CONFIDENT


def test_unclear_document_is_low_confidence_general():
    r = classify_by_rules("Just a short random note about lunch plans on Friday.")
    assert r.doc_type == DocType.GENERAL and r.confidence < CONFIDENT


def test_every_type_has_a_spec_with_required_fields_in_detailed():
    for doc_type in DocType:
        spec = SPECS[doc_type]
        assert set(spec.required) <= set(spec.detailed) and set(spec.basic) <= set(spec.standard) <= set(spec.detailed)


def test_user_override_wins_without_api_call():
    client = MockClient()
    r = resolve_doc_type(client, "anything", AnalysisConfig(document_type=DocType.SYLLABUS))
    assert r.doc_type == DocType.SYLLABUS and r.source == "user" and client.calls == 0


def test_confident_rules_skip_the_api():
    client = MockClient()
    text = dp.load_sample(SAMPLES["sample_business_report.txt"]).text
    r = resolve_doc_type(client, text, AnalysisConfig())
    assert r.doc_type == DocType.BUSINESS_REPORT and r.source == "rules" and client.calls == 0


def test_low_confidence_asks_the_ai():
    client = MockClient([{"document_type": "resume", "confidence": 0.9, "reason": "Lists skills and jobs."}])
    text = dp.load_sample(SAMPLES["sample_resume_incomplete.txt"]).text
    r = resolve_doc_type(client, text, AnalysisConfig())
    assert r.doc_type == DocType.RESUME and r.source == "ai" and client.calls == 1


@pytest.mark.parametrize("reply", [ai_service.AIServiceError("down"), "not json", {"document_type": "poem"}])
def test_ai_classification_failure_falls_back_to_rules(reply):
    r = resolve_doc_type(MockClient([reply]), "Short unclear text.", AnalysisConfig())
    assert r.source == "rules" and r.confidence < CONFIDENT and "confirm" in r.reason


def test_ai_classify_clamps_confidence():
    r = ai_classify(MockClient([{"document_type": "general", "confidence": 7}]), "text")
    assert r.confidence == 1.0
