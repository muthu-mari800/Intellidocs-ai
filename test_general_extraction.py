import json
from datetime import date

import pytest

from backend import document_processor as dp
from backend.doctype import SPECS
from backend.extraction import (build_extraction_prompt, build_summary_prompt, extract_document,
                                get_active_fields, generate_summary)
from backend.insights import count_mentions, parse_date, parse_metric
from backend.validation import FieldStatus, compute_completeness, export_dict
from components import charts
from models.schemas import (AnalysisConfig, AnalysisLevel, DocType, GeneralData, MetricItem, SummaryStyle)

SAMPLES = {p.name: p for p in dp.list_samples()}
ACADEMIC = dp.load_sample(SAMPLES["sample_academic_notes.txt"]).text
BUSINESS = dp.load_sample(SAMPLES["sample_business_report.txt"]).text


class MockClient:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, system_instruction, json_mode=False):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        return r if isinstance(r, str) else json.dumps(r)


def cfg(level=AnalysisLevel.STANDARD, focus=(), doc_type=None, style=SummaryStyle.BULLETS):
    return AnalysisConfig(level=level, focus=list(focus), document_type=doc_type, summary_style=style)


ACADEMIC_REPLY = {
    "title": "Operating Systems - Lecture Notes",
    "author_or_issuer": "Department of Computer Science, Riverside University",
    "main_topics": ["Paging", "Segmentation", "Virtual memory", "Page replacement"],
    "key_points": ["Paging divides memory into fixed-size frames and pages",
                   "Virtual memory relies on demand paging",
                   "Thrashing happens when a process spends more time paging than executing"],
    "important_facts": ["FIFO can suffer from Belady's anomaly"],
    "dates": [{"date": "14 August 2025", "event": "Lecture date"}],
    "entities": [{"name": "Riverside University", "type": "Organization"}],
}
BUSINESS_REPLY = {
    "title": "Quarterly Business Report - Q3 FY2025",
    "main_topics": ["Revenue", "Operating margin", "Customer churn", "Online sales"],
    "key_points": ["Revenue reached $48.2 million, up 12.5% year-over-year", "Operating margin improved to 14.3%"],
    "findings": ["Online sales grew faster than store sales"],
    "metrics": [{"name": "Revenue", "value": "$48.2 million"}, {"name": "Operating margin", "value": "14.3%"},
                {"name": "Customer churn", "value": "3.1%"}, {"name": "Market share", "value": "8.7%"},
                {"name": "Invented figure", "value": "99.9%"}],
    "dates": [{"date": "30 September 2025", "event": "Period end"}, {"date": "1 January 2030", "event": "invented"}],
}


# ---- adaptive logic per type
def test_active_fields_change_with_type_and_level():
    academic = {lv: get_active_fields(cfg(lv), DocType.ACADEMIC_NOTES) for lv in AnalysisLevel}
    assert academic[AnalysisLevel.BASIC] == ["title", "main_topics", "key_points"]
    assert "important_facts" in academic[AnalysisLevel.STANDARD] and "entities" in academic[AnalysisLevel.DETAILED]
    assert "quality_notes" in academic[AnalysisLevel.DETAILED] and "quality_notes" not in academic[AnalysisLevel.STANDARD]
    business = get_active_fields(cfg(), DocType.BUSINESS_REPORT)
    assert "metrics" in business and "findings" in business and "skills" not in business
    assert get_active_fields(cfg(), DocType.CERTIFICATE) == ["title", "author_or_issuer", "entities", "dates", "important_facts"]


def test_focus_applies_to_general_types_and_ignores_foreign_topics():
    assert get_active_fields(cfg(focus=["Dates", "Key Points"]), DocType.SYLLABUS) == ["key_points", "dates"]
    # resume labels on a non-resume doc -> level defaults; general labels on a resume -> resume level defaults
    assert get_active_fields(cfg(focus=["Skills"]), DocType.BUSINESS_REPORT) == list(SPECS[DocType.BUSINESS_REPORT].standard)
    assert "full_name" in get_active_fields(cfg(focus=["Dates"]), DocType.RESUME)


def test_prompts_are_type_specific():
    p = build_extraction_prompt(BUSINESS, cfg(), doc_type=DocType.BUSINESS_REPORT)
    assert "Business Report" in p and '"metrics"' in p and '"findings"' in p and "skills" not in p.lower().split("<document>")[0]
    resume = build_extraction_prompt(BUSINESS, cfg(), doc_type=DocType.RESUME)
    assert '"skills"' in resume and '"metrics"' not in resume
    s1 = build_summary_prompt(ACADEMIC, cfg(), DocType.ACADEMIC_NOTES)
    s2 = build_summary_prompt(ACADEMIC, cfg(), DocType.RESUME)
    assert "academic notes" in s1 and "recruiter" not in s1 and "recruiter" in s2


# ---- academic document end to end (mock AI)
def test_academic_extraction_detailed():
    client = MockClient([ACADEMIC_REPLY])
    res = extract_document(client, ACADEMIC, cfg(AnalysisLevel.DETAILED))
    assert res.doc_type == DocType.ACADEMIC_NOTES and res.type_source == "rules" and len(client.prompts) == 1
    assert res.data.title.startswith("Operating Systems") and len(res.data.main_topics) == 4
    assert res.data.dates[0].date == "14 August 2025" and res.uncertain == []
    report = compute_completeness(res.data, res.active_fields, res.uncertain, res.required_fields, res.universe)
    assert report.percent == 100 and not report.missing


def test_academic_basic_drops_out_of_scope_fields():
    res = extract_document(MockClient([ACADEMIC_REPLY]), ACADEMIC, cfg(AnalysisLevel.BASIC))
    assert res.data.dates == [] and res.data.entities == [] and res.data.important_facts == []
    assert list(export_dict(res.data, res.active_fields)) == ["title", "main_topics", "key_points"]


def test_missing_information_is_reported_not_invented():
    reply = {"title": None, "main_topics": [], "key_points": ["Paging divides memory"]}
    res = extract_document(MockClient([reply]), ACADEMIC, cfg())
    report = compute_completeness(res.data, res.active_fields, res.uncertain, res.required_fields, res.universe)
    assert res.data.title is None and set(report.missing) == {"title", "main_topics"}
    assert report.percent == 33 and report.statuses["key_points"] == FieldStatus.PRESENT


# ---- business report end to end (mock AI)
def test_business_report_grounding_removes_invented_values():
    res = extract_document(MockClient([BUSINESS_REPLY]), BUSINESS, cfg())
    assert res.doc_type == DocType.BUSINESS_REPORT
    assert [m.name for m in res.data.metrics] == ["Revenue", "Operating margin", "Customer churn", "Market share"]
    assert [d.date for d in res.data.dates] == ["30 September 2025"]
    assert sum("removed" in w for w in res.warnings) == 2


def test_ungrounded_text_fields_are_flagged_uncertain():
    reply = {"title": "Global Quantum Strategy Handbook", "main_topics": ["Revenue"],
             "key_points": ["Blockchain pilots across Antarctica", "Interplanetary taxation rules"], "findings": ["x"]}
    res = extract_document(MockClient([reply]), BUSINESS, cfg())
    assert {"title", "key_points"} <= set(res.uncertain)


def test_user_override_changes_extraction_type():
    res = extract_document(MockClient([ACADEMIC_REPLY]), BUSINESS, cfg(doc_type=DocType.ACADEMIC_NOTES))
    assert res.doc_type == DocType.ACADEMIC_NOTES and res.type_source == "user"


def test_general_summary():
    client = MockClient(["- Revenue grew."])
    assert generate_summary(client, BUSINESS, cfg(), DocType.BUSINESS_REPORT) == "- Revenue grew."
    assert "business report" in client.prompts[0]


# ---- schema, dates, metrics, charts
def test_general_schema_cleaning():
    d = GeneralData(title="  ", main_topics=["A", None, " "], dates=["2024"], entities=["Acme"],
                    metrics=[{"name": "x", "value": 5}])
    assert d.title is None and d.main_topics == ["A"] and d.dates[0].date == "2024"
    assert d.entities[0].name == "Acme" and d.metrics[0].value == "5"


@pytest.mark.parametrize("text,expected", [
    ("14 August 2025", date(2025, 8, 14)), ("August 14, 2025", date(2025, 8, 14)), ("2025-08-14", date(2025, 8, 14)),
    ("1st Oct 2025", date(2025, 10, 1)), ("25/12/2025", date(2025, 12, 25)), ("03/04/2025", None), ("someday", None),
])
def test_parse_date_formats(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("value,unit,expected", [
    ("$48.2 million", None, (48_200_000, "$")), ("14.3%", None, (14.3, "%")), ("12,400", None, (12400, "count")),
    ("3.5", "crore INR", (35_000_000, "₹")), ("n/a", None, (None, "")),
])
def test_parse_metric(value, unit, expected):
    number, group = parse_metric(value, unit)
    assert group == expected[1] and (number == pytest.approx(expected[0]) if expected[0] else number is None)


def test_count_mentions():
    assert count_mentions("Virtual memory", ACADEMIC) >= 2 and count_mentions("quantum", ACADEMIC) == 0


def test_charts_depend_on_content():
    res = extract_document(MockClient([BUSINESS_REPLY]), BUSINESS, cfg())
    groups = charts.metrics_charts(res.data.metrics)
    assert [g for g, _ in groups] == ["%"]  # only the unit with 2+ comparable figures is charted
    assert charts.topic_mentions_chart(res.data.main_topics, BUSINESS) is not None
    assert charts.dates_timeline(res.data.dates) is not None
    assert charts.entity_type_chart([]) is None and charts.dates_timeline([]) is None
    assert charts.metrics_charts([MetricItem(name="One", value="5")]) == []
    assert charts.topic_mentions_chart(["Quantum"], BUSINESS) is None
