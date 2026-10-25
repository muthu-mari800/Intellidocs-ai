"""IntelliDocs AI - Intelligent Document Analysis & Insight Platform (Streamlit)."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from backend import ai_service
from backend import document_processor as dp
from backend.config import get_settings
from backend.doctype import CONFIDENT, SPECS, classify_by_rules
from backend.extraction import (ExtractionError, extract_document_chunks, generate_summary_chunks,
                                get_active_fields, level_description)
from backend.insights import total_experience_months
from backend.validation import (FIELD_LABELS, compute_completeness, effective_uncertain, export_dict)
from components import charts, ui
from models.schemas import (DOC_TYPE_LABELS, FOCUS_TOPICS, GENERAL_FOCUS_TOPICS, AnalysisConfig, AnalysisLevel,
                            DocType, GeneralData, ResumeData, SummaryStyle)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("intellidocs")

st.set_page_config(page_title="IntelliDocs AI", page_icon="📄", layout="wide")
ui.inject_css()

PAGES = ["Overview", "AI Analyzer", "Data Extraction", "Insights Hub", "Preferences"]
PAGE_ICONS = {"Overview": ":material/home:", "AI Analyzer": ":material/plagiarism:", "Data Extraction": ":material/database:",
              "Insights Hub": ":material/bar_chart:", "Preferences": ":material/settings:"}
MODEL_DEFAULT = "Default (from .env)"
MODEL_CHOICES = [MODEL_DEFAULT, "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.5-flash",
                 "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
STYLE_SHORT = {SummaryStyle.BRIEF.value: "Brief", SummaryStyle.DETAILED.value: "Detailed",
               SummaryStyle.BULLETS.value: "Bullet Points"}
PERSISTED_KEYS = ("cfg_style", "cfg_level", "cfg_focus", "cfg_doctype", "model_choice", "model_override")
EDU_COLS = ["degree", "field_of_study", "institution", "year"]
EXP_COLS = ["title", "company", "start_date", "end_date", "highlights"]
AUTO = "Auto-detect"
TYPE_OPTIONS = [AUTO] + [DOC_TYPE_LABELS[t] for t in DocType]
LABEL_TO_TYPE = {v: k for k, v in DOC_TYPE_LABELS.items()}


# ------------------------------------------------------------------ state
def init_state() -> None:
    defaults = {
        "nav": "Overview", "model_choice": MODEL_DEFAULT, "last_run_seconds": None, "doc": None, "last_upload_digest": None, "upload_error": None,
        "summary": None, "extraction": None, "analysis_config": None, "analysis_id": 0,
        "current": None, "recent": [], "feedback": [], "model_override": "", "analysis_status": None,
        "doc_type_guess": None, "cfg_doctype": AUTO, "cfg_level": AnalysisLevel.STANDARD.value, "cfg_style": SummaryStyle.BULLETS.value, "cfg_focus": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go_to(page: str) -> None:
    st.session_state["nav"] = page


def set_document(doc: dp.ProcessedDocument) -> None:
    st.session_state.update(doc=doc, doc_type_guess=classify_by_rules(doc.text), summary=None, extraction=None, current=None, analysis_config=None,
                            upload_error=None, feedback=[], analysis_status=None)
    st.session_state.analysis_id += 1


def selected_override() -> DocType | None:
    return LABEL_TO_TYPE.get(st.session_state.cfg_doctype)


def ui_doc_type() -> DocType:
    """Type used to decide which options to show before an analysis has run."""
    override = selected_override()
    if override:
        return override
    guess = st.session_state.doc_type_guess
    return guess.doc_type if guess else DocType.GENERAL


def focus_options(doc_type: DocType) -> list[str]:
    return list(FOCUS_TOPICS) if doc_type == DocType.RESUME else list(GENERAL_FOCUS_TOPICS)


def current_config() -> AnalysisConfig:
    return AnalysisConfig(level=AnalysisLevel(st.session_state.cfg_level or AnalysisLevel.STANDARD.value),
                          summary_style=SummaryStyle(st.session_state.cfg_style or SummaryStyle.BULLETS.value),
                          focus=list(st.session_state.cfg_focus), document_type=selected_override())


def effective_model() -> str | None:
    """Session text override (Preferences) > header model choice > GEMINI_MODEL from the environment."""
    custom = (st.session_state.model_override or "").strip()
    if custom:
        return custom
    choice = st.session_state.model_choice
    return None if choice == MODEL_DEFAULT else choice


def page_header(title: str, subtitle: str) -> None:
    left, mid, right = st.columns([3, 1.25, 1.25])
    left.markdown(f"<h1 class='idx-title'>{title}</h1><p class='idx-sub2'>{subtitle}</p>", unsafe_allow_html=True)
    mid.selectbox("Gemini Model", MODEL_CHOICES, key="model_choice",
                  help="Pick a model for this session. Default uses GEMINI_MODEL from your .env file.")
    right.selectbox("Document Type", TYPE_OPTIONS, key="cfg_doctype",
                    help="Auto-detect uses keyword rules and, when unsure, Gemini. Pick a type to override it.")


def current_data() -> ResumeData | None:
    ex = st.session_state.extraction
    if ex is None:
        return None
    return st.session_state.current or ex.data


def current_report():
    ex, data = st.session_state.extraction, current_data()
    if ex is None or data is None:
        return None
    unc = effective_uncertain(ex.data, data, ex.uncertain)
    return compute_completeness(data, ex.active_fields, unc, required=ex.required_fields, universe=ex.universe,
                                partial=ex.partial, not_applicable=ex.not_applicable)


def show_partial_status(container=st) -> None:
    """Explain clearly when only part of the document was analysed (never claim a full analysis)."""
    status = st.session_state.analysis_status
    if status and status["reasons"]:
        container.warning("**Partial analysis** — only part of the document was analysed. "
                          + " ".join(status["reasons"]) + " Results cover the analysed portion only.")


def need_document() -> bool:
    if st.session_state.doc is None:
        ui.empty_state("No document loaded", "Upload a PDF/TXT resume or load a sample from the Dashboard.")
        st.button("Go to Dashboard", on_click=go_to, args=("Overview",))
        return False
    return True


def need_analysis() -> bool:
    if st.session_state.extraction is None:
        ui.empty_state("No analysis yet", "Run an analysis on the Document Analysis page first.")
        st.button("Go to Document Analysis", on_click=go_to, args=("AI Analyzer",))
        return False
    return True


# ------------------------------------------------------------------ sidebar
def sidebar() -> None:
    with st.sidebar:
        ui.brand()
        st.write("")
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed",
                 format_func=lambda p: f"{PAGE_ICONS[p]} {p}")


def handle_upload(uploaded) -> None:
    if uploaded is None:
        st.session_state.last_upload_digest = None
        st.session_state.upload_error = None
        return
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if digest == st.session_state.last_upload_digest:
        return
    st.session_state.last_upload_digest = digest
    try:
        set_document(dp.process_upload(uploaded.name, data))
    except dp.DocumentError as exc:
        st.session_state.upload_error = str(exc)
    except Exception as exc:  # never crash the UI; never log document content
        logger.warning("Unexpected upload error: %s", type(exc).__name__)
        st.session_state.upload_error = "Something went wrong while reading this file."


def load_sample_doc(path) -> None:
    try:
        set_document(dp.load_sample(path))
    except dp.DocumentError as exc:
        st.session_state.upload_error = str(exc)


def analysis_controls() -> AnalysisConfig:
    """Summary style, analysis level and focus fields (shared by Overview and AI Analyzer)."""
    dtype = ui_doc_type()
    st.caption("Summary Style")
    st.segmented_control("Summary style", [x.value for x in SummaryStyle], key="cfg_style",
                         format_func=STYLE_SHORT.get, label_visibility="collapsed", width="stretch")
    st.caption("Analysis Level")
    st.segmented_control("Analysis level", [x.value for x in AnalysisLevel], key="cfg_level",
                         label_visibility="collapsed", width="stretch")
    level = AnalysisLevel(st.session_state.cfg_level or AnalysisLevel.STANDARD.value)
    st.caption(level_description(level, dtype))
    options = focus_options(dtype)
    st.session_state.cfg_focus = [t for t in st.session_state.cfg_focus if t in options]
    with st.expander("Focus on specific fields (optional)"):
        st.multiselect("Focus fields", options, key="cfg_focus", label_visibility="collapsed",
                       help="Leave empty to analyse all fields for the selected level and document type.")
    config = current_config()
    st.caption("Fields analysed: " + ", ".join(FIELD_LABELS[f] for f in get_active_fields(config, dtype)))
    return config


def page_overview() -> None:
    page_header("Welcome to IntelliDocs AI", "Turn your documents into valuable insights with AI.")
    settings = get_settings()

    left, right = st.columns([1.55, 1])
    with left, st.container(border=True):
        ui.card_title("☁️", "Upload Document", "Upload a PDF or TXT file. We'll analyze and extract key information.")
        up, samples = st.columns([1.1, 1])
        with up:
            uploaded = st.file_uploader("Drag and drop a PDF or TXT file", key="uploader",
                                        label_visibility="collapsed", type=["pdf", "txt"],
                                        max_upload_size=max(1, int(settings.max_file_mb)),
                                        help="Files are processed in memory and not stored.")
            st.caption(f"Supported: PDF, TXT (Max {settings.max_file_mb:g} MB)")
            handle_upload(uploaded)
        with samples:
            st.markdown("**Sample Documents**")
            with st.container(height=190):
                for path in dp.list_samples():
                    if st.button(f"📄  {path.name}", key=f"sample_{path.name}", width="stretch"):
                        load_sample_doc(path)
        if st.session_state.upload_error:
            st.error(st.session_state.upload_error)

    doc = st.session_state.doc
    with right, st.container(border=True):
        ui.card_title("✨", "Document Analysis", "Choose your analysis type and adjust settings for better results.")
        config = analysis_controls()
        if st.button("Run Analysis", type="primary", width="stretch", disabled=doc is None, icon=":material/bolt:"):
            run_analysis(doc, config)
        if doc is None:
            st.caption("Upload or select a document to enable analysis.")

    ex, report = st.session_state.extraction, current_report()
    left, right = st.columns([1.55, 1])
    with left, st.container(border=True):
        ui.card_title("📄", "Current Document")
        if doc is None:
            ui.empty_state("Nothing loaded yet", "Upload a resume, report, notes, syllabus or certificate, or pick a sample.")
        else:
            pages = f"{doc.page_count} page{'s' if doc.page_count != 1 else ''} · " if doc.page_count else ""
            meta = f"{doc.file_type} · {doc.size_bytes / 1024:.0f} KB · {pages}{'sample data' if doc.is_sample else 'uploaded'}"
            run = st.session_state.analysis_status
            if not ex:
                status = ui.badge("na", "Not analysed yet")
            elif ex.partial or (run and run["reasons"]):
                status = ui.badge("uncertain", "Partial analysis")
            else:
                status = ui.badge("present", "Analysis completed")
            ui.file_row(doc.filename, meta, status)
            if ex:
                show_partial_status()
                if run and run["parts"] > 1 and not run["reasons"]:
                    st.caption(f"Long document: the full text was analysed in {run['parts']} overlapping parts "
                               "and the results were combined.")
            guess = st.session_state.doc_type_guess
            if ex:
                st.markdown("Document type: " + ui.type_badge(DOC_TYPE_LABELS[ex.doc_type], ex.type_source,
                                                              ex.type_confidence), unsafe_allow_html=True)
            elif guess:
                st.markdown("Detected type (preliminary): " + ui.type_badge(guess.label, guess.source, guess.confidence),
                            unsafe_allow_html=True)
            for w in doc.warnings:
                st.warning(w)
            st.markdown("**Analysis Overview**")
            summary = st.session_state.summary
            secs = st.session_state.last_run_seconds
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pages", doc.page_count if doc.page_count else "n/a")
            m2.metric("Characters", f"{doc.char_count:,}")
            m3.metric("Summary Words", len(summary["text"].split()) if summary else "—")
            m4.metric("Processing Time", f"{secs:.1f}s" if secs is not None else "—")
            with st.expander("Preview extracted text"):
                st.text(doc.text[:3000] + ("\n…" if doc.char_count > 3000 else ""))
    with right, st.container(border=True):
        ui.card_title("🎯", "Document Completeness")
        if report is None or report.percent is None:
            ui.empty_state("No data yet", "Run an analysis to see how complete the extracted information is.")
        else:
            ring_col, legend_col = st.columns([1, 1.35])
            ring_col.markdown(ui.ring(report.percent), unsafe_allow_html=True)
            total = len(report.required)
            legend_col.markdown(
                ui.legend_row("#34d399", "Extracted Fields", f"{len(report.present) + len(report.uncertain)} / {total}")
                + ui.legend_row("#fbbf24", "Uncertain Fields", str(len(report.uncertain)))
                + ui.legend_row("#fb7185", "Missing Fields", str(len(report.missing)))
                + (ui.legend_row("#64748b", "Not Verified", str(len(report.not_verified))) if report.not_verified else "")
                + ui.legend_row("#5b6bff", "Type Confidence", f"{round(ex.type_confidence * 100)}%"),
                unsafe_allow_html=True)
            st.caption(f"Required fields for {DOC_TYPE_LABELS[ex.doc_type]}: "
                       + ", ".join(FIELD_LABELS[f] for f in report.required) + ".")
            if report.missing or report.uncertain:
                names = ", ".join(FIELD_LABELS[f] for f in report.missing + report.uncertain)
                st.warning(f"Missing or uncertain: {names}. Review the extraction results.")
            if report.not_verified:
                why = ("they were not found in the analysed part of the document" if report.partial
                       else "they were outside the selected analysis level or focus fields")
                st.info("Not verified: " + ", ".join(FIELD_LABELS[f] for f in report.not_verified)
                        + f" — {why}. They are not counted as found.")
            if report.missing or report.uncertain or report.not_verified:
                st.button("Review in Data Extraction", on_click=go_to, args=("Data Extraction",), key="ov_review")
            else:
                st.success("All required fields were checked and are present.")

    left, right = st.columns([1.55, 1])
    with left, st.container(border=True):
        ui.card_title("📈", "Document Insights")
        if ex is None:
            ui.empty_state("No insights yet", "Charts appear here after you run an analysis.")
        else:
            data = current_data()
            if ex.doc_type == DocType.RESUME:
                res = charts.skills_category_chart(data.skills, data.skill_categories)
                content_fig = res[0] if res else None
            else:
                content_fig = (charts.topic_mentions_chart(data.main_topics, doc.text)
                               if "main_topics" in ex.active_fields else None)
            figs = [(charts.field_completeness_chart(report), "No required fields were checked."),
                    (content_fig, "No skills or topics that appear in the document text were extracted."),
                    (charts.field_counts_chart(data, ex.active_fields), "No list items were extracted to count.")]
            if not any(fig for fig, _ in figs):
                ui.empty_state("No chart data", "The extracted information has no numeric or categorical values to chart.")
            else:
                for col, (fig, why) in zip(st.columns(3), figs):
                    with col:
                        if fig:
                            fig.update_layout(height=250, title_font_size=13)
                            st.plotly_chart(fig, width="stretch")
                        else:
                            ui.empty_state("Chart unavailable", why)
            facts = (data.skills[:8] if ex.doc_type == DocType.RESUME
                     else (data.key_points or data.findings or data.important_facts)[:4])
            if facts:
                st.markdown("**Text insights**" + ("  \n" + ", ".join(facts) if ex.doc_type == DocType.RESUME
                                                   else "\n" + "\n".join(f"- {f}" for f in facts)))
            if ex.partial:
                st.caption("Insights are based on the analysed portion of the document only.")
            st.button("Open Insights Hub", on_click=go_to, args=("Insights Hub",), key="ov_insights")
    with right, st.container(border=True):
        ui.card_title("{ }", "Extracted Information")
        if ex is None:
            ui.empty_state("Nothing extracted yet", "The validated JSON appears here after an analysis.")
        else:
            payload = {"document_type": ex.doc_type.value, **export_dict(current_data(), ex.active_fields)}
            text = json.dumps(payload, indent=2, ensure_ascii=False)
            with st.container(height=230):
                st.code(text, language="json")
            b1, b2 = st.columns(2)
            b1.download_button("Download JSON", text, file_name="document_extraction.json",
                               mime="application/json", width="stretch", key="ov_download")
            b2.button("Edit fields", on_click=go_to, args=("Data Extraction",), width="stretch", key="ov_edit")

    if st.session_state.recent:
        with st.expander("Recent analyses (this session)"):
            st.dataframe(pd.DataFrame(st.session_state.recent[::-1]), hide_index=True, width="stretch")


def run_analysis(doc: dp.ProcessedDocument, config: AnalysisConfig, summary_only: bool = False) -> None:
    settings = get_settings()
    plan = dp.plan_chunks(doc.text, settings.max_chars, settings.chunk_overlap, settings.max_chunks)
    doc_reasons: list[str] = []
    if doc.page_count and doc.page_count > settings.max_pdf_pages:
        doc_reasons.append(f"Only the first {settings.max_pdf_pages} of {doc.page_count} PDF pages were read "
                           "(MAX_PDF_PAGES limit).")
    if not plan.complete:
        doc_reasons.append(
            f"The text needs {plan.total_parts} parts of up to {settings.max_chars:,} characters, but at most "
            f"{len(plan.chunks)} are processed (MAX_CHUNKS limit); the first {plan.analysed_chars:,} of "
            f"{plan.total_chars:,} characters ({plan.analysed_chars * 100 // plan.total_chars}%) were analysed.")
    try:
        client = ai_service.get_llm_client(effective_model())
    except ai_service.AIServiceError as exc:
        st.error(str(exc))
        return
    errors: list[str] = []
    summary_reasons: list[str] = []
    started = time.perf_counter()
    parts = f" ({len(plan.chunks)} parts)" if len(plan.chunks) > 1 else ""
    with st.spinner(f"Analysing document with Gemini…{parts}"):
        if not summary_only:
            try:
                ex_new = extract_document_chunks(client, plan.chunks, config)
                ex_new.partial_reasons = doc_reasons + ex_new.partial_reasons
                st.session_state.extraction = ex_new
                st.session_state.current = None
                st.session_state.analysis_config = config
                st.session_state.analysis_id += 1
            except (ai_service.AIServiceError, ExtractionError) as exc:
                errors.append(f"Extraction failed: {exc}")
        ex = st.session_state.extraction
        doc_type = ex.doc_type if ex else ui_doc_type()
        try:
            text, summary_reasons = generate_summary_chunks(client, plan.chunks, config, doc_type)
            st.session_state.summary = {"text": text, "level": config.level.value, "style": config.summary_style.value,
                                        "type": DOC_TYPE_LABELS[doc_type], "partial_reasons": summary_reasons}
        except (ai_service.AIServiceError, ExtractionError) as exc:
            errors.append(f"Summary failed: {exc}")
    st.session_state.last_run_seconds = round(time.perf_counter() - started, 1)
    ex = st.session_state.extraction
    extraction_reasons = [r for r in (ex.partial_reasons if ex else []) if r not in doc_reasons]
    failed_steps = [e.split(":")[0] + "." for e in errors]
    reasons = list(dict.fromkeys(doc_reasons + extraction_reasons + summary_reasons + failed_steps))
    st.session_state.analysis_status = {"reasons": reasons, "parts": len(plan.chunks)}
    if len(plan.chunks) > 1 and not reasons:
        st.info(f"The document is long ({plan.total_chars:,} characters), so it was analysed in {len(plan.chunks)} "
                f"overlapping parts of up to {settings.max_chars:,} characters and the results were combined.")
    show_partial_status()
    for e in errors:
        st.error(e)
    if not summary_only and st.session_state.extraction is not None and not errors:
        report = current_report()
        st.session_state.recent.append({
            "Document": doc.filename, "Type": DOC_TYPE_LABELS[st.session_state.extraction.doc_type],
            "Level": config.level.value, "Summary": config.summary_style.value,
            "Completeness": f"{report.percent}%" if report and report.percent is not None else "n/a",
            "Coverage": "Partial" if reasons else "Full",
            "Time": datetime.now().strftime("%H:%M:%S")})
        del st.session_state.recent[:-10]


def page_analysis() -> None:
    page_header("AI Analyzer", "Summaries, findings and data-quality checks for your document.")
    if not need_document():
        return
    doc = st.session_state.doc
    with st.container(border=True):
        ui.card_title("✨", "Analysis Settings", "Document type and model are chosen in the header above.")
        guess = st.session_state.doc_type_guess
        if selected_override() is None and guess:
            st.caption(f"Preliminary detection: {guess.label} ({round(guess.confidence * 100)}% confidence, keyword rules). "
                       "The final type is confirmed when you run the analysis.")
        config = analysis_controls()
        c1, c2 = st.columns(2)
        if c1.button("Run Analysis", type="primary", width="stretch", icon=":material/bolt:"):
            run_analysis(doc, config)
        if c2.button("Regenerate summary", width="stretch", disabled=st.session_state.summary is None):
            run_analysis(doc, config, summary_only=True)

    if st.session_state.extraction is not None and st.session_state.analysis_config != config:
        st.info("Settings changed since the last extraction. Run the analysis again to apply them to the extracted fields.")

    summary, ex = st.session_state.summary, st.session_state.extraction
    if summary is None and ex is None:
        ui.empty_state("No results yet", "Choose your settings and click Run Analysis.")
        return
    if summary:
        with st.container(border=True):
            ui.card_title("📝", "AI Summary")
            st.caption(f"{summary['style']} · {summary['level']} level · {summary.get('type', 'Resume / CV')}")
            for reason in summary.get("partial_reasons", []):
                st.warning(reason)
            st.markdown(summary["text"])
            st.caption("AI-generated from the uploaded document. Please review for accuracy.")
    if ex:
        report = current_report()
        with st.container(border=True):
            ui.card_title("🔎", "Findings")
            st.markdown("Document type: " + ui.type_badge(DOC_TYPE_LABELS[ex.doc_type], ex.type_source, ex.type_confidence),
                        unsafe_allow_html=True)
            if ex.type_reason and ex.type_source != "user":
                st.caption(ex.type_reason)
            if ex.type_source == "rules" and ex.type_confidence < CONFIDENT:
                st.warning("The document type is uncertain. Choose the correct type in the header and run the analysis again.")
            show_partial_status()
            for w in ex.warnings:
                st.warning(w)
            if report.percent is not None:
                st.progress(report.percent / 100, text=f"Completeness: {report.percent}%")
            if report.missing:
                st.error("Missing required fields: " + ", ".join(FIELD_LABELS[f] for f in report.missing))
            if report.uncertain:
                st.warning("Uncertain fields to review: " + ", ".join(FIELD_LABELS[f] for f in report.uncertain))
            if report.not_verified:
                st.info("Not verified (not checked or not in the analysed text): "
                        + ", ".join(FIELD_LABELS[f] for f in report.not_verified))
            if ex.data.quality_notes:
                st.markdown("**Data quality observations**")
                for note in ex.data.quality_notes:
                    st.markdown(f"- {note}")
            st.button("Review & edit extracted fields", on_click=go_to, args=("Data Extraction",))


def _df(items, cols) -> pd.DataFrame:
    rows = []
    for item in items:
        row = item.model_dump()
        if "highlights" in row:
            row["highlights"] = " | ".join(row["highlights"])
        rows.append({c: row.get(c) for c in cols})
    return pd.DataFrame(rows, columns=cols).astype("object")


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for rec in df.to_dict("records"):
        rec = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        if any(v not in (None, "") for v in rec.values()):
            if "highlights" in rec:
                rec["highlights"] = [h.strip() for h in (rec["highlights"] or "").split("|") if h.strip()]
            out.append(rec)
    return out


def _lines(text: str, split_commas: bool = False) -> list[str]:
    parts = re.split(r"[\n,]+" if split_commas else r"\n+", text or "")
    return [p.strip() for p in parts if p.strip()]


def add_feedback() -> None:
    aid = st.session_state.analysis_id
    text = (st.session_state.get(f"fb_{aid}") or "").strip()
    if text:
        st.session_state.feedback.append({"time": datetime.now().strftime("%H:%M:%S"), "comment": text})
        st.session_state[f"fb_{aid}"] = ""


def _resume_form(orig: ResumeData, active: set, aid: int) -> dict:
    values: dict = {}
    if {"full_name", "email"} & active:
        c1, c2 = st.columns(2)
        if "full_name" in active:
            values["full_name"] = c1.text_input("Full name", orig.full_name or "", key=f"name_{aid}") or None
        if "email" in active:
            values["email"] = c2.text_input("Email address", orig.email or "", key=f"email_{aid}") or None
    if "skills" in active:
        values["skills"] = _lines(st.text_area("Skills (one per line or comma-separated)",
                                               "\n".join(orig.skills), key=f"skills_{aid}", height=120), True)
    if "education" in active:
        st.markdown("**Education**")
        values["education"] = _records(st.data_editor(_df(orig.education, EDU_COLS), num_rows="dynamic",
                                                      key=f"edu_{aid}", width="stretch"))
    if "experience" in active:
        st.markdown("**Work experience** (separate highlights with `|`)")
        values["experience"] = _records(st.data_editor(_df(orig.experience, EXP_COLS), num_rows="dynamic",
                                                       key=f"exp_{aid}", width="stretch"))
    if "certifications" in active:
        values["certifications"] = _lines(st.text_area("Certifications (one per line)",
                                                       "\n".join(orig.certifications), key=f"cert_{aid}"))
    if "projects" in active:
        values["projects"] = _lines(st.text_area("Projects (one per line)",
                                                 "\n".join(orig.projects), key=f"proj_{aid}"))
    return values


def _general_form(orig: GeneralData, active: set, aid: int) -> dict:
    values: dict = {}
    if {"title", "author_or_issuer"} & active:
        c1, c2 = st.columns(2)
        if "title" in active:
            values["title"] = c1.text_input("Title", orig.title or "", key=f"title_{aid}") or None
        if "author_or_issuer" in active:
            values["author_or_issuer"] = c2.text_input("Author / issuer", orig.author_or_issuer or "",
                                                       key=f"author_{aid}") or None
    for name, label, height in [("main_topics", "Main topics (one per line)", 100),
                                ("key_points", "Key points (one per line)", 150),
                                ("findings", "Findings (one per line)", 120),
                                ("important_facts", "Important facts (one per line)", 120)]:
        if name in active:
            values[name] = _lines(st.text_area(label, "\n".join(getattr(orig, name)), key=f"{name}_{aid}", height=height))
    for name, label, cols in [("dates", "Dates", ["date", "event"]), ("entities", "Entities", ["name", "type"]),
                              ("metrics", "Metrics & figures", ["name", "value", "unit"])]:
        if name in active:
            st.markdown(f"**{label}**")
            values[name] = _records(st.data_editor(_df(getattr(orig, name), cols), num_rows="dynamic",
                                                   key=f"{name}_{aid}", width="stretch"))
    return values


def page_extracted() -> None:
    page_header("Data Extraction", "Review, correct and download the validated structured data.")
    if not need_analysis():
        return
    ex, aid = st.session_state.extraction, st.session_state.analysis_id
    orig, active = ex.data, set(ex.active_fields)
    st.markdown("Document type: " + ui.type_badge(DOC_TYPE_LABELS[ex.doc_type], ex.type_source, ex.type_confidence),
                unsafe_allow_html=True)
    summary_box = st.container(border=True)

    with st.container(border=True):
        st.subheader("Review and correct")
        st.caption("Edit any value. Results, completeness and downloads update immediately.")
        values = _resume_form(orig, active, aid) if ex.doc_type == DocType.RESUME else _general_form(orig, active, aid)

    try:
        if ex.doc_type == DocType.RESUME:
            kept = {x.lower() for x in values.get("skills", orig.skills)}
            data = ResumeData.model_validate({**values, "skill_categories": [
                {"category": c.category, "skills": [s for s in c.skills if s.lower() in kept]}
                for c in orig.skill_categories], "quality_notes": orig.quality_notes})
        else:
            data = GeneralData.model_validate({**values, "quality_notes": orig.quality_notes})
    except Exception:
        st.error("Some edited values are invalid. Please check the tables.")
        return
    st.session_state.current = data
    report = current_report()

    with summary_box:
        st.subheader("Validation & completeness")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Completeness", f"{report.percent}%" if report.percent is not None else "n/a")
        m2.metric("Present", len(report.present))
        m3.metric("Uncertain", len(report.uncertain))
        m4.metric("Missing / invalid", len(report.missing))
        if report.percent is not None:
            st.progress(report.percent / 100)
        show_partial_status()
        if report.missing:
            st.error("Missing required fields: " + ", ".join(FIELD_LABELS[f] for f in report.missing))
        if report.not_verified:
            st.info("Not verified: " + ", ".join(FIELD_LABELS[f] for f in report.not_verified)
                    + ". These required fields were not checked, so they are neither counted as found nor as missing.")
        ui.field_status_table(report, data)
        st.caption("Completeness counts the required fields of this document type that were analysed: "
                   "present = 1, uncertain = 0.5, missing/invalid = 0.")

    with st.container(border=True):
        st.subheader("Validated JSON")
        payload = {"document_type": ex.doc_type.value, **export_dict(data, ex.active_fields)}
        st.json(payload)
        d1, d2 = st.columns(2)
        d1.download_button("Download corrected JSON", json.dumps(payload, indent=2, ensure_ascii=False),
                           file_name="document_extraction.json", mime="application/json", width="stretch")
        report_payload = {"document_type": ex.doc_type.value, "completeness_percent": report.percent,
                          "field_status": {f: report.statuses[f].value for f in report.statuses},
                          "missing_fields": report.missing, "uncertain_fields": report.uncertain,
                          "not_verified_fields": report.not_verified, "partial_analysis": report.partial,
                          "reviewer_feedback": st.session_state.feedback}
        d2.download_button("Download validation report", json.dumps(report_payload, indent=2),
                           file_name="validation_report.json", mime="application/json", width="stretch")

    with st.container(border=True):
        st.subheader("Feedback")
        st.text_area("Notes about extraction quality or corrections you made", key=f"fb_{aid}")
        st.button("Submit feedback", on_click=add_feedback)
        for item in st.session_state.feedback:
            st.markdown(f"- **{item['time']}** — {item['comment']}")
        if st.session_state.feedback:
            st.caption("Feedback is kept for this session only and is included in the validation report download.")


def _resume_insights(data: ResumeData, report) -> None:
    months = total_experience_months(data.experience)
    m1, m2, m3 = st.columns(3)
    m1.metric("Skills found", len(data.skills))
    m2.metric("Education entries", len(data.education))
    m3.metric("Dated experience", f"{months // 12}y {months % 12}m" if months is not None else "n/a")

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        res = charts.skills_category_chart(data.skills, data.skill_categories)
        if res:
            st.plotly_chart(res[0], width="stretch")
            st.caption(f"Skills {res[1]}.")
        else:
            ui.empty_state("No skills to chart", "No skills were extracted from this document.")
    with c2, st.container(border=True):
        _completeness_card(report)

    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        fig = charts.experience_timeline(data.experience)
        if fig:
            st.plotly_chart(fig, width="stretch")
        else:
            ui.empty_state("Experience timeline unavailable",
                           "Needs work experience with both a start and end date (or 'Present').")
    with c4, st.container(border=True):
        fig = charts.education_chart(data.education)
        if fig:
            st.plotly_chart(fig, width="stretch")
        else:
            ui.empty_state("Education overview unavailable", "Needs education entries that include a year.")


def _completeness_card(report) -> None:
    gauge = charts.completeness_gauge(report.percent)
    bars = charts.field_completeness_chart(report)
    if gauge:
        st.plotly_chart(gauge, width="stretch")
        st.plotly_chart(bars, width="stretch")
    else:
        ui.empty_state("No required fields analysed", "Select fields that count toward completeness to see this chart.")


def _general_insights(ex, data: GeneralData, report) -> None:
    spec, active, text = SPECS[ex.doc_type], set(ex.active_fields), st.session_state.doc.text
    counts = [(label, len(getattr(data, f))) for f, label in
              [("main_topics", "Topics"), ("key_points", "Key points"), ("findings", "Findings"),
               ("dates", "Dates"), ("entities", "Entities"), ("metrics", "Figures")] if f in active][:4]
    if counts:
        for col, (label, n) in zip(st.columns(len(counts)), counts):
            col.metric(label, n)

    kinds = ["completeness"] + [k for k in spec.charts
                                if {"topics": "main_topics", "entities": "entities", "dates": "dates",
                                    "metrics": "metrics"}[k] in active]
    for i in range(0, len(kinds), 2):
        for col, kind in zip(st.columns(2), kinds[i:i + 2]):
            with col, st.container(border=True):
                if kind == "completeness":
                    _completeness_card(report)
                elif kind == "topics":
                    fig = charts.topic_mentions_chart(data.main_topics, text)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        ui.empty_state("Topic chart unavailable", "No topics were found, or none appear in the document text.")
                elif kind == "entities":
                    fig = charts.entity_type_chart(data.entities)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        ui.empty_state("Entity chart unavailable", "No named entities were found in this document.")
                elif kind == "dates":
                    fig = charts.dates_timeline(data.dates)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        ui.empty_state("Date timeline unavailable", "No dates that can be read unambiguously were found.")
                elif kind == "metrics":
                    figs = charts.metrics_charts(data.metrics)
                    for _, fig in figs:
                        st.plotly_chart(fig, width="stretch")
                    if not figs:
                        ui.empty_state("Figures chart unavailable",
                                       "Needs at least two numeric figures that share a unit (for example two percentages).")
    if len(kinds) == 1:
        st.info("No content charts at this depth. Choose a higher analysis level or select more fields.")

    takeaways = [("Key points", data.key_points), ("Findings", data.findings)]
    if any(items for _, items in takeaways):
        with st.container(border=True):
            st.subheader("Key takeaways")
            for title, items in takeaways:
                if items:
                    st.markdown(f"**{title}**")
                    for item in items[:6]:
                        st.markdown(f"- {item}")
    if data.metrics:
        with st.container(border=True):
            st.subheader("Figures found in the document")
            st.dataframe(pd.DataFrame([m.model_dump() for m in data.metrics]), hide_index=True, width="stretch")


def page_insights() -> None:
    page_header("Insights Hub", "Charts built only from what the document actually contains.")
    if not need_analysis():
        return
    data, report, ex = current_data(), current_report(), st.session_state.extraction
    st.markdown("Document type: " + ui.type_badge(DOC_TYPE_LABELS[ex.doc_type], ex.type_source, ex.type_confidence),
                unsafe_allow_html=True)
    st.caption("Charts are built from the extracted (and corrected) data and depend on this document type. "
               "Nothing is estimated or invented.")
    show_partial_status()
    if ex.doc_type == DocType.RESUME:
        _resume_insights(data, report)
    else:
        _general_insights(ex, data, report)


def page_settings() -> None:
    page_header("Preferences", "AI configuration, limits and privacy.")
    s = get_settings()
    with st.container(border=True):
        st.subheader("AI configuration")
        st.write(f"API key: **{'configured' if s.api_key else 'not set'}** (read from the GEMINI_API_KEY environment variable)")
        st.text_input("Custom Gemini model name (this session only, optional)", key="model_override", placeholder=s.model,
                      help="Leave empty to use GEMINI_MODEL from the environment.")
        st.caption(f"Active model: {effective_model() or s.model} · fallback: {s.fallback_model or 'none'} · timeout {s.timeout_seconds}s")
        if st.button("Test AI connection"):
            try:
                client = ai_service.get_llm_client(effective_model())
                with st.spinner("Contacting Gemini…"):
                    client.generate("Reply with the single word OK.", "You are a connectivity test.")
                st.success("Connection successful.")
            except ai_service.AIServiceError as exc:
                st.error(str(exc))
    with st.container(border=True):
        st.subheader("Limits & privacy")
        st.write(f"Max upload size: {s.max_file_mb:g} MB · PDF pages read: {s.max_pdf_pages} · "
                 f"Characters per AI request: {s.max_chars:,} (longer documents are split into up to "
                 f"{s.max_chunks} overlapping parts)")
        st.caption("Uploaded files are processed in memory only. Document text is not logged or stored.")
        if st.button("Clear session data"):
            for key in ["last_run_seconds", "doc", "doc_type_guess", "summary", "extraction", "current", "analysis_config", "upload_error", "last_upload_digest", "analysis_status"]:
                st.session_state[key] = None
            st.session_state.recent, st.session_state.feedback = [], []
            st.session_state.analysis_id += 1
            st.success("Session data cleared.")


def main() -> None:
    init_state()
    for key in PERSISTED_KEYS:  # keep settings when a page that shows the widget is not rendered
        st.session_state[key] = st.session_state[key]
    sidebar()
    {"Overview": page_overview, "AI Analyzer": page_analysis, "Data Extraction": page_extracted,
     "Insights Hub": page_insights, "Preferences": page_settings}[st.session_state.nav]()




main()
