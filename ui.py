"""Reusable UI helpers: theme CSS, badges, empty states and tables. All dynamic text is HTML-escaped."""
from __future__ import annotations

import html

import streamlit as st

from backend.validation import FIELD_LABELS, CompletenessReport, FieldStatus

CSS = """
<style>
.block-container{padding-top:1.6rem;max-width:1250px}
[data-testid="stSidebar"]{border-right:1px solid #1d2a55}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px;border-color:#1d2a55 !important;background:#0f1a3d}
.idx-brand{font-size:1.35rem;font-weight:700;letter-spacing:.2px}
.idx-sub{color:#9fb0e6;font-size:.8rem;margin-top:-2px}
.idx-hero h1{margin:0;font-size:2rem;background:linear-gradient(90deg,#8b9bff,#22d3ee);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.idx-hero p{color:#9fb0e6;margin:.2rem 0 1rem}
.idx-badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.75rem;font-weight:600}
.idx-present{background:#12372b;color:#6ee7b7}
.idx-uncertain{background:#3d3210;color:#fcd34d}
.idx-missing,.idx-invalid{background:#3d1a24;color:#fda4af}
.idx-na,.idx-notapplicable{background:#1d2a55;color:#9fb0e6}
.idx-notverified{background:#2a2350;color:#c4b5fd}
.idx-sample{background:#2a1f5c;color:#c4b5fd}
table.idx-table{width:100%;border-collapse:collapse;font-size:.88rem}
.idx-table th{text-align:left;color:#9fb0e6;font-weight:500;padding:6px 8px;border-bottom:1px solid #1d2a55}
.idx-table td{padding:8px;border-bottom:1px solid #162250;vertical-align:top}
.idx-empty{text-align:center;color:#9fb0e6;padding:1.2rem .5rem;border:1px dashed #2a3a7a;border-radius:12px}
.idx-empty b{display:block;color:#e6ebff;margin-bottom:.2rem}

.stApp{background:#070d24}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#0a1230}
[data-testid="stSidebar"] [role="radiogroup"]{gap:4px;width:100%}
[data-testid="stSidebar"] [role="radiogroup"] > div{width:100%}
[data-testid="stSidebar"] label[data-baseweb="radio"],
[data-testid="stSidebar"] label[data-testid="stRadioOption"]{width:100%;padding:9px 12px;border-radius:8px;margin:2px 0;cursor:pointer;
  border:1px solid transparent;color:#dbe4ff}
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]),
[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"]{display:none !important}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked),
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked){background:#2f3bd6;border-color:#5b6bff;color:#fff}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover,
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover{background:#131f52}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked):hover,
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked):hover{background:#2f3bd6}
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] p{font-size:.95rem;display:flex;align-items:center;gap:12px;margin:0}
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] span[data-testid="stIconMaterial"]{color:#c7d2fe;font-size:1.25rem}
[data-testid="stSidebar"] label:has(input:checked) span[data-testid="stIconMaterial"]{color:#fff}
[data-testid="stSidebar"] [data-testid="stRadio"], [data-testid="stSidebar"] [data-testid="stRadio"] > div,
[data-testid="stSidebar"] [role="radiogroup"] div[data-rac]{width:100% !important;max-width:100% !important}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]){width:100% !important}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]{box-sizing:border-box;width:100% !important;padding:7px 12px;margin:0}
[data-testid="stSidebar"] [role="radiogroup"]{gap:2px !important}
[data-testid="stSidebar"] [role="radiogroup"] p span{font-size:1.3rem !important;color:#c7d2fe}
[data-testid="stSidebar"] label:has(input:checked) p span{color:#fff}
.idx-brandrow .idx-sub{font-size:.66rem;white-space:nowrap}
.idx-brandrow{display:flex;gap:12px;align-items:center;margin-bottom:.4rem}
.idx-brandrow svg{flex:none}
[data-testid="stMetric"]{background:#0a1233;border:1px solid #1d2a55;border-radius:10px;padding:10px 12px}
[data-testid="stMetricValue"]{font-size:1.4rem}
button[kind="primary"]{background:linear-gradient(90deg,#3b4cf5,#5b6bff);border:0;font-weight:600}
.idx-title{margin:0;font-size:1.7rem;font-weight:700}
.idx-sub2{color:#9fb0e6;margin:.1rem 0 .6rem;font-size:.9rem}
.idx-ct{display:flex;gap:10px;align-items:center;margin-bottom:.5rem}
.idx-ct .ic{width:34px;height:34px;border-radius:9px;background:#2a35b8;display:grid;place-items:center;font-size:1rem;flex:none}
.idx-ct b{font-size:1.02rem}
.idx-ct span.sub{display:block;color:#9fb0e6;font-size:.78rem}
.idx-ring{width:132px;height:132px;border-radius:50%;display:grid;place-items:center;
  background:conic-gradient(#5b6bff calc(var(--p) * 1%), #1d2a55 0)}
.idx-ring div{width:100px;height:100px;border-radius:50%;background:#0f1a3d;display:grid;place-items:center;text-align:center;line-height:1.1}
.idx-ring b{font-size:1.5rem;display:block}.idx-ring small{color:#9fb0e6;font-size:.7rem}
.idx-leg{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #162250;font-size:.88rem}
.idx-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}
.idx-file{display:flex;gap:12px;align-items:center;justify-content:space-between;background:#0a1233;
  border:1px solid #1d2a55;border-radius:10px;padding:10px 14px;margin-bottom:.6rem}
.idx-file small{color:#9fb0e6;display:block}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        "<div class='idx-hero'><h1>IntelliDocs AI</h1>"
        "<p>Intelligent Document Analysis &amp; Insight Platform &mdash; Transform Documents into Intelligent Insights.</p></div>",
        unsafe_allow_html=True,
    )


def brand() -> None:
    star = ("<svg width='34' height='34' viewBox='0 0 24 24'><defs><linearGradient id='idxg' x1='0' y1='0' x2='1' y2='1'>"
            "<stop offset='0' stop-color='#c4b5fd'/><stop offset='1' stop-color='#7c5cff'/></linearGradient></defs>"
            "<path fill='url(#idxg)' d='M12 1.5c.6 5.6 4.9 9.9 10.5 10.5-5.6.6-9.9 4.9-10.5 10.5C11.4 16.9 7.1 12.6 1.5 12 "
            "7.1 11.4 11.4 7.1 12 1.5z'/></svg>")
    st.markdown(f"<div class='idx-brandrow'>{star}<div><div class='idx-brand'>IntelliDocs AI</div>"
                "<div class='idx-sub'>AI-Powered Document Analysis &amp; Insights</div></div></div>", unsafe_allow_html=True)


def card_title(icon: str, title: str, sub: str = "") -> None:
    sub_html = f"<span class='sub'>{html.escape(sub)}</span>" if sub else ""
    st.markdown(f"<div class='idx-ct'><div class='ic'>{icon}</div><div><b>{html.escape(title)}</b>{sub_html}</div></div>",
                unsafe_allow_html=True)


def ring(percent) -> str:
    value = 0 if percent is None else int(percent)
    label = "n/a" if percent is None else f"{value}%"
    return f"<div class='idx-ring' style='--p:{value}'><div><span><b>{label}</b><small>Complete</small></span></div></div>"


def legend_row(color: str, label: str, value: str) -> str:
    return (f"<div class='idx-leg'><span><i class='idx-dot' style='background:{color}'></i>{html.escape(label)}</span>"
            f"<b>{html.escape(value)}</b></div>")


def file_row(name: str, meta: str, status_html: str) -> None:
    st.markdown(f"<div class='idx-file'><div><b>📄 {html.escape(name)}</b><small>{html.escape(meta)}</small></div>"
                f"<div>{status_html}</div></div>", unsafe_allow_html=True)


def badge(status: str, text: str | None = None) -> str:
    key = status.replace(" ", "")
    css = {"notanalysed": "na"}.get(key, key)
    return f"<span class='idx-badge idx-{css}'>{html.escape(text or status.title())}</span>"


def type_badge(label: str, source: str = "", confidence: float | None = None) -> str:
    extra = []
    if source:
        extra.append({"rules": "keyword rules", "ai": "AI", "user": "chosen by you"}.get(source, source))
    if confidence is not None and source != "user":
        extra.append(f"{round(confidence * 100)}% confidence")
    suffix = f" · {', '.join(extra)}" if extra else ""
    return badge("sample", f"{label}{suffix}")


def empty_state(title: str, message: str) -> None:
    st.markdown(f"<div class='idx-empty'><b>{html.escape(title)}</b>{html.escape(message)}</div>", unsafe_allow_html=True)


def preview_value(data, field: str) -> str:
    v = getattr(data, field)
    if isinstance(v, list):
        if not v:
            return "—"
        if field == "dates":
            parts = [" – ".join(x for x in (d.date, d.event) if x) for d in v]
        elif field == "entities":
            parts = [f"{e.name} ({e.type})" if e.type else (e.name or "") for e in v]
        elif field == "metrics":
            parts = [f"{m.name}: {m.value}" + (f" {m.unit}" if m.unit else "") for m in v]
        elif field == "education":
            parts = [" · ".join(x for x in (e.degree, e.institution, e.year) if x) for e in v]
        elif field == "experience":
            parts = [" · ".join(x for x in (e.title, e.company) if x) for e in v]
        else:
            parts = [str(x) for x in v]
        shown = "; ".join(parts[:4])
        return shown + (f" (+{len(parts) - 4} more)" if len(parts) > 4 else "")
    return str(v) if v else "—"


def field_status_table(report: CompletenessReport, data) -> None:
    rows = []
    for name, status in report.statuses.items():
        rows.append(
            f"<tr><td>{html.escape(FIELD_LABELS[name])}</td><td>{badge(status.value)}</td>"
            f"<td>{html.escape(preview_value(data, name))}</td><td>{html.escape(report.notes.get(name, ''))}</td></tr>"
        )
    st.markdown(
        "<table class='idx-table'><tr><th>Field</th><th>Status</th><th>Extracted value</th><th>Note</th></tr>"
        + "".join(rows) + "</table>",
        unsafe_allow_html=True,
    )
