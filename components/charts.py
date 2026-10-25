"""Plotly charts built only from extracted data. Functions return None when data is insufficient."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backend.insights import (categorize_skills, count_mentions, experience_intervals, parse_date,
                              parse_metric, parse_year)
from backend.validation import FIELD_LABELS, CompletenessReport, FieldStatus

PALETTE = ["#5b6bff", "#8b5cf6", "#22d3ee", "#38bdf8", "#a78bfa", "#60a5fa", "#2dd4bf"]
STATUS_COLORS = {
    FieldStatus.PRESENT: "#34d399", FieldStatus.UNCERTAIN: "#fbbf24",
    FieldStatus.MISSING: "#fb7185", FieldStatus.INVALID: "#fb7185",
    FieldStatus.NOT_VERIFIED: "#64748b",
}


def _style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbe4ff"),
                      height=height, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(gridcolor="#1d2a55")
    fig.update_yaxes(gridcolor="#1d2a55")
    return fig


def skills_category_chart(skills: list[str], categories) -> Optional[tuple[go.Figure, str]]:
    if not skills:
        return None
    groups = {c.category: c.skills for c in categories if c.skills}
    source = "grouped by the AI (Detailed analysis)"
    if not groups:
        groups, source = categorize_skills(skills), "grouped by keyword rules"
    df = pd.DataFrame({"Category": list(groups), "Skills": [len(v) for v in groups.values()],
                       "Names": [", ".join(v) for v in groups.values()]}).sort_values("Skills")
    fig = px.bar(df, x="Skills", y="Category", orientation="h", hover_data={"Names": True},
                 color="Category", color_discrete_sequence=PALETTE, title="Skills by category")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(dtick=1)
    return _style(fig), source


def completeness_gauge(percent: Optional[int]) -> Optional[go.Figure]:
    if percent is None:
        return None
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=percent, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#5b6bff"},
               "bgcolor": "#1d2a55", "borderwidth": 0},
        title={"text": "Overall completeness"}))
    return _style(fig, 260)


def field_completeness_chart(report: CompletenessReport) -> Optional[go.Figure]:
    if not report.in_scope:
        return None
    scores = {FieldStatus.PRESENT: 100, FieldStatus.UNCERTAIN: 50, FieldStatus.MISSING: 0, FieldStatus.INVALID: 0,
              FieldStatus.NOT_VERIFIED: 0}
    fields = report.required or report.in_scope
    fig = go.Figure(go.Bar(
        x=[scores[report.statuses[f]] for f in fields], y=[FIELD_LABELS[f] for f in fields], orientation="h",
        marker_color=[STATUS_COLORS[report.statuses[f]] for f in fields],
        text=[report.statuses[f].value for f in fields], textposition="inside"))
    fig.update_layout(title="Field completeness", xaxis=dict(range=[0, 100], title="Score (%)"),
                      yaxis=dict(autorange="reversed"))
    return _style(fig)


def experience_timeline(experience) -> Optional[go.Figure]:
    spans = experience_intervals(experience)
    if not spans:
        return None
    df = pd.DataFrame([{
        "Role": f"{i.title or 'Role'} @ {i.company or 'n/a'}", "Start": s, "Finish": e} for s, e, i in spans])
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Role", color_discrete_sequence=PALETTE,
                      title="Work experience timeline")
    fig.update_yaxes(autorange="reversed")
    return _style(fig)


def education_chart(education) -> Optional[go.Figure]:
    rows = [(parse_year(e.year), e) for e in education if parse_year(e.year)]
    if not rows:
        return None
    df = pd.DataFrame([{"Year": y, "Label": e.degree or e.institution or "Education",
                        "Institution": e.institution or ""} for y, e in rows])
    fig = px.scatter(df, x="Year", y="Label", text="Institution", hover_data=["Institution"],
                     title="Education overview", color_discrete_sequence=PALETTE)
    fig.update_traces(marker=dict(size=16), textposition="top center")
    fig.update_xaxes(dtick=1)
    return _style(fig)


# ---------------------------------------------------------------- general-document charts
def topic_mentions_chart(topics: list[str], text: str) -> Optional[go.Figure]:
    """How often each extracted topic is mentioned in the document itself."""
    rows = [(t, count_mentions(t, text)) for t in topics]
    rows = [(t, n) for t, n in rows if n > 0]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["Topic", "Mentions"]).sort_values("Mentions")
    fig = px.bar(df, x="Mentions", y="Topic", orientation="h", color="Topic", color_discrete_sequence=PALETTE,
                 title="Topic mentions in the document")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(dtick=1)
    return _style(fig, max(260, 60 + 34 * len(df)))


def entity_type_chart(entities) -> Optional[go.Figure]:
    rows = [(e.type or "Unspecified", e.name) for e in entities if e.name]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["Type", "Name"])
    grouped = df.groupby("Type").agg(Count=("Name", "size"), Names=("Name", lambda v: ", ".join(v))).reset_index()
    fig = px.bar(grouped.sort_values("Count"), x="Count", y="Type", orientation="h", hover_data=["Names"],
                 color="Type", color_discrete_sequence=PALETTE, title="Entities by type")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(dtick=1)
    return _style(fig)


def dates_timeline(dates) -> Optional[go.Figure]:
    """Timeline of dates that could be parsed unambiguously from the document."""
    rows = [(parse_date(d.date), d.event or d.date) for d in dates if d.date]
    rows = [(when, label) for when, label in rows if when]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["Date", "Event"]).sort_values("Date")
    fig = px.scatter(df, x="Date", y="Event", title="Key dates timeline", color_discrete_sequence=PALETTE)
    fig.update_traces(marker=dict(size=14))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, max(260, 120 + 40 * len(df)))


def metrics_charts(metrics) -> list[tuple[str, go.Figure]]:
    """One bar chart per unit group (%, $, counts...) that has at least two comparable figures."""
    groups: dict[str, list[tuple[str, float, str]]] = {}
    for m in metrics:
        number, unit = parse_metric(m.value, m.unit)
        if number is not None and m.name:
            groups.setdefault(unit, []).append((m.name, number, f"{m.value}{(' ' + m.unit) if m.unit else ''}"))
    out = []
    for unit, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        if len(items) < 2:
            continue
        df = pd.DataFrame(items, columns=["Metric", "Value", "As written"])
        fig = px.bar(df, x="Value", y="Metric", orientation="h", hover_data=["As written"], color="Metric",
                     color_discrete_sequence=PALETTE, title=f"Key figures ({unit})")
        fig.update_layout(showlegend=False)
        out.append((unit, _style(fig, max(260, 70 + 40 * len(df)))))
    return out[:2]


def field_counts_chart(data, active_fields) -> Optional[go.Figure]:
    """Number of items extracted per analysed list field. None when nothing countable was extracted."""
    rows = []
    for f in active_fields:
        if f in ("quality_notes", "skill_categories"):
            continue
        value = getattr(data, f, None)
        if isinstance(value, list):  # single values (name, title, email) are not item counts
            rows.append((FIELD_LABELS[f], len(value)))
    if not rows or not any(n for _, n in rows):
        return None
    df = pd.DataFrame(rows, columns=["Field", "Items"])
    fig = px.bar(df, x="Field", y="Items", color="Field", color_discrete_sequence=PALETTE,
                 title="Extracted items per field")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-30, title=None)
    return _style(fig)
