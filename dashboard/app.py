"""Streamlit dashboard for IntelliDesk: triage UI, batch upload, analytics,
KB search, and drift reports."""

from __future__ import annotations

# Ensure project root is on sys.path BEFORE importing 'src.*'.
# Streamlit only puts dashboard/ on the path when it runs this script,
# so without this 'from src.config import ...' raises ModuleNotFoundError.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.config import CATEGORIES, PRIORITIES, TICKETS_CSV

API_URL = os.getenv("INTELLIDESK_API_URL", "http://localhost:8000")
API_TIMEOUT = 30  # seconds; first cold call can be slow on a fresh process

st.set_page_config(
    page_title="IntelliDesk — AI Service Desk",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _get_offline_pipeline():
    """
    Lazily load the pipeline as a fallback when the API isn't reachable.
    Cached as a Streamlit resource so it loads once per session.
    """
    from src.pipeline import TicketTriagePipeline
    p = TicketTriagePipeline()
    p.warm_up()
    return p


def api_is_healthy() -> bool:
    """Quick reachability check for the FastAPI service."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def api_info() -> Optional[Dict]:
    try:
        r = requests.get(f"{API_URL}/info", timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


def triage_via_api(text: str, top_k: int = 3) -> Dict:
    """Call FastAPI /triage endpoint."""
    r = requests.post(
        f"{API_URL}/triage",
        json={"text": text, "top_k": top_k},
        timeout=API_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def triage_batch_via_api(texts: List[str], top_k: int = 3) -> List[Dict]:
    r = requests.post(
        f"{API_URL}/triage/batch",
        json={"texts": texts, "top_k": top_k},
        timeout=API_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["results"]


def triage_offline(text: str, top_k: int = 3) -> Dict:
    """Fallback path: calls the pipeline directly. Same response shape."""
    pipeline = _get_offline_pipeline()
    result = pipeline.triage(text, top_k=top_k)
    return result.to_dict()


def triage_batch_offline(texts: List[str], top_k: int = 3) -> List[Dict]:
    pipeline = _get_offline_pipeline()
    return [r.to_dict() for r in pipeline.triage_batch(texts, top_k=top_k)]


def smart_triage(text: str, top_k: int = 3, use_api: bool = True) -> Dict:
    """Try API first if requested, fall back to offline pipeline."""
    if use_api and api_is_healthy():
        try:
            return triage_via_api(text, top_k=top_k)
        except requests.RequestException as exc:
            st.warning(f"API call failed ({exc}); falling back to offline pipeline.")
    return triage_offline(text, top_k=top_k)


def smart_triage_batch(texts: List[str], top_k: int = 3, use_api: bool = True) -> List[Dict]:
    if use_api and api_is_healthy():
        try:
            return triage_batch_via_api(texts, top_k=top_k)
        except requests.RequestException as exc:
            st.warning(f"API batch call failed ({exc}); using offline pipeline.")
    return triage_batch_offline(texts, top_k=top_k)


@st.cache_data(ttl=300)
def load_tickets() -> Optional[pd.DataFrame]:
    """Load the historical ticket dataset for analytics tab."""
    if not TICKETS_CSV.exists():
        return None
    df = pd.read_csv(
        TICKETS_CSV,
        parse_dates=["created_at", "first_response_at", "resolved_at"],
    )
    return df


def render_sidebar() -> Dict:
    """Render the sidebar and return user-selected settings."""
    st.sidebar.title("IntelliDesk")
    st.sidebar.caption("AI-Powered IT Service Desk")
    st.sidebar.divider()

    healthy = api_is_healthy()
    if healthy:
        st.sidebar.success(f"✅ API connected\n\n`{API_URL}`")
        info = api_info()
        if info:
            st.sidebar.caption(
                f"Service `{info['service']}` v{info['version']} — "
                f"{info['kb_articles']} KB articles indexed"
            )
    else:
        st.sidebar.warning(
            f"⚠️ API not reachable at `{API_URL}`\n\n"
            "Falling back to in-process pipeline. "
            "Run `.\\make.ps1 serve` in another terminal for full mode."
        )

    use_api = st.sidebar.checkbox(
        "Use API (vs in-process)",
        value=healthy,
        help="When checked, dashboard calls the FastAPI service. "
             "Uncheck to force in-process pipeline (useful for offline demos).",
    )

    st.sidebar.divider()
    top_k = st.sidebar.slider(
        "KB suggestions per ticket", min_value=1, max_value=10, value=3,
        help="Number of resolution articles to recommend.",
    )

    st.sidebar.divider()
    st.sidebar.caption("Built with FastAPI · Streamlit · XGBoost · FAISS")

    return {"use_api": use_api, "top_k": top_k, "api_healthy": healthy}


def tab_triage(settings: Dict) -> None:
    st.header("🎯 Single Ticket Triage")
    st.write("Paste a ticket description and see how the model triages it.")

    samples = {
        "VPN issue": "VPN keeps disconnecting every 10 minutes from home office",
        "Outlook crash": "Outlook crashes when opening large email attachments",
        "Account lockout": "Locked out of my AD account after too many login attempts",
        "Major outage": "Major SSO outage affecting our entire finance team this morning",
        "Printer": "Printer on floor 5 has paper jam and wont clear",
    }
    sample_label = st.selectbox(
        "Or pick a sample:",
        options=["(custom input)"] + list(samples.keys()),
        index=0,
    )
    default_text = samples.get(sample_label, "")

    ticket_text = st.text_area(
        "Ticket description",
        value=default_text,
        height=120,
        placeholder="e.g. Outlook crashes whenever I open a large email attachment...",
    )

    if st.button("Triage ticket", type="primary", use_container_width=True):
        if not ticket_text.strip():
            st.error("Please enter a ticket description.")
            return

        with st.spinner("Classifying and searching KB..."):
            result = smart_triage(
                ticket_text,
                top_k=settings["top_k"],
                use_api=settings["use_api"],
            )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Category", result["category"],
                    f"{result['category_confidence']:.0%} conf")
        col2.metric("Priority", result["priority"],
                    f"{result['priority_confidence']:.0%} conf")
        col3.metric("Latency", f"{result['latency_ms']:.0f} ms")
        col4.metric(
            "Routing",
            "🔍 Review" if result["needs_human_review"] else "🤖 Auto",
        )

        if result["needs_human_review"]:
            st.warning("**Routed to human review** — " +
                       "; ".join(result["review_reasons"]))
        else:
            st.success("**Confident classification — auto-handle eligible**")

        st.subheader("📚 Recommended Resolution Articles")
        if not result["kb_suggestions"]:
            st.info("No KB articles passed the similarity threshold.")
        else:
            for i, kb in enumerate(result["kb_suggestions"], 1):
                with st.expander(
                    f"{i}. [{kb['kb_id']}] {kb['title']} — "
                    f"score {kb['score']:.3f} · {kb['category']}",
                    expanded=(i == 1),
                ):
                    st.write(kb["content"])
                    if kb.get("tags"):
                        st.caption("Tags: " + ", ".join(f"`{t}`" for t in kb["tags"]))

        with st.expander("🔧 Raw API response (JSON)"):
            st.json(result)


def tab_batch(settings: Dict) -> None:
    st.header("📦 Batch Triage")
    st.write(
        "Upload a CSV with a `description` column to triage many tickets at once. "
        "Useful for backlog processing or migrating from a legacy system."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("👆 Upload a CSV to begin. Expected column: `description`.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return

    if "description" not in df.columns:
        st.error(
            "CSV must contain a `description` column. "
            f"Found: {list(df.columns)}"
        )
        return

    st.write(f"Loaded **{len(df):,} tickets**.")
    max_batch = 100
    if len(df) > max_batch:
        st.warning(
            f"Processing first {max_batch} for the demo. "
            f"In production, the API supports streaming for unlimited size."
        )
        df = df.head(max_batch)

    if st.button("Run batch triage", type="primary"):
        texts = df["description"].fillna("").tolist()
        with st.spinner(f"Triaging {len(texts)} tickets..."):
            results = smart_triage_batch(
                texts, top_k=settings["top_k"], use_api=settings["use_api"]
            )

        out = pd.DataFrame([
            {
                "ticket": r["ticket_text"][:80] + ("..." if len(r["ticket_text"]) > 80 else ""),
                "category": r["category"],
                "category_conf": round(r["category_confidence"], 2),
                "priority": r["priority"],
                "priority_conf": round(r["priority_confidence"], 2),
                "needs_review": r["needs_human_review"],
                "top_kb": r["kb_suggestions"][0]["kb_id"] if r["kb_suggestions"] else "",
            }
            for r in results
        ])

        st.subheader("Results")
        st.dataframe(out, use_container_width=True, height=400)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total processed", f"{len(out):,}")
        col2.metric("Flagged for review",
                    f"{out['needs_review'].sum()} ({out['needs_review'].mean():.0%})")
        col3.metric("P1 detected", f"{(out['priority'] == 'P1').sum()}")

        col_a, col_b = st.columns(2)
        with col_a:
            cat_fig = px.bar(
                out["category"].value_counts().reset_index(),
                x="category", y="count",
                title="Predicted category distribution",
                color="category",
            )
            st.plotly_chart(cat_fig, use_container_width=True)
        with col_b:
            pri_fig = px.bar(
                out["priority"].value_counts().reset_index(),
                x="priority", y="count",
                title="Predicted priority distribution",
                color="priority",
                category_orders={"priority": list(PRIORITIES)},
            )
            st.plotly_chart(pri_fig, use_container_width=True)

        st.download_button(
            "Download results as CSV",
            out.to_csv(index=False).encode("utf-8"),
            "triage_results.csv",
            "text/csv",
        )


def tab_analytics() -> None:
    st.header("📊 Operational Analytics")
    st.write(
        "Historical ticket performance — SLA compliance, MTTR, volume trends, "
        "and priority mix. Sourced from `data/tickets.csv`."
    )

    df = load_tickets()
    if df is None:
        st.error(
            "No ticket data found. Run `.\\make.ps1 data` to generate it."
        )
        return

    min_date = df["created_at"].min().date()
    max_date = df["created_at"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        df = df[(df["created_at"].dt.date >= start) &
                (df["created_at"].dt.date <= end)]

    resolved = df[df["status"] == "Resolved"]
    sla_met_rate = resolved["sla_met"].mean() if len(resolved) > 0 else 0
    mttr_hours = resolved["resolution_hours"].median() if len(resolved) > 0 else 0
    p1_rate = (df["priority"] == "P1").mean() if len(df) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total tickets", f"{len(df):,}")
    col2.metric("SLA met", f"{sla_met_rate:.1%}")
    col3.metric("Median MTTR", f"{mttr_hours:.1f} hrs")
    col4.metric("P1 rate", f"{p1_rate:.1%}")

    st.divider()

    daily = df.set_index("created_at").resample("D").size().reset_index(name="tickets")
    fig_vol = px.line(daily, x="created_at", y="tickets",
                      title="Daily ticket volume", markers=False)
    st.plotly_chart(fig_vol, use_container_width=True)

    pivot = df.pivot_table(
        index="category", columns="priority", values="ticket_id", aggfunc="count"
    ).reindex(index=list(CATEGORIES), columns=list(PRIORITIES)).fillna(0)
    fig_heat = px.imshow(
        pivot, text_auto=True, aspect="auto",
        title="Ticket count by category × priority",
        labels=dict(x="Priority", y="Category", color="Tickets"),
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    if len(resolved) > 0:
        mttr_by_cat = resolved.groupby("category")["resolution_hours"].median().reset_index()
        fig_mttr = px.bar(
            mttr_by_cat, x="category", y="resolution_hours",
            title="Median MTTR by category (hours)", color="category",
        )
        st.plotly_chart(fig_mttr, use_container_width=True)

        sla_by_pri = resolved.groupby("priority").agg(
            sla_breach_rate=("sla_met", lambda s: 1 - s.mean())
        ).reset_index()
        fig_sla = px.bar(
            sla_by_pri, x="priority", y="sla_breach_rate",
            title="SLA breach rate by priority",
            category_orders={"priority": list(PRIORITIES)},
            color="priority",
        )
        fig_sla.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_sla, use_container_width=True)

    st.subheader("Top assigned groups")
    top_groups = (df["assigned_group"].value_counts()
                  .head(10).reset_index())
    top_groups.columns = ["assigned_group", "tickets"]
    st.dataframe(top_groups, use_container_width=True)


def tab_kb_search(settings: Dict) -> None:
    st.header("🔎 Knowledge Base Search")
    st.write("Semantic search over the IntelliDesk knowledge base.")

    query = st.text_input(
        "Search query",
        placeholder="e.g. VPN keeps dropping connection",
    )
    if not query.strip():
        st.info("👆 Enter a query to search the KB.")
        return

    with st.spinner("Searching KB..."):
        result = smart_triage(
            query, top_k=settings["top_k"], use_api=settings["use_api"]
        )

    if not result["kb_suggestions"]:
        st.warning("No KB articles matched above the similarity threshold.")
        return

    for i, kb in enumerate(result["kb_suggestions"], 1):
        with st.expander(
            f"{i}. [{kb['kb_id']}] {kb['title']} — "
            f"score {kb['score']:.3f} · {kb['category']}",
            expanded=(i == 1),
        ):
            st.write(kb["content"])
            if kb.get("tags"):
                st.caption("Tags: " + ", ".join(f"`{t}`" for t in kb["tags"]))

def tab_drift() -> None:
    """Render the latest drift report inline (or instructions to generate one)."""
    st.header("📈 Drift Monitoring")
    st.write(
        "Compares recent ticket distribution against the training-time reference. "
        "Watch for unusual P1 spikes, vocabulary shifts, or category-mix changes."
    )

    from src.config import REPORTS_DIR

    candidates = [
        ("Data drift", "data_latest.html"),
        ("Target drift (category)", "target_category_latest.html"),
        ("Target drift (priority)", "target_priority_latest.html"),
    ]
    available = [
        (label, REPORTS_DIR / fname)
        for label, fname in candidates
        if (REPORTS_DIR / fname).exists()
    ]

    if not available:
        st.info(
            "No drift reports generated yet. Run `.\\make.ps1 drift` "
            "from the project root to generate them."
        )
        return

    labels = [a[0] for a in available]
    inner_tabs = st.tabs(labels)
    for tab_obj, (label, path) in zip(inner_tabs, available):
        with tab_obj:
            st.caption(f"Source: `{path}`")
            html = path.read_text(encoding="utf-8")
            st.components.v1.html(html, height=900, scrolling=True)


def main() -> None:
    settings = render_sidebar()

    st.title("🎫 IntelliDesk")
    st.caption(
        "AI-Powered IT Service Desk — automated ticket triage, "
        "semantic KB retrieval, and operational analytics."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Triage",
        "📦 Batch",
        "📊 Analytics",
        "🔎 KB Search",
        "📈 Drift",
    ])

    with tab1:
        tab_triage(settings)
    with tab2:
        tab_batch(settings)
    with tab3:
        tab_analytics()
    with tab4:
        tab_kb_search(settings)
    with tab5:
        tab_drift()


if __name__ == "__main__":
    main()