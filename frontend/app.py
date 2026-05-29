"""
app.py — Streamlit frontend for the SGEIA Smart Grid Energy Intelligence Assistant.

Features:
  - Natural-language chat interface with streaming responses
  - Grid Health Score gauge
  - Retrieved incident cards with metadata badges
  - Root cause analysis expandable panel
  - Mitigation steps with quality scores
  - SHAP feature importance display
  - Anomaly count and stability status
  - Dashboard KPI sidebar

Run:
    streamlit run frontend/app.py
"""

import json
import time
import uuid
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SGEIA — Smart Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://127.0.0.1:8000"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .incident-card {
    background: #1e2d3d; border-left: 4px solid #2e75b6;
    padding: 12px 16px; border-radius: 6px; margin: 8px 0;
  }
  .severity-critical { border-left-color: #e74c3c !important; }
  .severity-high     { border-left-color: #e67e22 !important; }
  .severity-medium   { border-left-color: #f39c12 !important; }
  .severity-low      { border-left-color: #27ae60 !important; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: bold; margin-right: 4px;
  }
  .badge-critical { background: #e74c3c; color: white; }
  .badge-high     { background: #e67e22; color: white; }
  .badge-medium   { background: #f39c12; color: black; }
  .badge-low      { background: #27ae60; color: white; }
  .shap-bar { background: #2e75b6; height: 14px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ── Helper functions ────────────────────────────────────────────────────────────

def _health_colour(score: int) -> str:
    if score >= 70: return "green"
    if score >= 40: return "orange"
    return "red"


def _health_emoji(score: int) -> str:
    if score >= 70: return "🟢"
    if score >= 40: return "🟡"
    return "🔴"


def _severity_class(sev: str) -> str:
    return f"severity-{sev.lower()}" if sev.lower() in ["critical", "high", "medium", "low"] else ""


def _gauge_chart(health_score: int) -> go.Figure:
    """Create a Plotly gauge for the grid health score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Grid Health Score", "font": {"size": 18}},
        delta={"reference": 70, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar":   {"color": _health_colour(health_score)},
            "steps": [
                {"range": [0,  40], "color": "#3d1515"},
                {"range": [40, 70], "color": "#3d3115"},
                {"range": [70, 100], "color": "#153d15"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": 70,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=220,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    return fig


def _call_api_sync(query: str, widget_context: dict | None = None) -> dict:
    """Call the non-streaming /api/query endpoint."""
    payload = {
        "query":          query,
        "session_id":     st.session_state.session_id,
        "widget_context": widget_context,
    }
    try:
        resp = requests.post(f"{API_BASE}/api/query", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API server. Is it running? (uvicorn src.api.main:app)"}
    except requests.exceptions.HTTPError as e:
        try:
            detail = resp.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": detail}
    except Exception as e:
        return {"error": str(e)}


def _fetch_dashboard_metrics() -> dict:
    """Fetch dashboard KPI metrics from the API."""
    try:
        resp = requests.get(f"{API_BASE}/api/dashboard/metrics", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/electrical.png", width=60)
    st.title("⚡ SGEIA")
    st.caption("Smart Grid Energy Intelligence Assistant")
    st.divider()

    # Dashboard metrics
    st.subheader("📊 Live Dashboard")
    metrics = _fetch_dashboard_metrics()

    if metrics:
        inc_counts = metrics.get("incident_counts", {})
        col1, col2 = st.columns(2)
        col1.metric("🔴 Critical", inc_counts.get("critical", "—"))
        col2.metric("🟠 High",     inc_counts.get("high",     "—"))
        col1.metric("🟡 Medium",   inc_counts.get("medium",   "—"))
        col2.metric("🟢 Low",      inc_counts.get("low",      "—"))

        stab = metrics.get("stability_summary", {})
        if stab:
            st.metric("⚡ Unstable %",  f"{stab.get('unstable_pct', 0):.1f}%")
            st.metric("📡 Avg Freq",    f"{stab.get('mean_freq_hz', 50):.3f} Hz")
    else:
        st.info("API not reachable — start the backend first.")

    st.divider()

    # API status check
    st.subheader("🔧 System Status")
    try:
        health_resp = requests.get(f"{API_BASE}/api/health", timeout=5)
        h = health_resp.json()
        status = h.get("status", "unknown")
        icon = "✅" if status == "healthy" else "⚠️"
        st.success(f"{icon} API {status}")
        comps = h.get("components", {})
        for name, val in comps.items():
            if isinstance(val, dict):
                st.caption(f"• {name}: {val.get('status', '?')}")
    except Exception:
        st.error("❌ API offline")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}`")


# ── Main layout ─────────────────────────────────────────────────────────────────
st.title("⚡ Smart Grid Energy Intelligence Assistant")
st.caption(
    "Ask questions about grid stability, historical incidents, smart meter anomalies, "
    "or request mitigation recommendations."
)

# ── Grid Health Gauge (shown when a result is available) ─────────────────────
if st.session_state.last_result:
    res = st.session_state.last_result
    health = res.get("health_score")
    if health is not None:
        col_gauge, col_kpi = st.columns([1, 2])
        with col_gauge:
            st.plotly_chart(_gauge_chart(health), use_container_width=True)
        with col_kpi:
            st.metric(
                "Stability",
                f"{_health_emoji(health)} {res.get('stability_label','N/A').upper()}",
                delta=f"Health {health}/100",
            )
            if res.get("anomaly_flags"):
                st.metric("Anomalies Detected", len(res.get("anomaly_flags", [])), delta="⚠️")
            if res.get("retrieval_count"):
                st.metric("Incidents Retrieved", res.get("retrieval_count", 0))
            if res.get("routing_decision"):
                st.caption(f"🔀 Routing: `{res.get('routing_decision')}` | "
                           f"Intent: _{res.get('extracted_intent','N/A')}_")
    st.divider()


# ── Chat history display ───────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "⚡"):
            if msg["role"] == "assistant" and "result" in msg:
                _render_result(msg["result"], msg["content"])
            else:
                st.markdown(msg["content"])


def _render_result(result: dict, fallback_text: str = ""):
    """Render a full agent result with panels."""
    # Routing info
    if result.get("routing_decision"):
        st.caption(
            f"🔀 `{result['routing_decision']}` | "
            f"_{result.get('extracted_intent','')}_"
        )

    # Stability + health (already shown in gauge above, show compact here)
    if result.get("stability_label"):
        health = result.get("health_score", 0)
        st.info(
            f"{_health_emoji(health)} **Grid Health: {health}/100** — "
            f"{result.get('stability_label','').upper()} | "
            f"Frequency: {result.get('grid_freq_status', 'N/A')}"
        )

    # Retrieved incidents
    incidents = result.get("retrieved_incidents") or []
    if incidents:
        with st.expander(f"📋 {len(incidents)} Similar Historical Incidents", expanded=True):
            for inc in incidents:
                meta    = inc.get("metadata", {})
                sev     = meta.get("severity", "medium")
                sev_cls = _severity_class(sev)
                sim_pct = f"{inc.get('rrf_score', 0) * 100:.0f}%"
                st.markdown(
                    f"""<div class="incident-card {sev_cls}">
                    <span class="badge badge-{sev}">{sev.upper()}</span>
                    <strong>{inc['doc_id']}</strong> — {meta.get('region','?')} |
                    {meta.get('equipment_type','?')} | {meta.get('outage_event','?')} |
                    Similarity: {sim_pct}<br>
                    <small>{inc['document'][:350]}...</small>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # Root cause
    rc = result.get("root_cause") or {}
    if rc and rc.get("probable_cause"):
        with st.expander("🔍 Root Cause Analysis", expanded=True):
            st.markdown(f"**Probable cause:** {rc.get('probable_cause')}")
            st.markdown(f"**Evidence:** {rc.get('evidence','N/A')}")
            conf = rc.get('confidence', 0)
            st.progress(conf, text=f"Confidence: {conf:.0%}")
            related = rc.get("related_incident_ids", [])
            if related:
                st.caption(f"Related incidents: {', '.join(related)}")

    # Mitigation steps
    steps = result.get("mitigation_steps") or []
    if steps:
        judge  = result.get("judge_score",      0)
        faith  = result.get("faithfulness_score", 0)
        with st.expander("🛠️ Mitigation Recommendations", expanded=True):
            col_j, col_f = st.columns(2)
            col_j.metric("Judge Score",       f"{judge:.1f}/5")
            col_f.metric("Faithfulness",      f"{faith:.0%}")
            st.divider()
            for step in steps:
                st.markdown(f"- {step}")

    # SHAP
    shap = result.get("shap_top5") or []
    if shap:
        with st.expander("📊 SHAP Feature Importances (Instability Drivers)"):
            shap_df = pd.DataFrame(shap)
            shap_df["abs_shap"] = shap_df["shap_value"].abs()
            shap_df = shap_df.sort_values("abs_shap", ascending=False)
            st.dataframe(
                shap_df[["feature", "value", "shap_value"]].rename(
                    columns={"feature": "Feature", "value": "Value", "shap_value": "SHAP"}
                ),
                use_container_width=True,
            )

    # Anomalies
    anomalies = result.get("anomaly_flags") or []
    if anomalies:
        with st.expander(f"⚡ Anomaly Events ({len(anomalies)} detected)"):
            anom_df = pd.DataFrame(anomalies)
            st.dataframe(anom_df, use_container_width=True)

    # Fallback: plain markdown
    if not any([incidents, rc.get("probable_cause"), steps]):
        st.markdown(fallback_text or result.get("final_response", "No result."))


# ── Chat input ─────────────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "Analyse voltage instability in Zone_B transformers",
    "Find similar incidents to Zone_C partial outage with critical transformer status",
    "What is the current grid stability and health score?",
    "Detect smart meter consumption anomalies",
    "Why did the grid frequency deviate last week and what should we do?",
]

with st.expander("💡 Example queries"):
    for eq in EXAMPLE_QUERIES:
        if st.button(eq, key=f"ex_{eq[:20]}"):
            st.session_state["prefill_query"] = eq

user_input = st.chat_input(
    "Ask about grid stability, incidents, smart meter anomalies, or request mitigation steps...",
)

# Handle example query prefill
if "prefill_query" in st.session_state:
    user_input = st.session_state.pop("prefill_query")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_input)

    # Call API and stream response
    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("⚡ Analysing grid data..."):
            start_t = time.time()
            result  = _call_api_sync(user_input)
            elapsed = time.time() - start_t

        if "error" in result:
            st.error(f"❌ {result['error']}")
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"Error: {result['error']}",
            })
        else:
            st.session_state.last_result = result
            st.caption(f"⏱ {elapsed:.1f}s | Request ID: `{result.get('request_id','?')}`")
            _render_result(result, result.get("final_response", ""))
            st.session_state.messages.append({
                "role":    "assistant",
                "content": result.get("final_response", ""),
                "result":  result,
            })
    st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "SGEIA v1.0 | Prodapt FDE Capstone | "
    "Powered by GPT-4o · LangGraph · ChromaDB · XGBoost · DeepEval"
)
