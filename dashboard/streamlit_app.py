import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    page_title="RAG Eval Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Agentic Self-Healing RAG — Eval Dashboard")
st.caption("Live metrics, deployment gate status, and manual query testing")

METRICS_HISTORY_PATH = "data/metrics_history.json"
GOLDEN_DATASET_PATH = "data/golden_dataset.json"


def load_metrics_history():
    path = Path(METRICS_HISTORY_PATH)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_golden_dataset():
    path = Path(GOLDEN_DATASET_PATH)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pairs", [])


# ── Section 1: Latest Metrics ──────────────────────────────────────────
st.header("📊 Latest Eval Metrics")

history = load_metrics_history()

if not history:
    st.warning("No eval results yet. Run the eval pipeline first.")
else:
    latest = history[-1]
    metrics = latest["metrics"]
    gate = latest["gate_result"]

    # Deployment gate status
    if gate["approved"]:
        st.success("✅ DEPLOYMENT APPROVED — All metrics within thresholds")
    else:
        st.error("❌ DEPLOYMENT BLOCKED — Quality thresholds not met")
        for failure in gate.get("failures", []):
            st.error(f"  • {failure}")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rate = metrics.get("hallucination_rate", 0)
        color = "normal" if rate < 0.08 else "inverse"
        st.metric(
            "Hallucination Rate",
            f"{rate:.1%}",
            delta=f"Target: <8%",
            delta_color=color,
        )

    with col2:
        faith = metrics.get("avg_faithfulness", 0)
        st.metric(
            "Avg Faithfulness",
            f"{faith:.2f}",
            delta=f"Target: >0.88",
        )

    with col3:
        total = metrics.get("total_evaluated", 0)
        hallu = metrics.get("hallucination_count", 0)
        st.metric(
            "Questions Evaluated",
            total,
            delta=f"{hallu} hallucinated",
        )

    with col4:
        latency = metrics.get("p50_latency_ms", 0)
        st.metric(
            "P50 Latency",
            f"{latency/1000:.1f}s",
            delta="Target: <1.5s",
        )

    st.caption(f"Last run: {latest.get('timestamp', 'unknown')}")

# ── Section 2: Metrics Over Time ───────────────────────────────────────
st.header("📈 Metrics Over Time")

if len(history) < 2:
    st.info("Run the eval pipeline multiple times to see trends.")
else:
    timestamps = [h.get("timestamp", "")[:16] for h in history]
    hallucination_rates = [h["metrics"].get("hallucination_rate", 0) * 100 for h in history]
    faithfulness_scores = [h["metrics"].get("avg_faithfulness", 0) for h in history]

    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=timestamps, y=hallucination_rates,
            mode="lines+markers", name="Hallucination Rate %",
            line=dict(color="red", width=2),
        ))
        fig1.add_hline(y=5, line_dash="dash", line_color="orange",
                       annotation_text="5% gate threshold")
        fig1.add_hline(y=8, line_dash="dash", line_color="red",
                       annotation_text="8% CV target")
        fig1.update_layout(
            title="Hallucination Rate Over Time",
            yaxis_title="Hallucination Rate (%)",
            xaxis_title="Eval Run",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=timestamps, y=faithfulness_scores,
            mode="lines+markers", name="Faithfulness",
            line=dict(color="green", width=2),
        ))
        fig2.add_hline(y=0.88, line_dash="dash", line_color="orange",
                       annotation_text="0.88 threshold")
        fig2.update_layout(
            title="Faithfulness Score Over Time",
            yaxis_title="Faithfulness Score",
            xaxis_title="Eval Run",
            yaxis=dict(range=[0, 1.1]),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Section 3: Golden Dataset ───────────────────────────────────────────
st.header("📋 Golden Dataset")

pairs = load_golden_dataset()
if pairs:
    st.write(f"**{len(pairs)} QA pairs** across 5 policy domains")
    import pandas as pd
    df = pd.DataFrame(pairs)
    st.dataframe(df[["question", "ground_truth", "context"]], use_container_width=True)

# ── Section 4: Manual Query Test ───────────────────────────────────────
st.header("🔍 Manual Query Test")
st.caption("Test the RAG pipeline live — results show retrieve → generate → critic flow")

query = st.text_input("Enter your question:", placeholder="What is the refund policy?")

if st.button("Run Query", type="primary") and query:
    with st.spinner("Running RAG pipeline..."):
        try:
            from app.graph.builder import rag_graph

            initial_state = {
                "query": query,
                "reformulated_query": None,
                "retrieved_chunks": [],
                "answer": "",
                "faithfulness_score": 0.0,
                "is_hallucinated": False,
                "critic_reason": "",
                "retry_count": 0,
                "final_answer": "",
            }

            result = rag_graph.invoke(initial_state)

            st.success("✅ Query complete")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Faithfulness", f"{result['faithfulness_score']:.2f}")
            with col2:
                st.metric("Hallucinated", str(result['is_hallucinated']))
            with col3:
                st.metric("Retries", result['retry_count'])

            st.subheader("Answer")
            st.write(result["final_answer"])

            with st.expander("Retrieved Chunks"):
                for i, chunk in enumerate(result["retrieved_chunks"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(chunk)

            with st.expander("Critic Verdict"):
                st.write(result["critic_reason"])

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("Agentic Self-Healing RAG | LangGraph + ChromaDB + Gemini + RAGAS | IIT Kharagpur")