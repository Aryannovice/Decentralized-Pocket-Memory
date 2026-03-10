import requests
import streamlit as st


def inject_styles(theme_mode: str) -> None:
    if theme_mode == "Auto":
        return

    if theme_mode == "Light":
        muted_color = "#4f4f4f"
        card_bg = "rgba(248, 250, 252, 0.95)"
        card_border = "rgba(160, 174, 192, 0.65)"
        answer_bg = "rgba(66, 153, 225, 0.12)"
        panel_bg = "#f8fafc"
        text_color = "#1f2937"
    else:
        muted_color = "#b6b6b6"
        card_bg = "rgba(30, 41, 59, 0.55)"
        card_border = "rgba(100, 116, 139, 0.6)"
        answer_bg = "rgba(76, 139, 245, 0.18)"
        panel_bg = "#0f172a"
        text_color = "#e5e7eb"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {panel_bg};
            color: {text_color};
        }}
        .pm-card {{
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
            background: {card_bg};
            line-height: 1.5;
        }}
        .pm-answer {{
            border-left: 4px solid #4c8bf5;
            padding: 0.8rem 1rem;
            border-radius: 8px;
            background: {answer_bg};
            margin-bottom: 0.8rem;
        }}
        .pm-muted {{
            color: {muted_color};
            font-size: 0.95rem;
        }}
        .pm-section-title {{
            margin-top: 0.25rem;
            margin-bottom: 0.45rem;
            font-size: 1.15rem;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_get(url: str, timeout: int = 10) -> dict:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def safe_post(url: str, json_payload: dict | None = None, files=None, data=None, timeout: int = 60) -> dict:
    try:
        response = requests.post(url, json=json_payload, files=files, data=data, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def render_crystal_card(item: dict, rank: int) -> None:
    summary = item.get("fact_summary", "")
    short_summary = summary if len(summary) <= 280 else f"{summary[:280]}..."
    score = float(item.get("score", 0.0))
    source_type = item.get("source_type", "unknown")
    source_ref = item.get("source_ref", "unknown")
    crystal_id = item.get("crystal_id", "unknown")
    metadata = item.get("metadata", {})

    st.markdown(
        (
            f"<div class='pm-card'><b>#{rank}</b> | score: <b>{score:.4f}</b><br/>"
            f"source: <b>{source_type}</b> | ref: <code>{source_ref}</code><br/>"
            f"{short_summary}</div>"
        ),
        unsafe_allow_html=True,
    )
    with st.expander(f"Details for result #{rank}"):
        st.write(f"crystal_id: `{crystal_id}`")
        st.json(metadata)
        st.write("Full summary:")
        st.write(summary)


API_BASE = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")
health = safe_get(f"{API_BASE}/health")
current_mode = safe_get(f"{API_BASE}/index/mode").get("mode", "unknown")

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Auto"

st.sidebar.write("### Appearance")
theme_choice = st.sidebar.selectbox(
    "Theme",
    ["Auto", "Light", "Dark"],
    index=["Auto", "Light", "Dark"].index(st.session_state.theme_mode),
    help="Auto uses Streamlit default theme. Light/Dark apply in-app styles.",
)
st.session_state.theme_mode = theme_choice
inject_styles(st.session_state.theme_mode)

st.title("Decentralized Pocket Memory - MVP")
st.caption("Ingest docs, build memory crystals, and query locally.")
st.sidebar.write("### Status")
st.sidebar.write(f"Health: {health.get('status', 'unreachable')}")
st.sidebar.write(f"Current index mode: {current_mode}")

st.sidebar.write("### Retrieval")
mode_choice = st.sidebar.selectbox("Retrieval mode", ["flat", "hnsw", "ivfpq"])
if st.sidebar.button("Apply Retrieval Mode"):
    mode_resp = safe_post(f"{API_BASE}/index/mode", json_payload={"mode": mode_choice}, timeout=20)
    st.sidebar.json(mode_resp)

with st.expander("Source status"):
    if st.button("Refresh source status"):
        st.json(safe_get(f"{API_BASE}/sources/status"))

tab_ingest, tab_query, tab_metrics = st.tabs(["Ingest", "Query", "Metrics"])

with tab_ingest:
    st.markdown("<div class='pm-section-title'>Ingest Data</div>", unsafe_allow_html=True)
    st.markdown("<div class='pm-muted'>Start by ingesting a PDF, URL, or raw text.</div>", unsafe_allow_html=True)
    source_type = st.selectbox("Select source type", ["pdf", "url", "text", "slack", "discord", "github"])

    if source_type == "pdf":
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        source_ref = st.text_input("Source ref", value="uploaded_pdf")
        if st.button("Ingest PDF"):
            if not pdf_file:
                st.warning("Please upload a PDF file first.")
            else:
                files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                data = {"source_ref": source_ref}
                st.json(safe_post(f"{API_BASE}/ingest/pdf", files=files, data=data, timeout=60))
    elif source_type == "url":
        url = st.text_input("URL")
        if st.button("Ingest URL"):
            st.json(safe_post(f"{API_BASE}/ingest/url", json_payload={"url": url}, timeout=60))
    elif source_type == "github":
        github_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo/blob/main/README.md")
        if st.button("Ingest GitHub"):
            st.json(safe_post(f"{API_BASE}/ingest/github", json_payload={"url": github_url}, timeout=60))
    elif source_type == "text":
        text = st.text_area("Paste text")
        source_ref = st.text_input("Source ref", value="manual_text")
        if st.button("Ingest Text"):
            st.json(
                safe_post(
                    f"{API_BASE}/ingest/text",
                    json_payload={"text": text, "source_ref": source_ref, "source_type": "text"},
                    timeout=60,
                )
            )
    else:
        st.info("This source is planned but not configured yet. Fallback to PDF/URL/Text for now.")

with tab_query:
    st.markdown("<div class='pm-section-title'>Ask Your Memory</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pm-muted'>Run a question after ingestion. Increase Top K for broader retrieval context.</div>",
        unsafe_allow_html=True,
    )
    query = st.text_input("Ask your memory", placeholder="Example: What are the main architecture components?")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, help="Higher values return more crystals.")
    if st.button("Run Query"):
        data = safe_post(f"{API_BASE}/query", json_payload={"query": query, "top_k": top_k}, timeout=60)
        if "error" in data:
            st.error(data["error"])
        else:
            st.write("### Answer")
            st.markdown(f"<div class='pm-answer'>{data.get('answer', '')}</div>", unsafe_allow_html=True)

            st.write("### Query Stats")
            q_metrics = data.get("metrics", {})
            q1, q2 = st.columns(2)
            q1.metric("Latency (ms)", f"{float(q_metrics.get('query_latency_ms', 0.0)):.2f}")
            q2.metric("Top-K overlap vs exact", f"{float(q_metrics.get('topk_overlap_vs_exact', 0.0)):.2f}")

            st.write("### Retrieved Crystals")
            crystals = data.get("crystals", [])
            if not crystals:
                st.info("No relevant crystals returned. Try ingesting more data or increasing Top K.")
            else:
                crystals_sorted = sorted(crystals, key=lambda x: float(x.get("score", 0.0)), reverse=True)
                for rank, item in enumerate(crystals_sorted, start=1):
                    render_crystal_card(item, rank)

with tab_metrics:
    st.markdown("<div class='pm-section-title'>Metrics Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pm-muted'>Refresh to view latency, quality, throughput, and memory trends.</div>",
        unsafe_allow_html=True,
    )
    if st.button("Refresh Metrics"):
        metrics = safe_get(f"{API_BASE}/metrics")
        if "error" in metrics:
            st.error(metrics["error"])
        else:
            latency = metrics.get("query_latency_ms", {})
            throughput = metrics.get("throughput", {})
            retrieval = metrics.get("retrieval", {})
            citation = metrics.get("citation", {})
            memory = metrics.get("memory", {})
            sources = metrics.get("sources", {})
            series = metrics.get("series", {})

            c1, c2, c3 = st.columns(3)
            c1.metric("Latency p50 (ms)", f"{latency.get('p50', 0.0):.2f}")
            c2.metric("Latency p95 (ms)", f"{latency.get('p95', 0.0):.2f}")
            c3.metric("Queries/sec", f"{throughput.get('queries_per_sec', 0.0):.2f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("Ingest chunks/sec", f"{throughput.get('ingest_chunks_per_sec', 0.0):.2f}")
            c5.metric("Overlap vs exact", f"{retrieval.get('mean_topk_overlap_vs_exact', 0.0):.2f}")
            c6.metric("Citation precision proxy", f"{citation.get('precision_proxy', 0.0):.2f}")

            c7, c8, c9 = st.columns(3)
            c7.metric("Crystals", f"{memory.get('crystal_count', 0)}")
            c8.metric("Index size", f"{memory.get('index_size', 0)}")
            c9.metric("Compression ratio", f"{memory.get('compression_ratio', 1.0):.2f}")

            samples = metrics.get("samples", {})
            if samples.get("queries", 0) == 0:
                st.warning("No query samples yet. Run a few queries to populate latency and overlap charts.")

            st.write("### Source Reliability")
            s1, s2 = st.columns(2)
            s1.write("Source usage counts")
            s1.json(sources.get("source_counts", {}))
            s2.write("Fallback counts")
            s2.json(sources.get("fallback_counts", {}))

            recent_latencies = series.get("recent_query_latencies_ms", [])
            recent_overlap = series.get("recent_overlap_vs_exact", [])
            if recent_latencies:
                st.write("Recent Query Latencies (ms)")
                st.line_chart(recent_latencies)
            else:
                st.caption("No latency points yet.")
            if recent_overlap:
                st.write("Recent Overlap vs Exact")
                st.line_chart(recent_overlap)
            else:
                st.caption("No overlap points yet.")

            with st.expander("Raw metrics JSON (debug)"):
                st.json(metrics)
