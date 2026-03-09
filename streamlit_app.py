import requests
import streamlit as st


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


API_BASE = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")
health = safe_get(f"{API_BASE}/health")
current_mode = safe_get(f"{API_BASE}/index/mode").get("mode", "unknown")

st.title("Decentralized Pocket Memory - MVP")
st.caption("Ingest docs, build memory crystals, and query locally.")
st.sidebar.write(f"Health: {health.get('status', 'unreachable')}")
st.sidebar.write(f"Current index mode: {current_mode}")

mode_choice = st.sidebar.selectbox("Retrieval mode", ["flat", "hnsw", "ivfpq"])
if st.sidebar.button("Apply Retrieval Mode"):
    mode_resp = safe_post(f"{API_BASE}/index/mode", json_payload={"mode": mode_choice}, timeout=20)
    st.sidebar.json(mode_resp)

with st.expander("Source status"):
    if st.button("Refresh source status"):
        st.json(safe_get(f"{API_BASE}/sources/status"))

st.subheader("Ingest")
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

st.subheader("Query")
query = st.text_input("Ask your memory")
top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
if st.button("Run Query"):
    data = safe_post(f"{API_BASE}/query", json_payload={"query": query, "top_k": top_k}, timeout=60)
    st.write("### Answer")
    st.write(data.get("answer", ""))
    st.write("### Query Metrics")
    st.json(data.get("metrics", {}))
    st.write("### Retrieved Crystals")
    st.json(data.get("crystals", []))

st.subheader("Metrics Dashboard")
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

        st.write("### Source Reliability")
        st.json(sources)

        recent_latencies = series.get("recent_query_latencies_ms", [])
        recent_overlap = series.get("recent_overlap_vs_exact", [])
        if recent_latencies:
            st.write("Recent Query Latencies (ms)")
            st.line_chart(recent_latencies)
        if recent_overlap:
            st.write("Recent Overlap vs Exact")
            st.line_chart(recent_overlap)
