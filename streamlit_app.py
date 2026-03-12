import html
import re
import statistics

import requests
import streamlit as st


def inject_styles(theme_mode: str) -> None:
    if theme_mode == "Auto":
        st.markdown(
            """
            <style>
            .pm-section-title {
                margin-top: 0.20rem;
                margin-bottom: 0.30rem;
                font-size: 1.15rem;
                font-weight: 600;
            }
            .badge {
                display: inline-block;
                padding: 2px 9px;
                border-radius: 6px;
                font-size: 0.73rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                vertical-align: middle;
            }
            .badge-reddit  { background: #ff4500; color: #fff; }
            .badge-github  { background: #24292e; color: #fff; }
            .badge-url     { background: #0ea5e9; color: #fff; }
            .badge-pdf     { background: #e11d48; color: #fff; }
            .badge-text    { background: #6b7280; color: #fff; }
            .badge-unknown { background: #d1d5db; color: #374151; }
            .stSelectbox > div > div,
            .stMultiSelect > div > div,
            .stButton > button,
            .stFileUploader label,
            [data-testid="stFileUploadDropzone"],
            .stSlider [role="slider"],
            .stTabs [role="tab"],
            a { cursor: pointer !important; }
            .stSelectbox input,
            .stMultiSelect input { cursor: pointer !important; caret-color: transparent; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    if theme_mode == "Light":
        panel_bg = "#f8fafc"
        panel_text = "#1f2937"
        card_bg = "rgba(248, 250, 252, 0.95)"
        card_border = "rgba(160, 174, 192, 0.65)"
        answer_bg = "rgba(66, 153, 225, 0.10)"
        muted_text = "#4f4f4f"
        meta_text = "#6b7280"
        ref_text = "#9ca3af"
    else:
        # Soft dark defaults for lower eye strain.
        panel_bg = "#111827"
        panel_text = "#e5e7eb"
        card_bg = "rgba(30, 41, 59, 0.45)"
        card_border = "rgba(100, 116, 139, 0.45)"
        answer_bg = "rgba(76, 139, 245, 0.14)"
        muted_text = "#aab4c5"
        meta_text = "#bac3d4"
        ref_text = "#8fa1bb"

    css = """
    <style>
    :root {
        --pm-panel-bg: __PANEL_BG__;
        --pm-panel-text: __PANEL_TEXT__;
        --pm-card-bg: __CARD_BG__;
        --pm-card-border: __CARD_BORDER__;
        --pm-answer-bg: __ANSWER_BG__;
        --pm-muted-text: __MUTED_TEXT__;
        --pm-meta-text: __META_TEXT__;
        --pm-ref-text: __REF_TEXT__;
    }
    .stApp { background: var(--pm-panel-bg); color: var(--pm-panel-text); }
    .pm-card {
        border: 1px solid var(--pm-card-border);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        background: var(--pm-card-bg);
        line-height: 1.6;
    }
    .pm-answer {
        border-left: 4px solid #4c8bf5;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        background: var(--pm-answer-bg);
        margin-bottom: 0.9rem;
        line-height: 1.75;
    }
    .pm-muted { color: var(--pm-muted-text); font-size: 0.95rem; }
    .pm-section-title {
        margin-top: 0.20rem;
        margin-bottom: 0.30rem;
        font-size: 1.15rem;
        font-weight: 600;
    }
    /* source badges */
    .badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 6px;
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        vertical-align: middle;
    }
    .badge-reddit  { background: #ff4500; color: #fff; }
    .badge-github  { background: #24292e; color: #fff; }
    .badge-url     { background: #0ea5e9; color: #fff; }
    .badge-pdf     { background: #e11d48; color: #fff; }
    .badge-text    { background: #6b7280; color: #fff; }
    .badge-unknown { background: #d1d5db; color: #374151; }
    /* cursor fixes */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stButton > button,
    .stFileUploader label,
    [data-testid="stFileUploadDropzone"],
    .stSlider [role="slider"],
    .stTabs [role="tab"],
    a { cursor: pointer !important; }
    .stSelectbox input,
    .stMultiSelect input { cursor: pointer !important; caret-color: transparent; }
    </style>
    """
    css = (
        css.replace("__PANEL_BG__", panel_bg)
        .replace("__PANEL_TEXT__", panel_text)
        .replace("__CARD_BG__", card_bg)
        .replace("__CARD_BORDER__", card_border)
        .replace("__ANSWER_BG__", answer_bg)
        .replace("__MUTED_TEXT__", muted_text)
        .replace("__META_TEXT__", meta_text)
        .replace("__REF_TEXT__", ref_text)
    )
    st.markdown(css, unsafe_allow_html=True)


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


def _clean_display(text: str) -> str:
    """Strip HTML tags/entities for user-facing display."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_BADGE: dict = {
    "reddit": ("badge-reddit", "\U0001f534"),
    "github": ("badge-github", "\U0001f431"),
    "url":    ("badge-url",    "\U0001f310"),
    "pdf":    ("badge-pdf",    "\U0001f4c4"),
    "text":   ("badge-text",   "\U0001f4dd"),
}


def record_mode_sample(mode: str, metrics: dict) -> None:
    entry = {
        "mode": mode,
        "latency_ms": float(metrics.get("query_latency_ms", 0.0)),
        "overlap": float(metrics.get("topk_overlap_vs_exact", 0.0)),
        "retrieval_ms": float(metrics.get("retrieval", {}).get("total_ms", 0.0)),
    }
    st.session_state.mode_samples.append(entry)
    st.session_state.mode_samples = st.session_state.mode_samples[-200:]


def mode_comparison_rows() -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for sample in st.session_state.mode_samples:
        grouped.setdefault(sample["mode"], []).append(sample)
    rows: list[dict] = []
    for mode, samples in grouped.items():
        rows.append(
            {
                "mode": mode,
                "queries": len(samples),
                "mean_latency_ms": statistics.mean(s["latency_ms"] for s in samples),
                "mean_overlap_vs_exact": statistics.mean(s["overlap"] for s in samples),
                "mean_retrieval_ms": statistics.mean(s["retrieval_ms"] for s in samples),
            }
        )
    rows.sort(key=lambda r: r["mode"])
    return rows


def summarize_query_usage_rows(items: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        used = item.get("crystals_used", [])
        entries = used if isinstance(used, list) else []
        top_id = ""
        top_similarity = 0.0
        contribution_sum = 0.0
        reward_sum = 0.0
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            crystal_id = str(entry.get("crystal_id", ""))
            similarity = float(entry.get("similarity", 0.0))
            contribution = float(entry.get("contribution_score", 0.0))
            reward_delta = float(entry.get("reward_delta", 0.0))
            contribution_sum += contribution
            reward_sum += reward_delta
            if idx == 0:
                top_id = crystal_id
                top_similarity = similarity
        rows.append(
            {
                "query_id": item.get("query_id", ""),
                "created_at": item.get("created_at", ""),
                "latency_ms": float(item.get("latency_ms", 0.0)),
                "reward_pool": float(item.get("reward_pool", 0.0)),
                "crystal_count": len(entries),
                "top_crystal_id": top_id,
                "top_similarity": top_similarity,
                "contribution_sum": contribution_sum,
                "reward_delta_sum": reward_sum,
            }
        )
    return rows


def render_crystal_card(item: dict, rank: int) -> None:
    summary = _clean_display(item.get("clean_summary") or item.get("fact_summary", ""))
    score = float(item.get("score", 0.0))
    source_type = item.get("source_type", "unknown")
    source_ref = item.get("source_ref", "unknown")
    crystal_id = item.get("crystal_id", "unknown")
    badge_cls, icon = _BADGE.get(source_type, ("badge-unknown", "\U0001f539"))
    preview = _clean_display(item.get("preview_summary") or "")
    if not preview:
        preview = summary if len(summary) <= 340 else summary[:340] + "\u2026"

    st.markdown(
        f"""<div class='pm-card'>
  <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.45rem'>
    <span style='font-weight:700;font-size:1rem'>#{rank}</span>
    <span class='badge {badge_cls}'>{icon} {source_type.upper()}</span>
    <span style='margin-left:auto;color:var(--pm-meta-text);font-size:0.82rem'>relevance&nbsp;<b>{score:.3f}</b></span>
  </div>
  <div style='font-size:0.91rem;color:var(--pm-panel-text);line-height:1.6;margin-bottom:0.45rem'>{preview}</div>
  <div style='font-size:0.76rem;color:var(--pm-ref-text);word-break:break-all'>&#128204; {source_ref}</div>
</div>""",
        unsafe_allow_html=True,
    )
    with st.expander(f"Full content \u2014 result #{rank}"):
        st.markdown(f"**Source:** `{source_type}` &nbsp;|&nbsp; **Ref:** `{source_ref}`", unsafe_allow_html=True)
        st.markdown(f"**Crystal ID:** `{crystal_id}`")
        st.divider()
        st.write(summary)


API_BASE = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")
health = safe_get(f"{API_BASE}/health")
current_mode = safe_get(f"{API_BASE}/index/mode").get("mode", "unknown")

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"
if "mode_samples" not in st.session_state:
    st.session_state.mode_samples = []

st.sidebar.write("### Appearance")
theme_mode = st.sidebar.selectbox(
    "Theme",
    ["Dark", "Light", "Auto"],
    index=["Dark", "Light", "Auto"].index(st.session_state.theme_mode),
    help="Dark is default (softer contrast). Auto follows Streamlit defaults.",
)
st.session_state.theme_mode = theme_mode
inject_styles(theme_mode)

st.title("Decentralized Pocket Memory - MVP")
st.caption("Ingest docs, build memory crystals, and query locally.")
st.sidebar.write("### Status")
st.sidebar.write(f"Health: {health.get('status', 'unreachable')}")
st.sidebar.write(f"Current index mode: {current_mode}")

st.sidebar.write("### Retrieval")
mode_choice = st.sidebar.selectbox("Retrieval mode", ["flat", "hnsw", "ivfpq", "hybrid_binary", "binary_only"])
if st.sidebar.button("Apply Retrieval Mode"):
    mode_resp = safe_post(f"{API_BASE}/index/mode", json_payload={"mode": mode_choice}, timeout=20)
    st.sidebar.json(mode_resp)

with st.expander("Source status"):
    if st.button("Refresh source status"):
        st.json(safe_get(f"{API_BASE}/sources/status"))

tab_ingest, tab_query, tab_metrics, tab_registry, tab_wallet = st.tabs(
    ["Ingest", "Query", "Metrics", "Registry", "Wallets"]
)

with tab_ingest:
    st.markdown("<div class='pm-section-title'>Ingest Data</div>", unsafe_allow_html=True)
    st.markdown("<div class='pm-muted'>Start by ingesting a PDF, URL, or raw text.</div>", unsafe_allow_html=True)
    source_type = st.selectbox("Select source type", ["pdf", "url", "text", "reddit", "github", "slack", "discord"])

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
    elif source_type == "reddit":
        reddit_url = st.text_input("Reddit URL", placeholder="https://reddit.com/r/python/comments/abc123/title")
        if st.button("Ingest Reddit"):
            st.json(safe_post(f"{API_BASE}/ingest/reddit", json_payload={"url": reddit_url}, timeout=60))
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
        "<div class='pm-muted'>Query spans all ingested sources. Use Top K + source filter for cleaner, focused results.</div>",
        unsafe_allow_html=True,
    )
    query_input = st.text_input("Ask your memory", placeholder="Example: What are the main architecture components?")
    qc1, qc2 = st.columns([2, 1])
    with qc1:
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5, help="How many crystals to retrieve.")
    with qc2:
        source_filter = st.multiselect(
            "Filter by source",
            ["pdf", "url", "text", "reddit", "github"],
            default=[],
            help="Leave empty to search across all ingested sources.",
        )
    if st.button("Run Query", type="primary"):
        q_payload: dict = {"query": query_input, "top_k": top_k}
        if source_filter:
            q_payload["source_types"] = source_filter
        resp = safe_post(f"{API_BASE}/query", json_payload=q_payload, timeout=60)
        if "error" in resp:
            st.error(resp["error"])
        else:
            active_mode = safe_get(f"{API_BASE}/index/mode").get("mode", "unknown")
            st.markdown("### Answer")
            answer_text = (resp.get("answer") or "").strip()
            if answer_text:
                safe_answer = html.escape(answer_text).replace("\n", "<br>")
                st.markdown(f"<div class='pm-answer'>{safe_answer}</div>", unsafe_allow_html=True)
            else:
                st.info("No answer generated yet.")

            st.markdown("### Query Stats")
            q_metrics = resp.get("metrics", {})
            record_mode_sample(active_mode, q_metrics)
            qm1, qm2 = st.columns(2)
            qm1.metric("Latency (ms)", f"{float(q_metrics.get('query_latency_ms', 0.0)):.2f}")
            qm2.metric("Top-K overlap vs exact", f"{float(q_metrics.get('topk_overlap_vs_exact', 0.0)):.2f}")
            if q_metrics.get("query_id"):
                st.caption(f"Query ID: `{q_metrics.get('query_id')}`")
            retrieval_stats = q_metrics.get("retrieval", {})
            if retrieval_stats:
                qm3, qm4, qm5 = st.columns(3)
                qm3.metric("Prefilter candidates", f"{int(retrieval_stats.get('prefilter_candidates', 0))}")
                qm4.metric("Prefilter ms", f"{float(retrieval_stats.get('prefilter_ms', 0.0)):.2f}")
                qm5.metric("Rerank ms", f"{float(retrieval_stats.get('rerank_ms', 0.0)):.2f}")

            st.markdown("### Retrieved Crystals")
            crystals = resp.get("crystals", [])
            if not crystals:
                st.info("No relevant crystals returned. Try ingesting more data or increasing Top K.")
            else:
                crystals_sorted = sorted(crystals, key=lambda x: float(x.get("score", 0.0)), reverse=True)
                for rank, item in enumerate(crystals_sorted, start=1):
                    render_crystal_card(item, rank)

with tab_metrics:
    st.markdown("<div class='pm-section-title'>Metrics Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pm-muted'>Refresh to view latency, quality, throughput, and memory trends in this session.</div>",
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

            c10, c11, c12 = st.columns(3)
            c10.metric("Float vector bytes", f"{int(memory.get('float_vector_bytes', 0))}")
            c11.metric("Binary vector bytes", f"{int(memory.get('binary_vector_bytes', 0))}")
            c12.metric("Vector compression", f"{float(memory.get('vector_compression_ratio', 0.0)):.2f}x")

            c13, _ = st.columns(2)
            c13.metric("Total footprint bytes", f"{int(memory.get('total_footprint_bytes', 0))}")

            st.write("### Hybrid Retrieval Breakdown")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Mean prefilter candidates", f"{retrieval.get('mean_prefilter_candidates', 0.0):.1f}")
            h2.metric("Mean prefilter ms", f"{retrieval.get('mean_prefilter_ms', 0.0):.2f}")
            h3.metric("Mean rerank ms", f"{retrieval.get('mean_rerank_ms', 0.0):.2f}")
            h4.metric("Mean retrieval ms", f"{retrieval.get('mean_total_retrieval_ms', 0.0):.2f}")

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

    st.write("### Mode Comparison (Session)")
    rows = mode_comparison_rows()
    if rows:
        st.dataframe(rows, width="stretch")
    else:
        st.caption("Run queries in multiple modes to compare flat/hybrid_binary/binary_only.")

with tab_registry:
    st.markdown("<div class='pm-section-title'>Crystal Registry</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pm-muted'>Track crystal provenance, usage counts, reward balances, and query-to-crystal usage logs.</div>",
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        crystal_limit = st.number_input("Crystal records", min_value=1, max_value=1000, value=100, step=10)
    with col_b:
        query_limit = st.number_input("Query usage records", min_value=1, max_value=1000, value=100, step=10)

    if st.button("Refresh Registry Data"):
        crystals_resp = safe_get(f"{API_BASE}/registry/crystals?limit={int(crystal_limit)}")
        leaderboard_resp = safe_get(f"{API_BASE}/registry/leaderboard?limit=20")
        queries_resp = safe_get(f"{API_BASE}/registry/queries?limit={int(query_limit)}")

        if "error" in crystals_resp:
            st.error(crystals_resp["error"])
        else:
            items = crystals_resp.get("items", [])
            st.write("### Crystal Records")
            st.caption(f"Total returned: {crystals_resp.get('count', len(items))}")
            if items:
                st.dataframe(items, width="stretch")
            else:
                st.info("No crystal registry records yet. Ingest data first.")

        if "error" in leaderboard_resp:
            st.error(leaderboard_resp["error"])
        else:
            st.write("### Leaderboard (Usage/Rewards)")
            litems = leaderboard_resp.get("items", [])
            if litems:
                st.dataframe(litems, width="stretch")
            else:
                st.caption("No leaderboard entries yet.")

        if "error" in queries_resp:
            st.error(queries_resp["error"])
        else:
            st.write("### Query Usage Log")
            qitems = queries_resp.get("items", [])
            if qitems:
                summary_rows = summarize_query_usage_rows(qitems)
                st.dataframe(summary_rows, width="stretch")
                with st.expander("Raw query usage JSON"):
                    st.json(qitems)
            else:
                st.info("No query usage logs yet. Run a few queries.")
    
    # ===== BLOCKCHAIN INTEGRATION SECTION =====
    st.write("### 🔗 Blockchain Verification")
    
    if st.button("Refresh Blockchain Status"):
        blockchain_resp = safe_get(f"{API_BASE}/blockchain/status")
        
        if "error" in blockchain_resp:
            st.error(f"Blockchain error: {blockchain_resp['error']}")
        else:
            # Show account info
            account = blockchain_resp.get("account", {})
            st.markdown("**Account Information:**")
            
            if account.get("connected", False):
                st.success(f"🟢 Connected to {account.get('network', 'Unknown')}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Address", account.get("address", "N/A")[:10] + "..." if account.get("address") else "N/A")
                with col2:
                    st.metric("Balance", account.get("balance", "0"))
            else:
                st.warning(f"🟡 Not connected: {account.get('error', 'Unknown error')}")
            
            # Show verification stats
            st.markdown("**Crystal Verification Status:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Verified on Blockchain", blockchain_resp.get("crystals_verified", 0))
            with col2:
                st.metric("Pending Verification", blockchain_resp.get("crystals_pending", 0))
            with col3:
                st.metric("Total Crystals", blockchain_resp.get("total_crystals", 0))
            
            # Crystal verification tool
            st.markdown("**Verify Individual Crystal:**")
            verify_crystal_id = st.text_input("Crystal ID to verify:", key="verify_crystal_id")
            if st.button("Verify on Blockchain") and verify_crystal_id:
                verify_resp = safe_post(f"{API_BASE}/blockchain/verify/{verify_crystal_id}", {})
                if "error" in verify_resp:
                    st.error(f"Verification failed: {verify_resp['error']}")
                else:
                    if verify_resp.get("verified", False):
                        st.success("✅ Crystal verified on blockchain!")
                        st.json(verify_resp)
                    else:
                        st.warning("❌ Crystal not found on blockchain")
                        if "error" in verify_resp:
                            st.error(verify_resp["error"])

with tab_wallet:
    st.markdown("<div class='pm-section-title'>Wallets and Transferability</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pm-muted'>View derived wallet balances from owned crystals and transfer crystal ownership.</div>",
        unsafe_allow_html=True,
    )

    owner_id = st.text_input("Owner ID", value="local_user", key="wallet_owner_id")
    wcol1, wcol2 = st.columns(2)
    with wcol1:
        crystal_limit = st.number_input("Owned crystals limit", min_value=1, max_value=1000, value=200, key="wallet_crystals_limit")
    with wcol2:
        transfer_limit = st.number_input("Transfer history limit", min_value=1, max_value=1000, value=200, key="wallet_transfers_limit")

    if st.button("Refresh Wallet View", key="wallet_refresh"):
        wallet_resp = safe_get(f"{API_BASE}/wallets/{owner_id}")
        crystals_resp = safe_get(f"{API_BASE}/wallets/{owner_id}/crystals?limit={int(crystal_limit)}")
        transfers_resp = safe_get(f"{API_BASE}/wallets/{owner_id}/transfers?limit={int(transfer_limit)}")

        if "error" in wallet_resp:
            st.error(wallet_resp["error"])
        else:
            st.write("### Wallet Snapshot")
            st.json(wallet_resp)

        if "error" in crystals_resp:
            st.error(crystals_resp["error"])
        else:
            st.write("### Owned Crystals")
            if crystals_resp:
                st.dataframe(crystals_resp, width="stretch")
            else:
                st.caption("No crystals owned by this wallet.")

        if "error" in transfers_resp:
            st.error(transfers_resp["error"])
        else:
            st.write("### Transfer History")
            transfer_items = transfers_resp.get("items", [])
            if transfer_items:
                st.dataframe(transfer_items, width="stretch")
            else:
                st.caption("No transfer events for this owner.")

    st.write("### Transfer Crystal")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        transfer_crystal_id = st.text_input("Crystal ID", key="transfer_crystal_id")
        transfer_actor_id = st.text_input("Actor ID (optional)", value=owner_id, key="transfer_actor_id")
    with tcol2:
        transfer_new_owner = st.text_input("New Owner ID", key="transfer_new_owner")
        transfer_reason = st.text_input("Reason (optional)", key="transfer_reason")
    if st.button("Transfer Crystal", type="primary", key="transfer_crystal_button"):
        payload = {
            "crystal_id": transfer_crystal_id.strip(),
            "new_owner_id": transfer_new_owner.strip(),
            "actor_id": transfer_actor_id.strip() or None,
            "reason": transfer_reason.strip() or None,
        }
        transfer_resp = safe_post(f"{API_BASE}/wallets/transfer", json_payload=payload, timeout=60)
        if "error" in transfer_resp:
            st.error(transfer_resp["error"])
        else:
            st.success("Transfer submitted.")
            st.json(transfer_resp)
