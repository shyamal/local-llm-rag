"""
ui.py — Streamlit entry point and chat interface.

Responsibilities:
- Page config: title, layout, sidebar
- Model selector dropdown (available Ollama models)
- Chat message display with st.chat_message
- st.chat_input prompt box
- Session history via st.session_state
- Document uploader wired to RAG ingestion pipeline
- Benchmarks tab with st.metric and st.dataframe
- Evaluation section with aggregated eval scores
"""

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app.*` imports resolve
# whether the script is launched as `streamlit run app/ui.py` or as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.chat import OllamaClient
from app.metrics import get_collector
from app.rag import ingest_document, load_index, rag_query
from benchmarks.benchmark import BENCHMARK_PROMPT, export_results, run_multi_model_benchmark

FALLBACK_MODELS = ["mistral", "llama3"]
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "data/documents"))

# Performance targets from CLAUDE.md
_TARGET_TTFT = 2.0       # seconds
_TARGET_TPS = 20.0       # tokens/sec
_TARGET_LATENCY = 5.0    # seconds


@st.cache_resource
def _get_client() -> OllamaClient:
    """Shared OllamaClient instance, created once per server lifetime."""
    return OllamaClient()


@st.cache_data(ttl=30)
def _fetch_models(base_url: str) -> tuple[list[str], bool]:
    """Return (model_list, ollama_reachable). Result cached for 30 seconds."""
    try:
        models = OllamaClient(base_url).list_models()
        return (models if models else FALLBACK_MODELS, bool(models))
    except Exception:
        return FALLBACK_MODELS, False


def _ingest_document(uploaded_file) -> int:
    """Save uploaded file, run ingestion pipeline, persist index. Returns chunk count."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name  # strip any directory components
    dest = DOCUMENTS_DIR / safe_name
    dest.write_bytes(uploaded_file.getvalue())
    return ingest_document(str(dest), source_name=safe_name)


def _render_chat_tab(client: OllamaClient, available_models: list[str]) -> None:
    mode_label = "Document QA" if st.session_state.rag_mode else "Chat"
    st.header(mode_label)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    placeholder = (
        "Ask a question about the document…"
        if st.session_state.rag_mode
        else "Ask anything…"
    )

    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        response_text = None
        with st.chat_message("assistant"):
            try:
                if st.session_state.rag_mode:
                    response_text = st.write_stream(
                        rag_query(prompt, model=st.session_state.selected_model)
                    )
                else:
                    response_text = st.write_stream(
                        client.chat_stream(
                            messages=st.session_state.messages,
                            model=st.session_state.selected_model,
                        )
                    )
            except RuntimeError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Error: {exc}")

        if response_text:
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )


def _render_benchmarks_tab(available_models: list[str]) -> None:
    st.header("Benchmarks")

    all_results = get_collector().get_results()

    # ── Last inference ─────────────────────────────────────────────────────────
    st.subheader("Last Inference")
    if all_results:
        last = all_results[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "TTFT",
            f"{last['ttft']:.3f}s",
            delta=f"{_TARGET_TTFT - last['ttft']:+.3f}s vs {_TARGET_TTFT}s target",
            delta_color="normal",
            help="Time To First Token — target < 2s",
        )
        col2.metric(
            "Tokens / sec",
            f"{last['tokens_per_sec']:.1f}",
            delta=f"{last['tokens_per_sec'] - _TARGET_TPS:+.1f} vs {_TARGET_TPS:.0f} target",
            delta_color="normal",
            help="Throughput — target > 20 tokens/s",
        )
        col3.metric(
            "Total Latency",
            f"{last['total_latency']:.3f}s",
            delta=f"{_TARGET_LATENCY - last['total_latency']:+.3f}s vs {_TARGET_LATENCY}s target",
            delta_color="normal",
            help="End-to-end generation time — target < 5s",
        )
    else:
        st.info("Send a message in the **Chat** tab to see live inference metrics here.")

    # ── Session history table ──────────────────────────────────────────────────
    st.subheader("Session History")
    if all_results:
        rows = [
            {
                "Model": r["model"],
                "Time": r["timestamp"][:19].replace("T", " "),
                "TTFT (s)": r["ttft"],
                "Tokens/s": r["tokens_per_sec"],
                "Latency (s)": r["total_latency"],
                "Tokens": r["token_count"],
            }
            for r in reversed(all_results)
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        if st.button("Clear session metrics"):
            get_collector().clear()
            st.rerun()
    else:
        st.caption("No inferences recorded yet.")

    st.divider()

    # ── Model comparison runner ────────────────────────────────────────────────
    st.subheader("Model Comparison")
    st.caption(f'Benchmark prompt: *"{BENCHMARK_PROMPT}"*')

    with st.form("benchmark_form"):
        bench_models = st.multiselect(
            "Models to benchmark",
            available_models,
            default=available_models[:2] if len(available_models) >= 2 else available_models,
        )
        bench_runs = st.number_input(
            "Runs per model", min_value=1, max_value=20, value=3, step=1
        )
        run_clicked = st.form_submit_button("Run Benchmark", use_container_width=True)

    if run_clicked:
        if not bench_models:
            st.warning("Select at least one model to benchmark.")
        else:
            with st.spinner(f"Benchmarking {', '.join(bench_models)} — {bench_runs} run(s) each…"):
                bench_results = run_multi_model_benchmark(
                    BENCHMARK_PROMPT, bench_models, n=int(bench_runs), verbose=False
                )
            dest = export_results(bench_results)
            st.session_state.comparison_results = bench_results
            st.success(f"Saved to `{dest.name}`")

    if st.session_state.get("comparison_results"):
        summary_rows = []
        for model, runs in st.session_state.comparison_results.items():
            if not runs:
                continue
            summary_rows.append(
                {
                    "Model": model,
                    "Avg TTFT (s)": round(sum(r["ttft"] for r in runs) / len(runs), 3),
                    "Avg Tokens/s": round(sum(r["tokens_per_sec"] for r in runs) / len(runs), 1),
                    "Avg Latency (s)": round(sum(r["total_latency"] for r in runs) / len(runs), 3),
                    "Runs": len(runs),
                }
            )
        if summary_rows:
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Local AI Assistant", layout="wide")

    client = _get_client()

    # --- Session state initialisation ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "mistral"
    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = False
    if "ingested_file" not in st.session_state:
        st.session_state.ingested_file = None
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = {}

    # Fetch model list once — used by both sidebar and benchmarks tab
    available_models, ollama_ok = _fetch_models(client.base_url)

    # --- Sidebar ---
    with st.sidebar:
        st.title("Local AI Assistant")
        st.caption("Fully offline · Powered by Ollama")

        st.divider()

        if not ollama_ok:
            st.warning("Ollama unreachable — using default model list.")

        default_idx = (
            available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models
            else 0
        )
        st.session_state.selected_model = st.selectbox(
            "Model",
            available_models,
            index=default_idx,
        )

        st.divider()

        # Document QA mode toggle
        st.session_state.rag_mode = st.toggle(
            "Document QA mode",
            value=st.session_state.rag_mode,
        )

        if st.session_state.rag_mode:
            st.caption("Upload a document to answer questions from its contents.")
            uploaded = st.file_uploader(
                "Upload document",
                type=["pdf", "txt", "md"],
                label_visibility="collapsed",
            )
            file_key = (uploaded.name, uploaded.size) if uploaded else None
            if uploaded and file_key != st.session_state.ingested_file:
                with st.spinner(f"Ingesting {uploaded.name}…"):
                    try:
                        n_chunks = _ingest_document(uploaded)
                        st.session_state.ingested_file = file_key
                        st.success(f"Indexed {n_chunks} chunks from {uploaded.name}")
                    except Exception as exc:
                        st.error(f"Ingestion failed: {exc}")

            if st.session_state.ingested_file:
                index_info = load_index()
                if index_info:
                    _, meta = index_info
                    active_name = st.session_state.ingested_file[0]
                    st.info(f"Active document: **{active_name}** ({len(meta)} chunks)")

        st.divider()

        if st.button("Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- Main content tabs ---
    chat_tab, benchmarks_tab = st.tabs(["Chat", "Benchmarks"])

    with chat_tab:
        _render_chat_tab(client, available_models)

    with benchmarks_tab:
        _render_benchmarks_tab(available_models)


if __name__ == "__main__":
    main()
