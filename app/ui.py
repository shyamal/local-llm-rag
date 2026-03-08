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