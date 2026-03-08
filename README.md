# Local AI Assistant

A fully offline, privacy-focused AI assistant built with Streamlit and Ollama. This project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline running entirely on local hardware.

## Features

- **Private & Offline**: Runs entirely on your local machine using Ollama.
- **Document QA (RAG)**: Upload PDF, TXT, or Markdown files and ask questions directly against their contents.
- **Streaming Responses**: Real-time token streaming for chat and document queries.
- **Inference Benchmarks**: Built-in dashboard to monitor Time To First Token (TTFT), throughput, and latency.
- **Model Comparison**: Compare performance across multiple local models using a benchmark prompt.

## Prerequisites

- [Ollama](https://ollama.ai/) installed and running locally
- Python 3.10 or higher
- At least one model pulled in Ollama (e.g., `ollama pull mistral`)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

2. **Run the application:**
   ```bash
   python -m streamlit run app/ui.py
   ```

3. **Open your browser:**
   Navigate to the Local URL provided (typically `http://localhost:8501`).

## Architecture

- **Frontend**: Streamlit
- **LLM Engine**: Ollama (REST API via `requests`)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Search**: FAISS
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`

## Project Structure

- `app/ui.py`: Main Streamlit frontend
- `app/chat.py`: Ollama API client with streaming and metrics collection
- `app/rag.py`: Document ingestion, chunking, and generation pipeline
- `app/retriever.py`: Vector search and hybrid search utilities
- `app/embedder.py`: Shared sentence-transformer singleton
- `app/metrics.py`: Singleton for collecting and storing inference metrics
- `benchmarks/benchmark.py`: Utilities for multi-model performance testing
