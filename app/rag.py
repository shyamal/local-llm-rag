"""
rag.py — Document ingestion and RAG query pipeline.

Responsibilities:
- load_document(): load PDF, TXT, or Markdown from disk
- chunk_document(): split into overlapping chunks via LangChain splitter
- embed_chunks(): produce embeddings via SentenceTransformer
- build_index(): create FAISS IndexFlatL2 from embeddings
- save_index() / load_index(): persist and reload from vector_store/
- build_context_prompt(): assemble retrieved chunks into an LLM prompt
- rewrite_query(): reformulate the user query for better retrieval
- rag_query(): full end-to-end RAG pipeline with streaming response
"""

import json
import tempfile
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.chat import OllamaClient
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K, VECTOR_STORE_PATH
from app.embedder import get_embedder
from app.retriever import hybrid_search

_INDEX_FILE = VECTOR_STORE_PATH / "index.faiss"
_META_FILE = VECTOR_STORE_PATH / "metadata.json"
_CORPUS_FILE = VECTOR_STORE_PATH / "corpus.json"


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def load_document(file_path: str) -> str:
    """Load a PDF, TXT, or Markdown file and return raw text."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Unsupported file type: {suffix!r}. Supported: .pdf, .txt, .md")


def chunk_document(text: str) -> list[str]:
    """Split text into overlapping chunks using RecursiveCharacterTextSplitter."""
    return _splitter.split_text(text)


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Embed a list of text chunks using the configured SentenceTransformer model.

    Returns an ndarray of shape (len(chunks), embedding_dim).
    """
    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        raise ValueError("No non-empty chunks to embed.")
    embedder = get_embedder()
    return embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create and return a FAISS IndexFlatL2 from a 2-D embeddings array."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"embeddings must be a non-empty 2-D array, got shape {embeddings.shape}")
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.IndexFlatL2, metadata: list[dict]) -> None:
    """Persist the FAISS index, chunk metadata, and BM25 corpus to vector_store/ atomically."""
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss", dir=VECTOR_STORE_PATH) as f:
        tmp_index = Path(f.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", dir=VECTOR_STORE_PATH) as f:
        tmp_meta = Path(f.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", dir=VECTOR_STORE_PATH) as f:
        tmp_corpus = Path(f.name)
    faiss.write_index(index, str(tmp_index))
    tmp_meta.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    corpus = [m["text"] for m in metadata]
    tmp_corpus.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    tmp_index.replace(_INDEX_FILE)
    tmp_meta.replace(_META_FILE)
    tmp_corpus.replace(_CORPUS_FILE)


def load_index() -> tuple[faiss.IndexFlatL2, list[dict], list[str]] | None:
    """Load and return (index, metadata, corpus) from vector_store/, or None if missing."""
    if not _INDEX_FILE.exists() or not _META_FILE.exists():
        return None
    index = faiss.read_index(str(_INDEX_FILE))
    try:
        metadata = json.loads(_META_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Corrupted metadata file at {_META_FILE}: {e}") from e
    if _CORPUS_FILE.exists():
        try:
            corpus = json.loads(_CORPUS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            corpus = [m["text"] for m in metadata]
    else:
        corpus = [m["text"] for m in metadata]
    if len(corpus) != len(metadata):
        corpus = [m["text"] for m in metadata]
    return index, metadata, corpus


def ingest_document(file_path: str, source_name: str | None = None) -> int:
    """Run the full ingestion pipeline for a single file.

    Loads, chunks, embeds, and persists the FAISS index + metadata.
    Returns the number of chunks indexed.
    """
    text = load_document(file_path)
    chunks = chunk_document(text)
    embeddings = embed_chunks(chunks)
    index = build_index(embeddings)
    name = source_name or Path(file_path).name
    metadata = [
        {"chunk_id": i, "text": chunk, "source": name}
        for i, chunk in enumerate(chunks)
    ]
    save_index(index, metadata)
    return len(chunks)


def build_context_prompt(query: str, chunks: list[str]) -> str:
    """Insert retrieved chunks into a prompt template and return the full prompt."""
    if not chunks:
        raise ValueError("No context chunks provided — cannot build a grounded prompt.")
    numbered = "\n\n".join(f"[{i + 1}] {chunk.strip()}" for i, chunk in enumerate(chunks))
    return (
        "You are a helpful assistant. Answer the question using only the context provided below. "
        "If the answer cannot be found in the context, say so.\n\n"
        f"Context:\n{numbered}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


_REWRITE_PROMPT_TEMPLATE = (
    "You are a search query optimizer. Reformulate the question below to improve "
    "retrieval from a document store. Rules:\n"
    "- Expand abbreviations and acronyms\n"
    "- Make implicit concepts explicit\n"
    "- Focus on the core information need\n"
    "- Return ONLY the reformulated query on a single line — no explanation, no preamble.\n\n"
    "Original question: {query}\n"
    "Reformulated query:"
)


def rewrite_query(query: str, model: str) -> str:
    """Use the local LLM to reformulate query for improved retrieval.

    Returns the rewritten query string. Falls back to the original query if
    the LLM returns an empty response.
    """
    prompt = _REWRITE_PROMPT_TEMPLATE.replace("{query}", query)
    raw = OllamaClient().generate(prompt, model)
    rewritten = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    return rewritten or query


def rag_query(query: str, model: str, rewrite: bool = False):
    """Run full RAG pipeline: retrieve → build prompt → stream response.

    Yields tokens as they arrive from OllamaClient.stream().
    Raises RuntimeError if no FAISS index exists (documents must be ingested first).
    If rewrite=True, the query is reformulated via rewrite_query() before retrieval.
    """
    loaded = load_index()
    if loaded is None:
        raise RuntimeError("No FAISS index found. Ingest documents before running a RAG query.")

    retrieval_query = query
    if rewrite:
        try:
            retrieval_query = rewrite_query(query, model)
        except Exception:
            retrieval_query = query  # rewrite is optional; fall back silently on failure

    index, metadata, corpus = loaded
    results = hybrid_search(retrieval_query, index, metadata, corpus, top_k=TOP_K)
    chunks = [r["text"] for r in results]
    if not chunks:
        yield "No relevant documents found for your query. Try rephrasing or ingesting more documents."
        return
    prompt = build_context_prompt(query, chunks)
    yield from OllamaClient().stream(prompt, model)
