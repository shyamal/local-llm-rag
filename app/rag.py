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