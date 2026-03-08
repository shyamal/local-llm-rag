"""
retriever.py — Hybrid retrieval with FAISS, BM25, and RRF fusion.

Responsibilities:
- vector_search(): embed query and retrieve top-k chunks from FAISS
- bm25_search(): keyword search over the chunk corpus using BM25Okapi
- reciprocal_rank_fusion(): merge ranked lists using RRF (k=60)
- hybrid_search(): run both retrievers and return fused top-k results
"""