# src/hybrid.py — hybrid retrieval: combine semantic (vector) + keyword (BM25) search.

from rank_bm25 import BM25Okapi

def build_bm25(chunks):
    """Tokenize chunks (simple lowercase split) and build a BM25 index."""
    tokenized = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized)

def hybrid_search(query, chunks, collection, model, bm25, n_results=8, k=60):
    """Return top-n chunk texts by fusing semantic and keyword rankings (RRF)."""
    # --- Semantic ranking (your existing vector search) ---
    q_emb = model.encode([query])
    sem = collection.query(query_embeddings=q_emb.tolist(), n_results=n_results)
    sem_ids = sem["ids"][0]            # chunk ids in semantic-rank order

    # --- Keyword ranking (BM25) ---
    scores = bm25.get_scores(query.lower().split())
    # indices of top-n chunks by BM25 score, best first
    bm_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    bm_ids = [f"chunk_{i}" for i in bm_idx]

    # --- Reciprocal Rank Fusion: combine the two rankings ---
    rrf = {}
    for rank, cid in enumerate(sem_ids):
        rrf[cid] = rrf.get(cid, 0) + 1 / (k + rank)
    for rank, cid in enumerate(bm_ids):
        rrf[cid] = rrf.get(cid, 0) + 1 / (k + rank)

    # Sort chunk ids by fused score, take top-n
    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:n_results]

    # Map ids back to chunk text (chunk_i -> chunks[i])
    return [chunks[int(cid.split("_")[1])] for cid in top_ids]