import chromadb
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize as sk_normalize
from src.cache import get_cache, SemanticCache, CACHE_PATH

# ---------------------------------------------------------------------------
# DESIGN DECISIONS
#
# Lifespan context manager:
#   - All heavy resources (embedding model, ChromaDB client, GMM, PCA,
#     semantic cache) are loaded ONCE at startup using FastAPI's lifespan.
#   - This avoids re-loading the 90MB model on every request, which would
#     make the API unusably slow.
#   - Resources are stored in app.state for access across all endpoints.
#
# Query result generation:
#   - On a cache miss, we query ChromaDB for the top-5 most semantically
#     similar documents and return a formatted result string.
#   - The result is then stored in the cache for future hits.
#
# State management:
#   - The SemanticCache singleton persists in memory across requests.
#   - On shutdown, the cache is saved to disk so state survives restarts.
#   - The DELETE /cache endpoint flushes in-memory state and deletes the
#     persisted file, giving a true clean reset.
# ---------------------------------------------------------------------------

# ── Request / Response schemas ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query           : str
    cache_hit       : bool
    matched_query   : str | None
    similarity_score: float | None
    result          : str
    dominant_cluster: int

class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count    : int
    miss_count   : int
    hit_rate     : float


# ── Lifespan: load all resources once at startup ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading resources...")

    # Embedding model
    app.state.model = SentenceTransformer('all-MiniLM-L6-v2')

    # ChromaDB
    app.state.chroma_client = chromadb.PersistentClient(path='embeddings/chroma_db')
    app.state.collection    = app.state.chroma_client.get_collection('newsgroups')

    # GMM + PCA for cluster assignment
    app.state.gmm = pickle.load(open('embeddings/gmm_model.pkl', 'rb'))
    app.state.pca = pickle.load(open('embeddings/pca_model.pkl', 'rb'))

    # Semantic cache singleton
    app.state.cache = get_cache(similarity_threshold=0.85)

    print(f"[startup] Ready. Collection has {app.state.collection.count()} documents.")
    print(f"[startup] Cache has {app.state.cache.get_stats()['total_entries']} entries.")

    yield  # ── app is running ──

    # Shutdown: persist cache to disk
    print("[shutdown] Saving cache state...")
    app.state.cache.save(CACHE_PATH)
    print("[shutdown] Done.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Trademarkia Semantic Search API",
    description=(
        "Semantic search over the 20 Newsgroups corpus with a cluster-partitioned "
        "semantic cache. Queries that are semantically equivalent to prior queries "
        "are served from cache without recomputation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Embed and L2-normalize a query. Returns shape (384,)."""
    return model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]


def get_dominant_cluster(app_state, query_embedding: np.ndarray) -> tuple[int, float]:
    """Project through PCA → GMM to get cluster assignment."""
    normed  = sk_normalize(query_embedding.reshape(1, -1), norm='l2')
    reduced = app_state.pca.transform(normed)
    probs   = app_state.gmm.predict_proba(reduced)[0]
    return int(probs.argmax()), float(probs.max())


def compute_result(app_state, query: str, query_embedding: np.ndarray,
                   dominant_cluster: int) -> str:
    """
    Query ChromaDB for top-5 semantically similar documents.
    Filters by dominant_cluster for faster retrieval.
    Returns a formatted result string that gets cached.
    """
    try:
        results = app_state.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5,
            where={'dominant_cluster': dominant_cluster},
            include=['documents', 'metadatas', 'distances'],
        )
    except Exception:
        # Fallback: no cluster filter (handles edge cases)
        results = app_state.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5,
            include=['documents', 'metadatas', 'distances'],
        )

    docs      = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    if not docs:
        return "No results found."

    lines = [f"Top {len(docs)} results for: \"{query}\"\n"]
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        similarity = round(1 - dist, 4)   # ChromaDB cosine distance → similarity
        lines.append(
            f"[{i+1}] Category: {meta.get('category', 'unknown')} | "
            f"Similarity: {similarity} | "
            f"Cluster: {meta.get('dominant_cluster', '?')}\n"
            f"    Subject: {meta.get('subject', '')[:100]}\n"
            f"    Preview: {doc[:200].strip()}\n"
        )
    return "\n".join(lines)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Semantic search with cache lookup.

    1. Embed the query
    2. Check the semantic cache
    3. On HIT  → return cached result immediately
    4. On MISS → query ChromaDB, store result in cache, return result
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query = request.query.strip()
    cache: SemanticCache = app.state.cache

    # Step 1: Check cache
    hit = cache.lookup(query)
    if hit:
        return QueryResponse(
            query            = query,
            cache_hit        = True,
            matched_query    = hit['matched_query'],
            similarity_score = hit['similarity_score'],
            result           = hit['result'],
            dominant_cluster = hit['dominant_cluster'],
        )

    # Step 2: Cache miss — compute result
    q_emb             = embed_query(app.state.model, query)
    dominant_cluster, _ = get_dominant_cluster(app.state, q_emb)
    result            = compute_result(app.state, query, q_emb, dominant_cluster)

    # Step 3: Store in cache
    cache.store(query, result)

    return QueryResponse(
        query            = query,
        cache_hit        = False,
        matched_query    = None,
        similarity_score = None,
        result           = result,
        dominant_cluster = dominant_cluster,
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Return current cache statistics."""
    stats = app.state.cache.get_stats()
    return CacheStatsResponse(**stats)


@app.delete("/cache")
async def flush_cache():
    """
    Flush the cache entirely and reset all stats.
    Also deletes the persisted cache file if it exists.
    """
    app.state.cache.flush()
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
    return {"message": "Cache flushed successfully.", "status": "ok"}


@app.get("/")
async def root():
    """Health check + system info."""
    stats = app.state.cache.get_stats()
    return {
        "status"          : "ok",
        "service"         : "Trademarkia Semantic Search API",
        "version"         : "1.0.0",
        "corpus_size"     : app.state.collection.count(),
        "cache_entries"   : stats['total_entries'],
        "cache_hit_rate"  : stats['hit_rate'],
        "n_clusters"      : app.state.gmm.n_components,
        "similarity_threshold": app.state.cache.similarity_threshold,
    }
