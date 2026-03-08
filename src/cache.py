import numpy as np
import time
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers import SentenceTransformer
from src.clustering import run_clustering

# ---------------------------------------------------------------------------
# DESIGN DECISIONS
#
# Data Structure: Cluster-partitioned dictionary
#   - The cache is a dict of lists: {cluster_id: [CacheEntry, ...]}
#   - When a query arrives, we embed it, find its dominant cluster, then
#     search ONLY within that cluster's cache entries.
#   - Without clustering, cache lookup is O(N) over all entries.
#     With cluster partitioning, lookup is O(N/K) on average — for K=10
#     clusters and N cached entries, this is a 10x speedup. At large N,
#     this is the difference between a usable and unusable cache.
#   - This is why Part 2 "should be doing real work" in Part 3 — the
#     cluster structure directly reduces the search space for every lookup.
#
# Similarity metric: Cosine similarity on L2-normalized embeddings
#   - Since embeddings are L2-normalized (normalize_embeddings=True in
#     embedder.py), cosine similarity reduces to a dot product: O(D)
#     per comparison where D=384. Fast and interpretable.
#
# THE ONE TUNABLE PARAMETER: similarity_threshold
#   - This is the single most consequential hyperparameter in the system.
#   - It determines what "close enough" means: two queries are considered
#     equivalent if their cosine similarity ≥ threshold.
#   - Behavior at different values:
#       0.70 → Very permissive. Semantically related but distinct queries
#              get cache hits. Risk: returning stale/wrong results.
#              Benefit: high hit rate, low recomputation.
#       0.80 → Balanced. Paraphrases reliably hit. Near-synonyms hit.
#              Different-angle queries on same topic may miss.
#       0.85 → Conservative. Only very close paraphrases hit.
#              Near-identical phrasing required for a hit.
#       0.90 → Very strict. Almost literal re-queries only.
#              Low hit rate but near-zero false positives.
#   - The interesting finding is NOT which value is "best" — it is what
#     each threshold reveals about the system's semantic resolution.
#     At 0.70, you discover that "gun laws" and "second amendment rights"
#     are neighbours in embedding space. At 0.90, they are not.
#   - Default: 0.85 — conservative enough for correctness, permissive
#     enough to demonstrate meaningful cache hits in testing.
#
# No external caching libraries:
#   - The entire cache is a Python dict + dataclasses + numpy.
#     No Redis, Memcached, diskcache, cachetools, or any caching library.
#     Every line of cache logic is written here from first principles.
# ---------------------------------------------------------------------------

MODEL_NAME  = 'all-MiniLM-L6-v2'
CACHE_PATH  = Path('embeddings/semantic_cache.pkl')


@dataclass
class CacheEntry:
    """A single cached query-result pair."""
    query_text      : str
    query_embedding : np.ndarray     # shape: (384,) — L2-normalized
    result          : str            # the search result returned
    dominant_cluster: int            # which cluster this query belongs to
    similarity_score: float = 1.0    # self-similarity on insertion
    timestamp       : float = field(default_factory=time.time)


class SemanticCache:
    """
    Cluster-partitioned semantic cache built from first principles.

    Architecture:
        _store: dict[int, list[CacheEntry]]
            Keys   = cluster IDs (0 to K-1)
            Values = list of CacheEntry objects in that cluster

    Lookup algorithm:
        1. Embed the incoming query (384-dim, L2-normalized)
        2. Find the query's dominant cluster via GMM predict_proba
        3. Search only within that cluster's cache entries
        4. Compute cosine similarity (= dot product on normalized vectors)
        5. If max similarity ≥ threshold → cache HIT, return stored result
        6. Otherwise → cache MISS, caller computes result and stores it
    """

    def __init__(self,
                 similarity_threshold: float = 0.85,
                 model_name: str = MODEL_NAME):

        self.similarity_threshold = similarity_threshold
        self._store  : dict[int, list[CacheEntry]] = {}
        self._hits   : int = 0
        self._misses : int = 0

        # Load embedding model — shared instance for encode() calls
        print(f"[cache] Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)

        # Load GMM + PCA for cluster assignment at query time
        print(f"[cache] Loading GMM and PCA models...")
        self._gmm = pickle.load(open('embeddings/gmm_model.pkl', 'rb'))
        self._pca = pickle.load(open('embeddings/pca_model.pkl', 'rb'))

        n_clusters = self._gmm.n_components
        for k in range(n_clusters):
            self._store[k] = []

        print(f"[cache] Initialized with {n_clusters} cluster partitions, "
              f"threshold={similarity_threshold}")

    # ------------------------------------------------------------------
    # Core public interface
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed and L2-normalize a query string. Returns shape (384,)."""
        return self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]

    def _get_dominant_cluster(self, query_embedding: np.ndarray) -> tuple[int, float]:
        """
        Project query embedding through PCA, then get GMM cluster assignment.
        Returns (dominant_cluster_id, confidence).
        """
        from sklearn.preprocessing import normalize as sk_normalize
        normed  = sk_normalize(query_embedding.reshape(1, -1), norm='l2')
        reduced = self._pca.transform(normed)        # (1, 50)
        probs   = self._gmm.predict_proba(reduced)[0]  # (K,)
        dominant   = int(probs.argmax())
        confidence = float(probs.max())
        return dominant, confidence

    def lookup(self, query: str) -> Optional[dict]:
        """
        Search the cache for a semantically equivalent prior query.

        Returns:
            dict with cache hit info if found, else None.
        """
        q_emb    = self._embed_query(query)
        cluster, confidence = self._get_dominant_cluster(q_emb)

        entries = self._store.get(cluster, [])
        if not entries:
            self._misses += 1
            return None

        # Vectorized cosine similarity over all entries in this cluster
        # (dot product on normalized vectors = cosine similarity)
        stored_embeddings = np.stack([e.query_embedding for e in entries])
        similarities      = stored_embeddings @ q_emb          # shape: (n_entries,)
        best_idx          = int(similarities.argmax())
        best_sim          = float(similarities[best_idx])

        if best_sim >= self.similarity_threshold:
            self._hits += 1
            best_entry = entries[best_idx]
            return {
                'cache_hit'      : True,
                'matched_query'  : best_entry.query_text,
                'similarity_score': round(best_sim, 4),
                'result'         : best_entry.result,
                'dominant_cluster': cluster,
            }

        self._misses += 1
        return None

    def store(self, query: str, result: str):
        """
        Store a new query-result pair in the cache.
        Deduplicates: if an identical query (sim=1.0) already exists, skips.
        """
        q_emb   = self._embed_query(query)
        cluster, _ = self._get_dominant_cluster(q_emb)

        # Prevent exact duplicate storage
        entries = self._store.get(cluster, [])
        if entries:
            stored_embeddings = np.stack([e.query_embedding for e in entries])
            sims = stored_embeddings @ q_emb
            if float(sims.max()) >= 0.9999:
                return  # exact duplicate, skip

        entry = CacheEntry(
            query_text       = query,
            query_embedding  = q_emb,
            result           = result,
            dominant_cluster = cluster,
        )
        self._store[cluster].append(entry)

    def get_stats(self) -> dict:
        """Return current cache statistics."""
        total   = sum(len(v) for v in self._store.values())
        total_queries = self._hits + self._misses
        hit_rate = round(self._hits / total_queries, 4) if total_queries > 0 else 0.0
        return {
            'total_entries': total,
            'hit_count'    : self._hits,
            'miss_count'   : self._misses,
            'hit_rate'     : hit_rate,
        }

    def flush(self):
        """Flush all cache entries and reset stats."""
        for k in self._store:
            self._store[k] = []
        self._hits   = 0
        self._misses = 0

    def save(self, path: Path = CACHE_PATH):
        """Persist the cache to disk (excludes model weights)."""
        state = {
            'store'               : self._store,
            'hits'                : self._hits,
            'misses'              : self._misses,
            'similarity_threshold': self.similarity_threshold,
        }
        pickle.dump(state, open(path, 'wb'))

    def load(self, path: Path = CACHE_PATH):
        """Restore a previously saved cache state."""
        if not path.exists():
            return
        state = pickle.load(open(path, 'rb'))
        self._store                = state['store']
        self._hits                 = state['hits']
        self._misses               = state['misses']
        self.similarity_threshold  = state['similarity_threshold']
        print(f"[cache] Restored {self.get_stats()['total_entries']} entries "
              f"(threshold={self.similarity_threshold})")


# ------------------------------------------------------------------
# Module-level singleton — shared across FastAPI app lifecycle
# ------------------------------------------------------------------
_cache_instance: Optional[SemanticCache] = None

def get_cache(similarity_threshold: float = 0.85) -> SemanticCache:
    """
    Returns the module-level cache singleton.
    Creates it on first call, reuses on subsequent calls.
    This is what FastAPI's dependency injection will call.
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache(similarity_threshold=similarity_threshold)
        _cache_instance.load()  # restore persisted state if it exists
    return _cache_instance


if __name__ == '__main__':
    # Smoke test + threshold exploration
    cache = SemanticCache(similarity_threshold=0.85)

    # Seed the cache with some queries
    test_pairs = [
        ("What are the best treatments for back pain?",
         "Result: sci.med documents about pain management and treatment options."),
        ("How does space shuttle launch work?",
         "Result: sci.space documents about shuttle propulsion and launch procedures."),
        ("What is the second amendment about?",
         "Result: talk.politics.guns documents about gun rights and US constitution."),
        ("Best graphics card for gaming",
         "Result: comp.sys.ibm.pc.hardware documents about GPU recommendations."),
    ]

    print("\n[smoke test] Seeding cache...")
    for q, r in test_pairs:
        cache.store(q, r)
    print(f"Cache entries: {cache.get_stats()['total_entries']}")

    # Test with paraphrases at different thresholds
    paraphrase_tests = [
        ("What medications help with back pain?",        "Should HIT — medical paraphrase"),
        ("Tell me about NASA rocket launches",            "Should HIT — space paraphrase"),
        ("Gun control and constitutional rights",         "Should HIT — guns/politics paraphrase"),
        ("Python programming tutorial for beginners",    "Should MISS — completely different topic"),
    ]

    print("\n[smoke test] Testing paraphrase hits at threshold=0.85:")
    for query, expectation in paraphrase_tests:
        result = cache.lookup(query)
        status = "HIT " if result else "MISS"
        sim    = f"sim={result['similarity_score']:.3f}" if result else "sim=N/A"
        print(f"  [{status}] ({sim}) {query[:50]:<50} ← {expectation}")

    print(f"\nFinal stats: {cache.get_stats()}")

    # Threshold sensitivity demonstration
    print("\n[smoke test] Threshold sensitivity — 'What medications help with back pain?':")
    for thresh in [0.70, 0.75, 0.80, 0.85, 0.90]:
        c = SemanticCache(similarity_threshold=thresh)
        for q, r in test_pairs:
            c.store(q, r)
        result = c.lookup("What medications help with back pain?")
        status = f"HIT  sim={result['similarity_score']:.3f}" if result else "MISS"
        print(f"  threshold={thresh} → {status}")
