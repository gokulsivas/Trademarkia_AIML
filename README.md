# Trademarkia AI/ML Engineer Task — Semantic Search System

A lightweight semantic search system over the **20 Newsgroups corpus** (~20,000 documents), featuring fuzzy clustering, a cluster-partitioned semantic cache, and a FastAPI service.

---

## System Architecture

```
20_newsgroups.csv
      │
      ▼
┌─────────────────┐
│  preprocess.py  │  Clean body text, strip noise, prepend subject
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  embedder.py    │  all-MiniLM-L6-v2 → 384-dim vectors → ChromaDB
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  clustering.py  │  PCA(50) → GMM(K=10) → soft cluster assignments
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  cache.py       │  Cluster-partitioned semantic cache (no Redis)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  main.py        │  FastAPI: /query · /cache/stats · DELETE /cache
└─────────────────┘
```

---

## Setup

```bash
git clone <repo-url>
cd AIML
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Build the pipeline (one-time)

```bash
# Step 1: Embed and store in ChromaDB (~70 seconds with GPU)
python -m src.embedder

# Step 2: Fit GMM clustering (~10 minutes)
python -m src.clustering
```

### Start the API

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Visit [[**http://localhost:8000/docs**](http://localhost:8000/docs)](http://localhost:8000/docs) for the interactive Swagger UI.

---

## API Endpoints

### `POST /query`
```json
{ "query": "How does the space shuttle launch work?" }
```
**Response:**
```json
{
  "query": "How does the space shuttle launch work?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "Top 5 results...",
  "dominant_cluster": 0
}
```
On a second semantically equivalent query:
```json
{
  "cache_hit": true,
  "matched_query": "How does the space shuttle launch work?",
  "similarity_score": 0.94,
  ...
}
```

### `GET /cache/stats`
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`
Flushes all cache entries and resets stats.

---

## Design Decisions

### Part 1 — Preprocessing
- Headers already decomposed into CSV columns — only `body` is cleaned
- Four quote styles handled: `>`, `:`, `->`, `}` (all confirmed in corpus)
- UUEncoded binary blocks removed (confirmed present in `alt.atheism` posts)
- Subject line prepended as topic anchor — densest semantic signal per post
- No lowercasing, stemming, or stopword removal — `all-MiniLM-L6-v2` is trained on cased subword tokens

### Part 2 — Fuzzy Clustering

**Algorithm: Gaussian Mixture Model (GMM)**
- Outputs a full probability distribution per document — satisfies the task's requirement for soft assignments
- Chosen over Fuzzy C-Means (less principled) and LDA (operates on token counts, not embeddings)

**Dimensionality: PCA to 50 dims before GMM**
- Raw 384-dim embeddings cause ill-conditioned covariance matrices in GMM
- 50 dims retains ~50% variance while making EM tractable

**K selection: BIC**

| K  | BIC          | AIC          | Silhouette |
|----|-------------|-------------|------------|
| 10 | -2,289,435  | -2,394,142  | 0.0793     |
| 15 | -2,279,631  | -2,436,696  | 0.0745     |
| 20 | -2,253,900  | -2,463,323  | 0.0726     |
| 25 | -2,219,807  | -2,481,587  | 0.0711     |
| 30 | -2,183,781  | -2,497,919  | 0.0766     |

**K=10 selected** — lowest BIC. BIC is preferred over silhouette here because silhouette penalises the intentional fuzzy overlaps that GMM produces.

**Discovered clusters:**

| Cluster | Semantic Identity | Boundary Docs |
|---------|------------------|---------------|
| 0 | Broad science (med/space/graphics) | 23 |
| 1 | Vehicles + electronics | 8 |
| 2 | PC/Mac hardware + Windows OS | 15 |
| 3 | Guns + politics + religion | 14 |
| 4 | Sports only (cleanest) | 0 |
| 5 | Christian religion + atheism | 6 |
| 6 | Cryptography + security | 2 |
| 7 | For-sale + hardware | 17 |
| 8 | Middle East politics | 3 |
| 9 | Windows/X11 + graphics | 15 |

**Notable boundary cases** (cross-posted, confidence ~0.50):
- `comp.sys.mac.hardware` + `cmu.comp.sys.mac` → Cluster 2 (confidence: 0.508)
- `sci.astro` + `sci.space` → Cluster 1 (confidence: 0.516)
- `alt.atheism` + `talk.bizarre` → Cluster 5 (confidence: 0.494)

### Part 3 — Semantic Cache

**Data structure:** `dict[cluster_id → list[CacheEntry]]`

**Why cluster-partitioned:**
- Naive cache lookup = O(N) over all entries
- Cluster-partitioned lookup = O(N/K) — 10x speedup at K=10
- At large N, this is the difference between linear scan and bounded search

**The one tunable parameter: `similarity_threshold`**

| Threshold | Behaviour |
|-----------|-----------|
| 0.70 | Very permissive — related-but-distinct queries hit. Risk of wrong results |
| 0.80 | Balanced — paraphrases hit, different-angle queries miss |
| 0.85 | Conservative — close paraphrases hit, near-literal match required (**default**) |
| 0.90 | Very strict — almost identical phrasing only, low false positives |

The insight is not which value is best — it is what each threshold reveals about semantic resolution in the embedding space. At 0.85, "What medications help with back pain?" matches "What are the best treatments for back pain?" (sim=0.892). At 0.90, it does not.

---

## Docker

```bash
docker build -t trademarkia-semantic-search .
docker run -p 8000:8000 trademarkia-semantic-search
```

---

## Project Structure

```
AIML/
├── data/
│   ├── 20_newsgroups.csv
│   └── mini_newsgroups.csv
├── embeddings/               # generated artifacts (not committed)
│   ├── chroma_db/
│   ├── embeddings_matrix.npy
│   ├── preprocessed_docs.csv
│   ├── clustered_docs.csv
│   ├── pca_model.pkl
│   ├── gmm_model.pkl
│   └── bic_scores.csv
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── embedder.py
│   ├── clustering.py
│   ├── cache.py
│   └── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```
