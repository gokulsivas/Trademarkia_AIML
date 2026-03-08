# Trademarkia AI/ML Engineer Task — Semantic Search System

A semantic search system built over the 20 Newsgroups corpus (~20,000 documents). It finds documents by meaning rather than keywords, groups them into topics using fuzzy clustering, and remembers past queries so it does not have to repeat expensive searches.

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
git clone https://github.com/gokulsivas/Trademarkia_AIML.git
cd Trademarkia_AIML
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Build the pipeline (one-time)

```bash
# Step 1: Embed corpus and store in ChromaDB (~70 seconds with GPU)
python -m src.embedder

# Step 2: Fit GMM clustering and update ChromaDB with cluster assignments
python -m src.clustering
```

### Start the API

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Visit [**http://localhost:8000/docs**](http://localhost:8000/docs) for the interactive Swagger UI.

---

## API Endpoints

### `POST /query`

Send any natural language query. The system checks the semantic cache first. If a similar query was seen before, it returns the cached result instantly. If not, it searches ChromaDB, stores the result, and returns it.

```json
{ "query": "How does the space shuttle launch work?" }
```

First time asking (cache miss):
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

Asking something semantically similar afterwards (cache hit):
```json
{
  "query": "Tell me about NASA rocket launches",
  "cache_hit": true,
  "matched_query": "How does the space shuttle launch work?",
  "similarity_score": 0.91,
  "result": "Top 5 results...",
  "dominant_cluster": 0
}
```

### `GET /cache/stats`

Returns how well the cache is performing at any point in time.

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`

Wipes all cached entries and resets the hit/miss counters to zero.

---

## How It Works

### Preprocessing (`preprocess.py`)

The raw data is a CSV of newsgroup posts from the 1990s. Each post has a category, a subject line, and a body. The problem is that the body is messy. People quote previous messages like this:

```
> John wrote:
> Yeah I agree with that
```

Those quoted lines are someone else's words being repeated. If we leave them in, the embedding model gets confused about what the post is actually about. So we strip them out along with email signatures, UUEncoded binary blocks (garbled file attachments), and extra whitespace.

One important step: we prepend the subject line to the cleaned body before embedding. The subject is usually the most descriptive sentence in the entire post, so putting it first gives the embedding model the best possible signal about the topic.

A few other things worth noting:

- No lowercasing, stemming, or stopword removal. The embedding model we use (`all-MiniLM-L6-v2`) is trained on natural cased text, so stripping morphology actually hurts quality rather than helping.
- Documents under 50 characters after cleaning are flagged as `short_doc=True`. They are kept in the search index but excluded from cluster fitting since they do not carry enough content to reliably represent a topic.

### Embedding (`embedder.py`)

Computers cannot understand text directly. We convert each document into 384 numbers called an embedding. These numbers capture the meaning of the text. Two documents about similar things will have similar numbers, even if they use completely different words.

We use `all-MiniLM-L6-v2` for this. It was chosen over larger models like `all-mpnet-base-v2` (768 dimensions) because the quality difference for retrieval tasks is minimal, and smaller embeddings mean faster cache lookups and less storage in ChromaDB.

All 20,000 documents are processed in batches of 128 using the GPU, which completes in about 70 seconds. The embeddings are stored in ChromaDB, a local vector database that supports both semantic search and metadata filtering.

One bug that was caught and fixed: the original `file_id` numbers are not unique across all 20 newsgroups. Article number 51127 exists in both `alt.atheism` and `sci.space`. Using raw file IDs caused 3,667 documents to silently overwrite each other in ChromaDB. The fix was to use `category_fileid` as the unique ID (for example, `sci.space_51127`).

### Fuzzy Clustering (`clustering.py`)

Normal clustering gives each document a single label. That is too rigid for a corpus like this. A post about gun laws and religious freedom genuinely belongs to both the guns cluster and the religion cluster. Forcing it into one loses information.

Gaussian Mixture Model (GMM) solves this by giving each document a probability score for every cluster instead of a single label:

```
"Re: Gun laws and religious freedom"
  Cluster 3 (Guns + Politics):  52%
  Cluster 5 (Religion):         44%
  Everything else:               4%
```

The cluster with the highest probability is the dominant cluster, but the full distribution is preserved. Documents with two high probabilities close together are called boundary documents and are the most semantically interesting ones in the corpus.

Before fitting GMM, the 384-dimensional embeddings are reduced to 50 dimensions using PCA. This is necessary because GMM needs to estimate the shape of each cluster, and doing that accurately in 384 dimensions requires far more data than we have. 50 dimensions retains about 50% of the variance while making the math stable.

**Choosing K:** We tried K = 10, 15, 20, 25, and 30, measuring BIC (Bayesian Information Criterion) for each. BIC rewards fit but penalizes complexity. K=10 gave the lowest BIC, so that is what we use.

| K  | BIC           | AIC           | Silhouette |
|----|---------------|---------------|------------|
| **10** | **-2,289,435** | -2,394,142 | 0.0793 |
| 15 | -2,279,631    | -2,436,696    | 0.0745     |
| 20 | -2,253,900    | -2,463,323    | 0.0726     |
| 25 | -2,219,807    | -2,481,587    | 0.0711     |
| 30 | -2,183,781    | -2,497,919    | 0.0766     |

![BIC Curve](assets/bic_curve.png)

**The 10 clusters found:**

| Cluster | Topic | Docs | Avg Confidence | Boundary Docs |
|---------|-------|------|----------------|---------------|
| 0 | Broad Science (medicine, space, graphics) | 2766 | 0.977 | 23 |
| 1 | Vehicles and Electronics | 2117 | 0.986 | 8 |
| 2 | PC and Mac Hardware | 2131 | 0.978 | 15 |
| 3 | Guns, Politics, Religion | 2378 | 0.985 | 14 |
| 4 | Sports (purest cluster) | 1911 | 0.999 | 0 |
| 5 | Christianity and Atheism | 2273 | 0.992 | 6 |
| 6 | Cryptography and Security | 1033 | 0.992 | 2 |
| 7 | For-Sale Posts and Hardware | 1954 | 0.971 | 17 |
| 8 | Middle East Politics | 1118 | 0.995 | 3 |
| 9 | Windows, X11, and Graphics | 2316 | 0.984 | 15 |

Cluster 4 (Sports) is the cleanest with zero boundary documents and 99.9% average confidence. Cluster 3 merging guns, politics, and religion makes intuitive sense since these topics genuinely overlap in the corpus.

Some interesting boundary cases from real cross-posted articles:
- `comp.sys.mac.hardware` + `cmu.comp.sys.mac` → Cluster 2 with only 50.8% confidence
- `sci.astro` + `sci.space` → Cluster 1 with only 51.6% confidence
- `alt.atheism` + `talk.bizarre` → Cluster 5 with only 49.4% confidence

### Cluster Visualizations

**UMAP projection of 5,000 documents, colored by cluster:**

![UMAP Clusters](assets/umap_clusters.png)

**Category to Cluster affinity heatmap:**

Each row is a newsgroup category. Each column is a cluster. Darker means a higher fraction of that category's documents landed in that cluster. The strong dark squares along roughly diagonal positions confirm the clusters align with the original topic categories.

![Category Heatmap](assets/category_heatmap.png)

**Confidence distribution, normal vs boundary documents:**

Normal documents pile up at confidence close to 1.0. Boundary documents spread between 0.3 and 0.6, which is exactly what we expect for documents that sit at the intersection of two topics.

![Confidence Distribution](assets/confidence_dist.png)

**Cross-posted documents vs single-group documents:**

Cross-posted articles show measurably lower cluster confidence than single-group articles. The model correctly identifies that these posts belong to more than one topic.

![Crosspost Confidence](assets/crosspost_confidence.png)

### Semantic Cache (`cache.py`)

The cache is built entirely from scratch using Python dicts, dataclasses, and NumPy. No Redis, no external caching library of any kind.

**The core idea:** If someone asks "what are treatments for back pain?" and later someone asks "what medications help with back pain?", these are the same question. We should return the stored answer rather than querying ChromaDB twice.

**How a lookup works:**
1. Embed the incoming query (384 numbers)
2. Find which cluster this query belongs to using the GMM model
3. Search only through cached queries in that same cluster
4. Compute cosine similarity between the new query and each cached entry
5. If the best match scores 0.85 or higher, return the cached result (hit)
6. If nothing is close enough, query ChromaDB and store the result (miss)

**Why cluster-partitioned?** Without clustering, step 3 would scan every single cached entry, which is O(N). With clustering, we only scan one cluster's worth of entries, which is O(N/K). At K=10 that is a 10x reduction in search space. The clustering work feeds directly into cache performance.

**The similarity threshold** is the one tunable parameter that matters:

| Threshold | What happens |
|-----------|-------------|
| 0.70 | Very permissive. Related but distinct queries might hit. Risk of returning wrong results |
| 0.80 | Balanced. Most paraphrases hit. Different-angle queries on the same topic may miss |
| 0.85 | Conservative. Close paraphrases hit reliably. Default choice |
| 0.90 | Very strict. Only near-identical phrasing hits. Low false positives but low hit rate |

![Threshold Sensitivity](assets/threshold_sensitivity.png)

> **Note on the threshold table:** The Space and Guns paraphrase misses are a consequence of cluster-partitioned lookup. The paraphrase query lands in a neighbouring cluster rather than the seed query's cluster, so it never finds the cached entry. This is the intentional speed vs recall tradeoff of cluster-scoped caching, documented in `src/cache.py`.

### FastAPI Service (`main.py`)

The API loads all resources once at startup: the embedding model, ChromaDB collection, GMM and PCA models, and the semantic cache. Loading once means every request is fast. Reloading a 90MB model on every request would make the API unusable.

On shutdown, the cache state is automatically saved to disk so it survives server restarts.

---

## Project Structure

```
AIML/
├── assets/                       <- cluster analysis visualizations
│   ├── bic_curve.png
│   ├── umap_clusters.png
│   ├── category_heatmap.png
│   ├── confidence_dist.png
│   ├── crosspost_confidence.png
│   └── threshold_sensitivity.png
├── data/
│   ├── 20_newsgroups.csv
│   └── mini_newsgroups.csv
├── embeddings/                   <- generated artifacts (not in git)
│   ├── chroma_db/
│   ├── embeddings_matrix.npy
│   ├── preprocessed_docs.csv
│   ├── clustered_docs.csv
│   ├── pca_model.pkl
│   ├── gmm_model.pkl
│   └── bic_scores.csv
├── notebooks/
│   └── analysis.ipynb            <- full cluster analysis notebook
├── src/
│   ├── __init__.py
│   ├── preprocess.py             <- Part 1: cleaning pipeline
│   ├── embedder.py               <- Part 1: embed and store in ChromaDB
│   ├── clustering.py             <- Part 2: PCA and GMM fuzzy clustering
│   ├── cache.py                  <- Part 3: semantic cache from scratch
│   └── main.py                   <- Part 4: FastAPI service
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Docker

```bash
docker build -t trademarkia-semantic-search .
docker run -p 8000:8000 trademarkia-semantic-search
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding model | `all-MiniLM-L6-v2` | Fast, 384-dim, strong retrieval quality |
| Vector store | ChromaDB | Local, persistent, metadata filtering built in |
| Clustering | GMM (scikit-learn) | Soft probability assignments per document |
| Dimensionality reduction | PCA 50 dims | Stabilizes GMM covariance estimation |
| Visualization | UMAP + matplotlib | Preserves local structure for 2D projection |
| API | FastAPI + uvicorn | Async, auto Swagger docs, production ready |
| Cache | Custom Python | Built from first principles, no external libraries |
