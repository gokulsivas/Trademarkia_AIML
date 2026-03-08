import os
import time
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
from src.preprocess import load_and_preprocess

# ---------------------------------------------------------------------------
# DESIGN DECISIONS
#
# Model: all-MiniLM-L6-v2
#   - 384-dimensional embeddings. Fast on CPU (~14k sentences/sec),
#     strong semantic quality for retrieval tasks.
#   - Chosen over all-mpnet-base-v2 (768-dim) because: (a) this system
#     runs downstream cache lookups at query time — lower dimensionality
#     means faster cosine similarity at inference; (b) the quality gap
#     between MiniLM and mpnet is minimal for retrieval on short-to-medium
#     texts like newsgroup posts; (c) ChromaDB storage is proportionally
#     smaller, keeping the vector store lightweight.
#
# Vector Store: ChromaDB (persistent)
#   - Native Python, no server required, persists to disk automatically.
#   - Supports metadata filtering — essential for cluster-scoped cache
#     lookup in Part 3 (filter by dominant_cluster before cosine search).
#   - Chosen over FAISS because FAISS requires manual index serialization
#     and does not support metadata natively. ChromaDB handles both.
#
# Batching: batch_size=128
#   - Balances GPU/CPU memory usage against throughput.
#   - SentenceTransformer internally tokenizes and pads to max_seq_length.
#     Larger batches mean more padding waste on short documents.
#     128 is the empirically stable sweet spot for this corpus length range.
#
# Short documents:
#   - Included in the index (they are valid search targets) but flagged
#     with short_doc=True in metadata. The clustering step excludes them
#     from centroid calculations to avoid pulling cluster centers toward
#     semantically underspecified embeddings.
#
# ID uniqueness:
#   - file_id alone is not unique across newsgroups — the same article ID
#     can appear in multiple categories. IDs are prefixed with category
#     to guarantee uniqueness across the full 20-newsgroup corpus.
# ---------------------------------------------------------------------------

CHROMA_PATH = Path('embeddings/chroma_db')
COLLECTION  = 'newsgroups'
MODEL_NAME  = 'all-MiniLM-L6-v2'
BATCH_SIZE  = 128


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_or_create_collection(client: chromadb.PersistentClient):
    """
    Use cosine distance — appropriate for semantic similarity on
    normalized sentence embeddings. Inner product is equivalent when
    vectors are L2-normalized (which SentenceTransformer does by default
    with normalize_embeddings=True), but cosine is explicit about intent.
    """
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={'hnsw:space': 'cosine'}
    )


def embed_and_store(csv_path: str | Path = 'data/20_newsgroups.csv',
                    force_reembed: bool = False) -> tuple[pd.DataFrame, chromadb.Collection]:
    """
    Full pipeline: preprocess → embed → store in ChromaDB.

    Args:
        csv_path      : Path to the newsgroups CSV.
        force_reembed : If True, drops and rebuilds the collection even if
                        it already exists. Set False to resume interrupted runs.

    Returns:
        (df, collection) — preprocessed DataFrame and the ChromaDB collection.
    """
    client     = get_chroma_client()
    collection = get_or_create_collection(client)

    # If collection already has documents, skip re-embedding unless forced
    existing_count = collection.count()
    if existing_count > 0 and not force_reembed:
        print(f"[embedder] Collection already has {existing_count} documents. Loading CSV only.")
        df = load_and_preprocess(csv_path)
        return df, collection

    if force_reembed and existing_count > 0:
        print(f"[embedder] force_reembed=True. Dropping existing collection.")
        client.delete_collection(COLLECTION)
        collection = get_or_create_collection(client)

    # Preprocess
    df = load_and_preprocess(csv_path)

    # Load the embedding model
    print(f"[embedder] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = df['clean_text'].tolist()

    # ← FIXED: Combine category + file_id to guarantee uniqueness across all 20 newsgroups.
    # file_id alone is not unique — the same numeric ID can appear in multiple categories.
    ids = (df['category'] + '_' + df['file_id'].astype(str)).tolist()

    # Build metadata dicts — stored alongside each vector in ChromaDB
    # These enable filtered retrieval (e.g. by cluster, category, crosspost status)
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            'file_id'       : str(row['file_id']),        # ← FIXED: preserve original file_id
            'category'      : str(row['category']),
            'subject'       : str(row['subject'])[:500],  # ChromaDB metadata cap
            'newsgroups'    : str(row['newsgroups'])[:200],
            'is_crossposted': str(row['is_crossposted']),
            'short_doc'     : str(row['short_doc']),
            'body_len_clean': int(row['body_len_clean']),
        })

    # Embed in batches with progress reporting
    print(f"[embedder] Embedding {len(texts)} documents in batches of {BATCH_SIZE}...")
    t0 = time.time()

    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch      = texts[i : i + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.append(embeddings)

        if (i // BATCH_SIZE) % 10 == 0:
            done    = min(i + BATCH_SIZE, len(texts))
            elapsed = time.time() - t0
            print(f"  [{done}/{len(texts)}] {elapsed:.1f}s elapsed")

    all_embeddings = np.vstack(all_embeddings)
    print(f"[embedder] Embedding complete in {time.time()-t0:.1f}s. Shape: {all_embeddings.shape}")

    # Store in ChromaDB in batches (ChromaDB has a default 5461-item upsert limit)
    print(f"[embedder] Storing in ChromaDB...")
    CHROMA_BATCH = 500
    for i in range(0, len(texts), CHROMA_BATCH):
        collection.upsert(
            ids        = ids[i : i + CHROMA_BATCH],
            embeddings = all_embeddings[i : i + CHROMA_BATCH].tolist(),
            documents  = texts[i : i + CHROMA_BATCH],
            metadatas  = metadatas[i : i + CHROMA_BATCH],
        )
        if (i // CHROMA_BATCH) % 5 == 0:
            print(f"  Stored {min(i + CHROMA_BATCH, len(texts))}/{len(texts)}")

    print(f"[embedder] Done. Collection size: {collection.count()}")

    # Save embeddings as numpy array for use in clustering (Part 2)
    # ChromaDB can return them, but numpy is faster for GMM fitting
    np.save('embeddings/embeddings_matrix.npy', all_embeddings)
    df.to_csv('embeddings/preprocessed_docs.csv', index=False)
    print(f"[embedder] Saved embeddings matrix → embeddings/embeddings_matrix.npy")
    print(f"[embedder] Saved preprocessed docs  → embeddings/preprocessed_docs.csv")

    return df, collection


if __name__ == '__main__':
    df, collection = embed_and_store(
        csv_path='data/20_newsgroups.csv',
        force_reembed=False    # ← back to False
    )
    print(f"\nCollection count: {collection.count()}")
    print(f"Embedding matrix shape: {np.load('embeddings/embeddings_matrix.npy').shape}")

    # Sanity check: query the collection
    model      = SentenceTransformer(MODEL_NAME)
    test_query = "space shuttle launch NASA"
    q_emb      = model.encode([test_query], normalize_embeddings=True).tolist()
    results    = collection.query(query_embeddings=q_emb, n_results=3)
    print(f"\nSanity check — query: '{test_query}'")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"  Result {i+1}: [{meta['category']}] {doc[:120]}...")
