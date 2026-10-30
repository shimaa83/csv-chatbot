
import threading
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    ENABLE_RERANKER,
    HYBRID_ALPHA,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
)

_MODEL_LOCK = threading.Lock()

_EMBEDDER = None
_RERANKER = None

# Each dataset has an independent in-memory RAG index.
_INDEXES = {}


# ============================================================
# Model loading
# ============================================================

def get_embedder():
    """Return the shared embedding model singleton."""

    global _EMBEDDER

    with _MODEL_LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)

        return _EMBEDDER


def get_reranker():
    """Return the shared reranker model singleton."""

    global _RERANKER

    if not ENABLE_RERANKER:
        return None

    with _MODEL_LOCK:
        if _RERANKER is None:
            _RERANKER = CrossEncoder(RERANKER_MODEL)

        return _RERANKER


def load_models():
    """
    Load all enabled models once during application startup.

    Returns references to the shared singleton instances.
    """

    embedder = get_embedder()

    reranker = None

    if ENABLE_RERANKER:
        reranker = get_reranker()

    return {
        "embedder": embedder,
        "reranker": reranker,
    }


# ============================================================
# Build RAG documents
# ============================================================

def build_documents(
    metadata: dict,
    schema_profile: dict | None = None,
) -> list[dict]:
    """Build RAG documents for every table and column."""

    documents = []

    profile_tables = {
        item.get("table_name"): item
        for item in (schema_profile or {}).get("tables", [])
        if isinstance(item, dict)
    }

    tables = metadata.get("tables")

    if not tables:
        # Backward compatibility with the original single-CSV format.
        tables = [{
            "table_name": metadata.get("table_name", "dataset"),
            "original_filename": metadata.get(
                "original_filename",
                "uploaded.csv",
            ),
            "row_count": metadata.get("row_count", 0),
            "column_count": metadata.get(
                "column_count",
                len(metadata.get("columns", [])),
            ),
            "columns": metadata.get("columns", []),
        }]

    for table in tables:

        table_name = table["table_name"]

        table_profile = profile_tables.get(table_name, {})

        table_text = (
            f"Table name: {table_name}. "
            f"Original CSV filename: "
            f"{table.get('original_filename', table_name + '.csv')}. "
            f"Rows: {table.get('row_count', 0)}. "
            f"Columns: "
            f"{', '.join(c['name'] for c in table.get('columns', []))}. "
            f"Arabic description: "
            f"{table_profile.get('description_ar', '')}. "
            f"English description: "
            f"{table_profile.get('description_en', '')}."
        )

        documents.append({
            "text": table_text,
            "kind": "table",
            "table": table_name,
            "column": None,
        })

        profile_columns = {
            item.get("name"): item
            for item in table_profile.get("columns", [])
            if isinstance(item, dict)
        }

        for column in table.get("columns", []):

            name = column["name"]

            dtype = column.get(
                "dtype",
                column.get("type", "unknown"),
            )

            null_count = column.get("null_count", "unknown")

            samples = column.get("sample_values", [])

            profile_column = profile_columns.get(name, {})

            aliases_ar = profile_column.get("aliases_ar", [])
            aliases_en = profile_column.get("aliases_en", [])

            text = (
                f"Table: {table_name}. "
                f"Column name: {name}. "
                f"Data type: {dtype}. "
                f"Missing values: {null_count}. "
                f"Sample values: {', '.join(map(str, samples))}. "
                f"Arabic description: "
                f"{profile_column.get('description_ar', '')}. "
                f"English description: "
                f"{profile_column.get('description_en', '')}. "
                f"Arabic aliases: {', '.join(map(str, aliases_ar))}. "
                f"English aliases: {', '.join(map(str, aliases_en))}."
            )

            documents.append({
                "text": text,
                "kind": "column",
                "table": table_name,
                "column": name,
            })

    documents.append({
        "text": (
            f"CSV workspace contains {len(tables)} tables: "
            + "; ".join(
                f"{table['table_name']} "
                f"({table.get('row_count', 0)} rows, "
                f"columns: "
                f"{', '.join(c['name'] for c in table.get('columns', []))})"
                for table in tables
            )
        ),
        "kind": "workspace",
        "table": None,
        "column": None,
    })

    return documents


# ============================================================
# Index dataset
# ============================================================

def index_dataset(
    dataset_id: str,
    metadata: dict,
    schema_profile: dict | None = None,
) -> dict:
    """Create an independent hybrid RAG index for a dataset."""

    documents = build_documents(metadata, schema_profile)

    if not documents:
        raise ValueError("No documents available for indexing.")

    texts = [doc["text"] for doc in documents]

    embedder = get_embedder()

    embeddings = embedder.encode(
        [f"passage: {text}" for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    word_matrix = word_vectorizer.fit_transform(texts)

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        lowercase=True,
    )

    char_matrix = char_vectorizer.fit_transform(texts)

    _INDEXES[dataset_id] = {
        "documents": documents,
        "embeddings": embeddings,
        "vectorizer": word_vectorizer,
        "word_matrix": word_matrix,
        "char_vectorizer": char_vectorizer,
        "char_matrix": char_matrix,
    }

    return {
        "dataset_id": dataset_id,
        "indexed_documents": len(documents),
    }


# ============================================================
# Score normalization
# ============================================================

def _normalize_scores(scores: np.ndarray) -> np.ndarray:

    if len(scores) == 0:
        return scores

    minimum = float(scores.min())
    maximum = float(scores.max())

    if maximum == minimum:
        return np.ones_like(scores, dtype=float)

    return (scores - minimum) / (maximum - minimum)


# ============================================================
# Hybrid Search + Reranking
# ============================================================

def hybrid_search(
    dataset_id: str,
    question: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """Retrieve relevant documents using hybrid search and reranking."""

    index = _INDEXES.get(dataset_id)

    if index is None:
        raise ValueError(
            "Dataset has not been indexed for RAG."
        )

    embedder = get_embedder()

    query_embedding = embedder.encode(
        [f"query: {question}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    # Semantic search
    vector_scores = index["embeddings"] @ query_embedding

    # Word-level search
    word_query = index["vectorizer"].transform([question])

    word_scores = (
        index["word_matrix"] @ word_query.T
    ).toarray().ravel()

    # Character-level search
    char_query = index["char_vectorizer"].transform([question])

    char_scores = (
        index["char_matrix"] @ char_query.T
    ).toarray().ravel()

    vector_scores = _normalize_scores(vector_scores)
    word_scores = _normalize_scores(word_scores)
    char_scores = _normalize_scores(char_scores)

    alpha = HYBRID_ALPHA

    combined = (
        alpha * vector_scores
        + (1 - alpha) * (
            0.65 * word_scores
            + 0.35 * char_scores
        )
    )

    candidate_count = min(
        max(top_k * 3, top_k),
        len(combined),
    )

    candidate_ids = np.argsort(combined)[::-1][:candidate_count]

    # Reciprocal Rank Fusion
    rankings = [
        np.argsort(vector_scores)[::-1],
        np.argsort(word_scores)[::-1],
        np.argsort(char_scores)[::-1],
    ]

    rrf_scores = defaultdict(float)

    rrf_k = 60

    for ranking in rankings:

        for rank, doc_id in enumerate(ranking, start=1):

            rrf_scores[int(doc_id)] += 1.0 / (rrf_k + rank)

    candidates = sorted(
        candidate_ids,
        key=lambda i: rrf_scores[int(i)],
        reverse=True,
    )

    docs = index["documents"]

    results = [
        {
            **docs[int(i)],
            "hybrid_score": float(combined[int(i)]),
            "rrf_score": float(rrf_scores[int(i)]),
        }
        for i in candidates
    ]

    # Cross-encoder reranking
    if ENABLE_RERANKER and results:

        reranker = get_reranker()

        pairs = [
            (question, item["text"])
            for item in results
        ]

        scores = reranker.predict(pairs)

        for item, score in zip(results, scores):
            item["rerank_score"] = float(score)

        results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return results[:RERANK_TOP_K]

    return results[:top_k]


# ============================================================
# Format context
# ============================================================

def format_context(results: list[dict]) -> str:

    return "\n\n".join(
        f"[{i}] {item['text']}"
        for i, item in enumerate(results, start=1)
    )
