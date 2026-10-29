
import re
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

# In-memory index for the first local version.
# Each dataset gets an independent index.
_INDEXES = {}


def get_embedder():
    global _EMBEDDER
    with _MODEL_LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
        return _EMBEDDER


def get_reranker():
    global _RERANKER
    with _MODEL_LOCK:
        if _RERANKER is None:
            _RERANKER = CrossEncoder(RERANKER_MODEL)
        return _RERANKER


def build_documents(metadata: dict) -> list[dict]:
    documents = []

    for column in metadata["columns"]:
        name = column["name"]
        dtype = column["dtype"]
        null_count = column["null_count"]
        samples = column["sample_values"]

        text = (
            f"Column name: {name}. "
            f"Data type: {dtype}. "
            f"Missing values: {null_count}. "
            f"Sample values: {', '.join(samples)}."
        )

        documents.append({
            "text": text,
            "kind": "column",
            "column": name,
        })

    # A schema overview helps retrieve information across columns.
    overview = (
        f"Dataset filename: {metadata['original_filename']}. "
        f"Rows: {metadata['row_count']}. "
        f"Columns: {', '.join(c['name'] for c in metadata['columns'])}."
    )

    documents.append({
        "text": overview,
        "kind": "dataset",
        "column": None,
    })

    return documents


def index_dataset(dataset_id: str, metadata: dict) -> dict:
    documents = build_documents(metadata)
    texts = [doc["text"] for doc in documents]

    embedder = get_embedder()

    # E5 models use passage: for indexed documents.
    embeddings = embedder.encode(
        [f"passage: {text}" for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Word and character features help with column names and exact terms.
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    word_matrix = vectorizer.fit_transform(texts)

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        lowercase=True,
    )
    char_matrix = char_vectorizer.fit_transform(texts)

    _INDEXES[dataset_id] = {
        "documents": documents,
        "embeddings": embeddings,
        "vectorizer": vectorizer,
        "word_matrix": word_matrix,
        "char_vectorizer": char_vectorizer,
        "char_matrix": char_matrix,
    }

    return {
        "dataset_id": dataset_id,
        "indexed_documents": len(documents),
    }


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores

    minimum = float(scores.min())
    maximum = float(scores.max())

    if maximum == minimum:
        return np.ones_like(scores, dtype=float)

    return (scores - minimum) / (maximum - minimum)


def hybrid_search(
    dataset_id: str,
    question: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    index = _INDEXES.get(dataset_id)

    if index is None:
        raise ValueError("Dataset has not been indexed for RAG.")

    embedder = get_embedder()
    query_embedding = embedder.encode(
        [f"query: {question}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    vector_scores = index["embeddings"] @ query_embedding

    word_query = index["vectorizer"].transform([question])
    word_scores = (
        index["word_matrix"] @ word_query.T
    ).toarray().ravel()

    char_query = index["char_vectorizer"].transform([question])
    char_scores = (
        index["char_matrix"] @ char_query.T
    ).toarray().ravel()

    vector_scores = _normalize_scores(vector_scores)
    word_scores = _normalize_scores(word_scores)
    char_scores = _normalize_scores(char_scores)

    alpha = HYBRID_ALPHA

    # Combine semantic, word, and character search.
    combined = (
        alpha * vector_scores
        + (1 - alpha) * (
            0.65 * word_scores + 0.35 * char_scores
        )
    )

    candidate_count = min(max(top_k * 3, top_k), len(combined))
    candidate_ids = np.argsort(combined)[::-1][:candidate_count]

    # Reciprocal Rank Fusion across the three rankings.
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

    # Only rerank candidates selected by the hybrid scores.
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


def format_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {item['text']}"
        for i, item in enumerate(results, start=1)
    )
