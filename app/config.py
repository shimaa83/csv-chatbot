
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = PROJECT_ROOT / os.getenv("UPLOAD_DIR", "data/uploads")
INDEX_DIR = PROJECT_ROOT / os.getenv("INDEX_DIR", "data/indexes")
DATABASE_DIR = DATA_DIR / "databases"

for directory in (UPLOAD_DIR, INDEX_DIR, DATABASE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL", "http://localhost:11434"
).rstrip("/")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "Horizon-Labs/multilingual-reranker-small",
)

ENABLE_RERANKER = (
    os.getenv("ENABLE_RERANKER", "true").lower() == "true"
)

HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "500000"))

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_CSV_FILES_PER_SESSION = 5
