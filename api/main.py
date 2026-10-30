
import asyncio
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any

import requests

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    File,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    ENABLE_RERANKER,
    MAX_UPLOAD_BYTES,
    MAX_CSV_FILES_PER_SESSION,
)

from app.rag import load_models, index_dataset
from app.chatbot import chat
from app.csv_loader import load_csv_bytes
from app.database import (
    get_table_names,
    register_dataframe,
    make_table_name,
)
from app.schema_profiler import profile_dataset


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Ollama warmup
# ============================================================

def warmup_ollama() -> None:
    """Load the Ollama model into memory."""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with OK.",
                }
            ],
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": 0,
                "num_predict": 5,
            },
        },
        timeout=300,
    )

    response.raise_for_status()

    logger.info("Ollama model loaded: %s", OLLAMA_MODEL)


# ============================================================
# Model initialization
# ============================================================

def initialize_models() -> dict[str, Any]:

    logger.info("Loading embedding and reranker models...")

    models = load_models()

    logger.info("Embedding and reranker initialization completed.")

    logger.info("Loading Ollama model...")

    warmup_ollama()

    models["ollama"] = OLLAMA_MODEL

    return models


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting CSV Chatbot API...")

    app.state.ready = False
    app.state.models = None

    try:

        models = await asyncio.to_thread(
            initialize_models
        )

        app.state.models = models
        app.state.ready = True

        logger.info("All models loaded successfully.")

    except Exception:

        logger.exception(
            "Application startup failed."
        )

        raise

    yield

    logger.info("Shutting down CSV Chatbot API.")

    app.state.ready = False
    app.state.models = None


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="CSV Chatbot API",
    description=(
        "Multi-CSV Chatbot using Text-to-SQL, Hybrid RAG, "
        "DuckDB, Qwen, Embeddings and Reranking."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# Helpers
# ============================================================

def ensure_ready(request: Request) -> None:

    if not getattr(request.app.state, "ready", False):

        raise HTTPException(
            status_code=503,
            detail="Models are not ready.",
        )


def validate_dataset_id(value: str) -> str:

    value = value.strip()

    if not re.fullmatch(r"[a-f0-9]{32}", value):

        raise ValueError(
            "dataset_id must contain exactly "
            "32 lowercase hexadecimal characters."
        )

    return value


# ============================================================
# Request schemas
# ============================================================

class PredictRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    dataset_id: str = Field(
        ...,
        min_length=32,
        max_length=32,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Question cannot be empty."
            )

        return value

    @field_validator("dataset_id")
    @classmethod
    def validate_id(cls, value: str) -> str:

        return validate_dataset_id(value)


# ============================================================
# Response schemas
# ============================================================

class UploadResponse(BaseModel):

    success: bool
    dataset_id: str
    workspace: dict[str, Any]
    schema_profile: dict[str, Any]
    indexed_documents: int
    preview: list[dict[str, Any]]


class PredictResponse(BaseModel):

    success: bool
    route: str
    answer: str

    sql: str | None = None
    result: dict[str, Any] | None = None
    context: str | None = None
    sources: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):

    status: str
    ready: bool
    models: dict[str, str]


# ============================================================
# Process CSV workspace
# Runs inside FastAPI process
# ============================================================

def process_uploaded_files(
    dataset_id: str,
    parsed_files: list[tuple[Any, dict]],
) -> dict[str, Any]:
    """
    Register CSV files in DuckDB, profile the workspace,
    and build the RAG index.
    """

    used_names = set()
    tables = []
    preview_frames = []

    import pandas as pd

    # --------------------------------------------------------
    # Register CSV tables
    # --------------------------------------------------------

    for df, metadata in parsed_files:

        table_name = make_table_name(
            metadata["original_filename"],
            used_names,
        )

        used_names.add(table_name)

        metadata["dataset_id"] = dataset_id
        metadata["table_name"] = table_name

        registered = register_dataframe(
            df,
            metadata,
        )

        tables.append({
            "table_name": registered["table_name"],
            "original_filename": registered["original_filename"],
            "row_count": registered["row_count"],
            "column_count": registered["column_count"],
            "columns": registered["columns"],
        })

        sample = df.head(10).copy()

        source_column = "__source_table__"

        while source_column in sample.columns:
            source_column = "_" + source_column

        sample.insert(
            0,
            source_column,
            registered["table_name"],
        )

        preview_frames.append(sample)

    # --------------------------------------------------------
    # Workspace metadata
    # --------------------------------------------------------

    workspace = {
        "dataset_id": dataset_id,

        "original_filename": ", ".join(
            table["original_filename"]
            for table in tables
        ),

        "row_count": sum(
            table["row_count"]
            for table in tables
        ),

        "column_count": sum(
            table["column_count"]
            for table in tables
        ),

        "tables": tables,

        "columns": [
            {
                **column,
                "table_name": table["table_name"],
            }
            for table in tables
            for column in table["columns"]
        ],
    }

    # --------------------------------------------------------
    # Schema profiling
    # --------------------------------------------------------

    schema_profile = profile_dataset(
        dataset_id,
        force_refresh=True,
    )

    # --------------------------------------------------------
    # RAG indexing
    # --------------------------------------------------------

    index_result = index_dataset(
        dataset_id,
        workspace,
        schema_profile,
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    preview_df = pd.concat(
        preview_frames,
        ignore_index=True,
        sort=False,
    ).head(30)

    preview_records = json.loads(
        preview_df.to_json(
            orient="records",
            date_format="iso",
        )
    )

    return {
        "workspace": workspace,
        "schema_profile": schema_profile,
        "indexed_documents": index_result["indexed_documents"],
        "preview": preview_records,
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
async def health(request: Request):

    ensure_ready(request)

    return HealthResponse(
        status="healthy",
        ready=True,
        models={
            "ollama": OLLAMA_MODEL,
            "embedding": EMBEDDING_MODEL,
            "reranker": (
                RERANKER_MODEL
                if ENABLE_RERANKER
                else "disabled"
            ),
        },
    )


# ============================================================
# Upload endpoint
# ============================================================

@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Upload"],
)
async def upload_workspace(
    request: Request,
    files: list[UploadFile] = File(...),
):

    ensure_ready(request)

    if not files:

        raise HTTPException(
            status_code=400,
            detail="Upload at least one CSV file.",
        )

    if len(files) > MAX_CSV_FILES_PER_SESSION:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Maximum allowed files: "
                f"{MAX_CSV_FILES_PER_SESSION}."
            ),
        )

    dataset_id = uuid.uuid4().hex

    parsed_files = []

    try:

        # ----------------------------------------------------
        # Read and validate files
        # ----------------------------------------------------

        for file in files:

            filename = file.filename or "uploaded.csv"

            if not filename.lower().endswith(".csv"):

                raise HTTPException(
                    status_code=400,
                    detail=f"{filename}: CSV files only.",
                )

            content = await file.read()

            if len(content) > MAX_UPLOAD_BYTES:

                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{filename}: File exceeds "
                        f"the upload size limit."
                    ),
                )

            df, metadata = load_csv_bytes(
                content,
                filename,
            )

            # Keep original filename in metadata.
            metadata["original_filename"] = filename

            parsed_files.append(
                (df, metadata)
            )

        # ----------------------------------------------------
        # Register + profile + index in API process
        # ----------------------------------------------------

        output = await run_in_threadpool(
            process_uploaded_files,
            dataset_id,
            parsed_files,
        )

        logger.info(
            "Workspace indexed: %s",
            dataset_id,
        )

        return UploadResponse(
            success=True,
            dataset_id=dataset_id,
            workspace=output["workspace"],
            schema_profile=output["schema_profile"],
            indexed_documents=output["indexed_documents"],
            preview=output["preview"],
        )

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            "Workspace upload/indexing failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to upload and index CSV files. "
                "Check the API logs for details."
            ),
        )

    finally:

        for file in files:
            await file.close()


# ============================================================
# Predict endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
)
async def predict(
    payload: PredictRequest,
    request: Request,
):

    ensure_ready(request)

    try:

        # Verify workspace tables.
        table_names = await run_in_threadpool(
            get_table_names,
            payload.dataset_id,
        )

        if not table_names:

            raise HTTPException(
                status_code=404,
                detail="No CSV tables found for this dataset.",
            )

        # Execute chatbot.
        output = await run_in_threadpool(
            chat,
            payload.question,
            payload.dataset_id,
        )

        if not isinstance(output, dict):

            raise ValueError(
                "Invalid chatbot response."
            )

        route = output.get(
            "route",
            "unknown",
        )

        answer = output.get(
            "answer",
            "",
        )

        if route == "error":

            raise HTTPException(
                status_code=400,
                detail=answer or "Unable to process question.",
            )

        return PredictResponse(
            success=True,
            route=route,
            answer=answer,
            sql=output.get("sql"),
            result=output.get("result"),
            context=output.get("context"),
            sources=output.get("sources", []),
        )

    except HTTPException:
        raise

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:

        logger.exception(
            "Prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred "
                "while processing the question."
            ),
        )
