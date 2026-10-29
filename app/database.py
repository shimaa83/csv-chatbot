
import re
import threading
import uuid

import duckdb
import pandas as pd

from app.config import DATABASE_DIR

_LOCK = threading.RLock()
_CONNECTIONS: dict[str, duckdb.DuckDBPyConnection] = {}

MAX_RESULT_ROWS = 200


def _quote_identifier(name: str) -> str:
    """Safely quote a DuckDB identifier."""
    return '"' + name.replace('"', '""') + '"'


def _get_connection(dataset_id: str):
    """Get or create a DuckDB connection for a dataset."""
    if not re.fullmatch(r"[a-f0-9]{32}", dataset_id):
        raise ValueError("Invalid dataset ID.")

    with _LOCK:
        if dataset_id not in _CONNECTIONS:
            db_path = DATABASE_DIR / f"{dataset_id}.duckdb"
            _CONNECTIONS[dataset_id] = duckdb.connect(str(db_path))

        return _CONNECTIONS[dataset_id]


def register_dataframe(
    df: pd.DataFrame,
    metadata: dict,
) -> dict:
    """Register an uploaded DataFrame as the dataset table."""
    dataset_id = metadata.get("dataset_id") or uuid.uuid4().hex

    if not re.fullmatch(r"[a-f0-9]{32}", dataset_id):
        raise ValueError("Invalid dataset ID.")

    table_name = "dataset"

    with _LOCK:
        con = _get_connection(dataset_id)
        con.register("_uploaded_df", df)

        try:
            con.execute(
                f"CREATE OR REPLACE TABLE {_quote_identifier(table_name)} "
                "AS SELECT * FROM _uploaded_df"
            )
        finally:
            con.unregister("_uploaded_df")

    return {
        **metadata,
        "dataset_id": dataset_id,
        "table_name": table_name,
    }


def get_schema(dataset_id: str) -> list[dict]:
    """Return the dataset column names and DuckDB data types."""
    con = _get_connection(dataset_id)

    with _LOCK:
        rows = con.execute(
            "DESCRIBE SELECT * FROM dataset"
        ).fetchall()

    return [
        {"name": row[0], "type": row[1]}
        for row in rows
    ]


def validate_readonly_sql(sql: str) -> str:
    """
    Basic validation for LLM-generated DuckDB SQL.

    This is a restrictive first-pass validator, not a complete
    security boundary for an internet-facing application.
    """
    if not sql or not sql.strip():
        raise ValueError("SQL query is empty.")

    query = sql.strip()

    # Allow one optional trailing semicolon.
    if query.endswith(";"):
        query = query[:-1].rstrip()

    if not query:
        raise ValueError("SQL query is empty.")

    # Reject multiple statements and SQL comments.
    if ";" in query:
        raise ValueError("Multiple SQL statements are not allowed.")

    if "--" in query or "/*" in query or "*/" in query:
        raise ValueError("SQL comments are not allowed.")

    if not re.match(r"^(SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    # Reject common write, administrative, and file-access operations.
    forbidden = re.compile(
        r"\b("
        r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
        r"ATTACH|DETACH|COPY|EXPORT|IMPORT|INSTALL|LOAD|"
        r"PRAGMA|CALL|EXECUTE|EXEC|TRUNCATE|GRANT|REVOKE|"
        r"SET|RESET|VACUUM|CHECKPOINT|SEQUENCE|MACRO|"
        r"READ_CSV|READ_CSV_AUTO|READ_PARQUET|READ_JSON|"
        r"READ_JSON_AUTO|READ_TEXT|READ_BLOB|"
        r"QUERY|QUERY_TABLE|GLOB|"
        r"HTTPFS|FORCE_CHECKPOINT"
        r")\b",
        re.IGNORECASE,
    )

    if forbidden.search(query):
        raise ValueError("Query contains a forbidden SQL operation.")

    # Reject table functions and table references other than dataset.
    # This intentionally supports a simple, single-table first version.
    references = re.findall(
        r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        query,
        re.IGNORECASE,
    )

    if not references:
        raise ValueError("Query must read from the dataset table.")

    if any(table.lower() != "dataset" for table in references):
        raise ValueError("Only the dataset table can be queried.")

    return query


def execute_readonly_sql(
    dataset_id: str,
    sql: str,
    max_rows: int = MAX_RESULT_ROWS,
) -> dict:
    """
    Execute a validated query and return at most max_rows rows.

    Wrapping the query avoids adding a second LIMIT when the generated
    SQL already contains LIMIT, ORDER BY, or GROUP BY.
    """
    if not isinstance(max_rows, int) or isinstance(max_rows, bool):
        raise ValueError("max_rows must be an integer.")

    if not 1 <= max_rows <= 10_000:
        raise ValueError("max_rows must be between 1 and 10000.")

    query = validate_readonly_sql(sql)
    con = _get_connection(dataset_id)

    # The outer LIMIT caps returned rows without modifying the inner query.
    safe_query = (
        f"SELECT * FROM ({query}) AS _safe_result "
        f"LIMIT {max_rows + 1}"
    )

    with _LOCK:
        result = con.execute(safe_query).fetchdf()

    truncated = len(result) > max_rows
    result = result.head(max_rows)

    # Convert pandas/NumPy values into ordinary Python values and nulls.
    rows = result.astype(object).where(pd.notna(result), None).values.tolist()

    return {
        "columns": [str(column) for column in result.columns],
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }
