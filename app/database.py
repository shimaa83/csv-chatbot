import re
import threading
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from app.config import DATABASE_DIR

_LOCK = threading.RLock()
_CONNECTIONS: dict[str, duckdb.DuckDBPyConnection] = {}

MAX_RESULT_ROWS = 200
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _quote_identifier(name: str) -> str:
    """Safely quote a DuckDB identifier."""
    return '"' + name.replace('"', '""') + '"'


def make_table_name(filename: str, used_names=None) -> str:
    """Create a predictable ASCII DuckDB table name from a CSV filename."""
    stem = Path(filename).stem.strip().lower()
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")

    if not stem:
        stem = "csv_table"
    if stem[0].isdigit():
        stem = f"table_{stem}"

    # Avoid SQL keywords and make the intended role explicit for edge cases.
    if stem.lower() in {
        "select", "from", "where", "group", "order", "by", "join",
        "table", "with", "limit", "having", "union", "values",
    }:
        stem = f"table_{stem}"

    used = {str(name).lower() for name in (used_names or set())}
    candidate = stem
    suffix = 2
    while candidate.lower() in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1

    return candidate


def _get_connection(dataset_id: str):
    """Get or create a DuckDB connection for a workspace."""
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
    """Register one uploaded CSV as a named table in a shared workspace DB."""
    dataset_id = metadata.get("dataset_id") or uuid.uuid4().hex

    if not re.fullmatch(r"[a-f0-9]{32}", dataset_id):
        raise ValueError("Invalid dataset ID.")

    table_name = metadata.get("table_name") or "dataset"
    if not _IDENTIFIER_RE.fullmatch(table_name):
        raise ValueError("Invalid table name.")

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


def get_table_names(dataset_id: str) -> list[str]:
    """Return user tables in a workspace, sorted by table name."""
    con = _get_connection(dataset_id)

    with _LOCK:
        rows = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        ).fetchall()

    return [str(row[0]) for row in rows]


def get_table_schemas(dataset_id: str) -> list[dict]:
    """Return every table and its columns for a multi-CSV workspace."""
    con = _get_connection(dataset_id)
    table_names = get_table_names(dataset_id)
    tables = []

    with _LOCK:
        for table_name in table_names:
            rows = con.execute(
                f"DESCRIBE SELECT * FROM {_quote_identifier(table_name)}"
            ).fetchall()
            tables.append({
                "table_name": table_name,
                "columns": [
                    {"name": str(row[0]), "type": str(row[1])}
                    for row in rows
                ],
            })

    return tables


def get_schema(dataset_id: str, table_name: str | None = None) -> list[dict]:
    """
    Return columns for one table.

    Backward compatibility: a single-table workspace can omit table_name.
    For multi-table workspaces, omitting table_name returns flattened columns
    with a table_name field so callers cannot confuse same-named columns.
    """
    schemas = get_table_schemas(dataset_id)

    if table_name is not None:
        for table in schemas:
            if table["table_name"].lower() == table_name.lower():
                return table["columns"]
        raise ValueError(f"Unknown table: {table_name}")

    if len(schemas) == 1:
        return schemas[0]["columns"]

    return [
        {"table_name": table["table_name"], **column}
        for table in schemas
        for column in table["columns"]
    ]


def validate_readonly_sql(
    sql: str,
    allowed_tables: set[str] | list[str] | None = None,
) -> str:
    """
    Validate a generated read-only query against known workspace tables.

    This is a restrictive first-pass validator, not a complete security
    boundary for an internet-facing application.
    """
    if not sql or not sql.strip():
        raise ValueError("SQL query is empty.")

    query = sql.strip()
    if query.endswith(";"):
        query = query[:-1].rstrip()

    if not query:
        raise ValueError("SQL query is empty.")
    if ";" in query:
        raise ValueError("Multiple SQL statements are not allowed.")
    if "--" in query or "/*" in query or "*/" in query:
        raise ValueError("SQL comments are not allowed.")
    if not re.match(r"^(SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    forbidden = re.compile(
        r"\b("
        r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
        r"ATTACH|DETACH|COPY|EXPORT|IMPORT|INSTALL|LOAD|"
        r"PRAGMA|CALL|EXECUTE|EXEC|TRUNCATE|GRANT|REVOKE|"
        r"SET|RESET|VACUUM|CHECKPOINT|SEQUENCE|MACRO|"
        r"READ_CSV|READ_CSV_AUTO|READ_PARQUET|READ_JSON|"
        r"READ_JSON_AUTO|READ_TEXT|READ_BLOB|"
        r"GLOB|HTTPFS|FORCE_CHECKPOINT"
        r")\b",
        re.IGNORECASE,
    )
    if forbidden.search(query):
        raise ValueError("Query contains a forbidden SQL operation.")

    allowed = {
        str(name).lower()
        for name in (allowed_tables if allowed_tables is not None else {"dataset"})
    }
    if not allowed:
        raise ValueError("No tables are available. Upload at least one CSV.")

    # CTE names are allowed as intermediate query sources, but physical table
    # references must still resolve to a known table in this workspace.
    cte_names = {
        match.group(1) or match.group(2)
        for match in re.finditer(
            r"(?:\bWITH\b|,)\s*(?:RECURSIVE\s+)?"
            r'(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*))\s+AS\s*\(',
            query,
            re.IGNORECASE,
        )
    }
    allowed_sources = allowed | {name.lower() for name in cte_names}

    references = []
    for match in re.finditer(
        r'\b(?:FROM|JOIN)\s+(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*))',
        query,
        re.IGNORECASE,
    ):
        references.append((match.group(1) or match.group(2)).lower())

    if not references:
        raise ValueError("Query must read from an uploaded CSV table.")

    unknown = sorted(set(references) - allowed_sources)
    if unknown:
        raise ValueError(
            "Query references unknown table(s): " + ", ".join(unknown)
        )

    if not any(reference in allowed for reference in references):
        raise ValueError("Query must read from an uploaded CSV table.")

    return query


def execute_readonly_sql(
    dataset_id: str,
    sql: str,
    max_rows: int = MAX_RESULT_ROWS,
) -> dict:
    """Execute validated SQL and return at most max_rows rows."""
    if not isinstance(max_rows, int) or isinstance(max_rows, bool):
        raise ValueError("max_rows must be an integer.")
    if not 1 <= max_rows <= 10_000:
        raise ValueError("max_rows must be between 1 and 10000.")

    allowed_tables = get_table_names(dataset_id)
    query = validate_readonly_sql(sql, allowed_tables=allowed_tables)
    con = _get_connection(dataset_id)

    safe_query = (
        f"SELECT * FROM ({query}) AS _safe_result "
        f"LIMIT {max_rows + 1}"
    )

    with _LOCK:
        result = con.execute(safe_query).fetchdf()

    truncated = len(result) > max_rows
    result = result.head(max_rows)
    rows = result.astype(object).where(pd.notna(result), None).values.tolist()

    return {
        "columns": [str(column) for column in result.columns],
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }
