
import csv
import io
import re
import uuid
from pathlib import Path

import pandas as pd
from charset_normalizer import from_bytes

from app.config import MAX_ROWS


def normalize_column_name(name: str, index: int) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w]", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("_")

    if not name:
        name = f"column_{index}"

    if name[0].isdigit():
        name = f"col_{name}"

    return name


def make_unique_columns(columns):
    result = []
    used = set()

    for index, column in enumerate(columns, start=1):
        base = normalize_column_name(column, index)
        name = base
        suffix = 2

        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1

        used.add(name)
        result.append(name)

    return result


def load_csv_bytes(
    content: bytes,
    filename: str,
) -> tuple[pd.DataFrame, dict]:
    if not content:
        raise ValueError("The uploaded CSV file is empty.")

    detected = from_bytes(content).best()
    encoding = detected.encoding if detected else "utf-8"

    try:
        decoded = content.decode(encoding or "utf-8-sig")
    except (UnicodeDecodeError, LookupError):
        decoded = content.decode("utf-8-sig", errors="replace")
        encoding = "utf-8-sig"

    try:
        dialect = csv.Sniffer().sniff(
            decoded[:100_000],
            delimiters=",;\t|",
        )
        separator = dialect.delimiter
    except csv.Error:
        separator = ","

    try:
        df = pd.read_csv(
            io.StringIO(decoded),
            sep=separator,
            dtype_backend="numpy_nullable",
            on_bad_lines="error",
            nrows=MAX_ROWS + 1,
        )
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if len(df) > MAX_ROWS:
        raise ValueError(
            f"File exceeds the configured limit of {MAX_ROWS:,} rows."
        )

    if df.empty and len(df.columns) == 0:
        raise ValueError("CSV has no recognizable columns.")

    df.columns = make_unique_columns(df.columns)

    # DuckDB will receive clean, unique column names.
    metadata = {
        "dataset_id": uuid.uuid4().hex,
        "original_filename": Path(filename).name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": [
            {
                "name": str(column),
                "dtype": str(df[column].dtype),
                "null_count": int(df[column].isna().sum()),
                "sample_values": [
                    str(value)
                    for value in df[column].dropna().head(3).tolist()
                ],
            }
            for column in df.columns
        ],
        "encoding": encoding,
        "separator": separator,
    }

    return df, metadata
