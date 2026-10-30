import json
import re

import requests

from app.config import OLLAMA_URL, OLLAMA_MODEL, INDEX_DIR
from app.database import get_table_schemas, _get_connection


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def collect_column_samples(dataset_id: str) -> list[dict]:
    """Collect schema and representative values from every uploaded CSV table."""
    schemas = get_table_schemas(dataset_id)
    con = _get_connection(dataset_id)
    tables = []

    for table in schemas:
        table_name = table["table_name"]
        columns = []

        for column in table["columns"]:
            name = column["name"]
            dtype = column["type"]
            quoted_column = _quote_identifier(name)
            quoted_table = _quote_identifier(table_name)

            with con.cursor() as cursor:
                rows = cursor.execute(
                    f"""
                    SELECT DISTINCT {quoted_column}
                    FROM {quoted_table}
                    WHERE {quoted_column} IS NOT NULL
                    LIMIT 10
                    """
                ).fetchall()

            columns.append({
                "name": name,
                "dtype": dtype,
                "sample_values": [str(row[0])[:200] for row in rows],
            })

        with con.cursor() as cursor:
            row_count = cursor.execute(
                f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}"
            ).fetchone()[0]

        tables.append({
            "table_name": table_name,
            "row_count": int(row_count),
            "columns": columns,
        })

    return tables


def _fallback_profile(tables: list[dict]) -> dict:
    """Create a usable bilingual profile if the LLM response is invalid."""
    profiled_tables = []
    flat_columns = []

    for table in tables:
        columns = []
        for column in table["columns"]:
            item = {
                **column,
                "table_name": table["table_name"],
                "description_ar": f"العمود {column['name']} في جدول {table['table_name']}",
                "description_en": f"Column {column['name']} in table {table['table_name']}",
                "aliases_ar": [],
                "aliases_en": [],
            }
            columns.append(item)
            flat_columns.append(item)

        profiled_tables.append({
            **table,
            "description_ar": f"جدول يحتوي على بيانات {table['table_name']}.",
            "description_en": f"A table containing {table['table_name']} data.",
            "columns": columns,
        })

    return {
        "dataset_summary_ar": "مساحة عمل تحتوي على جداول مستخرجة من ملفات CSV.",
        "dataset_summary_en": "A workspace containing tables loaded from CSV files.",
        "tables": profiled_tables,
        "columns": flat_columns,
    }


def profile_dataset(
    dataset_id: str,
    force_refresh: bool = False,
) -> dict:
    """Build and cache a bilingual profile for every table in a workspace."""
    profile_path = INDEX_DIR / f"{dataset_id}_schema_profile.json"

    if profile_path.exists() and not force_refresh:
        try:
            cached = json.loads(profile_path.read_text(encoding="utf-8"))
            if isinstance(cached.get("tables"), list):
                return cached
            # Old single-CSV profiles need rebuilding for the new structure.
        except (OSError, json.JSONDecodeError):
            pass

    tables = collect_column_samples(dataset_id)
    if not tables:
        return _fallback_profile([])

    schema_json = json.dumps(tables, ensure_ascii=False, indent=2)

    prompt = f"""
You are a bilingual data dictionary generator for a workspace containing
multiple related or unrelated CSV tables.

Analyze the actual table schemas and sample values below. Help users ask
questions in Arabic or English, including questions that may need joins.

Workspace tables and sample values:
{schema_json}

Return one valid JSON object with this structure:
{{
  "dataset_summary_ar": "short Arabic workspace summary",
  "dataset_summary_en": "short English workspace summary",
  "tables": [
    {{
      "table_name": "EXACT TABLE NAME",
      "description_ar": "Arabic table description",
      "description_en": "English table description",
      "columns": [
        {{
          "name": "EXACT ORIGINAL COLUMN NAME",
          "description_ar": "Arabic description",
          "description_en": "English description",
          "aliases_ar": ["Arabic search terms"],
          "aliases_en": ["English search terms"]
        }}
      ]
    }}
  ]
}}

Rules:
- Include every table and every column exactly once.
- Preserve all original table and column names character-for-character.
- Do not create tables or columns.
- Infer meanings conservatively from names and sample values.
- You may describe possible relationships only when supported by column names
  and sample values; do not claim a join key is certain without evidence.
- Never translate, modify, or invent actual data values.
- Return JSON only, without Markdown fences.
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return valid JSON only. Do not invent schema fields.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"].strip()
        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()
        generated = json.loads(raw)

        actual_tables = {table["table_name"]: table for table in tables}
        generated_tables = {
            item["table_name"]: item
            for item in generated.get("tables", [])
            if isinstance(item, dict)
            and item.get("table_name") in actual_tables
        }

        profiled_tables = []
        flat_columns = []

        for table in tables:
            generated_table = generated_tables.get(table["table_name"], {})
            generated_columns = {
                item["name"]: item
                for item in generated_table.get("columns", [])
                if isinstance(item, dict)
                and item.get("name") in {c["name"] for c in table["columns"]}
            }

            profile_columns = []
            for column in table["columns"]:
                item = generated_columns.get(column["name"], {})
                profiled_column = {
                    **column,
                    "table_name": table["table_name"],
                    "description_ar": str(item.get(
                        "description_ar",
                        f"العمود {column['name']} في جدول {table['table_name']}",
                    )),
                    "description_en": str(item.get(
                        "description_en",
                        f"Column {column['name']} in table {table['table_name']}",
                    )),
                    "aliases_ar": item.get("aliases_ar", [])
                    if isinstance(item.get("aliases_ar", []), list) else [],
                    "aliases_en": item.get("aliases_en", [])
                    if isinstance(item.get("aliases_en", []), list) else [],
                }
                profile_columns.append(profiled_column)
                flat_columns.append(profiled_column)

            profiled_tables.append({
                **table,
                "description_ar": str(generated_table.get(
                    "description_ar", f"جدول {table['table_name']}.",
                )),
                "description_en": str(generated_table.get(
                    "description_en", f"Table {table['table_name']}.",
                )),
                "columns": profile_columns,
            })

        profile = {
            "dataset_summary_ar": str(generated.get(
                "dataset_summary_ar", "مساحة عمل تحتوي على جداول CSV.",
            )),
            "dataset_summary_en": str(generated.get(
                "dataset_summary_en", "A workspace containing CSV tables.",
            )),
            "tables": profiled_tables,
            "columns": flat_columns,
        }

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
    ):
        profile = _fallback_profile(tables)

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return profile


def load_schema_profile(dataset_id: str) -> dict:
    """Load the cached profile or generate it when missing."""
    return profile_dataset(dataset_id)
