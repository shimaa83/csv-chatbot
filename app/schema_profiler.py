
import json
import re
from pathlib import Path

import requests

from app.config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    INDEX_DIR,
)
from app.database import get_schema, _get_connection


def _quote_identifier(name: str) -> str:
    """Safely quote a DuckDB column name."""
    return '"' + name.replace('"', '""') + '"'


def collect_column_samples(dataset_id: str) -> list[dict]:
    """Collect schema and representative values from the uploaded CSV."""
    schema = get_schema(dataset_id)
    con = _get_connection(dataset_id)

    columns = []

    for column in schema:
        name = column["name"]
        dtype = column["type"]

        quoted_name = _quote_identifier(name)

        with con.cursor() as cursor:
            rows = cursor.execute(
                f"""
                SELECT DISTINCT {quoted_name}
                FROM dataset
                WHERE {quoted_name} IS NOT NULL
                LIMIT 10
                """
            ).fetchall()

        samples = [
            str(row[0])[:200]
            for row in rows
        ]

        columns.append({
            "name": name,
            "dtype": dtype,
            "sample_values": samples,
        })

    return columns


def _fallback_profile(columns: list[dict]) -> dict:
    """Create a usable profile if the LLM response is invalid."""
    return {
        "dataset_summary_ar": "ملف بيانات يحتوي على الأعمدة الموضحة أدناه.",
        "dataset_summary_en": "A dataset containing the columns listed below.",
        "columns": [
            {
                **column,
                "description_ar": f"العمود {column['name']}",
                "description_en": f"Column {column['name']}",
                "aliases_ar": [],
                "aliases_en": [],
            }
            for column in columns
        ],
    }


def profile_dataset(
    dataset_id: str,
    force_refresh: bool = False,
) -> dict:
    """
    Build and cache a bilingual schema profile for one CSV dataset.
    The original column names and values remain unchanged.
    """
    profile_path = INDEX_DIR / f"{dataset_id}_schema_profile.json"

    if profile_path.exists() and not force_refresh:
        try:
            cached = json.loads(
                profile_path.read_text(encoding="utf-8")
            )

            if isinstance(cached.get("columns"), list):
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    columns = collect_column_samples(dataset_id)

    if not columns:
        return _fallback_profile(columns)

    schema_json = json.dumps(
        columns,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are a bilingual data dictionary generator.

Analyze the following real CSV schema and sample values.
Create descriptions that help users ask questions in Arabic or English.

Schema and sample values:
{schema_json}

Return one valid JSON object with this exact structure:
{{
  "dataset_summary_ar": "short Arabic summary",
  "dataset_summary_en": "short English summary",
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

Rules:
- Include every original column exactly once.
- Preserve each original column name character-for-character.
- Do not create new columns.
- Infer meanings conservatively from column names and sample values.
- Do not claim an uncertain meaning is definitely correct.
- Aliases are alternative ways users might refer to the column.
- Never modify, translate, or invent actual data values.
- Do not include sample values in aliases unless they are useful names
  for the column itself.
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
                        "content": (
                            "Return valid JSON only. "
                            "Do not invent schema fields."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
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

        # Only accept descriptions for columns that really exist.
        generated_columns = {
            item["name"]: item
            for item in generated.get("columns", [])
            if isinstance(item, dict)
            and item.get("name") in {c["name"] for c in columns}
        }

        profile_columns = []

        for column in columns:
            item = generated_columns.get(column["name"], {})

            profile_columns.append({
                **column,
                "description_ar": str(
                    item.get(
                        "description_ar",
                        f"العمود {column['name']}",
                    )
                ),
                "description_en": str(
                    item.get(
                        "description_en",
                        f"Column {column['name']}",
                    )
                ),
                "aliases_ar": item.get("aliases_ar", [])
                if isinstance(item.get("aliases_ar", []), list)
                else [],
                "aliases_en": item.get("aliases_en", [])
                if isinstance(item.get("aliases_en", []), list)
                else [],
            })

        profile = {
            "dataset_summary_ar": str(
                generated.get(
                    "dataset_summary_ar",
                    "ملف بيانات.",
                )
            ),
            "dataset_summary_en": str(
                generated.get(
                    "dataset_summary_en",
                    "A dataset.",
                )
            ),
            "columns": profile_columns,
        }

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
    ):
        profile = _fallback_profile(columns)

    profile_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_path.write_text(
        json.dumps(
            profile,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return profile


def load_schema_profile(dataset_id: str) -> dict:
    """Load the cached profile or generate it when missing."""
    return profile_dataset(dataset_id)
