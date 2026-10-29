
import re
import json
import requests

from app.config import OLLAMA_URL, OLLAMA_MODEL
from app.database import get_schema
from app.schema_profiler import load_schema_profile


def generate_sql(question: str, dataset_id: str) -> str:
    """Generate SQL using the original schema, bilingual descriptions, and actual values."""

    schema = get_schema(dataset_id)
    profile = load_schema_profile(dataset_id)

    schema_json = json.dumps(
        schema,
        ensure_ascii=False,
        indent=2,
    )

    profile_json = json.dumps(
        profile,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are an expert DuckDB Text-to-SQL assistant.

User question:
{question}

Actual database schema:
{schema_json}

Bilingual schema profile with real sample values:
{profile_json}

Database table: dataset

Rules:
1. Return exactly one SQL query and nothing else.
2. Only SELECT or WITH ... SELECT queries are allowed.
3. Query only the dataset table.
4. Use only original column names from the actual schema.
5. Never use translated column names in SQL unless they are
   the actual original names.
6. Understand the user's question in Arabic or English.
7. Use the bilingual descriptions and aliases to identify columns.
8. Match filter values against actual sample values when possible.
9. Never assume that a translated value exists in the database.
10. For example, if the user asks about القاهرة and the actual
    sample value is Cairo, use Cairo as the filter value.
11. If the actual data contains القاهرة, use that original value.
12. Use SUM for totals, AVG for averages, and COUNT(*) for row counts.
13. Use GROUP BY for comparisons between groups.
14. Preserve actual column names and data values.
15. Do not use file-reading functions, external tables, or write operations.
16. Do not add a LIMIT unless needed by the user's question.
17. Use valid DuckDB SQL syntax.
18. Return SQL only, with no Markdown fences or explanation.
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Generate valid, read-only DuckDB SQL. "
                        "Return only the SQL statement."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )

    response.raise_for_status()

    sql = response.json()["message"]["content"].strip()

    sql = re.sub(
        r"^```(?:sql)?\s*|\s*```$",
        "",
        sql,
        flags=re.IGNORECASE,
    ).strip()

    return sql
