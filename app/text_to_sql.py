
import json
import re

import requests

from app.config import OLLAMA_URL, OLLAMA_MODEL
from app.database import get_table_schemas
from app.schema_profiler import load_schema_profile


def _clean_sql(sql: str) -> str:
    """Remove Markdown fences and normalize generated SQL."""
    sql = sql.strip()

    sql = re.sub(
        r"^```(?:sql)?\s*|\s*```$",
        "",
        sql,
        flags=re.IGNORECASE,
    ).strip()

    return sql


def _call_ollama(prompt: str, system_prompt: str) -> str:
    """Send a prompt to Ollama and return the generated text."""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0,
            },
        },
        timeout=180,
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]

    return _clean_sql(content)


def _get_workspace_context(dataset_id: str) -> tuple[list, dict]:
    """Load actual schemas and schema profiles for all workspace tables."""

    schemas = get_table_schemas(dataset_id)
    profile = load_schema_profile(dataset_id)

    if not schemas:
        raise ValueError(
            "No CSV tables found. Please upload at least one CSV file."
        )

    return schemas, profile


def generate_sql(question: str, dataset_id: str) -> str:
    """Generate read-only DuckDB SQL using all workspace CSV tables."""

    schemas, profile = _get_workspace_context(dataset_id)

    schema_json = json.dumps(
        schemas,
        ensure_ascii=False,
        indent=2,
    )

    profile_json = json.dumps(
        profile,
        ensure_ascii=False,
        indent=2,
    )

    table_names = [
        table["table_name"]
        for table in schemas
    ]

    prompt = f"""
You are an expert DuckDB Text-to-SQL assistant.

The database contains multiple CSV files.
Each CSV file is stored as a separate physical table.

USER QUESTION:
{question}

ACTUAL DATABASE SCHEMA:
{schema_json}

BILINGUAL SCHEMA PROFILE AND SAMPLE VALUES:
{profile_json}

AVAILABLE PHYSICAL TABLES:
{json.dumps(table_names, ensure_ascii=False)}

IMPORTANT SQL RULES:

1. Return exactly one SQL query and nothing else.
2. Only SELECT or WITH ... SELECT queries are allowed.
3. Use only the available physical tables.
4. Use only columns that exist in the actual schema.
5. Never invent column names.
6. Never translate physical column names into SQL identifiers.
7. Use the original table and column names exactly.
8. When a question needs information from multiple tables,
   you MUST use explicit JOIN ... ON conditions.
9. Use table aliases to avoid ambiguous column names.
10. Qualify columns with their table aliases when using JOIN.
11. Determine relationships using matching column names,
    schema meaning, and sample values.
12. Do not assume that every table should be joined.
13. Use SUM for totals.
14. Use AVG for averages.
15. Use COUNT(*) for counting rows.
16. Use GROUP BY for comparisons between groups.
17. Use valid DuckDB SQL syntax.
18. Do not use external files or file-reading functions.
19. Do not use INSERT, UPDATE, DELETE, DROP, CREATE, or ALTER.
20. Do not add LIMIT unless the question requires it.
21. Understand Arabic and English questions.
22. Match Arabic filter values to actual data values when possible.

JOIN EXAMPLE:

If customers contains:
customer_id, customer_name, city

And orders contains:
order_id, customer_id, quantity

For a question asking for total quantity per customer,
the correct query pattern is:

SELECT
    c.customer_name,
    SUM(o.quantity) AS total_quantity
FROM customers AS c
JOIN orders AS o
    ON c.customer_id = o.customer_id
GROUP BY
    c.customer_id,
    c.customer_name
ORDER BY total_quantity DESC;

Return SQL only.
"""

    system_prompt = """
You generate valid read-only DuckDB SQL.

You must inspect the supplied schema carefully.
When a requested column belongs to another table,
use the correct JOIN.

Never invent tables or columns.
Return SQL only.
"""

    return _call_ollama(prompt, system_prompt)


def fix_sql(
    question: str,
    failed_sql: str,
    error: str,
    dataset_id: str,
) -> str:
    """
    Correct a failed SQL query using the real database schema.

    This function is called only once after an execution failure.
    """

    schemas, profile = _get_workspace_context(dataset_id)

    schema_json = json.dumps(
        schemas,
        ensure_ascii=False,
        indent=2,
    )

    profile_json = json.dumps(
        profile,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are an expert DuckDB SQL debugging assistant.

The generated SQL failed during execution.

USER QUESTION:
{question}

ACTUAL DATABASE SCHEMA:
{schema_json}

SCHEMA PROFILE AND SAMPLE VALUES:
{profile_json}

FAILED SQL:
{failed_sql}

DUCKDB ERROR:
{error}

YOUR TASK:
Correct the SQL query while preserving the user's original intention.

IMPORTANT RULES:

1. Return exactly one corrected SQL query.
2. Use only SELECT or WITH ... SELECT.
3. Use only the actual tables and columns in the schema.
4. Never invent columns or tables.
5. If a requested column belongs to another table,
   add the correct JOIN.
6. Use explicit JOIN ... ON conditions.
7. Use the actual relationship between the tables.
8. Qualify ambiguous column names with table aliases.
9. Keep the original aggregation and filtering intention.
10. Use valid DuckDB syntax.
11. Do not use external files or file-reading functions.
12. Do not use write operations.
13. Return SQL only, without Markdown or explanations.

Remember:
The error message identifies a problem in the previous SQL.
Do not repeat the same mistake.
"""

    system_prompt = """
You are a DuckDB SQL error correction assistant.

Analyze the actual schema and the execution error.
Correct the query using existing tables and columns only.
Return one read-only SQL query and nothing else.
"""

    return _call_ollama(prompt, system_prompt)
