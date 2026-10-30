
import json
import requests

from app.config import OLLAMA_URL, OLLAMA_MODEL

from app.database import (
    execute_readonly_sql,
)

from app.text_to_sql import (
    generate_sql,
    fix_sql,
)

from app.rag import (
    hybrid_search,
    format_context,
)


def detect_language(text: str) -> str:
    """Detect whether the question is Arabic or English."""

    arabic_count = sum(
        "\u0600" <= char <= "\u06FF"
        for char in text
    )

    english_count = sum(
        char.isascii() and char.isalpha()
        for char in text
    )

    return "Arabic" if arabic_count > english_count else "English"


def call_llm(
    prompt: str,
    system_prompt: str = "",
    timeout: int = 180,
) -> str:
    """Generate a natural-language answer using Ollama."""

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
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()["message"]["content"].strip()


def detect_route(question: str) -> str:
    """Choose SQL for analytical questions and RAG for schema questions."""

    q = question.lower()

    rag_keywords = [
        "what columns",
        "which columns",
        "schema",
        "structure",
        "describe table",
        "explain column",
        "ما هي الأعمدة",
        "ايه الأعمدة",
        "ما هى الأعمدة",
        "تركيب الجدول",
        "هيكل الجدول",
        "وصف الجدول",
        "اشرح العمود",
    ]

    sql_keywords = [
        "sum",
        "total",
        "average",
        "avg",
        "count",
        "maximum",
        "minimum",
        "max",
        "min",
        "highest",
        "lowest",
        "compare",
        "sales",
        "revenue",
        "quantity",
        "orders",
        "customers",
        "top",
        "أعلى",
        "أقل",
        "إجمالي",
        "اجمالي",
        "مجموع",
        "متوسط",
        "عدد",
        "كم",
        "قارن",
        "مبيعات",
        "العملاء",
        "الطلبات",
        "الكمية",
        "الأكثر",
        "الأقل",
    ]

    if any(keyword in q for keyword in rag_keywords):
        return "rag"

    if any(keyword in q for keyword in sql_keywords):
        return "sql"

    return "rag"


def answer_sql_question(
    question: str,
    dataset_id: str,
) -> dict:
    """
    Generate SQL, execute it, and retry once if execution fails.

    The retry uses the actual schema and DuckDB error.
    """

    sql = generate_sql(
        question=question,
        dataset_id=dataset_id,
    )

    try:
        result = execute_readonly_sql(
            dataset_id=dataset_id,
            sql=sql,
        )

    except Exception as first_error:

        # One correction attempt only.
        corrected_sql = fix_sql(
            question=question,
            failed_sql=sql,
            error=str(first_error),
            dataset_id=dataset_id,
        )

        # execute_readonly_sql validates the corrected SQL again.
        result = execute_readonly_sql(
            dataset_id=dataset_id,
            sql=corrected_sql,
        )

        sql = corrected_sql

    result_json = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    language = detect_language(question)

    prompt = f"""
User question:
{question}

Executed SQL:
{sql}

SQL result:
{result_json}

Instructions:
1. Answer the user's question using the SQL result only.
2. Do not invent numbers or information.
3. Explain the result clearly.
4. If the result is empty, say no matching records were found.
5. Mention if the result was truncated.
6. Answer in {language}.
"""

    system_prompt = f"""
You are a helpful data analyst.

Answer questions based only on the supplied SQL result.
Never invent data.
Always respond in {language}.
"""

    answer = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
    )

    return {
        "route": "sql",
        "answer": answer,
        "sql": sql,
        "result": result,
    }


def answer_rag_question(
    question: str,
    dataset_id: str,
) -> dict:
    """Answer schema and metadata questions using hybrid RAG."""

    documents = hybrid_search(
        question=question,
        dataset_id=dataset_id,
    )

    context = format_context(documents)

    language = detect_language(question)

    prompt = f"""
User question:
{question}

Retrieved context:
{context}

Instructions:
1. Answer using the retrieved context only.
2. Do not invent table names or columns.
3. If the context does not contain the answer,
   clearly say that the information is unavailable.
4. Answer in {language}.
"""

    system_prompt = f"""
You are a CSV data assistant.

Answer using the provided context only.
Do not invent information.
Respond in {language}.
"""

    answer = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
    )

    return {
        "route": "rag",
        "answer": answer,
        "context": context,
    }


def chat(
    question: str,
    dataset_id: str,
) -> dict:
    """Main chatbot entry point."""

    if not question or not question.strip():
        return {
            "route": "error",
            "answer": "Please enter a question.",
        }

    try:
        route = detect_route(question)

        if route == "sql":
            return answer_sql_question(
                question=question,
                dataset_id=dataset_id,
            )

        return answer_rag_question(
            question=question,
            dataset_id=dataset_id,
        )

    except Exception as error:
        return {
            "route": "error",
            "answer": (
                "I couldn't process your question. "
                f"Details: {str(error)}"
            ),
        }
