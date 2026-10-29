
import json
import re

import requests

from app.config import OLLAMA_URL, OLLAMA_MODEL
from app.database import execute_readonly_sql
from app.rag import hybrid_search, format_context
from app.text_to_sql import generate_sql


def detect_language(question: str) -> str:
    """Detect whether the question is primarily Arabic or English."""
    arabic_chars = len(
        re.findall(r"[\u0600-\u06FF]", question)
    )
    latin_chars = len(
        re.findall(r"[A-Za-z]", question)
    )

    if arabic_chars > latin_chars:
        return "Arabic"

    return "English"


def call_llm(prompt: str, language: str) -> str:
    """Call Qwen through Ollama and instruct it to answer in the requested language."""
    language_instruction = (
        "Answer entirely in Arabic. Use clear, natural Arabic."
        if language == "Arabic"
        else "Answer entirely in English."
    )

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful CSV data assistant. "
                        f"{language_instruction} "
                        "Use only the information provided. "
                        "Never invent data or statistics."
                    ),
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

    return response.json()["message"]["content"].strip()


def detect_route(question: str) -> str:
    """
    Route analytical questions to SQL and dataset metadata questions to RAG.
    Supports Arabic and English.
    """
    q = question.lower().strip()

    # Questions about schema and dataset descriptions should use RAG.
    metadata_patterns = [
        r"\b(columns?|column names|field names|schema|"
        r"data types?|dataset information|file information|"
        r"describe the dataset|what does this dataset contain)\b",
        r"الأعمدة|اسماء الأعمدة|أسماء الأعمدة|"
        r"أسماء الحقول|اسماء الحقول|"
        r"معلومات الملف|معلومات البيانات|"
        r"وصف الملف|وصف البيانات|"
        r"نوع البيانات|أنواع البيانات|"
        r"العمود الذي|العمود اللى|"
        r"أي عمود|ايه الأعمدة|ايه الاعمدة",
    ]

    if any(
        re.search(pattern, q, re.IGNORECASE)
        for pattern in metadata_patterns
    ):
        return "rag"

    # Analytical and numeric questions should use SQL.
    sql_patterns = [
        r"\b(sum|total|average|avg|mean|median|count|"
        r"how many|number of|"
        r"highest|lowest|maximum|minimum|max|min|"
        r"compare|comparison|group by|greater than|less than|"
        r"most|least|top|bottom|sort|rank|"
        r"calculate|percentage|percent|"
        r"sales|revenue|profit|price|quantity|"
        r"orders|customers|products|"
        r"greater|fewer|more than|less than)\b",
        r"إجمالي|إجمالى|مجموع|متوسط|المتوسط|"
        r"عدد|كام|كم|"
        r"أعلى|اعلى|أكبر|اكبر|"
        r"أقل|اقل|أصغر|اصغر|"
        r"أقصى|اقصى|أدنى|ادنى|"
        r"احسب|حساب|قارن|مقارنة|"
        r"الأكثر|الاكثر|الأقل|الاقل|"
        r"ترتيب|رتب|نسبة|نسبه|"
        r"مبيعات|المبيعات|إيرادات|ايرادات|"
        r"أرباح|ارباح|ربح|سعر|أسعار|اسعار|"
        r"كمية|كميات|طلبات|الطلبات|"
        r"عملاء|العملاء|منتجات|المنتجات|"
        r"تجاوز|أكبر من|اكبر من|أقل من|اقل من",
    ]

    if any(
        re.search(pattern, q, re.IGNORECASE)
        for pattern in sql_patterns
    ):
        return "sql"

    # Default: questions about meaning and dataset metadata use RAG.
    return "rag"


def answer_sql_question(
    question: str,
    dataset_id: str,
    language: str,
) -> dict:
    """Generate SQL, execute it, and explain the result."""
    sql = generate_sql(question, dataset_id)

    result = execute_readonly_sql(
        dataset_id=dataset_id,
        sql=sql,
    )

    result_json = json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    )

    if language == "Arabic":
        prompt = f"""
أجب عن سؤال المستخدم باللغة العربية فقط.

سؤال المستخدم:
{question}

استعلام SQL الذي تم تنفيذه:
{sql}

نتيجة SQL:
{result_json}

التعليمات:
- استخدم القيم الموجودة في النتيجة فقط.
- لا تخترع أرقامًا أو إحصائيات.
- إذا كانت النتيجة فارغة، وضح أنه لا توجد سجلات مطابقة.
- إذا كانت النتائج مختصرة، وضح ذلك للمستخدم.
- اشرح النتيجة ببساطة وباختصار.
- لا تترجم أسماء الأعمدة أو القيم الأصلية الموجودة في البيانات.
"""
    else:
        prompt = f"""
Answer the user's question in English only.

User question:
{question}

Executed SQL:
{sql}

SQL result:
{result_json}

Rules:
- Use only values present in the result.
- Never invent numbers or statistics.
- If the result is empty, say no matching records were found.
- Mention if the results were truncated.
- Explain the result clearly and concisely.
- Do not translate original column names or data values.
"""

    answer = call_llm(
        prompt=prompt,
        language=language,
    )

    return {
        "question": question,
        "route": "sql",
        "language": language,
        "answer": answer,
        "sql": sql,
        "result": result,
        "sources": [],
    }


def answer_rag_question(
    question: str,
    dataset_id: str,
    language: str,
) -> dict:
    """Retrieve relevant dataset metadata and generate a grounded answer."""
    documents = hybrid_search(
        dataset_id=dataset_id,
        question=question,
    )

    if not documents:
        message = (
            "لم أجد معلومات كافية للإجابة عن السؤال."
            if language == "Arabic"
            else "I could not find enough information to answer the question."
        )

        return {
            "question": question,
            "route": "rag",
            "language": language,
            "answer": message,
            "sql": None,
            "result": None,
            "sources": [],
        }

    context = format_context(documents)

    if language == "Arabic":
        prompt = f"""
أجب عن سؤال المستخدم باللغة العربية فقط، بالاعتماد على المعلومات المسترجعة.

سؤال المستخدم:
{question}

معلومات الملف المسترجعة:
{context}

التعليمات:
- استخدم المعلومات الموجودة في السياق فقط.
- لا تخترع أسماء أعمدة أو قيمًا أو إحصائيات.
- السياق يصف الملف وأعمدته، لكنه قد لا يحتوي على جميع صفوف CSV.
- إذا لم تكن المعلومات كافية، وضح ذلك بوضوح.
- احتفظ بأسماء الأعمدة الأصلية كما هي.
"""
    else:
        prompt = f"""
Answer the user's question in English only, using the retrieved information.

User question:
{question}

Retrieved dataset information:
{context}

Rules:
- Use only information supported by the context.
- Do not invent column names, values, or statistics.
- The context describes the dataset and its columns;
  it may not contain every row of the CSV.
- If the information is insufficient, say so clearly.
- Preserve original column names exactly as they appear.
"""

    answer = call_llm(
        prompt=prompt,
        language=language,
    )

    sources = [
        {
            "text": item["text"],
            "kind": item["kind"],
            "column": item.get("column"),
            "score": item.get(
                "rerank_score",
                item.get("rrf_score"),
            ),
        }
        for item in documents
    ]

    return {
        "question": question,
        "route": "rag",
        "language": language,
        "answer": answer,
        "sql": None,
        "result": None,
        "sources": sources,
    }


def chat(question: str, dataset_id: str) -> dict:
    """Main entry point for the bilingual CSV chatbot."""
    if not question or not question.strip():
        raise ValueError(
            "Please enter a question."
        )

    question = question.strip()
    language = detect_language(question)
    route = detect_route(question)

    if route == "sql":
        return answer_sql_question(
            question=question,
            dataset_id=dataset_id,
            language=language,
        )

    return answer_rag_question(
        question=question,
        dataset_id=dataset_id,
        language=language,
    )
