
import inspect
import json
import os
from pathlib import Path
from contextlib import ExitStack

import pandas as pd
import requests
import gradio as gr

from app.chatbot import detect_language


# ============================================================
# FastAPI configuration
# ============================================================

API_BASE_URL = os.getenv(
    "CSV_CHATBOT_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

API_TIMEOUT = 600


# ============================================================
# API helpers
# ============================================================

def get_api_error(response):
    try:
        data = response.json()
        return data.get("detail", response.text)
    except ValueError:
        return response.text or "Unknown API error."


def check_api_health():
    """Check whether FastAPI is ready."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("ready"):
            return False, "API is not ready."

        return True, data

    except requests.RequestException as exc:
        return False, str(exc)


def call_api(endpoint, payload=None, timeout=API_TIMEOUT):
    """Send JSON request to FastAPI."""

    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload or {},
            timeout=timeout,
        )

    except requests.Timeout as exc:
        raise RuntimeError(
            "The API request timed out. Please try again."
        ) from exc

    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to FastAPI at {API_BASE_URL}. "
            "Make sure the API is running."
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError(
            f"API request failed: {exc}"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"API Error ({response.status_code}): "
            f"{get_api_error(response)}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            "FastAPI returned an invalid JSON response."
        ) from exc


# ============================================================
# Upload CSV files through FastAPI
# ============================================================

def upload_csv(file_paths):
    """
    Send CSV files to FastAPI.

    FastAPI handles:
    - CSV parsing
    - DuckDB registration
    - Schema profiling
    - RAG indexing
    """

    if not file_paths:
        return (
            None,
            "ارفع ملف CSV واحد على الأقل / Please upload at least one CSV file.",
            None,
            [],
            "",
        )

    if isinstance(file_paths, (str, Path)):
        file_paths = [str(file_paths)]

    try:
        paths = [
            Path(file_path)
            for file_path in file_paths
        ]

        if not paths:
            raise ValueError("No CSV files were selected.")

        for path in paths:
            if path.suffix.lower() != ".csv":
                raise ValueError(
                    f"{path.name}: CSV files only."
                )

        # ----------------------------------------------------
        # Multipart upload
        # ----------------------------------------------------

        with ExitStack() as stack:

            multipart_files = []

            for path in paths:

                file_handle = stack.enter_context(
                    path.open("rb")
                )

                multipart_files.append(
                    (
                        "files",
                        (
                            path.name,
                            file_handle,
                            "text/csv",
                        ),
                    )
                )

            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=multipart_files,
                timeout=API_TIMEOUT,
            )

        if not response.ok:
            raise RuntimeError(
                f"API Error ({response.status_code}): "
                f"{get_api_error(response)}"
            )

        data = response.json()

        if not data.get("success"):
            raise RuntimeError(
                "API upload was not successful."
            )

        # ----------------------------------------------------
        # API response
        # ----------------------------------------------------

        dataset_id = data["dataset_id"]
        workspace = data["workspace"]
        schema_profile = data["schema_profile"]
        indexed_documents = data["indexed_documents"]

        # Preview returned by FastAPI.
        preview = pd.DataFrame(
            data.get("preview", [])
        )

        # ----------------------------------------------------
        # Format table descriptions
        # ----------------------------------------------------

        profile_tables = {
            table.get("table_name"): table
            for table in schema_profile.get("tables", [])
        }

        table_sections = []

        for table in workspace.get("tables", []):

            table_profile = profile_tables.get(
                table["table_name"],
                {},
            )

            profile_columns = {
                column.get("name"): column
                for column in table_profile.get("columns", [])
            }

            column_lines = []

            for column in table.get("columns", []):

                profiled = profile_columns.get(
                    column["name"],
                    {},
                )

                column_lines.append(
                    f"- **`{column['name']}`** "
                    f"(`{column['dtype']}`): "
                    f"{profiled.get('description_ar', '')} / "
                    f"{profiled.get('description_en', '')}"
                )

            table_sections.append(
                f"### `{table['table_name']}` "
                f"← `{table['original_filename']}`\n\n"
                f"- **الصفوف / Rows:** "
                f"{table['row_count']:,}\n"
                f"- **الأعمدة / Columns:** "
                f"{table['column_count']}\n\n"
                f"**وصف الجدول / Table description:** "
                f"{table_profile.get('description_ar', '')} "
                f"{table_profile.get('description_en', '')}\n\n"
                + "\n".join(column_lines)
            )

        # ----------------------------------------------------
        # Upload status
        # ----------------------------------------------------

        status = (
            "## تم تحميل الملفات بنجاح / CSV files uploaded successfully\n\n"
            f"- **عدد الملفات والجداول / Files and tables:** "
            f"{len(workspace.get('tables', []))}\n"
            f"- **إجمالي الصفوف / Total rows:** "
            f"{workspace['row_count']:,}\n"
            f"- **إجمالي الأعمدة / Total columns:** "
            f"{workspace['column_count']}\n"
            f"- **مستندات RAG المفهرسة / Indexed RAG documents:** "
            f"{indexed_documents}\n\n"
            "### وصف مساحة العمل / Workspace description\n\n"
            f"{schema_profile.get('dataset_summary_ar', '')}\n\n"
            f"{schema_profile.get('dataset_summary_en', '')}\n\n"
            "### الجداول / Tables\n\n"
            + "\n\n---\n\n".join(table_sections)
            + "\n\n> كل الملفات في مساحة العمل دي موجودة في قاعدة بيانات واحدة، "
            "ويمكن استخدام أكثر من جدول في استعلام SQL عند وجود علاقة مناسبة."
        )

        return (
            preview,
            status,
            dataset_id,
            [],
            "",
        )

    except Exception as exc:

        return (
            None,
            f"**تعذر تحميل الملفات / Upload failed:** {str(exc)}",
            None,
            [],
            "",
        )


# ============================================================
# Chat response through FastAPI
# ============================================================

def respond(question, history, dataset_id):
    """Send user question to FastAPI and display the response."""

    history = history or []

    if not question or not question.strip():
        return (
            history,
            "",
            "اكتب سؤالًا الأول / Please enter a question.",
        )

    if not dataset_id:
        return (
            history,
            "",
            "ارفع ملف CSV وانتظر اكتمال الفهرسة / "
            "Upload CSV files and wait for indexing.",
        )

    question = question.strip()
    language = detect_language(question)

    try:

        # ----------------------------------------------------
        # Call FastAPI /predict
        # ----------------------------------------------------

        response = call_api(
            "/predict",
            {
                "question": question,
                "dataset_id": dataset_id,
            },
            timeout=API_TIMEOUT,
        )

        answer = response["answer"]

        updated_history = history + [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        # ----------------------------------------------------
        # SQL details
        # ----------------------------------------------------

        if response.get("route") == "sql":

            result = response.get("result") or {}

            result_json = json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

            sql_query = response.get("sql") or ""

            if language == "Arabic":

                details = (
                    "### نوع السؤال: Text-to-SQL\n\n"
                    "**استعلام SQL المنفذ:**\n\n"
                    f"```sql\n{sql_query}\n```\n\n"
                    "**نتيجة الاستعلام:**\n\n"
                    f"```json\n{result_json}\n```"
                )

                if result.get("truncated"):
                    details += (
                        "\n\n**تنبيه:** تم اختصار النتائج."
                    )

            else:

                details = (
                    "### Question type: Text-to-SQL\n\n"
                    "**Executed SQL query:**\n\n"
                    f"```sql\n{sql_query}\n```\n\n"
                    "**Query result:**\n\n"
                    f"```json\n{result_json}\n```"
                )

                if result.get("truncated"):
                    details += (
                        "\n\n**Note:** Results were truncated."
                    )

        # ----------------------------------------------------
        # RAG details
        # ----------------------------------------------------

        else:

            sources = response.get("sources") or []

            if language == "Arabic":

                details = "### نوع السؤال: RAG\n\n"

                if sources:

                    details += "### المصادر المسترجعة\n\n"

                    for i, source in enumerate(sources, start=1):

                        details += (
                            f"**{i}. النوع:** "
                            f"`{source.get('kind', 'unknown')}`\n\n"
                            f"**المحتوى:** "
                            f"{source.get('text', '')}\n\n"
                        )

                        if source.get("table"):
                            details += (
                                f"**الجدول:** "
                                f"`{source['table']}`\n\n"
                            )

                        if source.get("column"):
                            details += (
                                f"**العمود:** "
                                f"`{source['column']}`\n\n"
                            )

                        score = source.get("score")

                        if score is not None:
                            details += (
                                f"**درجة الترتيب:** "
                                f"`{score:.4f}`\n\n"
                            )

                        details += "---\n\n"

                else:
                    details += "لم يتم إرجاع مصادر."

            else:

                details = "### Question type: RAG\n\n"

                if sources:

                    details += "### Retrieved sources\n\n"

                    for i, source in enumerate(sources, start=1):

                        details += (
                            f"**{i}. Type:** "
                            f"`{source.get('kind', 'unknown')}`\n\n"
                            f"**Content:** "
                            f"{source.get('text', '')}\n\n"
                        )

                        if source.get("table"):
                            details += (
                                f"**Table:** "
                                f"`{source['table']}`\n\n"
                            )

                        if source.get("column"):
                            details += (
                                f"**Column:** "
                                f"`{source['column']}`\n\n"
                            )

                        score = source.get("score")

                        if score is not None:
                            details += (
                                f"**Ranking score:** "
                                f"`{score:.4f}`\n\n"
                            )

                        details += "---\n\n"

                else:
                    details += "No sources were returned."

        return (
            updated_history,
            "",
            details,
        )

    except Exception as exc:

        error_message = (
            f"حصل خطأ أثناء الاتصال بالـ API: {exc}"
            if language == "Arabic"
            else f"API request failed: {exc}"
        )

        updated_history = history + [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": error_message,
            },
        ]

        return (
            updated_history,
            "",
            error_message,
        )


# ============================================================
# Clear chat
# ============================================================

def clear_chat():
    return [], ""


# ============================================================
# Theme
# ============================================================

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_md,
    font=[
        gr.themes.GoogleFont("IBM Plex Sans Arabic"),
        "Segoe UI",
        "Tahoma",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("IBM Plex Mono"),
        "monospace",
    ],
).set(
    body_background_fill="#F4EBDA",
    block_background_fill="#FBF7EE",
    block_border_color="#E0D2B6",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(62, 42, 30, 0.08)",
    block_label_background_fill="#FBF7EE",
    block_label_text_color="#5A3A22",
    body_text_color="#3E2A1E",
    button_primary_background_fill="#6B4423",
    button_primary_background_fill_hover="#4E3019",
    button_primary_text_color="#FBF3DF",
    button_secondary_background_fill="#FBF7EE",
    button_secondary_background_fill_hover="#EFE3CB",
    button_secondary_border_color="#CDB88F",
    button_secondary_text_color="#5A3A22",
    input_background_fill="#FFFDF8",
    input_border_color="#D6C4A0",
    input_border_color_focus="#B8892F",
)


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

:root {
    --brown-900: #3E2A1E;
    --brown-700: #6B4423;
    --gold: #C9A24B;
    --gold-soft: #E6C97E;
    --beige: #F4EBDA;
    --beige-card: #FBF7EE;
    --line: #E0D2B6;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: 'IBM Plex Sans Arabic', 'Segoe UI', Tahoma, sans-serif !important;
}

footer {
    display: none !important;
}

#app-header {
    direction: rtl;
    text-align: right;
    background: linear-gradient(135deg, #3E2A1E 0%, #5A3A22 100%);
    border-radius: 14px;
    padding: 28px 34px;
    margin-bottom: 8px;
    border-right: 6px solid var(--gold);
}

#app-header h1 {
    margin: 0 0 8px 0;
    font-size: 2rem;
    font-weight: 700;
    color: #FBF3DF;
    direction: ltr;
    text-align: right;
}

#app-header .sub {
    margin: 0 0 10px 0;
    font-size: 1.15rem;
    font-weight: 500;
    color: var(--gold-soft);
}

#app-header .hint {
    margin: 0 0 18px 0;
    font-size: 1rem;
    line-height: 1.9;
    color: #E8DAB8;
}

#app-header .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: flex-start;
}

#app-header .chip {
    font-size: 0.9rem;
    padding: 6px 14px;
    border-radius: 999px;
    color: #F4EBDA;
    background: rgba(230, 201, 126, 0.12);
    border: 1px solid rgba(230, 201, 126, 0.4);
}

#app-header .chip bdi {
    color: var(--gold-soft);
    font-weight: 700;
    margin-inline-start: 4px;
}

#upload-status {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--beige-card);
    padding: 4px 18px;
    max-height: 420px;
    overflow-y: auto;
    line-height: 1.8;
    unicode-bidi: plaintext;
}

#upload-status:empty {
    display: none;
}

#upload-status h2 {
    font-size: 1.15rem;
    color: var(--brown-700);
}

#upload-status h3 {
    font-size: 1rem;
    color: var(--brown-900);
}

#chat-box {
    border-radius: 14px !important;
    border: 1px solid var(--line) !important;
}

#chat-box .message,
#chat-box .message * {
    unicode-bidi: plaintext;
    text-align: start;
    line-height: 1.8;
}

#question-box textarea {
    font-size: 1rem;
    line-height: 1.7;
    unicode-bidi: plaintext;
}

#send-btn {
    min-height: 56px;
    font-weight: 600;
}

#details-box {
    unicode-bidi: plaintext;
    line-height: 1.8;
}

#details-box pre {
    border-radius: 10px;
}

#preview-table {
    border-radius: 12px;
    overflow: hidden;
}

button:focus-visible,
textarea:focus-visible,
input:focus-visible {
    outline: 2px solid var(--gold) !important;
    outline-offset: 2px;
}

@media (max-width: 768px) {
    #app-header {
        padding: 20px;
    }

    #app-header h1 {
        font-size: 1.5rem;
    }
}
"""


# ============================================================
# Gradio version compatibility
# ============================================================

_THEME_IN_BLOCKS = (
    "css" in inspect.signature(gr.Blocks.__init__).parameters
)

_BLOCKS_STYLE = (
    {
        "theme": THEME,
        "css": CUSTOM_CSS,
    }
    if _THEME_IN_BLOCKS
    else {}
)

_LAUNCH_STYLE = (
    {}
    if _THEME_IN_BLOCKS
    else {
        "theme": THEME,
        "css": CUSTOM_CSS,
    }
)


# ============================================================
# Gradio application
# ============================================================

with gr.Blocks(
    title="Bilingual CSV Chatbot | Text-to-SQL + RAG",
    **_BLOCKS_STYLE,
) as demo:

    dataset_state = gr.State(value=None)

    gr.HTML(
        """
        <div id="app-header" dir="rtl">
            <h1>CSV Chatbot</h1>

            <p class="sub">
                مساعد ذكي لتحليل ملفات
                <bdi>CSV</bdi>
                بالعربي والإنجليزي
            </p>

            <p class="hint">
                ارفعي ملف CSV واحد أو أكتر، وانتظري انتهاء التحليل والفهرسة،
                وبعدها اسألي عن جدول واحد أو عن بيانات من أكتر من جدول.
            </p>

            <div class="chips">
                <span class="chip">
                    <bdi>Text-to-SQL</bdi>
                    للحسابات والتحليلات على البيانات
                </span>

                <span class="chip">
                    <bdi>RAG</bdi>
                    لفهم أسماء الأعمدة ووصف البيانات
                </span>

                <span class="chip">
                    <bdi>Bilingual Schema Profiler</bdi>
                    قاموس أعمدة بالعربي والإنجليزي
                </span>
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):

        # ----------------------------------------------------
        # Upload section
        # ----------------------------------------------------

        with gr.Column(scale=1, min_width=340):

            file_input = gr.File(
                label="رفع ملفات CSV متعددة / Upload multiple CSV files",
                file_types=[".csv"],
                file_count="multiple",
                type="filepath",
            )

            upload_button = gr.Button(
                "تحميل الملفات وتحليلها / Upload and analyze",
                variant="primary",
                size="lg",
            )

            upload_status = gr.Markdown(
                elem_id="upload-status"
            )

            preview = gr.Dataframe(
                label="معاينة أول 30 صفًا من الملفات / Preview first 30 rows",
                interactive=False,
                elem_id="preview-table",
            )

        # ----------------------------------------------------
        # Chat section
        # ----------------------------------------------------

        with gr.Column(scale=2, min_width=420):

            chatbot = gr.Chatbot(
                label="المحادثة / Chat",
                height=450,
                elem_id="chat-box",
            )

            with gr.Row():

                question_input = gr.Textbox(
                    label="سؤالك / Your question",
                    placeholder=(
                        "ما إجمالي المبيعات في القاهرة؟ "
                        "/ What is the total sales in Cairo?"
                    ),
                    lines=2,
                    scale=5,
                    elem_id="question-box",
                )

                send_button = gr.Button(
                    "إرسال / Send",
                    variant="primary",
                    scale=1,
                    min_width=120,
                    elem_id="send-btn",
                )

            clear_button = gr.Button(
                "مسح المحادثة / Clear chat",
                variant="secondary",
            )

            with gr.Accordion(
                "تفاصيل الإجابة / Answer details",
                open=False,
            ):

                details_output = gr.Markdown(
                    elem_id="details-box"
                )

    # --------------------------------------------------------
    # Upload event
    # --------------------------------------------------------

    upload_button.click(
        fn=upload_csv,
        inputs=[file_input],
        outputs=[
            preview,
            upload_status,
            dataset_state,
            chatbot,
            details_output,
        ],
    )

    # --------------------------------------------------------
    # Send button event
    # --------------------------------------------------------

    send_button.click(
        fn=respond,
        inputs=[
            question_input,
            chatbot,
            dataset_state,
        ],
        outputs=[
            chatbot,
            question_input,
            details_output,
        ],
    )

    # --------------------------------------------------------
    # Enter key event
    # --------------------------------------------------------

    question_input.submit(
        fn=respond,
        inputs=[
            question_input,
            chatbot,
            dataset_state,
        ],
        outputs=[
            chatbot,
            question_input,
            details_output,
        ],
    )

    # --------------------------------------------------------
    # Clear chat event
    # --------------------------------------------------------

    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[
            chatbot,
            details_output,
        ],
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    demo.queue().launch(
        **_LAUNCH_STYLE
    )
