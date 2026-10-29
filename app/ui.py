import inspect
import json
from pathlib import Path

import gradio as gr

from app.config import MAX_UPLOAD_BYTES
from app.csv_loader import load_csv_bytes
from app.database import register_dataframe
from app.rag import index_dataset
from app.schema_profiler import profile_dataset
from app.chatbot import chat, detect_language


def upload_csv(file_path):
    """Load a CSV, profile its schema, register it, and index its metadata."""

    if not file_path:
        return (
            None,
            "ارفع ملف CSV اولا / Please upload a CSV file first.",
            None,
            [],
            "",
        )

    path = Path(file_path)

    try:
        if path.suffix.lower() != ".csv":
            raise ValueError(
                "من فضلك ارفع ملف CSV فقط / Please upload a CSV file only."
            )

        if path.stat().st_size > MAX_UPLOAD_BYTES:
            max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise ValueError(
                f"حجم الملف أكبر من {max_mb} MB."
            )

        # 1. Read CSV.
        content = path.read_bytes()
        df, metadata = load_csv_bytes(content, path.name)

        # 2. Register dataset in DuckDB.
        dataset = register_dataframe(df, metadata)
        dataset_id = dataset["dataset_id"]

        # 3. Generate bilingual schema profile.
        schema_profile = profile_dataset(dataset_id)

        # 4. Build the existing RAG index.
        index_result = index_dataset(dataset_id, dataset)

        # 5. Preview the first 10 rows.
        preview = df.head(10)

        columns_info = "\n".join(
            (
                f"- `{column['name']}` "
                f"({column['dtype']})"
            )
            for column in metadata["columns"]
        )

        profile_columns = []

        for column in schema_profile.get("columns", []):
            profile_columns.append(
                f"**{column['name']}**\n\n"
                f"- الوصف بالعربي: "
                f"{column.get('description_ar', '')}\n"
                f"- English description: "
                f"{column.get('description_en', '')}\n"
            )

        profile_details = "\n\n".join(profile_columns)

        status = (
            "## تم تحميل الملف بنجاح / File uploaded successfully\n\n"
            f"- **اسم الملف / Filename:** `{path.name}`\n"
            f"- **عدد الصفوف / Rows:** {metadata['row_count']:,}\n"
            f"- **عدد الأعمدة / Columns:** {metadata['column_count']}\n"
            f"- **مستندات RAG المفهرسة / Indexed documents:** "
            f"{index_result['indexed_documents']}\n\n"
            "### وصف الملف بالعربي\n\n"
            f"{schema_profile.get('dataset_summary_ar', '')}\n\n"
            "### Dataset description\n\n"
            f"{schema_profile.get('dataset_summary_en', '')}\n\n"
            "### الأعمدة الأصلية / Original columns\n\n"
            f"{columns_info}\n\n"
            "### قاموس الأعمدة / Bilingual column dictionary\n\n"
            f"{profile_details}"
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
            f"**تعذر تحميل الملف / Upload failed:** {str(exc)}",
            None,
            [],
            "",
        )


def respond(question, history, dataset_id):
    """Answer questions using Text-to-SQL or RAG."""

    history = history or []

    if not question or not question.strip():
        return history, "", "اكتب سؤالًا الأول / Please enter a question."

    if not dataset_id:
        return (
            history,
            "",
            "ارفع ملف CSV وانتظر اكتمال الفهرسة / "
            "Upload a CSV and wait for indexing to finish.",
        )

    question = question.strip()
    language = detect_language(question)

    try:
        response = chat(
            question=question,
            dataset_id=dataset_id,
        )

        answer = response["answer"]

        # Gradio messages format.
        updated_history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        # Text-to-SQL details.
        if response["route"] == "sql":
            result_json = json.dumps(
                response["result"],
                ensure_ascii=False,
                indent=2,
                default=str,
            )

            if language == "Arabic":
                details = (
                    "### نوع السؤال: Text-to-SQL\n\n"
                    "**استعلام SQL المنفذ:**\n\n"
                    f"```sql\n{response['sql']}\n```\n\n"
                    "**نتيجة الاستعلام:**\n\n"
                    f"```json\n{result_json}\n```"
                )

                if response["result"].get("truncated"):
                    details += (
                        "\n\n**تنبيه:** تم اختصار النتائج."
                    )

            else:
                details = (
                    "### Question type: Text-to-SQL\n\n"
                    "**Executed SQL query:**\n\n"
                    f"```sql\n{response['sql']}\n```\n\n"
                    "**Query result:**\n\n"
                    f"```json\n{result_json}\n```"
                )

                if response["result"].get("truncated"):
                    details += "\n\n**Note:** Results were truncated."

        # RAG details.
        else:
            sources = response.get("sources", [])

            if language == "Arabic":
                details = "### نوع السؤال: RAG\n\n"

                if sources:
                    details += "### المصادر المسترجعة\n\n"

                    for i, source in enumerate(sources, start=1):
                        details += (
                            f"**{i}. النوع:** `{source['kind']}`\n\n"
                            f"**المحتوى:** {source['text']}\n\n"
                        )

                        if source.get("column"):
                            details += (
                                f"**العمود:** `{source['column']}`\n\n"
                            )

                        score = source.get("score")

                        if score is not None:
                            details += (
                                f"**درجة الترتيب:** `{score:.4f}`\n\n"
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
                            f"**{i}. Type:** `{source['kind']}`\n\n"
                            f"**Content:** {source['text']}\n\n"
                        )

                        if source.get("column"):
                            details += (
                                f"**Column:** `{source['column']}`\n\n"
                            )

                        score = source.get("score")

                        if score is not None:
                            details += (
                                f"**Ranking score:** `{score:.4f}`\n\n"
                            )

                        details += "---\n\n"
                else:
                    details += "No sources were returned."

        return updated_history, "", details

    except Exception as exc:
        error_message = (
            f"حصل خطأ أثناء الإجابة: {exc}"
            if language == "Arabic"
            else f"An error occurred while answering: {exc}"
        )

        updated_history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_message},
        ]

        return updated_history, "", error_message


def clear_chat():
    """Clear the conversation and the displayed details."""
    return [], ""


# ---------------------------------------------------------------------------
# Visual design only (theme + CSS + layout). No application logic below.
# ---------------------------------------------------------------------------

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
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
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

footer { display: none !important; }

/* Header */
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

/* Upload status card */
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
#upload-status:empty { display: none; }
#upload-status h2 { font-size: 1.15rem; color: var(--brown-700); }
#upload-status h3 { font-size: 1rem; color: var(--brown-900); }

/* Chat */
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

/* Question row */
#question-box textarea {
    font-size: 1rem;
    line-height: 1.7;
    unicode-bidi: plaintext;
}
#send-btn { min-height: 56px; font-weight: 600; }

/* Answer details */
#details-box {
    unicode-bidi: plaintext;
    line-height: 1.8;
}
#details-box pre { border-radius: 10px; }

/* Data preview */
#preview-table { border-radius: 12px; overflow: hidden; }

/* Keyboard focus */
button:focus-visible, textarea:focus-visible, input:focus-visible {
    outline: 2px solid var(--gold) !important;
    outline-offset: 2px;
}

@media (max-width: 768px) {
    #app-header { padding: 20px; }
    #app-header h1 { font-size: 1.5rem; }
}
"""

# Gradio >= 6 takes theme/css in launch(); older versions take them in Blocks().
_THEME_IN_BLOCKS = "css" in inspect.signature(gr.Blocks.__init__).parameters
_BLOCKS_STYLE = {"theme": THEME, "css": CUSTOM_CSS} if _THEME_IN_BLOCKS else {}
_LAUNCH_STYLE = {} if _THEME_IN_BLOCKS else {"theme": THEME, "css": CUSTOM_CSS}


with gr.Blocks(
    title="Bilingual CSV Chatbot | Text-to-SQL + RAG",
    **_BLOCKS_STYLE,
) as demo:

    dataset_state = gr.State(value=None)

    gr.HTML(
        """
        <div id="app-header" dir="rtl">
            <h1>CSV Chatbot</h1>
            <p class="sub">مساعد ذكي لتحليل ملفات <bdi>CSV</bdi> بالعربي والإنجليزي</p>
            <p class="hint">
                ارفعي أي ملف <bdi>CSV</bdi>، وانتظر انتهاء التحليل والفهرسة،
                وبعدها اسأل باللغة التى تفضلها.
            </p>
            <div class="chips">
                <span class="chip"><bdi>Text-to-SQL</bdi> للحسابات والتحليلات على البيانات</span>
                <span class="chip"><bdi>RAG</bdi> لفهم أسماء الأعمدة ووصف البيانات</span>
                <span class="chip"><bdi>Bilingual Schema Profiler</bdi> قاموس أعمدة بالعربي والإنجليزي</span>
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):

        # Upload and preview.
        with gr.Column(scale=1, min_width=340):

            file_input = gr.File(
                label="رفع ملف CSV / Upload CSV",
                file_types=[".csv"],
                type="filepath",
            )

            upload_button = gr.Button(
                "تحميل الملف وتحليله / Upload and analyze",
                variant="primary",
                size="lg",
            )

            upload_status = gr.Markdown(elem_id="upload-status")

            preview = gr.Dataframe(
                label="معاينة البيانات / Data preview",
                interactive=False,
                elem_id="preview-table",
            )

        # Chat interface.
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
                details_output = gr.Markdown(elem_id="details-box")

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

    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[
            chatbot,
            details_output,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(**_LAUNCH_STYLE)