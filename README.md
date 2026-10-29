::: {align="center"}
# CSV Chatbot

### Ask Questions About Any CSV in Arabic or English

**A bilingual AI data assistant powered by Text-to-SQL,
Retrieval-Augmented Generation (RAG), DuckDB, and Qwen 2.5.**

Upload a CSV file, ask a question in natural language, and get an answer
with SQL and query results when the question requires data analysis.
:::

------------------------------------------------------------------------

## Overview

CSV Chatbot turns tabular data into a conversational interface. Instead
of manually filtering spreadsheets or writing SQL queries, users can
upload a CSV file and ask questions in Arabic or English.

The application uses two complementary approaches:

-   **Text-to-SQL:** Converts analytical questions into read-only DuckDB
    SQL queries, executes them against the uploaded dataset, and
    explains the results.
-   **RAG (Retrieval-Augmented Generation):** Retrieves relevant dataset
    metadata and schema information to answer questions about the file,
    its columns, and their meanings.

A bilingual **Schema Profiler** uses the language model to generate
Arabic and English descriptions and aliases for the actual columns in
each uploaded dataset. The original column names and stored values
remain unchanged.

## Key Features

-   **Dynamic CSV ingestion:** Upload a CSV without defining its column
    names in advance.
-   **Arabic and English questions:** Ask questions in either language.
-   **Text-to-SQL analytics:** Supports questions involving totals,
    averages, counts, grouping, comparisons, and filtering, depending on
    the dataset and generated query.
-   **Metadata RAG:** Retrieves relevant schema and dataset information
    for descriptive questions.
-   **Bilingual schema profiling:** Generates a data dictionary with
    Arabic and English column descriptions and aliases.
-   **Query transparency:** Displays the generated SQL and returned data
    for SQL-routed questions.
-   **Dataset preview:** Shows a preview of uploaded rows.
-   **Local model integration:** Uses Ollama to run Qwen 2.5 locally,
    avoiding a required hosted LLM API.
-   **Web interface:** Provides a conversational UI built with Gradio.
-   **DuckDB analytics:** Executes SQL against the uploaded tabular
    dataset.

## Architecture

``` mermaid
flowchart TD
    A[Upload CSV] --> B[CSV Loader]
    B --> C[DataFrame and Metadata]
    C --> D[Register Dataset in DuckDB]
    D --> E[Bilingual Schema Profiler]
    E --> F[Schema and Sample Values]
    F --> G[Text-to-SQL]
    F --> H[RAG Metadata Index]
    I[User Question: Arabic or English] --> J[Question Routing]
    J -->|Analytical question| G
    J -->|Metadata or schema question| H
    G --> K[Read-only SQL Validation]
    K --> L[Execute SQL in DuckDB]
    L --> M[Qwen Answer Generation]
    H --> M
    M --> N[Gradio Chat UI]
```

### Request flow

1.  The user uploads a CSV file.
2.  The loader reads the file and extracts its tabular data and
    metadata.
3.  The dataset is registered in DuckDB.
4.  The schema profiler collects column names, data types, and
    representative values, then generates a bilingual data dictionary.
5.  The application routes each question to Text-to-SQL or RAG.
6.  For analytical questions, the system generates and executes a
    read-only SQL query.
7.  For metadata questions, the system retrieves relevant indexed
    documents.
8.  The language model prepares a response, which is shown in the Gradio
    interface.

## Technology Stack

  --------------------------------------------------------------------------------------------
  Component               Technology                                   Purpose
  ----------------------- -------------------------------------------- -----------------------
  Language model          Qwen 2.5 3B via Ollama                       SQL generation and
                                                                       natural-language
                                                                       responses

  Application language    Python                                       Application logic

  User interface          Gradio                                       CSV upload, preview,
                                                                       chat, and result
                                                                       details

  Analytical database     DuckDB                                       Store and query
                                                                       uploaded tabular
                                                                       datasets

  Embeddings              `intfloat/multilingual-e5-small`             Multilingual semantic
                                                                       retrieval

  Reranker                `Horizon-Labs/multilingual-reranker-small`   Re-rank retrieved
                                                                       candidates, when
                                                                       enabled in the
                                                                       retrieval pipeline

  Retrieval               Hybrid search / RAG components               Retrieve relevant
                                                                       schema and metadata
                                                                       documents

  Configuration           `.env`                                       Local model and
                                                                       application settings

  Dependency management   `uv`                                         Install dependencies
                                                                       and run the application
  --------------------------------------------------------------------------------------------

> The embedding model and reranker are intended for the retrieval
> pipeline. They are not needed to execute SQL queries themselves.

## Project Structure

The following is the intended high-level structure. Adjust names if your
local repository differs.

``` text
csv-chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── csv_loader.py
│   ├── database.py
│   ├── schema_profiler.py
│   ├── text_to_sql.py
│   ├── rag.py
│   ├── chatbot.py
│   └── ui.py
├── data/
│   └── sample_sales.csv
├── indexes/
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

-   `csv_loader.py` --- reads CSV uploads and extracts file metadata.
-   `database.py` --- registers datasets in DuckDB and executes
    validated read-only queries.
-   `schema_profiler.py` --- creates and caches bilingual schema
    descriptions and aliases.
-   `text_to_sql.py` --- prompts the LLM to generate SQL from a question
    and the dataset schema.
-   `rag.py` --- indexes and retrieves dataset metadata for RAG.
-   `chatbot.py` --- routes questions and coordinates the SQL and RAG
    paths.
-   `ui.py` --- launches the Gradio interface.
-   `config.py` --- loads application configuration and directory
    settings.

## Requirements

-   Windows, macOS, or Linux
-   Python version compatible with the project's `pyproject.toml`
-   [`uv`](https://docs.astral.sh/uv/)
-   [Ollama](https://ollama.com/) installed and running
-   Enough RAM and disk space for the selected local model and embedding
    models

## Installation

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd csv-chatbot
```

Replace `<YOUR_REPOSITORY_URL>` with your repository URL.

### 2. Install dependencies

If the repository includes `pyproject.toml` and `uv.lock`:

``` bash
uv sync
```

### 3. Download the language model

Start Ollama, then pull the model:

``` bash
ollama pull qwen2.5:3b
```

Verify that the model is available:

``` bash
ollama list
```

The application expects the Ollama API at `http://localhost:11434` by
default.

### 4. Configure environment variables

Create a `.env` file in the project root. Use the variable names
expected by `app/config.py`. A typical configuration may look like this:

``` dotenv
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b

EMBEDDING_MODEL=intfloat/multilingual-e5-small
RERANKER_MODEL=Horizon-Labs/multilingual-reranker-small
```

Add any other required settings from your project's `app/config.py`,
such as upload-size limits and data/index directories.

**Important:** Do not commit `.env` files, API keys, credentials, or
private datasets. Keep secrets in environment variables and commit only
a sanitized `.env.example`.

## Run the Application

From the project root:

``` bash
uv run python -m app.ui
```

Open the local URL shown in the terminal. With the default Gradio
configuration, it is usually:

``` text
http://127.0.0.1:7860
```

Make sure Ollama is running before submitting questions.

## How to Use

1.  Open the web interface.
2.  Upload a `.csv` file.
3.  Wait for the dataset to load and the schema profile/indexing steps
    to finish.
4.  Review the data preview and generated column descriptions.
5.  Ask a question in Arabic or English.
6.  Review the answer. For SQL-routed questions, inspect the generated
    query and returned result in the answer details.

### Example dataset

`data/sample_sales.csv`

``` csv
order_id,product,city,sales
1,Laptop,Cairo,25000
2,Phone,Giza,12000
3,Laptop,Cairo,18000
4,Tablet,Alexandria,9000
```

### Example questions

  -----------------------------------------------------------------------
  Question                            Expected operation
  ----------------------------------- -----------------------------------
  What is the total sales in Cairo?   Filter by city and calculate
                                      `SUM(sales)`

  ما إجمالي المبيعات في القاهرة؟      Arabic analytical question; filter
                                      and calculate a total

  What is the average sale?           Calculate `AVG(sales)`

  How many orders are in the dataset? Calculate `COUNT(*)`

  Show total sales by city.           Group by city and calculate totals

  Which product has the highest sale? Sort or aggregate sales according
                                      to the question

  What columns are available?         Retrieve schema metadata

  اشرحلي معنى كل عمود.                Retrieve or explain the column
                                      descriptions
  -----------------------------------------------------------------------

For the example dataset, the expected values are:

-   Total sales in Cairo: **43,000**
-   Average of the four `sales` values: **16,000**
-   Number of rows/orders: **4**
-   Largest individual `sales` value: **25,000**

These are example expectations for this sample file, not guaranteed
results for arbitrary uploaded data.

## Text-to-SQL vs. RAG

  -----------------------------------------------------------------------
  Capability              Text-to-SQL             RAG
  ----------------------- ----------------------- -----------------------
  Main purpose            Calculate and filter    Retrieve relevant
                          values in rows          schema and metadata

  Example                 "What is the total      "What does the `sales`
                          sales in Cairo?"        column represent?"

  Uses SQL execution      Yes                     Not necessarily

  Uses retrieved          Not as its primary      Yes
  documents               operation               

  Best suited for         Counts, totals,         Column descriptions and
                          averages, filters, and  dataset context
                          grouped analysis        
  -----------------------------------------------------------------------

The routing logic determines which path handles a question. Accurate
routing is important: a question that requires an exact calculation
should reach SQL, while a question about the meaning of a column should
use metadata retrieval.

## Bilingual Data and Value Matching

The schema profiler is designed to help the model understand unfamiliar
column names across Arabic and English. For example, a file may use
`city` as a column and contain `Cairo` as a stored value, while a user
asks about `القاهرة`.

The original database values are not automatically translated or
rewritten. Cross-language value matching depends on the generated SQL
logic and the available sample values. For production use, consider
adding a deterministic value-mapping layer, fuzzy matching with
confidence thresholds, and clarification when a requested value cannot
be mapped reliably. This helps avoid incorrect empty results or
unintended filters.

## Configuration

Review `app/config.py` and `.env.example` for the exact settings
supported by your checkout.

  Setting             Purpose
  ------------------- -----------------------------------------------
  `OLLAMA_URL`        Base URL for the local Ollama API
  `OLLAMA_MODEL`      Name of the chat/generation model
  `EMBEDDING_MODEL`   Embedding model used for retrieval
  `RERANKER_MODEL`    Reranking model used by retrieval, if enabled

The project may define additional configuration values for upload
limits, index storage, caching, or timeouts. Use the names implemented
in the code rather than adding unsupported environment variables.

## Troubleshooting

### Ollama connection error

-   Confirm Ollama is installed and running.
-   Check that `OLLAMA_URL` points to the correct host and port.
-   Run `ollama list` and confirm `qwen2.5:3b` is available.
-   Try a simple request through the local Ollama installation before
    debugging the application.

### The answer is zero or a filter returns no rows

-   Inspect the generated SQL shown in the UI.
-   Check the actual values in the relevant CSV column.
-   Remember that `Cairo` and `القاهرة` are different strings in the
    database.
-   Add or improve value normalization/mapping rather than changing the
    original data silently.

### SQL generation fails

-   Check that the generated query uses the real column names and the
    `dataset` table.
-   Check data types and null values.
-   Confirm the SQL validator allows the query shape.
-   Rephrase the question more precisely and retry.

### RAG does not answer a schema question well

-   Confirm that indexing completed after upload.
-   Inspect the retrieved sources in the UI.
-   Ensure the index contains the generated schema descriptions if you
    expect RAG to use them.
-   Rebuild or refresh the dataset index after changing the schema
    profiling or indexing logic.

### The first request is slow

The first request may need to load the local language, embedding, or
reranking model. Performance also depends on available RAM/VRAM, dataset
size, and model settings.

## Limitations and Security Notes

-   **LLM output can be wrong.** Always inspect SQL and verify important
    results.
-   **Schema descriptions are inferred.** Column meanings generated by a
    model may be incomplete or inaccurate.
-   **Value translation is not guaranteed.** A bilingual column
    dictionary does not, by itself, guarantee that Arabic and English
    values will match correctly.
-   **Read-only validation matters.** The application should reject
    non-read-only SQL and restrict access to the intended dataset.
    Regex-based validation alone is not a complete security boundary.
-   **Treat uploaded CSV files as untrusted input.** Validate file size,
    encoding, column names, and data types.
-   **Do not expose the local app publicly without additional
    security.** Add authentication, request limits, robust SQL
    parsing/validation, isolation, and logging controls before
    deployment.
-   **Privacy depends on the environment.** With a local Ollama model,
    prompts can be processed locally, but any external services added to
    the application may have separate data-handling policies.

## Roadmap

Potential improvements:

-   More robust Arabic/English value normalization and entity matching.
-   Better question routing and SQL-generation evaluation.
-   SQL parsing and strict query validation.
-   Automatic data-quality and column profiling.
-   Support for Excel and other tabular formats.
-   Conversation-aware follow-up questions.
-   Query-result visualizations.
-   Unit tests and a benchmark dataset for Text-to-SQL accuracy.
-   Docker or cloud deployment, if needed.

## Development and Testing

Before releasing a change, test at least:

-   CSV upload and parsing.
-   Numeric aggregation (`SUM`, `AVG`, `COUNT`).
-   Filtering by text and numeric values.
-   Grouped comparisons.
-   Arabic and English versions of equivalent questions.
-   Schema/metadata questions routed to RAG.
-   Invalid SQL and empty-result behavior.
-   Large files and missing values.

Example local run:

``` bash
uv run python -m app.ui
```

Add test commands here once a test suite is configured in the
repository.

## Contributing

Contributions and suggestions are welcome. For substantial changes,
consider opening an issue first to discuss the intended behavior.

1.  Fork the repository.
2.  Create a focused feature branch.
3.  Add or update tests.
4.  Update documentation.
5.  Open a pull request with a clear description of the change.

## License

No license has been specified in this README. Add a `LICENSE` file and
update this section before distributing the project.

------------------------------------------------------------------------

::: {align="center"}
**CSV Chatbot --- From raw CSV files to conversational data analysis.**
:::
