
from pathlib import Path

from app.csv_loader import load_csv_bytes
from app.database import register_dataframe
from app.rag import index_dataset
from app.chatbot import chat


# Load the sample CSV
path = Path("data/sample_sales.csv")

df, metadata = load_csv_bytes(
    path.read_bytes(),
    path.name,
)

# Register data in DuckDB
dataset = register_dataframe(df, metadata)
dataset_id = dataset["dataset_id"]

# Build the RAG index for this dataset
print("Building RAG index...")
print(index_dataset(dataset_id, dataset))

# Test 1: SQL question
question1 = "What is the total sales in Cairo?"
response1 = chat(question1, dataset_id)

print("\n--- SQL TEST ---")
print("Route:", response1["route"])
print("Answer:", response1["answer"])
print("SQL:", response1["sql"])
print("Result:", response1["result"])

# Test 2: RAG question
question2 = "Which column contains the city?"
response2 = chat(question2, dataset_id)

print("\n--- RAG TEST ---")
print("Route:", response2["route"])
print("Answer:", response2["answer"])
print("Sources:", response2["sources"])
