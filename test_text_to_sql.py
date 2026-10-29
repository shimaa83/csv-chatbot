
from pathlib import Path

from app.csv_loader import load_csv_bytes
from app.database import register_dataframe
from app.text_to_sql import ask_with_sql


path = Path("data/sample_sales.csv")

df, metadata = load_csv_bytes(path.read_bytes(), path.name)
dataset = register_dataframe(df, metadata)

question = "Which city has the highest total sales?"

response = ask_with_sql(
    question=question,
    dataset_id=dataset["dataset_id"],
)

print("Question:", response["question"])
print("Generated SQL:", response["sql"])
print("Result:", response["result"])
