
from pathlib import Path

from app.csv_loader import load_csv_bytes
from app.database import register_dataframe
from app.rag import index_dataset, hybrid_search

path = Path("data/sample_sales.csv")

df, metadata = load_csv_bytes(path.read_bytes(), path.name)
dataset = register_dataframe(df, metadata)

index_result = index_dataset(
    dataset["dataset_id"],
    dataset,
)

print("Index:", index_result)

results = hybrid_search(
    dataset["dataset_id"],
    "Which column contains the city?",
)

for item in results:
    print(
        item.get("rerank_score", item["rrf_score"]),
        item["text"],
    )
