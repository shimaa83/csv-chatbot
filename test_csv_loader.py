
from pathlib import Path

from app.csv_loader import load_csv_bytes

path = Path("data/sample_sales.csv")

df, metadata = load_csv_bytes(
    path.read_bytes(),
    path.name,
)

print("Dataset ID:", metadata["dataset_id"])
print("Rows:", metadata["row_count"])
print("Columns:", metadata["columns"])
print(df.head())
