
from pathlib import Path

from app.csv_loader import load_csv_bytes
from app.database import register_dataframe, execute_readonly_sql

path = Path("data/sample_sales.csv")

df, metadata = load_csv_bytes(path.read_bytes(), path.name)
dataset = register_dataframe(df, metadata)

result = execute_readonly_sql(
    dataset["dataset_id"],
    """
    SELECT city, SUM(sales) AS total_sales
    FROM dataset
    GROUP BY city
    ORDER BY total_sales DESC
    """,
)

print("SQL result:")
print(result)
