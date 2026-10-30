import uuid

import pandas as pd
import pytest

from app.config import DATABASE_DIR
from app.database import (
    _CONNECTIONS,
    _LOCK,
    execute_readonly_sql,
    get_table_names,
    get_table_schemas,
    make_table_name,
    register_dataframe,
    validate_readonly_sql,
)


def test_make_table_name_is_safe_and_unique():
    used = set()
    first = make_table_name("Order Details.csv", used)
    used.add(first)
    second = make_table_name("Order Details.csv", used)

    assert first == "order_details"
    assert second == "order_details_2"
    assert first.replace("_", "").isalnum()


def test_two_csv_tables_can_be_joined():
    workspace_id = uuid.uuid4().hex
    db_path = DATABASE_DIR / f"{workspace_id}.duckdb"

    try:
        customers = pd.DataFrame({
            "customer_id": [1, 2],
            "customer_name": ["Mona", "Omar"],
        })
        orders = pd.DataFrame({
            "order_id": [101, 102, 103],
            "customer_id": [1, 1, 2],
            "amount": [100, 50, 75],
        })

        register_dataframe(customers, {
            "dataset_id": workspace_id,
            "table_name": "customers",
            "original_filename": "customers.csv",
        })
        register_dataframe(orders, {
            "dataset_id": workspace_id,
            "table_name": "orders",
            "original_filename": "orders.csv",
        })

        assert get_table_names(workspace_id) == ["customers", "orders"]
        schemas = get_table_schemas(workspace_id)
        assert {table["table_name"] for table in schemas} == {
            "customers", "orders"
        }

        result = execute_readonly_sql(
            workspace_id,
            """
            SELECT c.customer_name, SUM(o.amount) AS total_amount
            FROM customers AS c
            JOIN orders AS o ON c.customer_id = o.customer_id
            GROUP BY c.customer_name
            ORDER BY c.customer_name
            """,
        )

        assert result["columns"] == ["customer_name", "total_amount"]
        assert result["rows"] == [["Mona", 150], ["Omar", 75]]
    finally:
        with _LOCK:
            connection = _CONNECTIONS.pop(workspace_id, None)
            if connection is not None:
                connection.close()
        for suffix in ("", ".wal"):
            file_path = db_path.with_name(db_path.name + suffix)
            if file_path.exists():
                file_path.unlink()


def test_sql_validator_rejects_unknown_tables():
    with pytest.raises(ValueError, match="unknown table"):
        validate_readonly_sql(
            "SELECT * FROM private_table",
            allowed_tables={"customers", "orders"},
        )


def test_sql_validator_rejects_write_operations():
    with pytest.raises(ValueError, match="Only SELECT"):
        validate_readonly_sql(
            "DROP TABLE customers",
            allowed_tables={"customers"},
        )
