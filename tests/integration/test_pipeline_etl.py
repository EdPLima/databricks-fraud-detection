import pytest

from databricks_fraud_model.features import MODEL_FEATURES
from databricks_fraud_model.utils import load_config

pytestmark = pytest.mark.integration

cfg = load_config()
CATALOG = cfg["catalog"]


def test_bronze_tables_exist():
    tables = spark.sql(f"SHOW TABLES IN {CATALOG}.bronze").toPandas()
    table_names = set(tables["tableName"].tolist())
    assert "train_transaction_raw" in table_names
    assert "prd_transaction_raw" in table_names


def test_silver_tables_exist():
    tables = spark.sql(f"SHOW TABLES IN {CATALOG}.silver").toPandas()
    table_names = set(tables["tableName"].tolist())
    assert "fraud_data_clean" in table_names
    assert "prd_data_clean" in table_names


def test_gold_tables_exist():
    tables = spark.sql(f"SHOW TABLES IN {CATALOG}.gold").toPandas()
    table_names = set(tables["tableName"].tolist())
    assert "fraud_features" in table_names
    assert "fraud_features_prd" in table_names
    assert "model_predictions" in table_names


def test_gold_features_has_split_column():
    cols = spark.sql(f"SHOW COLUMNS FROM {CATALOG}.gold.fraud_features").toPandas()
    col_names = set(cols.iloc[:, 0].tolist())
    assert "split" in col_names


def test_gold_features_train_has_rows():
    count = spark.sql(f"""
        SELECT count(*) FROM {CATALOG}.gold.fraud_features WHERE split = 'train'
    """).collect()[0][0]
    assert count > 0


def test_gold_features_prd_has_all_model_features():
    cols = spark.sql(f"SHOW COLUMNS FROM {CATALOG}.gold.fraud_features_prd").toPandas()
    col_set = set(cols.iloc[:, 0].tolist())
    missing = [f for f in MODEL_FEATURES if f not in col_set]
    assert len(missing) == 0, f"Missing features in prd gold: {missing}"


def test_model_predictions_has_expected_columns():
    cols = spark.sql(f"SHOW COLUMNS FROM {CATALOG}.gold.model_predictions").toPandas()
    col_names = set(cols.iloc[:, 0].tolist())
    required = {"TransactionID", "probability_fraud", "is_fraud", "prediction_timestamp"}
    assert required.issubset(col_names)


def test_model_predictions_has_rows():
    count = spark.sql(f"SELECT count(*) FROM {CATALOG}.gold.model_predictions").collect()[0][0]
    assert count > 0
