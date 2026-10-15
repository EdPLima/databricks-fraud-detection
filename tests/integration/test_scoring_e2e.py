import pandas as pd
import pytest
from databricks.sdk import WorkspaceClient

from databricks_fraud_model import score_batch, save_predictions
from databricks_fraud_model.features import MODEL_FEATURES
from databricks_fraud_model.utils import load_config

pytestmark = pytest.mark.integration

cfg = load_config()
CATALOG = cfg["catalog"]
GOLD_TABLE = f"{CATALOG}.gold.fraud_features_prd"
PREDICTIONS_TABLE = f"{CATALOG}.gold.model_predictions"
BRONZE_TABLE = f"{CATALOG}.bronze.monitoring_raw_events"


def test_endpoint_is_ready():
    w = WorkspaceClient()
    ep = w.serving_endpoints.get("model_fraud")
    assert ep.state.ready.name == "READY"


def test_prod_features_table_has_all_model_features():
    cols = spark.sql(f"SHOW COLUMNS FROM {GOLD_TABLE}").toPandas()
    col_set = set(cols.iloc[:, 0].tolist())
    missing = [f for f in MODEL_FEATURES if f not in col_set]
    assert len(missing) == 0, f"Missing features in prod table: {missing}"


def test_score_batch_with_real_endpoint():
    dados = spark.sql(f"SELECT * FROM {GOLD_TABLE}").toPandas()
    preds = score_batch(dados, n=5)

    assert len(preds) == 5
    assert "probability_fraud" in preds.columns
    assert "is_fraud" in preds.columns
    assert "latency_ms" in preds.columns
    assert preds["latency_ms"].iloc[0] >= 0
    assert preds["probability_fraud"].between(0, 1).all()


def test_save_predictions_writes_to_gold_and_bronze():
    dados = spark.sql(f"SELECT * FROM {GOLD_TABLE}").toPandas()
    preds = score_batch(dados, n=3)
    save_predictions(dados, preds)

    gold_count = spark.sql(f"SELECT count(*) FROM {PREDICTIONS_TABLE}").collect()[0][0]
    assert gold_count >= 3

    bronze_count = spark.sql(f"SELECT count(*) FROM {BRONZE_TABLE}").collect()[0][0]
    assert bronze_count >= 3


def test_bronze_events_have_latency():
    latest = spark.sql(f"""
        SELECT latency_ms, status, endpoint_name
        FROM {BRONZE_TABLE}
        ORDER BY event_timestamp DESC
        LIMIT 1
    """).toPandas()

    assert len(latest) == 1
    assert latest["latency_ms"].iloc[0] >= 0
    assert latest["status"].iloc[0] == "success"
    assert latest["endpoint_name"].iloc[0] == "model_fraud"
