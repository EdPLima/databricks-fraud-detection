from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from databricks_fraud_model.model.scoring import (
    ENDPOINT_NAME,
    _clean_records,
    save_to_bronze,
    score_batch,
)


def test_clean_records_selects_features(sample_df):
    features = ["TransactionAmt", "ProductCD", "card4"]
    records = _clean_records(sample_df, features)
    assert len(records) == 3
    for r in records:
        assert set(r.keys()) == set(features)


def test_clean_records_respects_n(sample_df):
    features = ["TransactionAmt", "ProductCD"]
    records = _clean_records(sample_df, features, n=2)
    assert len(records) == 2


def test_clean_records_handles_nan():
    df = pd.DataFrame({
        "TransactionAmt": [100.0, np.nan, 50.0],
        "ProductCD": ["W", None, "C"],
    })
    records = _clean_records(df, ["TransactionAmt", "ProductCD"])
    assert records[1]["TransactionAmt"] is None
    assert records[1]["ProductCD"] is None
    assert records[0]["TransactionAmt"] == 100.0
    assert records[2]["ProductCD"] == "C"


def test_clean_records_returns_all_when_no_n(sample_df):
    features = ["TransactionAmt", "ProductCD", "card4"]
    records = _clean_records(sample_df, features)
    assert len(records) == len(sample_df)


@patch("databricks_fraud_model.model.scoring.WorkspaceClient")
def test_score_batch_returns_predictions_with_latency(mock_ws, sample_df):
    mock_client = MagicMock()
    mock_ws.return_value = mock_client
    mock_client.api_client.do.return_value = {
        "predictions": [
            {"probability_fraud": 0.1, "is_fraud": 0},
            {"probability_fraud": 0.9, "is_fraud": 1},
            {"probability_fraud": 0.5, "is_fraud": 0},
        ]
    }

    features = ["TransactionAmt", "ProductCD", "card4"]
    preds = score_batch(sample_df, features=features, n=3)

    assert len(preds) == 3
    assert "latency_ms" in preds.columns
    assert "endpoint_name" in preds.columns
    assert "n_records" in preds.columns
    assert "request_id" in preds.columns
    assert "status" in preds.columns
    assert preds["endpoint_name"].iloc[0] == ENDPOINT_NAME
    assert preds["n_records"].iloc[0] == 3
    assert preds["status"].iloc[0] == "success"
    assert "TransactionID" in preds.columns
    assert preds["probability_fraud"].iloc[1] == 0.9
    assert len(preds["request_id"].iloc[0]) > 0  # UUID is non-empty


@patch("databricks_fraud_model.model.scoring.WorkspaceClient")
def test_score_batch_latency_is_positive(mock_ws, sample_df):
    mock_client = MagicMock()
    mock_ws.return_value = mock_client
    mock_client.api_client.do.return_value = {
        "predictions": [{"probability_fraud": 0.1, "is_fraud": 0}]
    }

    preds = score_batch(sample_df, features=["TransactionAmt"], n=1)
    assert preds["latency_ms"].iloc[0] >= 0


def test_save_to_bronze_includes_request_id_and_contract(sample_df):
    preds_df = pd.DataFrame({
        "TransactionID": [1, 2],
        "probability_fraud": [0.1, 0.9],
        "is_fraud": [0, 1],
        "latency_ms": [50.0, 60.0],
        "endpoint_name": ["model_fraud", "model_fraud"],
        "n_records": [2, 2],
        "request_id": ["req-1", "req-2"],
        "status": ["success", "success"],
    })

    with patch("pyspark.sql.SparkSession") as mock_spark_cls:
        mock_spark = MagicMock()
        mock_spark_cls.builder.getOrCreate.return_value = mock_spark
        mock_spark.createDataFrame.return_value = MagicMock()

        events = save_to_bronze(preds_df, table_name="test_catalog.test_schema.test_table")

    assert "request_id" in events.columns
    assert "feature_contract_version" in events.columns
    assert (events["request_id"] == ["req-1", "req-2"]).all()
    assert (events["feature_contract_version"] == "1.0.0").all()
