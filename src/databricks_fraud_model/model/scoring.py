"""
Scoring via REST endpoint do modelo em producao.

Funcoes:
- score_batch(): envia N registros ao endpoint, retorna predicoes + latency_ms
- score_single(): scoring de um unico registro
- save_to_bronze(): salva eventos brutos (com latencia) em bronze.monitoring_raw_events
- save_predictions(): salva features + predicoes em gold.model_predictions
"""

import time
import pandas as pd
from datetime import datetime, timezone
from databricks.sdk import WorkspaceClient
from databricks_fraud_model.features import MODEL_FEATURES
from databricks_fraud_model.utils import load_config

_cfg = load_config()
ENDPOINT_NAME = "model_fraud"
BRONZE_TABLE = f"{_cfg['catalog']}.bronze.monitoring_raw_events"
GOLD_TABLE = f"{_cfg['catalog']}.gold.model_predictions"


def _clean_records(df, features, n=None):
    """Seleciona features, converte para records, replace NaN -> None para JSON compliance."""
    sample = df[features].head(n) if n else df[features]
    records = sample.to_dict(orient='records')
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in records]


def score_batch(df, features=None, n=None, endpoint_name=ENDPOINT_NAME):
    """
    Envia features diretamente no payload REST

    Args:
        df: DataFrame pandas com os dados de producao (deve conter as features).
        features: lista de colunas a enviar. Default: MODEL_FEATURES (152).
        n: numero de registros a enviar. Default: todos.
        endpoint_name: nome do serving endpoint.

    Returns:
        DataFrame com TransactionID, probability_fraud, is_fraud, latency_ms, endpoint_name, n_records.
    """
    if features is None:
        features = MODEL_FEATURES

    w = WorkspaceClient()
    clean = _clean_records(df, features, n)
    payload = {'dataframe_records': clean}

    t0 = time.perf_counter()
    response = w.api_client.do(
        'POST',
        f'/serving-endpoints/{endpoint_name}/invocations',
        body=payload,
        headers={'Content-Type': 'application/json'},
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    preds = pd.DataFrame(response['predictions'])
    preds['latency_ms'] = latency_ms
    preds['endpoint_name'] = endpoint_name
    preds['n_records'] = len(preds)

    if 'TransactionID' in df.columns:
        ids = df['TransactionID'].head(n) if n else df['TransactionID']
        preds['TransactionID'] = ids.values

    return preds


def score_single(record, features=None, endpoint_name=ENDPOINT_NAME):
    """
    Scoring de um unico registro.

    Args:
        record: dict com as features do modelo.
        features: lista de colunas a enviar. Default: MODEL_FEATURES.

    Returns:
        dict com probability_fraud, is_fraud, latency_ms.
    """
    if features is None:
        features = MODEL_FEATURES

    w = WorkspaceClient()
    clean = {k: (None if pd.isna(v) else v) for k, v in record.items() if k in features}

    t0 = time.perf_counter()
    response = w.api_client.do(
        'POST',
        f'/serving-endpoints/{endpoint_name}/invocations',
        body={'dataframe_records': [clean]},
        headers={'Content-Type': 'application/json'},
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    result = response['predictions'][0]
    result['latency_ms'] = latency_ms
    return result


def save_to_bronze(preds_df, table_name=BRONZE_TABLE):
    """
    Salva eventos brutos de scoring na camada Bronze para observabilidade.

    Registra: TransactionID, probability_fraud, is_fraud, latency_ms,
    endpoint_name, n_records, status, event_timestamp.

    Esta tabela e a fonte para o pipeline ETL de monitoring (Bronze->Silver->Gold).
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    schema_name = '.'.join(table_name.split('.')[:2])
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

    events = pd.DataFrame({
        'event_timestamp': [datetime.now(timezone.utc)] * len(preds_df),
        'TransactionID': preds_df['TransactionID'].values if 'TransactionID' in preds_df.columns else [None] * len(preds_df),
        'probability_fraud': preds_df['probability_fraud'].values,
        'is_fraud': preds_df['is_fraud'].values,
        'latency_ms': preds_df['latency_ms'].values,
        'endpoint_name': preds_df['endpoint_name'].values if 'endpoint_name' in preds_df.columns else [ENDPOINT_NAME] * len(preds_df),
        'n_records': preds_df['n_records'].values if 'n_records' in preds_df.columns else [len(preds_df)] * len(preds_df),
        'status': ['success'] * len(preds_df),
    })

    spark_df = spark.createDataFrame(events)
    spark_df.write.mode("append").saveAsTable(table_name)
    return events


def save_predictions(df, preds_df, features=None, table_name=GOLD_TABLE):
    """
    Salva predicoes + features usadas em tabela Gold para monitoring.

    Args:
        df: DataFrame original com os dados de producao.
        preds_df: DataFrame retornado por score_batch().
        features: lista de features a salvar. Default: MODEL_FEATURES.
        table_name: nome da tabela Gold onde salvar.
    """
    from pyspark.sql import SparkSession
    if features is None:
        features = MODEL_FEATURES

    spark = SparkSession.builder.getOrCreate()

    save_to_bronze(preds_df)

    n = len(preds_df)
    result = df[features].head(n).copy()
    result['TransactionID'] = preds_df['TransactionID'].values if 'TransactionID' in preds_df.columns else df['TransactionID'].head(n).values
    result['probability_fraud'] = preds_df['probability_fraud'].values
    result['is_fraud'] = preds_df['is_fraud'].values
    result['prediction_timestamp'] = datetime.now(timezone.utc)

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {'.'.join(table_name.split('.')[:2])}")

    spark_df = spark.createDataFrame(result)
    spark_df.write.mode("append").saveAsTable(table_name)
    return result