# Databricks notebook source
import os, sys, argparse, pandas as pd

_nb = dbutils.notebook.entry_point.getNotebookPath()
_r = _nb.split("databricks-fraud-detection")[0] + "databricks-fraud-detection"
sys.path.insert(0, f"{_r}/src")
from databricks_fraud_model.features import MODEL_FEATURES
from databricks_fraud_model.utils import load_config
from databricks_fraud_model.monitoring import (
    calculate_ks_drift,
    calculate_categorical_drift,
    calculate_prediction_drift,
    summarize_endpoint_health,
)

cfg = load_config()

def _get_param(name, default=None):
    try:
        return dbutils.widgets.get(name)
    except Exception:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(f"--{name}", default=default)
        args, _ = parser.parse_known_args()
        return getattr(args, name)

CATALOG = _get_param("catalog", cfg["catalog"])

CATEGORICAL_FEATURES = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_12', 'id_16', 'id_31', 'id_36', 'id_37', 'id_38',
    'DeviceType', 'DeviceInfo',
    'DeviceType_isnull', 'TransactionAmt_round',
    'is_morning', 'is_afternoon', 'is_night', 'is_weekend',
    'email_match', 'P_email_free', 'R_email_free', 'device_mismatch',
]
NUMERIC_FEATURES = [f for f in MODEL_FEATURES if f not in CATEGORICAL_FEATURES]

df_train = spark.sql(f"""SELECT * FROM {CATALOG}.gold.fraud_features WHERE split = 'train'""").toPandas()
df_prod = spark.sql(f"""SELECT * FROM {CATALOG}.gold.model_predictions""").toPandas()

try:
    df_latency = spark.sql(f"""
        SELECT * FROM {CATALOG}.bronze.monitoring_raw_events
        ORDER BY event_timestamp DESC
    """).toPandas()
    has_latency = True
except Exception:
    has_latency = False

ks_df = calculate_ks_drift(df_train, df_prod, NUMERIC_FEATURES)
drifted_numeric = ks_df[ks_df['drift']]
print(ks_df.head(15).to_string())

if has_latency:
    health = summarize_endpoint_health(df_latency)
    lat = health['latency']
    errs = health['errors']
    cold = health['cold_starts']
    warm = health['warm_starts']

    obs_summary = pd.DataFrame([
        {"metric": "avg_ms",           "value": lat['avg_ms']},
        {"metric": "p50_ms",           "value": lat['p50_ms']},
        {"metric": "p95_ms",           "value": lat['p95_ms']},
        {"metric": "p99_ms",           "value": lat['p99_ms']},
        {"metric": "n_requests",       "value": lat['n_requests']},
        {"metric": "total_predictions", "value": lat['total_predictions']},
        {"metric": "ms_per_prediction", "value": lat['avg_per_record_ms']},
        {"metric": "error_rate",        "value": errs['error_rate']},
        {"metric": "n_errors",           "value": errs['n_errors']},
        {"metric": "cold_starts",        "value": cold['count']},
        {"metric": "warm_starts",        "value": warm['count']},
    ])
    print(obs_summary.to_string())
else:
    print("Dados de latencia indisponiveis")

psi_df = calculate_categorical_drift(df_train, df_prod, CATEGORICAL_FEATURES)
drifted_categorical = psi_df[psi_df['psi'] > 0.25]
print(psi_df.head(15).to_string())

pred_drift = calculate_prediction_drift(df_train, df_prod)
pred_summary = pd.DataFrame([
    {"metric": "n_predictions",       "value": pred_drift['n_predictions']},
    {"metric": "mean_score",          "value": pred_drift['mean_score']},
    {"metric": "median_score",        "value": pred_drift['median_score']},
    {"metric": "std_score",           "value": pred_drift['std_score']},
    {"metric": "min_score",            "value": pred_drift['min_score']},
    {"metric": "max_score",            "value": pred_drift['max_score']},
    {"metric": "predicted_fraud_rate", "value": pred_drift.get('predicted_fraud_rate')},
    {"metric": "train_fraud_rate",    "value": pred_drift.get('train_fraud_rate')},
])
print(pred_summary.to_string())

all_drift = []
for _, row in ks_df.iterrows():
    all_drift.append({'feature': row['feature'], 'metric': 'KS', 'value': row['ks_statistic'], 'p_value': row['p_value'], 'drifted': row['drift']})
for _, row in psi_df.iterrows():
    all_drift.append({'feature': row['feature'], 'metric': 'PSI', 'value': row['psi'], 'p_value': None, 'drifted': row['drift']})

drift_summary = pd.DataFrame(all_drift).sort_values(['drifted', 'value'], ascending=[False, False])
total_drifted = int(drift_summary['drifted'].sum())
total_tested = len(drift_summary)
drift_rate = total_drifted / total_tested if total_tested > 0 else 0

summary_df = pd.DataFrame([
    {"category": "observability", "metric": "p95_ms",           "value": lat['p95_ms'] if has_latency else None},
    {"category": "observability", "metric": "error_rate",        "value": errs['error_rate'] if has_latency else None},
    {"category": "observability", "metric": "n_requests",        "value": lat['n_requests'] if has_latency else None},
    {"category": "data_drift",    "metric": "features_tested",   "value": total_tested},
    {"category": "data_drift",    "metric": "features_drifted",  "value": total_drifted},
    {"category": "data_drift",    "metric": "drift_rate",       "value": drift_rate},
    {"category": "data_drift",    "metric": "drifted_numeric",   "value": len(drifted_numeric)},
    {"category": "data_drift",    "metric": "drifted_categorical", "value": len(drifted_categorical)},
    {"category": "pred_drift",   "metric": "n_predictions",    "value": pred_drift['n_predictions']},
    {"category": "pred_drift",   "metric": "mean_score",       "value": pred_drift['mean_score']},
    {"category": "pred_drift",   "metric": "predicted_fraud_rate", "value": pred_drift.get('predicted_fraud_rate')},
    {"category": "pred_drift",   "metric": "train_fraud_rate",  "value": pred_drift.get('train_fraud_rate')},
])
print(summary_df.to_string())
print(drift_summary[drift_summary['drifted']].head(10).to_string())