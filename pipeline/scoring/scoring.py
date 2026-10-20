# Databricks notebook source
import os, sys, argparse

_nb = dbutils.notebook.entry_point.getNotebookPath()
_r = _nb.split("databricks-fraud-detection")[0] + "databricks-fraud-detection"
sys.path.insert(0, f"{_r}/src")
from databricks_fraud_model import MODEL_FEATURES, score_batch, save_predictions
from databricks_fraud_model.utils import load_config

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
GOLD_TABLE = f"{CATALOG}.gold.fraud_features_prd"

# Carrega dados de producao
dados = spark.sql(f"SELECT * FROM {GOLD_TABLE}").toPandas()

# Scoring via endpoint REST (mede latencia do endpoint)
preds = score_batch(dados, n=1000)

# Salva predicoes + features em Gold + eventos brutos em Bronze (com latency_ms)
save_predictions(dados, preds)