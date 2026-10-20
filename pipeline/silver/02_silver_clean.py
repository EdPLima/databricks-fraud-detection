# Databricks notebook source
from pyspark.sql import functions as F
import os, sys, argparse

PROJECT_ROOT = "/Workspace/Users/eduardolimaofficial410@gmail.com/databricks-fraud-detection"
sys.path.insert(0, f"{PROJECT_ROOT}/src")
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
batch_type = _get_param("batch_type", "train")

BRONZE_SCHEMA = f"{CATALOG}.bronze"
SILVER_SCHEMA = f"{CATALOG}.silver"

TRANSACTION_TABLE = f"{BRONZE_SCHEMA}.{batch_type}_transaction_raw"
IDENTITY_TABLE = f"{BRONZE_SCHEMA}.{batch_type}_identity_raw"
SILVER_OUTPUT = f"{SILVER_SCHEMA}.{batch_type}_data_clean"

# Join bronze
dados_bronze = spark.sql(f"""
    SELECT 
        trans.*, 
        iden.* EXCEPT (TransactionID, ingestion_timestamp, dataset_type)
    FROM {TRANSACTION_TABLE} trans
    LEFT JOIN {IDENTITY_TABLE} iden
        ON trans.TransactionID = iden.TransactionID
""").dropDuplicates()

dados_silver = dados_bronze

# Renomear colunas id-XX para id_XX (hifen -> underline)
for col_name in dados_silver.columns:
    if col_name.startswith("id-"):
        dados_silver = dados_silver.withColumnRenamed(col_name, col_name.replace("-", "_"))

# Transformacoes
if "isFraud" in dados_silver.columns:
    dados_silver = dados_silver.withColumnRenamed("isFraud", "target")
if "id_34" in dados_silver.columns:
    extracted = F.regexp_extract(F.col("id_34"), r":(\d+)", 1)
    dados_silver = dados_silver.withColumn(
        "id_34", F.when(extracted != "", extracted.cast("float")).otherwise(None)
    )

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER_SCHEMA}")

(
    dados_silver
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_OUTPUT)
)
spark.sql(f"ANALYZE TABLE {SILVER_OUTPUT} COMPUTE STATISTICS FOR ALL COLUMNS")
spark.sql(f"OPTIMIZE {SILVER_OUTPUT} ZORDER BY (TransactionID)")