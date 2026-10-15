# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Camada Bronze - Ingestão de Dados
# MAGIC %md
# MAGIC # 01 — Bronze: Ingestão de Dados Brutos
# MAGIC
# MAGIC Ingesta arquivos CSV de transação e identidade para tabelas bronze no Unity Catalog.
# MAGIC
# MAGIC **Parâmetro:** `batch_type` (train, test, val, prod) — escalável para qualquer novo batch
# MAGIC **Input:** Arquivos CSV em `/Volumes/workspace/bronze/raw/`
# MAGIC **Output:** `fraud_detection_dev.bronze.{batch_type}_transaction_raw` + `{batch_type}_identity_raw`

# COMMAND ----------

# DBTITLE 1,Bronze Ingestion (Unified)
from pyspark.sql import functions as F
import sys

sys.path.insert(0, "/Workspace/Users/eduardolimaofficial410@gmail.com/databricks-fraud-detection/src")
from databricks_fraud_model.utils import load_config

cfg = load_config()
CATALOG = cfg["catalog"]
BRONZE_SCHEMA = f"{CATALOG}.bronze"
VOLUME_PATH = cfg.get("volumes", {}).get("bronze_raw", "/Volumes/workspace/bronze/raw")

# batch_type: train, test, val, prod — escalavel para qualquer novo batch
try:
    batch_type = dbutils.widgets.get("batch_type")
except Exception:
    batch_type = "train"

file_name_trans = f"{batch_type}_transaction.csv"
file_name_id = f"{batch_type}_identity.csv"
table_name_trans = f"{BRONZE_SCHEMA}.{batch_type}_transaction_raw"
table_name_id = f"{BRONZE_SCHEMA}.{batch_type}_identity_raw"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {BRONZE_SCHEMA}")

def ingest_bronze_csv(file_name, table_name, zorder_cols=None):
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(f"{VOLUME_PATH}/{file_name}")
        .withColumn("ingestion_timestamp", F.current_timestamp())
        .withColumn("dataset_type", F.lit(batch_type))
    )

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )

    if zorder_cols:
        cols_str = ','.join(zorder_cols)
        spark.sql(f"OPTIMIZE {table_name} ZORDER BY ({cols_str})")
    else:
        spark.sql(f"OPTIMIZE {table_name} ZORDER BY (TransactionID)")

ingest_bronze_csv(
    file_name=file_name_trans,
    table_name=table_name_trans,
    zorder_cols=["TransactionID"]
)
ingest_bronze_csv(
    file_name=file_name_id,
    table_name=table_name_id,
    zorder_cols=["TransactionID"]
)