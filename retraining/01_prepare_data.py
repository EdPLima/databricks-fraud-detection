# Databricks notebook source
# DBTITLE 1,01_prepare_data
# MAGIC %md
# MAGIC # 01 -- Preparo de Dados para Retreino
# MAGIC Carrega dados da Silver, renomeia colunas id-XX, deduplica e salva tabela para retreino.
# MAGIC
# MAGIC **Entrada:** `{catalog}.silver.{batch_type}_data_clean`
# MAGIC **Saida:** `{catalog}.silver.retrain_data_clean`

# COMMAND ----------

from pyspark.sql import functions as F
import sys

import os, sys

PROJECT_ROOT = "/Workspace/Users/eduardolimaofficial410@gmail.com/databricks-fraud-detection"
sys.path.insert(0, f"{PROJECT_ROOT}/src")
from databricks_fraud_model.utils import load_config

cfg = load_config()
try:
    CATALOG = dbutils.widgets.get("catalog")
except Exception:
    CATALOG = cfg["catalog"]
SILVER = f"{CATALOG}.silver"

try:
    batch_type = dbutils.widgets.get("batch_type")
except Exception:
    batch_type = "train"

SOURCE = f"{SILVER}.{batch_type}_data_clean"
OUTPUT = f"{SILVER}.retrain_data_clean"

# COMMAND ----------

dados = spark.sql(f"SELECT * FROM {SOURCE}")

if "isFraud" in dados.columns:
    dados = dados.withColumnRenamed("isFraud", "target")

for col in dados.columns:
    if col.startswith("id-"):
        dados = dados.withColumnRenamed(col, col.replace("-", "_"))

dados = dados.dropDuplicates()

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER}")
(dados.write.format("delta").mode("overwrite")
     .option("overwriteSchema", "true").saveAsTable(OUTPUT))
spark.sql(f"OPTIMIZE {OUTPUT} ZORDER BY (TransactionID)")

display(spark.sql(f"SELECT count(*) as total, sum(target) as frauds FROM {OUTPUT}"))