# Databricks notebook source
# DBTITLE 1,02_feature_engineering
# MAGIC %md
# MAGIC # 02 -- Feature Engineering para Retreino
# MAGIC Aplica feature engineering, split temporal 60/20/20 e publica feature table no UC.
# MAGIC
# MAGIC **Entrada:** `{catalog}.silver.retrain_data_clean`
# MAGIC **Saida:** `{catalog}.gold.fraud_features_retrain`

# COMMAND ----------

import os, sys
from pyspark.sql import functions as F

_nb = dbutils.notebook.entry_point.getNotebookPath()
_r = _nb.split("databricks-fraud-detection")[0] + "databricks-fraud-detection"
sys.path.insert(0, f"{_r}/src")
from databricks_fraud_model.utils import load_config
from databricks_fraud_model.features import apply_feature_engineering_spark

cfg = load_config()
try:
    CATALOG = dbutils.widgets.get("catalog")
except Exception:
    CATALOG = cfg["catalog"]
SOURCE = f"{CATALOG}.silver.retrain_data_clean"
FEATURE_TABLE = f"{CATALOG}.gold.fraud_features_retrain"

# COMMAND ----------

dados = spark.sql(f"SELECT * FROM {SOURCE}")
dados_fe = apply_feature_engineering_spark(dados)

# Split temporal 60/20/20
q60, q80 = dados_fe.approxQuantile('TransactionDT', [0.60, 0.80], 0.01)

full_df = (
    dados_fe.filter(F.col('TransactionDT') <= q60).withColumn("split", F.lit("train"))
    .unionByName(dados_fe.filter((F.col('TransactionDT') > q60) & (F.col('TransactionDT') <= q80)).withColumn("split", F.lit("test")))
    .unionByName(dados_fe.filter(F.col('TransactionDT') > q80).withColumn("split", F.lit("val")))
    .filter(F.col("TransactionID").isNotNull())
    .withColumn("TransactionDT", F.from_unixtime(F.col("TransactionDT")).cast("timestamp"))
)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold")
spark.sql(f"DROP TABLE IF EXISTS {FEATURE_TABLE}")

(full_df.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(FEATURE_TABLE))
spark.sql(f"OPTIMIZE {FEATURE_TABLE} ZORDER BY (TransactionID)")

display(spark.sql(f"SELECT split, count(*) as n FROM {FEATURE_TABLE} GROUP BY split"))