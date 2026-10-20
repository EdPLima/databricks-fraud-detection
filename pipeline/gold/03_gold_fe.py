# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
import os, sys, argparse

from pyspark.sql import functions as F, Window

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
FEATURE_TABLE = cfg['feature_store']['table']
batch_type = _get_param("batch_type", "train")

SOURCE_TABLE = f"{CATALOG}.silver.{batch_type}_data_clean"
dados = spark.sql(f"SELECT * FROM {SOURCE_TABLE}")

from databricks_fraud_model.features import apply_feature_engineering_spark

if batch_type == "train":
    q60, q80 = dados.approxQuantile('TransactionDT', [0.60, 0.80], 0.01)
    train_df = dados.filter(F.col('TransactionDT') <= q60)
    test_df = dados.filter((F.col('TransactionDT') > q60) & (F.col('TransactionDT') <= q80))
    val_df = dados.filter(F.col('TransactionDT') > q80)

    train_s = apply_feature_engineering_spark(train_df)
    test_s = apply_feature_engineering_spark(test_df)
    val_s = apply_feature_engineering_spark(val_df)

    uc_parts = FEATURE_TABLE.split(".")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {uc_parts[0]}.{uc_parts[1]}")

    full_df = (
        train_s.withColumn("split", F.lit("train"))
        .unionByName(test_s.withColumn("split", F.lit("test")))
        .unionByName(val_s.withColumn("split", F.lit("val")))
        .filter(F.col("TransactionID").isNotNull())
        .withColumn("TransactionDT", F.from_unixtime(F.col("TransactionDT")).cast("timestamp"))
    )

    spark.sql(f"DROP TABLE IF EXISTS {FEATURE_TABLE}")

    (full_df.write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").saveAsTable(FEATURE_TABLE))
    spark.sql(f"OPTIMIZE {FEATURE_TABLE} ZORDER BY (TransactionID)")
    spark.sql(f"COMMENT ON COLUMN {FEATURE_TABLE}.split IS 'Conjunto de dados: train, test ou val (split temporal)'")
else:
    prod_s = apply_feature_engineering_spark(dados)

    GOLD_PRD = f"{CATALOG}.gold.fraud_features_prd"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold")

    (
        prod_s.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(GOLD_PRD)
    )
    spark.sql(f"OPTIMIZE {GOLD_PRD} ZORDER BY (TransactionID)")