# Databricks notebook source
from pyspark.sql import functions as F, Window
from delta.tables import DeltaTable
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

VALID_CATALOGS = {"fraud_detection_dev", "fraud_detection_hml", "fraud_detection_prod"}
VALID_BATCHES = {"train", "test", "val", "prd", "prod"}

if CATALOG not in VALID_CATALOGS:
    raise ValueError(f"Catálogo não permitido: {CATALOG}")
if batch_type not in VALID_BATCHES:
    raise ValueError(f"batch_type inválido: {batch_type}")

BRONZE_SCHEMA = f"{CATALOG}.bronze"
VOLUME_PATH = cfg.get("volumes", {}).get("bronze_raw", "/Volumes/workspace/bronze/raw")
file_name_trans = f"{batch_type}_transaction.csv"
file_name_id = f"{batch_type}_identity.csv"
path_trans = f"{VOLUME_PATH.rstrip('/')}/{file_name_trans}"
path_id = f"{VOLUME_PATH.rstrip('/')}/{file_name_id}"
table_name_trans = f"{BRONZE_SCHEMA}.{batch_type}_transaction_raw"
table_name_id = f"{BRONZE_SCHEMA}.{batch_type}_identity_raw"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {BRONZE_SCHEMA}")

def read_csv(path, dataset_type):
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    if "TransactionID" not in df.columns:
        raise ValueError(f"TransactionID ausente em {path}")
    df = df.withColumn("dataset_type", F.lit(dataset_type))
    df = df.withColumn("ingestion_timestamp", F.current_timestamp())
    return df

transactions = read_csv(path_trans, "transaction")

try:
    identities = read_csv(path_id, "identity")
except Exception:
    if batch_type != "prod":
        raise
    identities = None

def validate_unique_transactions(df, dataset_name):
    duplicated = df.groupBy("TransactionID").count().filter(F.col("count") > 1).limit(1).count()
    if duplicated:
        raise ValueError(f"TransactionID duplicado em {dataset_name}")

validate_unique_transactions(transactions, "transactions")
if identities is not None:
    validate_unique_transactions(identities, "identity")

def persist_delta(df, table_name, incremental):
    table_exists = spark.catalog.tableExists(table_name)
    if not incremental:
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
        return
    if not table_exists:
        df.write.format("delta").mode("errorifexists").saveAsTable(table_name)
        return
    delta_table = DeltaTable.forName(spark, table_name)
    delta_table.alias("destino").merge(
        df.alias("origem"), "destino.TransactionID = origem.TransactionID"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

is_incremental = batch_type == "prod"
persist_delta(transactions, table_name_trans, incremental=is_incremental)

if identities is not None:
    persist_delta(identities, table_name_id, incremental=is_incremental)
elif not spark.catalog.tableExists(table_name_id):
    transactions.select("TransactionID").limit(0).write.format("delta").mode("errorifexists").saveAsTable(table_name_id)

print(f"Catálogo: {CATALOG}")
print(f"Batch type: {batch_type}")
print(f"Arquivo de transações: {path_trans}")
print(f"Arquivo de identidade: {path_id}")
print(f"Tabela Bronze transações: {table_name_trans}")
print(f"Tabela Bronze identidade: {table_name_id}")
print("Ingestão Bronze concluída.")