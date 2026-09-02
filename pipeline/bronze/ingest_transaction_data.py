# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Bronze Layer - Transaction Data Ingestion
# MAGIC %md
# MAGIC # Bronze Layer - Transaction Data Ingestion
# MAGIC
# MAGIC This notebook ingests raw CSV transaction data into Unity Catalog bronze tables.
# MAGIC
# MAGIC **Input:** CSV files from `/Volumes/workspace/bronze/raw/`
# MAGIC **Output:** Delta tables in `fraud_detection_dev.bronze`
# MAGIC
# MAGIC **Tables Created:**
# MAGIC * `train_transaction_raw` - Training transaction data (590K records)
# MAGIC * `test_transaction_raw` - Test transaction data (507K records)

# COMMAND ----------

# DBTITLE 1,Configuration
# Configuration
from pyspark.sql import functions as F

# Environment - change this for hml/prd deployments
ENVIRONMENT = "dev"  # dev, hml, prd
CATALOG = f"fraud_detection_{ENVIRONMENT}"

# Source data path
VOLUME_PATH = "/Volumes/workspace/bronze/raw"

# Target schema
BRONZE_SCHEMA = f"{CATALOG}.bronze"

print(f"Environment: {ENVIRONMENT}")
print(f"Target Catalog: {CATALOG}")
print(f"Bronze Schema: {BRONZE_SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Ingest Train Transaction Data
from pyspark.sql import functions as F

TABLE_NAME = f"{BRONZE_SCHEMA}.train_transaction_raw"

# =============================================================================
# 1. Read raw CSV
# =============================================================================
df_train_transaction = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{VOLUME_PATH}/train_transaction.csv")
    .withColumn("ingestion_timestamp", F.current_timestamp())
    .withColumn("dataset_type", F.lit("train"))
)

# =============================================================================
# 2. Write Bronze Delta table
# =============================================================================
(
    df_train_transaction
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

record_count = spark.table(TABLE_NAME).count()

print(
    f"✓ {TABLE_NAME} created: "
    f"{record_count:,} records"
)

# =============================================================================
# 3. Configure Delta data skipping statistics
# =============================================================================
spark.sql(f"""
    ALTER TABLE {TABLE_NAME}
    SET TBLPROPERTIES (
        'delta.dataSkippingStatsColumns' = 'TransactionID'
    )
""")

# =============================================================================
# 4. Recompute Delta statistics for existing files
# =============================================================================
print("  Computing Delta statistics...")

spark.sql(f"""
    ANALYZE TABLE {TABLE_NAME}
    COMPUTE DELTA STATISTICS
""")

# Optional: SQL optimizer statistics
spark.sql(f"""
    ANALYZE TABLE {TABLE_NAME}
    COMPUTE STATISTICS
""")

# =============================================================================
# 5. Optimize physical layout
# =============================================================================
print("  Optimizing table...")

spark.sql(f"""
    OPTIMIZE {TABLE_NAME}
    ZORDER BY (TransactionID)
""")

print("✓ Optimization complete")

# COMMAND ----------

# DBTITLE 1,Ingest Test Transaction Data
from pyspark.sql import functions as F

TABLE_NAME = f"{BRONZE_SCHEMA}.test_transaction_raw"

# =============================================================================
# 1. Read raw CSV
# =============================================================================
df_test_transaction = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{VOLUME_PATH}/test_transaction.csv")
    .withColumn("ingestion_timestamp", F.current_timestamp())
    .withColumn("dataset_type", F.lit("test"))
)

# =============================================================================
# 2. Write Bronze Delta table
# =============================================================================
(
    df_test_transaction
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

record_count = spark.table(TABLE_NAME).count()

print(
    f"✓ {TABLE_NAME} created: "
    f"{record_count:,} records"
)

# =============================================================================
# 3. Configure Delta data skipping statistics
# =============================================================================
spark.sql(f"""
    ALTER TABLE {TABLE_NAME}
    SET TBLPROPERTIES (
        'delta.dataSkippingStatsColumns' = 'TransactionID'
    )
""")

# =============================================================================
# 4. Recompute Delta statistics
# =============================================================================
print("  Computing Delta statistics...")

spark.sql(f"""
    ANALYZE TABLE {TABLE_NAME}
    COMPUTE DELTA STATISTICS
""")

# Optional: statistics used by the SQL optimizer
spark.sql(f"""
    ANALYZE TABLE {TABLE_NAME}
    COMPUTE STATISTICS
""")

# =============================================================================
# 5. Optimize physical layout
# =============================================================================
print("  Optimizing table...")

spark.sql(f"""
    OPTIMIZE {TABLE_NAME}
    ZORDER BY (TransactionID)
""")

print("✓ Optimization complete")

# COMMAND ----------

# DBTITLE 1,Verify Transaction Tables
# MAGIC %sql
# MAGIC -- Verify transaction tables
# MAGIC SELECT 
# MAGIC   'train_transaction_raw' as table_name,
# MAGIC   COUNT(*) as total_records,
# MAGIC   COUNT(DISTINCT TransactionID) as unique_transactions,
# MAGIC   SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count,
# MAGIC   ROUND(AVG(TransactionAmt), 2) as avg_amount
# MAGIC FROM fraud_detection_dev.bronze.train_transaction_raw
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 
# MAGIC   'test_transaction_raw',
# MAGIC   COUNT(*),
# MAGIC   COUNT(DISTINCT TransactionID),
# MAGIC   NULL,
# MAGIC   ROUND(AVG(TransactionAmt), 2)
# MAGIC FROM fraud_detection_dev.bronze.test_transaction_raw;