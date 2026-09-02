# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Bronze Layer - Identity Data Ingestion
# MAGIC %md
# MAGIC # Bronze Layer - Identity Data Ingestion
# MAGIC
# MAGIC This notebook ingests raw CSV identity data into Unity Catalog bronze tables.
# MAGIC
# MAGIC **Input:** CSV files from `/Volumes/workspace/bronze/raw/`
# MAGIC **Output:** Delta tables in `fraud_detection_dev.bronze`
# MAGIC
# MAGIC **Tables Created:**
# MAGIC * `train_identity_raw` - Training identity data (144K records)
# MAGIC * `test_identity_raw` - Test identity data (142K records)
# MAGIC
# MAGIC **Note:** Identity data is only available for ~24% of transactions (device fingerprinting limitations).

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

# DBTITLE 1,Ingest Train Identity Data
from pyspark.sql import functions as F

TABLE_NAME = f"{BRONZE_SCHEMA}.train_identity_raw"

# =============================================================================
# 1. Read raw CSV
# =============================================================================
df_train_identity = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{VOLUME_PATH}/train_identity.csv")
    .withColumn("ingestion_timestamp", F.current_timestamp())
    .withColumn("dataset_type", F.lit("train"))
)

# =============================================================================
# 2. Write Bronze Delta table
# =============================================================================
(
    df_train_identity
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
# 3. Configure columns used for Delta data skipping / ZORDER
# =============================================================================
spark.sql(f"""
    ALTER TABLE {TABLE_NAME}
    SET TBLPROPERTIES (
        'delta.dataSkippingStatsColumns' =
        'TransactionID,DeviceType,DeviceInfo'
    )
""")

# =============================================================================
# 4. Recompute Delta statistics for existing files
# =============================================================================
spark.sql(f"""
    ANALYZE TABLE {TABLE_NAME}
    COMPUTE DELTA STATISTICS
""")

# Optional: statistics used by the SQL query optimizer
spark.sql(f"""
    ANALYZE TABLE {TABLE_NAME}
    COMPUTE STATISTICS
""")

# =============================================================================
# 5. Optimize physical layout
# =============================================================================
spark.sql(f"""
    OPTIMIZE {TABLE_NAME}
    ZORDER BY (TransactionID, DeviceType, DeviceInfo)
""")

print("✓ Optimization complete")

# COMMAND ----------

# DBTITLE 1,Ingest Test Identity Data
from pyspark.sql import functions as F

TABLE_NAME = f"{BRONZE_SCHEMA}.test_identity_raw"

# =============================================================================
# 1. Read raw CSV
# =============================================================================
df_test_identity = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{VOLUME_PATH}/test_identity.csv")
    .withColumn("ingestion_timestamp", F.current_timestamp())
    .withColumn("dataset_type", F.lit("test"))
)

# =============================================================================
# 2. Write Bronze Delta table
# =============================================================================
(
    df_test_identity
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

# Optional: statistics used by the SQL query optimizer
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

# DBTITLE 1,Verify Identity Tables
# MAGIC %sql
# MAGIC -- Verify identity tables and device type distribution
# MAGIC SELECT 
# MAGIC   'train_identity_raw' as table_name,
# MAGIC   COUNT(*) as total_records,
# MAGIC   COUNT(DISTINCT TransactionID) as unique_transactions,
# MAGIC   COUNT(DISTINCT DeviceType) as device_types,
# MAGIC   COUNT(DISTINCT DeviceInfo) as device_info_variants
# MAGIC FROM fraud_detection_dev.bronze.train_identity_raw
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 
# MAGIC   'test_identity_raw',
# MAGIC   COUNT(*),
# MAGIC   COUNT(DISTINCT TransactionID),
# MAGIC   COUNT(DISTINCT DeviceType),
# MAGIC   COUNT(DISTINCT DeviceInfo)
# MAGIC FROM fraud_detection_dev.bronze.test_identity_raw;