# Databricks notebook source
# DBTITLE 1,06_promote_model
# MAGIC %md
# MAGIC # 06 -- Promocao do Challenger para Champion
# MAGIC Promove @challenger a @champion se os notebooks 04 e 05 aprovaram.
# MAGIC Exige `confirm_promote=yes` (widget) para executar a troca.

# COMMAND ----------

import mlflow, logging, datetime, sys
from mlflow.tracking import MlflowClient

import os, sys

PROJECT_ROOT = "/Workspace/Users/eduardolimaofficial410@gmail.com/databricks-fraud-detection"
sys.path.insert(0, f"{PROJECT_ROOT}/src")
from databricks_fraud_model.utils import load_config

cfg = load_config()
UC_MODEL_NAME = cfg["model"]["registered_name"]

logging.getLogger("mlflow.tracking.context.registry").disabled = True
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------

try: should_promote = dbutils.jobs.taskValues.get("promote", default=False)
except Exception: should_promote = False

try: ab_test_pass = dbutils.jobs.taskValues.get("ab_test_pass", default=False)
except Exception: ab_test_pass = False

if not should_promote:
    dbutils.notebook.exit("MANTER_CHAMPION")

if not ab_test_pass:
    dbutils.notebook.exit("AB_TEST_FAILED")

# COMMAND ----------

try:
    old_champion = client.get_model_version_by_alias(UC_MODEL_NAME, "champion").version
except Exception:
    old_champion = None

challenger_version = client.get_model_version_by_alias(UC_MODEL_NAME, "challenger").version

dbutils.widgets.text("confirm_promote", "no")
if dbutils.widgets.get("confirm_promote").lower() != "yes":
    dbutils.notebook.exit("CONFIRMATION_REQUIRED")

client.set_registered_model_alias(UC_MODEL_NAME, "champion", int(challenger_version))
try:
    client.delete_registered_model_alias(UC_MODEL_NAME, "challenger")
except Exception:
    pass

client.set_model_version_tag(UC_MODEL_NAME, challenger_version, "promoted_at", datetime.datetime.now().isoformat())
client.set_model_version_tag(UC_MODEL_NAME, challenger_version, "promoted_from", "challenger")
if old_champion:
    client.set_model_version_tag(UC_MODEL_NAME, challenger_version, "previous_champion_version", str(old_champion))

display(spark.sql(f"SELECT name, version, aliases FROM {UC_MODEL_NAME}.model_versions ORDER BY version DESC LIMIT 5"))