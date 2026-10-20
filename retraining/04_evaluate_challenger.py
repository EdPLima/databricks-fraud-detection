# Databricks notebook source
# DBTITLE 1,04_evaluate_challenger
# MAGIC %md
# MAGIC # 04 -- Avaliacao Challenger vs Champion
# MAGIC Compara metricas no validation set. Promove se PR-AUC e F1 do challenger forem superiores.
# MAGIC
# MAGIC **Entrada:** `{catalog}.gold.fraud_features_retrain` (split=val)
# MAGIC **Saida:** taskValues `promote` (bool), `pr_auc_delta`, `f1_delta`

# COMMAND ----------

import pandas as pd, mlflow, logging, sys
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, brier_score_loss
from scipy.stats import ks_2samp

import os, sys

PROJECT_ROOT = "/Workspace/Users/eduardolimaofficial410@gmail.com/databricks-fraud-detection"
sys.path.insert(0, f"{PROJECT_ROOT}/src")
from databricks_fraud_model.utils import load_config

cfg = load_config()
try:
    CATALOG = dbutils.widgets.get("catalog")
except Exception:
    CATALOG = cfg["catalog"]
FEATURE_TABLE = f"{CATALOG}.gold.fraud_features_retrain"
UC_MODEL_NAME = cfg["model"]["registered_name"]
THRESHOLD = cfg["model"]["optimal_threshold"]

logging.getLogger("mlflow.tracking.context.registry").disabled = True

# COMMAND ----------

df_val = spark.read.table(FEATURE_TABLE).filter("split = 'val'").toPandas()
exclude = {"TransactionID", "TransactionDT", "target", "split", "class_weight"}
feature_names = [c for c in df_val.columns if c not in exclude]
X_val = df_val[feature_names]
y_val = df_val["target"].values

def compute_metrics(y_true, y_proba):
    y_pred = (y_proba >= THRESHOLD).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "ks_statistic": float(ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }

# COMMAND ----------

champion = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@champion")
try:
    challenger = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@challenger")
except Exception:
    dbutils.jobs.taskValues.set("promote", False)
    dbutils.notebook.exit("NO_CHALLENGER")

champion_proba = champion.predict(X_val)["probability_fraud"].values
challenger_proba = challenger.predict(X_val)["probability_fraud"].values

champion_m = compute_metrics(y_val, champion_proba)
challenger_m = compute_metrics(y_val, challenger_proba)

comparison = pd.DataFrame([
    {"metric": k, "champion": v, "challenger": challenger_m[k],
     "delta": challenger_m[k] - v, "better": challenger_m[k] > v}
    for k, v in champion_m.items()
])
display(comparison)

pr_auc_delta = challenger_m["pr_auc"] - champion_m["pr_auc"]
f1_delta = challenger_m["f1_score"] - champion_m["f1_score"]
promote = pr_auc_delta > 0 and f1_delta > 0

dbutils.jobs.taskValues.set("promote", promote)
dbutils.jobs.taskValues.set("pr_auc_delta", pr_auc_delta)
dbutils.jobs.taskValues.set("f1_delta", f1_delta)