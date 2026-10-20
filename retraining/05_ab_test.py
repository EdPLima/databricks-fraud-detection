# Databricks notebook source
# DBTITLE 1,05_ab_test
# MAGIC %md
# MAGIC # 05 -- A/B Test (Champion vs Challenger)
# MAGIC Roteia dados de producao: grupo A via endpoint REST (champion), grupo B via pyfunc (challenger).
# MAGIC Compara distribuicoes de score, deteccao e latencia.
# MAGIC
# MAGIC **Entrada:** `{catalog}.gold.fraud_features_prd`
# MAGIC **Saida:** taskValues `ab_test_pass` (bool)

# COMMAND ----------

# MAGIC %pip install lightgbm -q
# MAGIC
# MAGIC import os, sys, logging
# MAGIC import numpy as np, pandas as pd, mlflow, time
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC _nb = dbutils.notebook.entry_point.getNotebookPath()
# MAGIC _r = _nb.split("databricks-fraud-detection")[0] + "databricks-fraud-detection"
# MAGIC sys.path.insert(0, f"{_r}/src")
# MAGIC from databricks_fraud_model.utils import load_config
# MAGIC from databricks_fraud_model import MODEL_FEATURES
# MAGIC
# MAGIC cfg = load_config()
# MAGIC try:
# MAGIC     CATALOG = dbutils.widgets.get("catalog")
# MAGIC except Exception:
# MAGIC     CATALOG = cfg["catalog"]
# MAGIC UC_MODEL_NAME = cfg["model"]["registered_name"]
# MAGIC FEATURES_PRD = f"{CATALOG}.gold.fraud_features_prd"
# MAGIC ENDPOINT = "model_fraud"
# MAGIC SAMPLE_SIZE = 1000
# MAGIC SPLIT_RATIO = 0.5
# MAGIC
# MAGIC logging.getLogger("mlflow.tracking.context.registry").disabled = True
# MAGIC
# MAGIC try:
# MAGIC     should_proceed = dbutils.jobs.taskValues.get("promote", default=True)
# MAGIC except Exception:
# MAGIC     should_proceed = True
# MAGIC
# MAGIC if not should_proceed:
# MAGIC     dbutils.jobs.taskValues.set("ab_test_pass", False)
# MAGIC     dbutils.notebook.exit("MANTER_CHAMPION")

# COMMAND ----------

df_prod = spark.sql(f"SELECT * FROM {FEATURES_PRD} LIMIT {SAMPLE_SIZE * 2}").toPandas()
features = [f for f in MODEL_FEATURES if f in df_prod.columns]

champion = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@champion")
try:
    challenger = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@challenger")
except Exception:
    dbutils.jobs.taskValues.set("ab_test_pass", False)
    dbutils.notebook.exit("NO_CHALLENGER")

np.random.seed(42)
df_prod["ab_group"] = np.where(np.random.rand(len(df_prod)) < SPLIT_RATIO, "A", "B")
group_a = df_prod[df_prod["ab_group"] == "A"].copy()
group_b = df_prod[df_prod["ab_group"] == "B"].copy()

# COMMAND ----------

# Grupo A: scoring via endpoint REST (champion em producao)
def score_via_endpoint(df, features, endpoint_name):
    w = WorkspaceClient()
    records = [{k: (None if pd.isna(v) else v) for k, v in r.items()}
               for r in df[features].to_dict(orient='records')]
    t0 = time.perf_counter()
    resp = w.api_client.do('POST', f'/serving-endpoints/{endpoint_name}/invocations',
                           body={'dataframe_records': records},
                           headers={'Content-Type': 'application/json'})
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    preds = pd.DataFrame(resp['predictions'])
    preds['latency_ms'] = latency_ms
    return preds

preds_a = score_via_endpoint(group_a, features, ENDPOINT)
group_a["champion_proba"] = preds_a["probability_fraud"].values
group_a["champion_pred"] = preds_a["is_fraud"].values
group_a["latency_ms"] = preds_a["latency_ms"].values

# Grupo B: scoring via pyfunc (challenger do UC)
t0 = time.perf_counter()
preds_b = challenger.predict(group_b[features])
challenger_latency = round((time.perf_counter() - t0) * 1000, 2)
group_b["challenger_proba"] = preds_b["probability_fraud"].values
group_b["challenger_pred"] = preds_b["is_fraud"].values
group_b["latency_ms"] = challenger_latency

# COMMAND ----------

results = pd.DataFrame({
    "metric": ["mean_proba", "median_proba", "std_proba", "fraud_rate_pred",
               "min_proba", "max_proba", "p95_proba", "n_records", "latency_ms"],
    "champion_A": [
        group_a["champion_proba"].mean(), group_a["champion_proba"].median(),
        group_a["champion_proba"].std(), group_a["champion_pred"].mean(),
        group_a["champion_proba"].min(), group_a["champion_proba"].max(),
        group_a["champion_proba"].quantile(0.95), len(group_a), group_a["latency_ms"].mean(),
    ],
    "challenger_B": [
        group_b["challenger_proba"].mean(), group_b["challenger_proba"].median(),
        group_b["challenger_proba"].std(), group_b["challenger_pred"].mean(),
        group_b["challenger_proba"].min(), group_b["challenger_proba"].max(),
        group_b["challenger_proba"].quantile(0.95), len(group_b), group_b["latency_ms"].mean(),
    ],
})
results["delta"] = results["challenger_B"] - results["champion_A"]
display(results)

# COMMAND ----------

mean_delta = abs(results.loc[results["metric"] == "mean_proba", "delta"].values[0])
fraud_delta = results.loc[results["metric"] == "fraud_rate_pred", "delta"].values[0]
p95_delta = results.loc[results["metric"] == "p95_proba", "delta"].values[0]

ab_test_pass = (mean_delta < 0.05) and (fraud_delta >= -0.02) and (p95_delta > -0.05)

dbutils.jobs.taskValues.set("ab_test_pass", ab_test_pass)
dbutils.jobs.taskValues.set("ab_mean_delta", float(mean_delta))
dbutils.jobs.taskValues.set("ab_fraud_rate_delta", float(fraud_delta))