# Databricks notebook source
# DBTITLE 1,03_train_challenger
# MAGIC %md
# MAGIC # 03 - Treino do Challenger (XGBoost + Optuna HPO)
# MAGIC Treina XGBoost com HPO via Optuna, registra no UC como @challenger.
# MAGIC
# MAGIC **Entrada:** `{catalog}.gold.fraud_features_retrain`
# MAGIC **Saida:** Modelo registrado no UC com alias @challenger

# COMMAND ----------

# MAGIC %pip install xgboost optuna scikit-learn==1.6.1 cloudpickle==3.1.1 -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys, logging
import numpy as np, pandas as pd, pickle, json, tempfile
import mlflow, mlflow.pyfunc, optuna
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, brier_score_loss
from scipy.stats import ks_2samp
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

_nb = dbutils.notebook.entry_point.getNotebookPath()
_r = _nb.split("databricks-fraud-detection")[0] + "databricks-fraud-detection"
sys.path.insert(0, f"{_r}/src")
from databricks_fraud_model.utils import load_config

cfg = load_config()
try:
    CATALOG = dbutils.widgets.get("catalog")
except Exception:
    CATALOG = cfg["catalog"]
FEATURE_TABLE = f"{CATALOG}.gold.fraud_features_retrain"
UC_MODEL_NAME = cfg["model"]["registered_name"]
OPTIMAL_THRESHOLD = cfg["model"]["optimal_threshold"]

logging.getLogger("mlflow.tracking.context.registry").disabled = True
mlflow.set_experiment(cfg["experiments"]["hpo"])
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

df_all = spark.read.table(FEATURE_TABLE).toPandas()
exclude_cols = {"TransactionID", "TransactionDT", "target", "split", "class_weight"}
feature_names = [c for c in df_all.columns if c not in exclude_cols]

def get_split(df, split_name):
    s = df[df["split"] == split_name].copy()
    return s[feature_names].copy(), s["target"].values

X_train, y_train = get_split(df_all, "train")
X_test, y_test = get_split(df_all, "test")
X_val, y_val = get_split(df_all, "val")

# Preprocessor
cat_cols = [c for c in feature_names if X_train[c].dtype == "object"]
num_cols = [c for c in feature_names if c not in cat_cols]

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols),
])

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

fraud_ratio = y_train.sum() / len(y_train)
sample_weight = np.where(y_train == 1, 1.0/fraud_ratio, 1.0/(1.0-fraud_ratio))

def compute_metrics(y_true, y_proba):
    y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
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

# MAGIC %md
# MAGIC ## Optuna HPO

# COMMAND ----------

N_TRIALS = 10
optuna.logging.set_verbosity(optuna.logging.WARNING)

with mlflow.start_run(run_name="Challenger_Optuna_HPO"):
    mlflow.set_tags({"model_type": "XGBoost", "pipeline": "challenger", "optimization_metric": "pr_auc"})
    mlflow.log_params({"n_trials": N_TRIALS, "random_state": 42})

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        }
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            model = XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                                  tree_method="hist", early_stopping_rounds=50,
                                  random_state=42, n_jobs=-1, verbosity=0, **params)
            model.fit(X_train_proc, y_train, sample_weight=sample_weight,
                      eval_set=[(X_val_proc, y_val)], verbose=False)
            val_proba = model.predict_proba(X_val_proc)[:, 1]
            mlflow.log_params(params)
            mlflow.log_metrics(compute_metrics(y_val, val_proba))
            trial.set_user_attr("best_iteration", getattr(model, "best_iteration", None))
            return average_precision_score(y_val, val_proba)

    study = optuna.create_study(direction="maximize", study_name="challenger_xgb",
                                sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_iter = study.best_trial.user_attrs.get("best_iteration")
    final_params = {**best_params, "n_estimators": int(best_iter) + 1 if best_iter else int(best_params["n_estimators"])}
    mlflow.log_params({f"best_{k}": v for k, v in final_params.items()})
    mlflow.log_metric("best_validation_pr_auc", float(study.best_value))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treino final + registro @challenger

# COMMAND ----------

best_model = XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                           tree_method="hist", random_state=42, n_jobs=-1, verbosity=0, **final_params)
best_model.fit(X_train_proc, y_train, sample_weight=sample_weight, verbose=False)

test_proba = best_model.predict_proba(X_test_proc)[:, 1]
test_metrics = compute_metrics(y_test, test_proba)

mlflow.set_experiment(cfg["experiments"]["deploy"])
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_MODEL_NAME.rsplit('.', 1)[0]}")

class FraudDetectionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle, json
        with open(context.artifacts["preprocessor"], "rb") as f: self.preprocessor = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f: self.model = pickle.load(f)
        with open(context.artifacts["config"], "r") as f:
            config = json.load(f)
            self.threshold = config["threshold"]
            self.feature_names = config["feature_names"]

    def predict(self, context, model_input, params=None):
        X = model_input[self.feature_names]
        proba = self.model.predict_proba(self.preprocessor.transform(X))[:, 1]
        return pd.DataFrame({"probability_fraud": proba, "is_fraud": (proba >= self.threshold).astype(int)})

with tempfile.TemporaryDirectory() as tmp:
    for name, obj in [("preprocessor", preprocessor), ("model", best_model)]:
        with open(os.path.join(tmp, f"{name}.pkl"), "wb") as f: pickle.dump(obj, f)
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump({"threshold": OPTIMAL_THRESHOLD, "feature_names": feature_names}, f)

    sample = X_test[feature_names].head(100)
    sample_proba = best_model.predict_proba(preprocessor.transform(sample))[:, 1]
    signature = infer_signature(sample, pd.DataFrame({
        "probability_fraud": sample_proba,
        "is_fraud": (sample_proba >= OPTIMAL_THRESHOLD).astype(int)}))

    with mlflow.start_run(run_name="challenger_pyfunc_deploy") as run:
        mlflow.log_params({"pipeline": "challenger", "optimal_threshold": OPTIMAL_THRESHOLD, **final_params})
        for k, v in test_metrics.items(): mlflow.log_metric(f"test_{k}", v)
        mlflow.pyfunc.log_model(
            python_model=FraudDetectionModel(), artifact_path="model",
            signature=signature, input_example=sample.head(3),
            artifacts={"preprocessor": os.path.join(tmp, "preprocessor.pkl"),
                       "model": os.path.join(tmp, "model.pkl"),
                       "config": os.path.join(tmp, "config.json")},
            pip_requirements=["mlflow==3.8.1", "xgboost", "scikit-learn==1.6.1", "pandas", "numpy", "cloudpickle==3.1.1"],
            registered_model_name=UC_MODEL_NAME,
        )
        challenger_run_id = run.info.run_id

client = MlflowClient(registry_uri="databricks-uc")
challenger_version = max(client.search_model_versions(f"name='{UC_MODEL_NAME}'"), key=lambda v: int(v.version)).version
client.set_registered_model_alias(UC_MODEL_NAME, "challenger", int(challenger_version))
for tag, val in [("algorithm", "XGBoost"), ("pipeline", "challenger"),
                 ("test_pr_auc", f"{test_metrics['pr_auc']:.4f}"),
                 ("test_f1_score", f"{test_metrics['f1_score']:.4f}")]:
    client.set_model_version_tag(UC_MODEL_NAME, challenger_version, tag, val)

display(pd.DataFrame([test_metrics]))