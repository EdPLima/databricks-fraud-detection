"""
databricks_fraud_model — package do modelo de deteccao de fraude.

Exports:
- MODEL_FEATURES: lista das 152 features do modelo
- apply_feature_engineering_spark: feature engineering em PySpark
- FEATURES_DROP, MODEL_KEEP: listas de feature selection
- score_batch, score_single, save_predictions: scoring via REST endpoint
- load_model: carrega pyfunc do Unity Catalog
- clean_silver_data: limpeza Silver (dedup, join, padronizacao)
- calculate_ks_drift, calculate_psi, calculate_prediction_drift: drift detection
- calculate_latency_metrics, summarize_endpoint_health: observabilidade
"""

from databricks_fraud_model.features import (
    FREE_DOMAINS,
    MODEL_FEATURES,
    FEATURE_CONTRACT_VERSION,
    FEATURE_CONTRACT_HASH,
    validate_features,
    apply_feature_engineering_spark,
)
from databricks_fraud_model.model.load import load_model
from databricks_fraud_model.model.scoring import (
    save_predictions,
    save_to_bronze,
    score_batch,
    score_single,
)
from databricks_fraud_model.data import clean_silver_data
from databricks_fraud_model.monitoring import (
    calculate_ks_drift,
    calculate_psi,
    calculate_categorical_drift,
    calculate_prediction_drift,
    calculate_latency_metrics,
    calculate_error_rate,
    detect_cold_starts,
    summarize_endpoint_health,
)

__all__ = [
    # Features
    "MODEL_FEATURES",
    "FEATURE_CONTRACT_VERSION",
    "FEATURE_CONTRACT_HASH",
    "validate_features",
    "apply_feature_engineering_spark",
    "FREE_DOMAINS",
    # Model
    "score_batch",
    "score_single",
    "save_predictions",
    "save_to_bronze",
    "load_model",
    # Data (Silver cleaning)
    "clean_silver_data",
    # Monitoring (drift)
    "calculate_ks_drift",
    "calculate_psi",
    "calculate_categorical_drift",
    "calculate_prediction_drift",
    # Monitoring (observability)
    "calculate_latency_metrics",
    "calculate_error_rate",
    "detect_cold_starts",
    "summarize_endpoint_health",
]
