"""
databricks_fraud_model — package do modelo de deteccao de fraude.

Exports:
- MODEL_FEATURES: lista das 152 features do modelo
- apply_feature_engineering_spark: feature engineering em PySpark
- FEATURES_DROP, MODEL_KEEP: listas de feature selection
- score_batch, score_single, save_predictions: scoring via REST endpoint
- load_model: carrega pyfunc do Unity Catalog
"""

from databricks_fraud_model.features import (
    FEATURES_DROP,
    MODEL_FEATURES,
    MODEL_KEEP,
    REMOVED_QUALITY,
    REMOVED_REDUNDANCY,
    apply_feature_engineering_spark,
)
from databricks_fraud_model.model.load import load_model
from databricks_fraud_model.model.scoring import (
    save_predictions,
    save_to_bronze,
    score_batch,
    score_single,
)

__all__ = [
    "MODEL_FEATURES",
    "apply_feature_engineering_spark",
    "FEATURES_DROP",
    "MODEL_KEEP",
    "REMOVED_QUALITY",
    "REMOVED_REDUNDANCY",
    "score_batch",
    "score_single",
    "save_predictions",
    "save_to_bronze",
    "load_model",
]
