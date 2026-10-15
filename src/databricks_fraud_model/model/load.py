"""
Carrega o modelo pyfunc do Unity Catalog para batch scoring.
"""

import mlflow.pyfunc

UC_MODEL_NAME = "fraud_detection_dev.models.xgboost_fraud_model"
ALIAS = "champion"


def load_model(model_name=UC_MODEL_NAME, alias=ALIAS):
    """
    Carrega o modelo pyfunc do Unity Catalog.

    Returns:
        mlflow pyfunc model — chamar .predict(df[features]) retorna predicoes.
    """
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")