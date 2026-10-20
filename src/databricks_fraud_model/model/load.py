"""
Carrega o modelo pyfunc do Unity Catalog para batch scoring.
Este modulo e usado quando o scoring e feito localmente,

"""

import mlflow.pyfunc

from databricks_fraud_model.utils import load_config

# Carrega o nome do modelo do config

_cfg = load_config()
UC_MODEL_NAME = _cfg["model"]["registered_name"]
ALIAS = "champion"  # Alias do modelo em producao no Unity Catalog


def load_model(model_name=UC_MODEL_NAME, alias=ALIAS):
    """
    Carrega o modelo pyfunc do Unity Catalog.

    Usa o formato models:/<catalog>.<schema>.<model>@<alias> para resolver
    a versao atual do modelo pelo alias (ex: @champion aponta para a versao
    em producao, @challenger aponta para a versao candidata).

    Args:
        model_name: nome completo do modelo no UC (catalog.schema.model_name).
                   Default: valor do config (fraud_detection_dev.models.lgbm_fraud_model).
        alias:     alias do modelo no UC. Default: 'champion'.

    Returns:
        mlflow pyfunc model — chamar .predict(df[features]) retorna predicoes
        como DataFrame com colunas 'probability_fraud' e 'is_fraud'.
    """
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
