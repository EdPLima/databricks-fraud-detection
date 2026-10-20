"""
Testes de contrato de dados e features.

Validam que o contrato de features (MODEL_FEATURES) e consistente,
versionado, e que as funcoes de validacao funcionam corretamente.
"""

import hashlib

from databricks_fraud_model.features import (
    FEATURE_CONTRACT_HASH,
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURES,
    validate_features,
)
from databricks_fraud_model.utils import load_config

# ============================================================
# 1. CONTRATO DE FEATURES — VERSIONAMENTO
# ============================================================

def test_feature_contract_version_is_string():
    """O contrato deve ter uma versao em formato string."""
    assert isinstance(FEATURE_CONTRACT_VERSION, str)
    assert len(FEATURE_CONTRACT_VERSION) > 0


def test_feature_contract_hash_is_stable():
    """O hash do contrato deve ser deterministico."""
    expected = hashlib.sha256(
        ",".join(MODEL_FEATURES).encode()
    ).hexdigest()[:16]
    assert FEATURE_CONTRACT_HASH == expected


def test_feature_contract_hash_changes_with_features():
    """Modificar a lista de features deve mudar o hash."""
    modified = MODEL_FEATURES + ["new_feature"]
    modified_hash = hashlib.sha256(
        ",".join(modified).encode()
    ).hexdigest()[:16]
    assert modified_hash != FEATURE_CONTRACT_HASH


# ============================================================
# 2. FUNCAO validate_features()
# ============================================================

def test_validate_features_all_present():
    """validate_features retorna True quando todas features estao presentes."""
    df_cols = list(MODEL_FEATURES) + ["TransactionID", "TransactionDT"]
    valid, missing = validate_features(df_cols)
    assert valid is True
    assert missing == []


def test_validate_features_missing_some():
    """validate_features detecta features faltantes."""
    partial = MODEL_FEATURES[:100]  # faltam 52
    valid, missing = validate_features(partial)
    assert valid is False
    assert len(missing) == 52


def test_validate_features_empty_columns():
    """validate_features com lista vazia retorna todas como faltantes."""
    valid, missing = validate_features([])
    assert valid is False
    assert len(missing) == 152


def test_validate_features_custom_list():
    """validate_features aceita lista customizada."""
    custom = ["a", "b", "c"]
    valid, missing = validate_features(["a", "b"], model_features=custom)
    assert valid is False
    assert missing == ["c"]


# ============================================================
# 3. CONFIG — RESOLUCAO DE PATHS E NOME DO MODELO
# ============================================================

def test_config_model_name_has_catalog_prefix():
    """load_config deve prefixar o catalogo no nome do modelo."""
    cfg = load_config()
    assert cfg["model"]["registered_name"].startswith(cfg["catalog"])
    assert "xgboost" in cfg["model"]["registered_name"]


def test_config_experiment_paths_are_absolute():
    """load_config deve resolver paths de experimentos para absolutos."""
    cfg = load_config()
    for key, path in cfg["experiments"].items():
        assert path.startswith("/"), f"Experiment path {key} is not absolute: {path}"


def test_config_experiment_paths_contain_project_name():
    """Paths de experimentos devem conter o nome do projeto."""
    cfg = load_config()
    for key, path in cfg["experiments"].items():
        assert "databricks-fraud-detection" in path or "fraud" in path


# ============================================================
# 4. CONTRATO DE DADOS — SCHEMA DA FEATURE TABLE
# ============================================================

def test_model_features_are_unique():
    """Nenhuma feature pode aparecer duas vezes no contrato."""
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))


def test_model_features_count_is_152():
    """O contrato deve ter exatamente 152 features."""
    assert len(MODEL_FEATURES) == 152


def test_model_features_are_strings():
    """Todas as features devem ser strings."""
    for f in MODEL_FEATURES:
        assert isinstance(f, str), f"Feature {f} is not a string"
        assert len(f) > 0, "Empty feature name found"


def test_model_features_no_whitespace():
    """Nomes de features nao devem ter espacos."""
    for f in MODEL_FEATURES:
        assert " " not in f, f"Feature '{f}' contains whitespace"
