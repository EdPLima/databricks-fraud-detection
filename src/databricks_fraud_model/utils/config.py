"""
Loader central de configuracao do projeto de deteccao de fraude.

Le o arquivo config/project_config.yml e expoe as configuracoes
como constantes Python para todos os notebooks do projeto.

Uso nos notebooks:
    import sys, os
    _nb = dbutils.notebook.entry_point.getNotebookPath()
    for _i in range(len(_nb.split("/")), 0, -1):
        _r = "/".join(_nb.split("/")[:_i])
        if os.path.isdir(f"{_r}/src/databricks_fraud_model"):
            sys.path.insert(0, f"{_r}/src")
            break
    from databricks_fraud_model.utils import load_config
    cfg = load_config()
    FEATURE_STORE_TABLE = cfg["feature_store"]["table"]
"""

import os

import yaml

# ============================================================
# RESOLUCAO DE CAMINHOS DO PROJETO
# ============================================================
# utils/config.py -> utils/ -> databricks_fraud_model/ -> src/ -> project_root
# Esta navegacao garante que o config seja encontrado independente do CWD.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "project_config.yml")


def load_config(config_path: str | None = None) -> dict:
    """
    Carrega a configuracao do projeto a partir do YAML.

    Resolve automaticamente:
    - Caminhos relativos de experimentos (prefixados com _PROJECT_ROOT sem /Workspace)
      para que funcione em Repos, DABs dev/hml/prod sem caminhos hardcoded.
    - Nome do modelo com prefixo do catalogo (ex: catalog.models.name)
      para que o config seja portatil entre ambientes.

    Args:
        config_path: Caminho alternativo para o arquivo de config.
                     Se None, usa config/project_config.yml na raiz do projeto.

    Returns:
        dict com todas as configuracoes do projeto, com paths e nome do
        modelo ja resolvidos para o ambiente atual.
    """
    path = config_path or _CONFIG_PATH
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Resolve paths relativos de experimentos para paths de workspace ---
    # MLflow espera paths de workspace (sem /Workspace), mas o config usa paths
    # relativos ao projeto. Remove /Workspace do _PROJECT_ROOT se presente.
    _root_ws = _PROJECT_ROOT.replace("/Workspace", "", 1) if _PROJECT_ROOT.startswith("/Workspace") else _PROJECT_ROOT
    for key, val in cfg.get("experiments", {}).items():
        if val and not val.startswith("/"):  # Apenas paths relativos
            cfg["experiments"][key] = f"{_root_ws}/{val}"

    # --- Resolve nome do modelo com prefixo do catalogo ---
    # O config tem apenas models.lgbm_fraud_model (sem catalog).
    # load_config() adiciona o prefixo: fraud_detection_dev.models.lgbm_fraud_model
    model_name = cfg.get("model", {}).get("registered_name", "")
    if model_name and not model_name.startswith(cfg.get("catalog", "")):
        cfg["model"]["registered_name"] = f"{cfg['catalog']}.{model_name}"

    return cfg


def get(key: str, default=None):
    """
    Atalho: retorna um valor top-level do config.
    Equivalente a load_config()[key], mas mais conciso.

    Ex: get("feature_store")["table"]
        get("nonexistent", default="fallback")

    Args:
        key:     chave top-level do config YAML.
        default: valor de retorno se a chave nao existir.

    Returns:
        Valor do config ou default.
    """
    return load_config().get(key, default)
