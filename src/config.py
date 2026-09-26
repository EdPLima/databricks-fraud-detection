"""
Loader central de configuracao do projeto de deteccao de fraude.

Le o arquivo config/project_config.yml e expoe as configuracoes
como constantes Python para todos os notebooks do projeto.

Uso nos notebooks:
    import sys
    sys.path.append("/Workspace/Users/eduardolimaofficial410@gmail.com/databricks-fraud-detection/src")
    from config import load_config
    cfg = load_config()
    FEATURE_STORE_TABLE = cfg["feature_store"]["table"]
"""

import yaml
import os

# Caminho do arquivo de configuracao (relativo a raiz do projeto)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "project_config.yml")


def load_config(config_path: str | None = None) -> dict:
    """
    Carrega a configuracao do projeto a partir do YAML.

    Args:
        config_path: Caminho alternativo para o arquivo de config.
                     Se None, usa config/project_config.yml na raiz do projeto.

    Returns:
        dict com todas as configuracoes do projeto.
    """
    path = config_path or _CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get(key: str, default=None):
    """
    Atalho: retorna um valor top-level do config.
    Ex: get("feature_store")["table"]
    """
    return load_config().get(key, default)
