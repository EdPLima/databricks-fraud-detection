from databricks_fraud_model.utils import get, load_config


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)


def test_load_config_has_required_keys():
    cfg = load_config()
    required_keys = ["catalog", "feature_store", "model", "preprocessing"]
    for key in required_keys:
        assert key in cfg, f"Missing required config key: {key}"


def test_catalog_value():
    cfg = load_config()
    assert cfg["catalog"] == "fraud_detection_dev"


def test_feature_store_table():
    cfg = load_config()
    assert "table" in cfg["feature_store"]
    assert cfg["feature_store"]["table"].startswith("fraud_detection_dev.")


def test_model_registered_name():
    cfg = load_config()
    assert "registered_name" in cfg["model"]
    assert "xgboost" in cfg["model"]["registered_name"]


def test_get_helper():
    assert get("catalog") == "fraud_detection_dev"
    assert get("nonexistent_key", default="fallback") == "fallback"
