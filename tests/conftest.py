import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = str(PROJECT_ROOT / "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


@pytest.fixture
def spark():
    try:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()
    except Exception:
        pytest.skip("Spark session not available in this context")


@pytest.fixture
def sample_df():
    import pandas as pd
    return pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionAmt": [100.0, 200.0, 50.0],
        "TransactionDT": [3600, 7200, 10800],
        "ProductCD": ["W", "H", "C"],
        "card4": ["visa", "mastercard", "visa"],
        "card6": ["debit", "credit", "debit"],
        "P_emaildomain": ["gmail.com", "yahoo.com", "hotmail.com"],
        "R_emaildomain": ["gmail.com", "yahoo.com", "unknown.com"],
        "DeviceType": ["desktop", None, "mobile"],
        "DeviceInfo": ["Windows", "iOS", "Android"],
        "addr1": [100.0, 200.0, 100.0],
        "dist1": [10.0, 20.0, None],
        "card1": [1000, 2000, 1000],
        "card2": [100.0, 200.0, 300.0],
        "card3": [150.0, 150.0, 180.0],
        "card5": [100.0, 200.0, 100.0],
    })
