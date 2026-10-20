"""
Testes para o modulo de monitoring (drift e observability).

Cobre:
- calculate_psi: PSI para distribuicoes categoricas
- calculate_ks_drift: KS test para features numericas
- calculate_categorical_drift: PSI em batch
- calculate_prediction_drift: drift na distribuicao de scores
- calculate_latency_metrics: metricas de latencia
- calculate_error_rate: taxa de erro
- detect_cold_starts: classificacao cold/warm start
- summarize_endpoint_health: sumario consolidado
"""

import numpy as np
import pandas as pd
import pytest

from databricks_fraud_model.monitoring.drift import (
    calculate_psi,
    calculate_ks_drift,
    calculate_categorical_drift,
    calculate_prediction_drift,
)
from databricks_fraud_model.monitoring.observability import (
    calculate_latency_metrics,
    calculate_error_rate,
    detect_cold_starts,
    summarize_endpoint_health,
)


# ============================================================
# TESTS: PSI (drift categorico)
# ============================================================

class TestCalculatePSI:
    """Testes para calculate_psi."""

    def test_psi_identical_distributions(self):
        """Distribuicoes identicas devem ter PSI proximo de zero."""
        train = pd.Series({"a": 50, "b": 30, "c": 20})
        prod = pd.Series({"a": 50, "b": 30, "c": 20})
        psi = calculate_psi(train, prod)
        assert psi < 0.01, f"PSI for identical distributions should be ~0, got {psi}"

    def test_psi_different_distributions(self):
        """Distribuicoes muito diferentes devem ter PSI alto (> 0.25)."""
        train = pd.Series({"a": 90, "b": 10})
        prod = pd.Series({"a": 10, "b": 90})
        psi = calculate_psi(train, prod)
        assert psi > 0.25, f"PSI for very different distributions should be > 0.25, got {psi}"

    def test_psi_missing_category_in_prod(self):
        """Categoria ausente em producao nao deve causar erro."""
        train = pd.Series({"a": 50, "b": 30, "c": 20})
        prod = pd.Series({"a": 50, "b": 50})  # 'c' ausente
        psi = calculate_psi(train, prod)
        assert psi > 0  # Deve haver algum drift

    def test_psi_new_category_in_prod(self):
        """Nova categoria em producao nao deve causar erro."""
        train = pd.Series({"a": 50, "b": 50})
        prod = pd.Series({"a": 40, "b": 40, "c": 20})  # 'c' nova
        psi = calculate_psi(train, prod)
        assert psi > 0


# ============================================================
# TESTS: KS DRIFT (features numericas)
# ============================================================

class TestCalculateKSDrift:
    """Testes para calculate_ks_drift."""

    def test_ks_no_drift(self):
        """Mesma distribuicao: nenhuma feature deve ter drift."""
        np.random.seed(42)
        data = {f"feature_{i}": np.random.normal(0, 1, 1000) for i in range(5)}
        train_df = pd.DataFrame(data)
        prod_df = pd.DataFrame(data)

        result = calculate_ks_drift(train_df, prod_df, list(data.keys()))

        assert len(result) == 5
        assert not result["drift"].any(), "Identical distributions should not have drift"

    def test_ks_with_drift(self):
        """Distribuicoes diferentes: deve detectar drift."""
        np.random.seed(42)
        train_df = pd.DataFrame({"feature_1": np.random.normal(0, 1, 1000)})
        prod_df = pd.DataFrame({"feature_1": np.random.normal(5, 1, 1000)})  # media muito diferente

        result = calculate_ks_drift(train_df, prod_df, ["feature_1"])

        assert len(result) == 1
        assert result["drift"].iloc[0], "Very different distributions should have drift"
        assert result["ks_statistic"].iloc[0] > 0.5

    def test_ks_skip_missing_columns(self):
        """Colunas ausentes devem ser ignoradas, nao causar erro."""
        train_df = pd.DataFrame({"feature_1": [1, 2, 3]})
        prod_df = pd.DataFrame({"feature_2": [1, 2, 3]})

        result = calculate_ks_drift(train_df, prod_df, ["feature_1", "feature_2", "feature_3"])

        assert len(result) == 0  # Nenhuma feature presente em ambos

    def test_ks_skip_insufficient_samples(self):
        """Features com poucas observacoes devem ser ignoradas."""
        train_df = pd.DataFrame({"feature_1": [1, 2]})  # < min_samples (10)
        prod_df = pd.DataFrame({"feature_1": [1, 2]})

        result = calculate_ks_drift(train_df, prod_df, ["feature_1"], min_samples=10)

        assert len(result) == 0  # Ignorada por insuficiencia de amostras


# ============================================================
# TESTS: CATEGORICAL DRIFT (PSI em batch)
# ============================================================

class TestCalculateCategoricalDrift:
    """Testes para calculate_categorical_drift."""

    def test_categorical_drift_identical(self):
        """Distribuicoes identicas nao devem ter drift."""
        train_df = pd.DataFrame({"cat_1": ["a", "b", "a", "b"] * 100})
        prod_df = pd.DataFrame({"cat_1": ["a", "b", "a", "b"] * 100})

        result = calculate_categorical_drift(train_df, prod_df, ["cat_1"])

        assert len(result) == 1
        assert not result["drift"].iloc[0]

    def test_categorical_drift_different(self):
        """Distribuicoes diferentes devem ter drift."""
        train_df = pd.DataFrame({"cat_1": ["a"] * 200})
        prod_df = pd.DataFrame({"cat_1": ["b"] * 200})

        result = calculate_categorical_drift(train_df, prod_df, ["cat_1"])

        assert len(result) == 1
        assert result["drift"].iloc[0]


# ============================================================
# TESTS: PREDICTION DRIFT
# ============================================================

class TestCalculatePredictionDrift:
    """Testes para calculate_prediction_drift."""

    def test_prediction_drift_with_scores(self):
        """Deve calcular estatisticas de score de producao."""
        train_df = pd.DataFrame({"target": [0, 1, 0, 1, 0]})
        prod_df = pd.DataFrame({
            "probability_fraud": [0.1, 0.9, 0.2, 0.8, 0.1],
            "is_fraud": [0, 1, 0, 1, 0],
        })

        result = calculate_prediction_drift(train_df, prod_df)

        assert result["n_predictions"] == 5
        assert "mean_score" in result
        assert "median_score" in result
        assert "std_score" in result
        assert result["predicted_fraud_rate"] == 0.4
        assert result["train_fraud_rate"] == 0.4

    def test_prediction_drift_no_target(self):
        """Sem target no treino: apenas estatisticas de score."""
        train_df = pd.DataFrame({"some_col": [1, 2, 3]})
        prod_df = pd.DataFrame({
            "probability_fraud": [0.1, 0.9, 0.2],
            "is_fraud": [0, 1, 0],
        })

        result = calculate_prediction_drift(train_df, prod_df)

        assert "train_fraud_rate" not in result
        assert result["predicted_fraud_rate"] is not None


# ============================================================
# TESTS: LATENCY METRICS
# ============================================================

class TestCalculateLatencyMetrics:
    """Testes para calculate_latency_metrics."""

    def test_latency_basic(self):
        """Deve calcular avg, p50, p95, p99 corretamente."""
        events = pd.DataFrame({
            "latency_ms": [10, 20, 30, 40, 50, 100, 200, 500],
            "n_records": [100, 100, 100, 100, 100, 100, 100, 100],
        })

        result = calculate_latency_metrics(events)

        assert result["n_requests"] == 8
        assert result["total_predictions"] == 800
        assert result["avg_ms"] == pytest.approx(events["latency_ms"].mean())
        assert result["p50_ms"] == pytest.approx(events["latency_ms"].median())
        assert result["avg_per_record_ms"] > 0

    def test_latency_empty_dataframe(self):
        """DataFrame vazio nao deve causar erro."""
        events = pd.DataFrame({"latency_ms": [], "n_records": []})

        result = calculate_latency_metrics(events)

        assert result["n_requests"] == 0


# ============================================================
# TESTS: ERROR RATE
# ============================================================

class TestCalculateErrorRate:
    """Testes para calculate_error_rate."""

    def test_error_rate_all_success(self):
        events = pd.DataFrame({"status": ["success"] * 10})
        result = calculate_error_rate(events)
        assert result["error_rate"] == 0.0
        assert result["n_errors"] == 0
        assert result["n_success"] == 10

    def test_error_rate_mixed(self):
        events = pd.DataFrame({"status": ["success", "error", "success", "error"]})
        result = calculate_error_rate(events)
        assert result["error_rate"] == 0.5
        assert result["n_errors"] == 2

    def test_error_rate_no_status_column(self):
        events = pd.DataFrame({"other_col": [1, 2, 3]})
        result = calculate_error_rate(events)
        assert result["error_rate"] == 0.0


# ============================================================
# TESTS: COLD START DETECTION
# ============================================================

class TestDetectColdStarts:
    """Testes para detect_cold_starts."""

    def test_cold_start_detection(self):
        """Primeira requisicao e gaps > 5min devem ser cold start."""
        events = pd.DataFrame({
            "event_timestamp": pd.to_datetime([
                "2026-01-01 10:00:00",
                "2026-01-01 10:01:00",  # 1 min gap = warm
                "2026-01-01 10:10:00",  # 9 min gap = cold
                "2026-01-01 10:11:00",  # 1 min gap = warm
            ]),
            "latency_ms": [50, 30, 60, 35],
        })

        result = detect_cold_starts(events, gap_threshold_minutes=5)

        assert result["start_type"].iloc[0] == "cold"  # primeira requisicao
        assert result["start_type"].iloc[1] == "warm"
        assert result["start_type"].iloc[2] == "cold"  # gap > 5 min
        assert result["start_type"].iloc[3] == "warm"


# ============================================================
# TESTS: SUMMARIZE ENDPOINT HEALTH
# ============================================================

class TestSummarizeEndpointHealth:
    """Testes para summarize_endpoint_health."""

    def test_summary_contains_all_sections(self):
        events = pd.DataFrame({
            "event_timestamp": pd.to_datetime([
                "2026-01-01 10:00:00",
                "2026-01-01 10:01:00",
            ]),
            "latency_ms": [50, 30],
            "n_records": [100, 100],
            "status": ["success", "success"],
        })

        result = summarize_endpoint_health(events)

        assert "latency" in result
        assert "errors" in result
        assert "cold_starts" in result
        assert "warm_starts" in result
        assert result["latency"]["n_requests"] == 2
        assert result["errors"]["error_rate"] == 0.0

    def test_summary_empty_dataframe(self):
        events = pd.DataFrame()
        result = summarize_endpoint_health(events)
        assert result == {"latency": {}, "errors": {}, "cold_starts": {}, "warm_starts": {}}