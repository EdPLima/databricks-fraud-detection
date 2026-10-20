import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from databricks_fraud_model.evaluation.evaluation import EvaluationModelFraud

# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def temporal_data():
    """Dataset temporal com 100 amostras, 2 features e target binario.
    Fraudes distribuidas aleatoriamente para garantir 2 classes por fold."""
    np.random.seed(42)
    n = 100
    target = np.zeros(n, dtype=int)
    fraud_idx = np.random.choice(n, size=20, replace=False)
    target[fraud_idx] = 1
    return pd.DataFrame({
        "TransactionDT": np.arange(n) * 100,
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "target": target,
    })


@pytest.fixture
def simple_model():
    return LogisticRegression(random_state=42, max_iter=200)


# ============================================================
# 1. TESTES DE _calculate_metrics
# ============================================================

class TestCalculateMetrics:

    def test_metrics_keys(self, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        metrics = evaluator._calculate_metrics([0, 1, 0, 1], [0.1, 0.9, 0.3, 0.8])
        assert set(metrics.keys()) == {
            "precision", "recall", "f1_score", "pr_auc",
            "roc_auc", "brier_score", "ks",
        }

    def test_perfect_predictions(self, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        metrics = evaluator._calculate_metrics([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0])
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_single_class_returns_nan(self, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        metrics = evaluator._calculate_metrics([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
        assert np.isnan(metrics["roc_auc"])
        assert np.isnan(metrics["ks"])

    def test_threshold_affects_predictions(self, simple_model):
        evaluator_default = EvaluationModelFraud(simple_model, threshold=0.5)
        evaluator_high = EvaluationModelFraud(simple_model, threshold=0.8)
        y_true = [0, 0, 1, 1]
        y_proba = [0.3, 0.6, 0.4, 0.7]
        m_default = evaluator_default._calculate_metrics(y_true, y_proba)
        m_high = evaluator_high._calculate_metrics(y_true, y_proba)
        assert m_high["recall"] <= m_default["recall"]


# ============================================================
# 2. TESTES DE evaluation_walk_forward
# ============================================================

class TestWalkForward:

    def test_returns_dataframe(self, temporal_data, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        result = evaluator.evaluation_walk_forward(
            df=temporal_data,
            feature_cols=["feature_1", "feature_2"],
            n_splits=3,
            initial_train_size=0.5,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        expected_cols = {
            "fold", "train_size", "val_size", "train_end", "val_start",
            "precision", "recall", "f1_score", "pr_auc", "roc_auc",
            "brier_score", "ks",
        }
        assert expected_cols.issubset(result.columns)

    def test_expanding_window_grows(self, temporal_data, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        result = evaluator.evaluation_walk_forward(
            df=temporal_data,
            feature_cols=["feature_1", "feature_2"],
            n_splits=3,
            initial_train_size=0.5,
            window_type="expanding",
        )
        assert result["train_size"].is_monotonic_increasing

    def test_sliding_window_fixed_size(self, temporal_data, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        result = evaluator.evaluation_walk_forward(
            df=temporal_data,
            feature_cols=["feature_1", "feature_2"],
            n_splits=3,
            initial_train_size=0.5,
            window_type="sliding",
            train_window_size=30,
        )
        assert result["train_size"].max() <= 31

    def test_gap_reduces_val_size(self, temporal_data, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        result_no_gap = evaluator.evaluation_walk_forward(
            df=temporal_data,
            feature_cols=["feature_1", "feature_2"],
            n_splits=3,
            initial_train_size=0.5,
            gap=0,
        )
        result_with_gap = evaluator.evaluation_walk_forward(
            df=temporal_data,
            feature_cols=["feature_1", "feature_2"],
            n_splits=3,
            initial_train_size=0.5,
            gap=3,
        )
        assert result_with_gap["val_size"].sum() < result_no_gap["val_size"].sum()

    def test_invalid_window_type_raises(self, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        with pytest.raises(ValueError, match="window_type"):
            evaluator.evaluation_walk_forward(
                df=pd.DataFrame({"t": [1, 2], "x": [1, 2], "target": [0, 1]}),
                feature_cols=["x"],
                window_type="invalid",
            )

    def test_invalid_n_splits_raises(self, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        with pytest.raises(ValueError, match="n_splits"):
            evaluator.evaluation_walk_forward(
                df=pd.DataFrame({"t": [1, 2], "x": [1, 2], "target": [0, 1]}),
                feature_cols=["x"],
                n_splits=0,
            )

    def test_invalid_initial_train_size_raises(self, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        with pytest.raises(ValueError, match="initial_train_size"):
            evaluator.evaluation_walk_forward(
                df=pd.DataFrame({"t": [1, 2], "x": [1, 2], "target": [0, 1]}),
                feature_cols=["x"],
                initial_train_size=0,
            )

    def test_negative_gap_raises(self, simple_model):
        evaluator = EvaluationModelFraud(simple_model)
        with pytest.raises(ValueError, match="gap"):
            evaluator.evaluation_walk_forward(
                df=pd.DataFrame({"t": [1, 2], "x": [1, 2], "target": [0, 1]}),
                feature_cols=["x"],
                gap=-1,
            )


# ============================================================
# 3. TESTES DE evaluation_out_of_time
# ============================================================

class TestOutOfTime:

    def test_returns_dataframe_with_metrics(self, simple_model):
        np.random.seed(42)
        train_df = pd.DataFrame({
            "TransactionDT": np.arange(50) * 100,
            "x1": np.random.randn(50),
            "target": np.concatenate([np.zeros(40), np.ones(10)]),
        })
        oot_df = pd.DataFrame({
            "TransactionDT": np.arange(50, 70) * 100,
            "x1": np.random.randn(20),
            "target": np.concatenate([np.zeros(16), np.ones(4)]),
        })
        evaluator = EvaluationModelFraud(simple_model)
        result = evaluator.evaluation_out_of_time(
            train_df=train_df,
            oot_df=oot_df,
            feature_cols=["x1"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        expected_cols = {
            "train_size", "oot_size", "train_end", "oot_start",
            "precision", "recall", "f1_score", "pr_auc",
            "roc_auc", "brier_score", "ks",
        }
        assert expected_cols.issubset(result.columns)

    def test_overlapping_periods_raises(self, simple_model):
        train_df = pd.DataFrame({
            "TransactionDT": [100, 200, 300],
            "x1": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        oot_df = pd.DataFrame({
            "TransactionDT": [200, 300, 400],
            "x1": [2.0, 3.0, 4.0],
            "target": [1, 0, 1],
        })
        evaluator = EvaluationModelFraud(simple_model)
        with pytest.raises(ValueError, match="OOT"):
            evaluator.evaluation_out_of_time(
                train_df=train_df,
                oot_df=oot_df,
                feature_cols=["x1"],
            )
