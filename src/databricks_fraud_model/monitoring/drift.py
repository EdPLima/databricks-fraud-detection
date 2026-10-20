"""
Drift detection para modelo de deteccao de fraude.

Funcoes reutilizaveis para detectar data drift (features) e prediction
drift (scores). Extraidas do notebook pipeline/monitoring/05_monitoring.

Funcoes publicas:
- calculate_ks_drift():        KS test para features numericas
- calculate_psi():             PSI para distribuicoes categoricas
- calculate_categorical_drift(): PSI em batch para features categoricas
- calculate_prediction_drift(): drift na distribuicao de probability_fraud

Fluxo tipico (no notebook pipeline/monitoring/05_monitoring):
    from databricks_fraud_model.monitoring import calculate_ks_drift

    ks_df = calculate_ks_drift(df_train, df_prod, numeric_features)
    display(ks_df)
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ============================================================
# 1. KS TEST — DRIFT EM FEATURES NUMERICAS
# ============================================================

def calculate_ks_drift(train_df, prod_df, numeric_features, alpha=0.05, min_samples=10):
    """
    Calcula KS test (Kolmogorov-Smirnov) para features numericas.

    Compara a distribuicao de cada feature entre treino e producao.
    KS statistic: 0 = distribuicoes identicas, 1 = completamente diferentes.
    p-valor < alpha indica drift significativo.

    Args:
        train_df:          DataFrame pandas com dados de treino.
        prod_df:           DataFrame pandas com dados de producao.
        numeric_features:  lista de colunas numericas para testar.
        alpha:             nivel de significancia (default: 0.05).
        min_samples:       minimo de observacoes nao-nulas para testar (default: 10).

    Returns:
        DataFrame pandas com colunas: feature, ks_statistic, p_value, drift.
        Ordenado por ks_statistic descendente (mais drift primeiro).
    """
    results = []
    for col in numeric_features:
        # Verifica se a coluna existe em ambos DataFrames
        if col not in train_df.columns or col not in prod_df.columns:
            continue

        train_vals = train_df[col].dropna().values
        prod_vals = prod_df[col].dropna().values

        # KS test requer amostras minimas
        if len(train_vals) < min_samples or len(prod_vals) < min_samples:
            continue

        statistic, p_value = ks_2samp(train_vals, prod_vals)
        results.append({
            "feature": col,
            "ks_statistic": round(statistic, 4),
            "p_value": p_value,
            "drift": p_value < alpha,
        })

    if not results:
        return pd.DataFrame(columns=["feature", "ks_statistic", "p_value", "drift"])
    return pd.DataFrame(results).sort_values("ks_statistic", ascending=False)


# ============================================================
# 2. PSI — DRIFT EM FEATURES CATEGORICAS
# ============================================================

def calculate_psi(train_dist, prod_dist):
    """
    Calcula PSI (Population Stability Index) para distribuicoes categoricas.

    PSI < 0.1:  sem drift
    0.1-0.25:   drift moderado
    > 0.25:     drift significativo

    Args:
        train_dist: Series pandas com contagens por categoria (treino).
        prod_dist:   Series pandas com contagens por categoria (producao).

    Returns:
        float: valor de PSI.
    """
    # Une todas as categorias possiveis
    all_categories = set(train_dist.index) | set(prod_dist.index)
    train_pct = np.array([train_dist.get(cat, 0) for cat in all_categories], dtype=float)
    prod_pct = np.array([prod_dist.get(cat, 0) for cat in all_categories], dtype=float)

    # Normaliza para proporcoes
    train_pct = train_pct / train_pct.sum()
    prod_pct = prod_pct / prod_pct.sum()

    # Evita divisao por zero
    epsilon = 1e-6
    train_pct = np.where(train_pct == 0, epsilon, train_pct)
    prod_pct = np.where(prod_pct == 0, epsilon, prod_pct)

    return float(np.sum((prod_pct - train_pct) * np.log(prod_pct / train_pct)))


def calculate_categorical_drift(train_df, prod_df, categorical_features, threshold=0.25):
    """
    Calcula PSI em batch para features categoricas.

    Args:
        train_df:             DataFrame pandas com dados de treino.
        prod_df:              DataFrame pandas com dados de producao.
        categorical_features: lista de colunas categoricas para testar.
        threshold:            limite de PSI para classificar como drift (default: 0.25).

    Returns:
        DataFrame pandas com colunas: feature, psi, drift.
    """
    results = []
    for col in categorical_features:
        if col not in train_df.columns or col not in prod_df.columns:
            continue

        # Conta frequencias por categoria
        train_dist = train_df[col].value_counts()
        prod_dist = prod_df[col].value_counts()

        psi_value = calculate_psi(train_dist, prod_dist)
        results.append({
            "feature": col,
            "psi": round(psi_value, 4),
            "drift": psi_value > threshold,
        })

    return pd.DataFrame(results).sort_values("psi", ascending=False)


# ============================================================
# 3. PREDICTION DRIFT — DISTRIBUICAO DE SCORES
# ============================================================

def calculate_prediction_drift(train_df, prod_df, score_col="probability_fraud", target_col="target"):
    """
    Analisa drift na distribuicao de predicoes (probability_fraud).

    Compara estatisticas de score entre treino e producao, e se houver
    target no treino, compara a taxa de fraude real vs predita.

    Args:
        train_df:    DataFrame pandas com dados de treino (tem target).
        prod_df:     DataFrame pandas com predicoes de producao (tem probability_fraud).
        score_col:   nome da coluna de score (default: probability_fraud).
        target_col:  nome da coluna de target no treino (default: target).

    Returns:
        dict com:
            - mean, median, std, min, max, p95 do score de producao
            - predicted_fraud_rate: taxa de fraudes preditas em producao
            - train_fraud_rate: taxa de fraudes reais no treino (se target existir)
            - fraud_rate_delta: diferenca entre producao e treino
    """
    result = {
        "n_predictions": len(prod_df),
        "mean_score": float(prod_df[score_col].mean()),
        "median_score": float(prod_df[score_col].median()),
        "std_score": float(prod_df[score_col].std()),
        "min_score": float(prod_df[score_col].min()),
        "max_score": float(prod_df[score_col].max()),
        "p95_score": float(prod_df[score_col].quantile(0.95)),
        "predicted_fraud_rate": float(prod_df["is_fraud"].mean()) if "is_fraud" in prod_df.columns else None,
    }

    # Compara com taxa real de fraude no treino
    if target_col in train_df.columns:
        result["train_fraud_rate"] = float(train_df[target_col].mean())
        if result["predicted_fraud_rate"] is not None:
            result["fraud_rate_delta"] = result["predicted_fraud_rate"] - result["train_fraud_rate"]

    return result