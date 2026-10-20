"""
Observabilidade de endpoint de serving: latencia, error rate, cold starts.

Funcoes reutilizaveis extraidas do notebook pipeline/monitoring/05_monitoring.
Cada funcao opera sobre um DataFrame pandas de eventos da Bronze
(bronze.monitoring_raw_events) e retorna metricas calculadas.

Funcoes publicas:
- calculate_latency_metrics():  avg, p50, p95, p99 de latencia por requisicao
- calculate_error_rate():        % de requisicoes com erro
- detect_cold_starts():          classifica cold vs warm start
- summarize_endpoint_health():   sumario consolidado de observabilidade

Fluxo tipico (no notebook pipeline/monitoring/05_monitoring):
    from databricks_fraud_model.monitoring import summarize_endpoint_health

    events_df = spark.sql("SELECT * FROM bronze.monitoring_raw_events").toPandas()
    health = summarize_endpoint_health(events_df)
    print(health)
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. METRICAS DE LATENCIA
# ============================================================

def calculate_latency_metrics(events_df, latency_col="latency_ms"):
    """
    Calcula metricas de latencia por requisicao HTTP.

    Args:
        events_df:    DataFrame pandas com eventos da Bronze (monitoring_raw_events).
        latency_col:  nome da coluna de latencia (default: latency_ms).

    Returns:
        dict com:
            - avg_ms:   latencia media por requisicao
            - p50_ms:   mediana
            - p95_ms:   percentil 95
            - p99_ms:   percentil 99
            - min_ms:   latencia minima
            - max_ms:   latencia maxima
            - n_requests: numero de requisicoes HTTP
            - total_predictions: SUM(n_records) — predicoes reais
            - avg_per_record_ms: latencia normalizada por predicao
    """
    lat = events_df[latency_col]
    total_predictions = events_df["n_records"].sum() if "n_records" in events_df.columns else len(events_df)

    return {
        "avg_ms": float(lat.mean()),
        "p50_ms": float(lat.median()),
        "p95_ms": float(lat.quantile(0.95)),
        "p99_ms": float(lat.quantile(0.99)),
        "min_ms": float(lat.min()),
        "max_ms": float(lat.max()),
        "n_requests": len(events_df),
        "total_predictions": int(total_predictions),
        "avg_per_record_ms": float(lat.sum() / total_predictions) if total_predictions > 0 else 0.0,
    }


# ============================================================
# 2. ERROR RATE
# ============================================================

def calculate_error_rate(events_df, status_col="status"):
    """
    Calcula a taxa de erro das requisicoes HTTP.

    Args:
        events_df:  DataFrame pandas com eventos da Bronze.
    status_col: nome da coluna de status (default: status).

    Returns:
        dict com:
            - error_rate:    % de requisicoes com status != 'success' (0.0 a 1.0)
            - n_errors:      numero absoluto de requisicoes com erro
            - n_success:     numero absoluto de requisicoes bem-sucedidas
            - n_total:       numero total de requisicoes
    """
    if status_col not in events_df.columns:
        return {"error_rate": 0.0, "n_errors": 0, "n_success": len(events_df), "n_total": len(events_df)}

    n_total = len(events_df)
    n_errors = (events_df[status_col] != "success").sum()
    n_success = n_total - n_errors

    return {
        "error_rate": float(n_errors / n_total) if n_total > 0 else 0.0,
        "n_errors": int(n_errors),
        "n_success": int(n_success),
        "n_total": int(n_total),
    }


# ============================================================
# 3. COLD START DETECTION
# ============================================================

def detect_cold_starts(events_df, timestamp_col="event_timestamp", gap_threshold_minutes=5):
    """
    Classifica requisicoes como cold start ou warm start.

    Cold start: primeira requisicao apos `gap_threshold_minutes` minutos
    de inatividade. Warm start: requisicoes subsequentes.

    Args:
        events_df:                DataFrame pandas com eventos da Bronze.
        timestamp_col:            nome da coluna de timestamp (default: event_timestamp).
        gap_threshold_minutes:    minutos de inatividade para classificar como cold start (default: 5).

    Returns:
        DataFrame pandas com coluna adicional `start_type` ('cold' ou 'warm').
    """
    df = events_df.copy()

    # Garante que o timestamp e datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Ordena por timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Calcula gap em minutos desde a requisicao anterior
    df["_gap_minutes"] = df[timestamp_col].diff().dt.total_seconds() / 60

    # Cold start: gap > threshold ou primeira requisicao
    df["start_type"] = np.where(
        (df["_gap_minutes"] > gap_threshold_minutes) | df["_gap_minutes"].isna(),
        "cold",
        "warm",
    )

    return df.drop(columns=["_gap_minutes"])


# ============================================================
# 4. SUMARIO CONSOLIDADO DE SAUDE DO ENDPOINT
# ============================================================

def summarize_endpoint_health(events_df, gap_threshold_minutes=5):
    """
    Sumario consolidado de observabilidade do endpoint de serving.

    Combina latencia, error rate e cold starts em um unico dict,
    facil de exibir em dashboards ou logar no MLflow.

    Args:
        events_df:             DataFrame pandas com eventos da Bronze.
        gap_threshold_minutes: minutos para classificar cold start (default: 5).

    Returns:
        dict com todas as metricas de observabilidade:
            - latency:    dict de calculate_latency_metrics()
            - errors:     dict de calculate_error_rate()
            - cold_starts: count e latencia media de cold starts
            - warm_starts: count e latencia media de warm starts
    """
    if events_df.empty:
        return {"latency": {}, "errors": {}, "cold_starts": {}, "warm_starts": {}}

    # Calcula metricas individuais
    latency = calculate_latency_metrics(events_df)
    errors = calculate_error_rate(events_df)

    # Detecta cold starts
    events_with_starts = detect_cold_starts(events_df, gap_threshold_minutes=gap_threshold_minutes)

    cold = events_with_starts[events_with_starts["start_type"] == "cold"]
    warm = events_with_starts[events_with_starts["start_type"] == "warm"]

    cold_starts = {
        "count": len(cold),
        "avg_latency_ms": float(cold["latency_ms"].mean()) if len(cold) > 0 else 0.0,
    }
    warm_starts = {
        "count": len(warm),
        "avg_latency_ms": float(warm["latency_ms"].mean()) if len(warm) > 0 else 0.0,
    }

    return {
        "latency": latency,
        "errors": errors,
        "cold_starts": cold_starts,
        "warm_starts": warm_starts,
    }