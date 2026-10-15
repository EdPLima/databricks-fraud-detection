# Monitoring & Observability

## Visao Geral

Pipeline de monitoramento MLOps do modelo de deteccao de fraude. Mede latencia do endpoint, drift de features e predicoes, e exibe alertas em dashboard.

Notebook: `pipeline/monitoring/05_monitoring`

## Arquitetura de Monitoramento

```
04_scoring
  -> score_batch() mede latency_ms (time.perf_counter)
  -> save_to_bronze() -> bronze.monitoring_raw_events (append)
  -> save_predictions() -> gold.model_predictions (append)

05_monitoring
  -> le gold.fraud_features (treino, baseline)
  -> le gold.model_predictions (producao)
  -> le bronze.monitoring_raw_events (latencia)
  -> dashboard: latency, drift, alertas
```

## Fontes de Dados

| Fonte | Tabela | Uso |
|---|---|---|
| Treino (baseline) | `{catalog}.gold.fraud_features` (split='train') | Distribuicao de referencia para drift |
| Producao | `{catalog}.gold.model_predictions` | Predicoes + features para comparacao |
| Latencia | `{catalog}.bronze.monitoring_raw_events` | Eventos brutos com `latency_ms`, `status`, `endpoint_name` |

## Metricas de Observabilidade

### Latencia do Endpoint

Mede o tempo de resposta do endpoint REST `model_fraud` usando `time.perf_counter()` em `score_batch()`. Cada linha em `bronze.monitoring_raw_events` representa uma requisicao HTTP individual com sua propria `latency_ms` e `n_records` (predicoes por batch):

```python
t0 = time.perf_counter()
response = w.api_client.do('POST', f'/serving-endpoints/{endpoint_name}/invocations', ...)
latency_ms = round((time.perf_counter() - t0) * 1000, 2)
```

#### Correcoes aplicadas (Out 2026)

As metricas foram revisadas para garantir granularidade por requisicao HTTP individual:

| Metrica | Expressao SQL | Descricao |
|---|---|---|
| `avg_latency_ms` | `AVG(latency_ms)` | Latencia media por requisicao HTTP (ms) |
| `p50_latency_ms` | `PERCENTILE(latency_ms, 0.5)` | Mediana das latencias individuais (ms) |
| `p95_latency_ms` | `PERCENTILE(latency_ms, 0.95)` | P95 das latencias individuais (ms) |
| `p99_latency_ms` | `PERCENTILE(latency_ms, 0.99)` | P99 das latencias individuais (ms) |
| `avg_latency_per_record_ms` | `SUM(latency_ms) / SUM(n_records)` | Latencia normalizada por predicao (ms) — **NOVO** |
| `total_predictions` | `SUM(n_records)` | Total de predicoes reais — **corrigido** (era `COUNT(*)`) |
| `total_requests` | `COUNT(*)` | Total de requisicoes HTTP — **renomeado** (era `total_batches = COUNT(DISTINCT event_timestamp)`) |
| `error_rate_pct` | `100.0 * SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) / COUNT(*)` | % de requisicoes HTTP com erro |

**Correcoes principais:**
- `total_predictions` agora soma `n_records` (predicoes reais), nao conta linhas (requisicoes)
- `total_batches` renomeado para `total_requests` = `COUNT(*)` — conta requisicoes HTTP reais, nao `COUNT(DISTINCT event_timestamp)` que agrupava requisicoes com mesmo timestamp
- `avg_latency_ms` e percentiles ja operavam sobre latencias individuais por requisicao — confirmado correto
- Nova metrica `avg_latency_per_record_ms` normaliza latencia pelo tamanho do batch

### Cold Start vs Warm Start

**Nova dimensao `start_type`** adicionada via funcao `LAG()` no source SQL da metric view:

```sql
CASE
  WHEN LAG(event_timestamp) OVER (PARTITION BY endpoint_name ORDER BY event_timestamp) IS NULL
    THEN 'Cold Start'
  WHEN event_timestamp - LAG(event_timestamp) OVER (PARTITION BY endpoint_name ORDER BY event_timestamp) > INTERVAL 5 MINUTES
    THEN 'Cold Start'
  ELSE 'Warm Start'
END AS start_type
```

- **Cold Start**: primeira requisicao apos 5+ minutos de inatividade no endpoint
- **Warm Start**: requisicoes subsequentes dentro de 5 minutos da requisicao anterior

### Resultados Atuais (Out 2026)

| Metrica | Valor |
|---|---|
| Requisicoes HTTP (total) | 2.000 |
| Predicoes (total) | 2.000.000 |
| Latencia media/requisicao | 19.034 ms |
| Latencia media/predicao | 19.03 ms |
| Cold Start — latencia media | 36.964 ms |
| Warm Start — latencia media | 19.025 ms |
| Error rate | 0% (todas success) |

Cold starts sao aproximadamente **2x mais lentos** que warm starts (36.964 ms vs 19.025 ms).

### Limitacao — Split de Timings

A tabela `bronze.monitoring_raw_events` rastreia apenas `latency_ms` (tempo total da requisicao HTTP). Nao ha colunas separadas para:
- **Feature Lookup**: o modelo v13 usa features diretas (sem FeatureLookup), entao essa fase nao existe
- **Model Execution**: tempo de inferencia do modelo XGBoost dentro do pyfunc

Para separar esses timings, seria necessario modificar `scoring.py` para capturar sub-timings (ex.: `feature_lookup_ms`, `model_execution_ms`) e adicionar essas colunas a `bronze.monitoring_raw_events`.

### Throughput

- Predicoes por batch (`n_records`) — media de 1.000 predicoes/requisicao
- Total de requisicoes HTTP (`total_requests`)
- Total de predicoes (`total_predictions = SUM(n_records)`)

### Error Rate

- % de requisicoes HTTP com `status != 'success'` em `bronze.monitoring_raw_events`
- Calculado por requisicao (nao por predicao individual)

## Metricas de Drift

### Data Drift — Numericas (KS Test)

Compara a distribuicao de cada feature numerica entre treino e producao usando o teste Kolmogorov-Smirnov (`scipy.stats.ks_2samp`):

- **KS statistic**: 0 = distribuicoes identicas, 1 = completamente diferentes
- **p-valor < 0.05**: indica drift significativo
- Features testadas: todas em `NUMERIC_FEATURES` (152 - categoricas)

### Data Drift — Categoricas (PSI)

Population Stability Index para features categoricas:

- **PSI < 0.1**: sem drift
- **0.1 <= PSI < 0.25**: drift moderado
- **PSI >= 0.25**: drift significativo

Funcao `calculate_psi()` compara a distribuicao de categorias entre treino e producao.

### Prediction Drift

Compara a distribuicao de `probability_fraud` entre treino e producao:
- Media, mediana, desvio padrao
- Taxa de fraude predita vs treino
- Range de scores

## Alertas

| Condicao | Acao |
|---|---|
| Features com drift > 10% | Atencao: monitorar proximos batches |
| Features com drift > 20% | Retreinar modelo |
| Latencia P95 > 500ms | Investigar saude do endpoint |
| Error rate > 5% | Verificar disponibilidade do endpoint |

## Escalabilidade

O pipeline e escalavel por design:
- `bronze.monitoring_raw_events` e `gold.model_predictions` usam modo **append** — cada batch de scoring adiciona novos registros
- O dashboard le automaticamente todos os batches disponiveis
- Novos batches de producao (`batch_type=prod`) fluem automaticamente pelo ETL

## Job de Monitoramento

Definido em `resources/jobs/inference_monitoring.job.yml` como task `model_monitoring` (depende de `scoring_model`):
- Schedule: diario as 8AM (America/Sao_Paulo)
- Task: `05_monitoring` notebook (executada apos `04_scoring`)
- Compute: serverless