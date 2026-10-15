# MLOps

## MLflow Tracking

3 experimentos registrados no MLflow:

| Experimento | ID | Link (leitura) |
|---|---|---|
| Model Selection (walk-forward CV) | 4081575290465878 | [ml/experiments/4081575290465878](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/4081575290465878) |
| HPO (Optuna) | 986719377495176 | [ml/experiments/986719377495176](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/986719377495176) |
| Deployment | 28212970340608 | [ml/experiments/28212970340608](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/28212970340608) |

Caminhos definidos em `config/project_config.yml`:
```yaml
experiments:
  hpo: .../experiments/fraud_detection_walkforward
  deploy: .../experiments/fraud_detection_deployment
```

### Runs principais

* Walk-forward CV: 4 modelos x 3 folds = 12 runs filhas + 4 runs agregadas (1 por modelo)
* HPO: 1 run pai (`XGBoost_Optuna_HPO`) + 20 runs filhas (Optuna trials)
* Deployment: 1 run por versao registrada do modelo

## Model Registry

* **Catalogo/Schema**: `fraud_detection_dev.models`
* **Modelo**: `fraud_detection_dev.models.xgboost_fraud_model`
* **Alias**: `@champion`
* **Formato**: pyfunc com FeatureLookup
* **Tags por versao**: `algorithm`, `model_type`, `serving_mode`, `serving_table`, `sklearn_version`, `xgboost_version`

### Pinning de Dependencias

**Versoes salvas no modelo (requirements.txt):**

```
mlflow==3.8.1
xgboost
scikit-learn==1.6.1
pandas
numpy
cloudpickle==3.1.1
```

⚠️ `xgboost`, `pandas` e `numpy` estao sem pin de versao. O serving endpoint instala a versao mais recente disponivel, que pode ser incompativel com o modelo treinado. No proximo registro, fixar todas as versoes:

```python
pip_requirements=[
    "scikit-learn==1.6.1",
    "xgboost==<versao_do_treino>",
    "cloudpickle==3.1.1",
    "pandas==<versao_do_treino>",
    "numpy==<versao_do_treino>",
]
```

Sem o pinning, o serving environment pode instalar versoes diferentes do treino, causando falha na desserializacao de objetos pickle.

## Feature Store

* **Offline**: `fraud_detection_dev.gold.fraud_features` (590k linhas, 156 colunas)
* **Online**: `fraud-detection-online-store` (CU_8, ~512 MB)
* **Online Table**: `fraud_detection_dev.gold.fraud_features_serving_online`
* **Serving table**: `fraud_detection_dev.gold.fraud_features_serving` (114k linhas, 154 colunas)
* **Sincronizacao**: pipeline continua via Change Data Feed

Detalhes em [serving.md](serving.md).

## Retraining Pipeline

Pasta: `retraining/`. Pipeline de retreino automatizado com padrao Champion/Challenger.

**Status:** Planejado — os 6 notebooks estao criados mas sem implementacao.

| Notebook | Funcao |
|---|---|
| `01_prepare_data` | Preparo dos dados ( Bronze -> Silver) |
| `02_feature_engineering` | Feature engineering + publicacao na Gold |
| `03_train_challenger` | Treino do modelo challenger com HPO |
| `04_evaluate_challenger` | Avaliacao do challenger vs champion |
| `05_ab_test` | A/B test entre champion e challenger |
| `06_promote_model` | Promocao do challenger a champion |

### Fluxo Champion/Challenger

```
01_prepare_data -> 02_feature_engineering -> 03_train_challenger
  -> 04_evaluate_challenger (metricas vs champion)
  -> 05_ab_test (trafego split entre os dois modelos)
  -> 06_promote_model (alias @champion atualizado)
```

## CI/CD

* Repositorio Git no Databricks (`.git` no diretorio do projeto)
* Configuracao central em `config/project_config.yml` carregada via `databricks_fraud_model.utils.load_config()`
* `pyproject.toml` para dependencias, pytest, ruff, mypy
* Declarative Automation Bundle (`databricks.yml`) para deploy de jobs e recursos

### Testes

| Tipo | Local | Cobertura |
|---|---|---|
| Unit | `tests/unit/test_config.py` | load_config, keys, get helper |
| Unit | `tests/unit/test_features.py` | 152 features, no dup, MODEL_KEEP vs DROP, FE columns |
| Unit | `tests/unit/test_scoring.py` | _clean_records, score_batch (mock), save_to_bronze (mock) |
| Integration | `tests/integration/test_scoring_e2e.py` | endpoint READY, real scoring, gold+bronze writes |
| Integration | `tests/integration/test_pipeline_etl.py` | bronze/silver/gold tables, model_predictions schema |

18 unitarios passando, 3 skipped (Spark). Integration requerem workspace ativo.

### Jobs (DABs)

| Job | Arquivo | Schedule | Tasks |
|---|---|---|---|
| ETL Pipeline | `resources/jobs/etl_pipeline.job.yml` | Manual | 01_bronze -> 02_silver -> 03_gold (batch_type=prd) |
| Inference & Monitoring | `resources/jobs/inference_monitoring.job.yml` | Diario 8AM | 04_scoring -> 05_monitoring |
| Retraining | `resources/jobs/retraining.job.yml` | Manual | 01_prepare -> 02_fe -> 03_train -> 04_evaluate -> 05_ab_test -> 06_promote |

Targets: `dev` (default, `fraud_detection_dev`), `hml` (`fraud_detection_hml`) e `prod` (`fraud_detection_prod`). Variaveis: `catalog`, `schema`, `batch_type`.

## Configuracao Central

Arquivo: `config/project_config.yml`

```yaml
catalog: fraud_detection_dev

feature_store:
  table: fraud_detection_dev.gold.fraud_features
  split_methodology: temporal

model:
  registered_name: fraud_detection_dev.models.xgboost_fraud_model
  optimal_threshold: 0.83

preprocessing:
  exclude_cols: [TransactionID, TransactionDT, target, class_weight]
  imputer_strategy: median
```

Loader: `databricks_fraud_model.utils.load_config()` com funcao `load_config()` e `get()`.

## Monitoramento de Producao

Implementado em `pipeline/monitoring/05_monitoring`. Detalhes em [monitoring.md](monitoring.md).

* **Latencia do endpoint**: avg, p50, p95, p99 (ms) via `bronze.monitoring_raw_events` — granularidade por requisicao HTTP individual
* **Latencia por predicao**: `avg_latency_per_record_ms` = `SUM(latency_ms) / SUM(n_records)` — normalizada por predicao
* **Cold vs Warm Start**: dimensao `start_type` via `LAG()` — Cold Start = primeira requisicao apos 5+ min de inatividade
* **Contagem corrigida**: `total_predictions = SUM(n_records)` (2.000.000 predicoes), `total_requests = COUNT(*)` (2.000 requisicoes HTTP)
* **Data drift**: KS test (numericas) + PSI (categoricas)
* **Prediction drift**: distribuicao de `probability_fraud` treino vs producao
* **Throughput**: predicoes por batch (`n_records`), total de requisicoes HTTP
* **Error rate**: % de requisicoes HTTP com `status != 'success'`
* **Alertas**: drift > 10% (atencao), > 20% (retreinar)

### Resultados atuais (Out 2026)

| Metrica | Valor |
|---|---|
| Requisicoes HTTP | 2.000 |
| Predicoes | 2.000.000 |
| Latencia media/requisicao | 19.034 ms |
| Latencia media/predicao | 19.03 ms |
| Cold Start latencia | 36.964 ms |
| Warm Start latencia | 19.025 ms |
| Error rate | 0% |

> **Limitacao**: A tabela nao separa Feature Lookup / Model Execution — o modelo v13 usa features diretas (sem FeatureLookup) e `scoring.py` rastreia apenas `latency_ms` unico por requisicao.