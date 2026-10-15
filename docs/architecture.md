# Arquitetura

## Visao Geral

Pipeline de deteccao de fraude em transacoes financeiras com medallion architecture (Bronze, Silver, Gold), Feature Store no Unity Catalog, modelo XGBoost otimizado via Optuna, serving via REST endpoint, e monitoramento MLOps com latencia, drift e observabilidade.

```
Bronze (raw) -> Silver (clean) -> Gold (features) -> Feature Store -> Modelo XGBoost
                                                                              |
                                                                              v
                                                                    REST Endpoint (model_fraud)
                                                                              |
                                                                              v
                                                          Scoring -> Bronze (events) + Gold (predictions)
                                                                              |
                                                                              v
                                                                    Monitoring Dashboard (MLOps)
```

## Estrutura do Projeto

```
databricks-fraud-detection/
  pipeline/
    bronze/01_bronze_ingest        # CSV -> Bronze (batch_type: train/test/val/prod)
    silver/02_silver_clean          # Bronze -> Silver (join, DQ, MODEL_FEATURES check)
    gold/03_gold_feature_engineering # Silver -> Gold (FE + Feature Store)
    scoring/04_scoring              # Gold -> endpoint REST -> predicoes + bronze (latency)
    monitoring/05_monitoring        # Dashboard MLOps (latency, drift, observability)
  src/databricks_fraud_model/      # Package Python (single source of truth)
    features/
      features.py                   # MODEL_FEATURES (152 features)
      feature_engineering.py        # apply_feature_engineering_spark(), FEATURES_DROP, MODEL_KEEP
    model/
      scoring.py                    # score_batch, score_single, save_to_bronze, save_predictions
      load.py                        # load_model (pyfunc do UC)
    utils/
      config.py                     # load_config() do project_config.yml
  tests/
    unit/                           # test_config, test_features, test_scoring
    integration/                    # test_scoring_e2e, test_pipeline_etl
  retraining/                       # Pipeline de retreino (Champion/Challenger) - futuro
  notebooks/                        # EDA, experimentacao
  config/
    project_config.yml              # Configuracao central (catalog, feature_store, model)
  resources/jobs/                   # Definicoes de jobs (DABs)
    etl_pipeline.job.yml            # Bronze -> Silver -> Gold (batch_type=prd)
    inference_monitoring.job.yml    # Scoring -> Monitoring (diario 8AM)
  docs/                             # Documentacao tecnica
  databricks.yml                    # Declarative Automation Bundle
  pyproject.toml                    # Dependencias, pytest, ruff, mypy
```

## Package Python (`src/databricks_fraud_model/`)

Single source of truth para features, scoring e configuracao. Todos os notebooks importam da package:

```python
from databricks_fraud_model import MODEL_FEATURES, score_batch, save_predictions
from databricks_fraud_model.utils import load_config
from databricks_fraud_model.features import apply_feature_engineering_spark
```

| Modulo | Exports |
|---|---|
| `features/features.py` | `MODEL_FEATURES` (152 features, assert de integridade) |
| `features/feature_engineering.py` | `apply_feature_engineering_spark()`, `FEATURES_DROP`, `MODEL_KEEP`, `FREE_DOMAINS` |
| `model/scoring.py` | `score_batch()`, `score_single()`, `save_to_bronze()`, `save_predictions()` |
| `model/load.py` | `load_model()` (pyfunc do Unity Catalog via alias @champion) |
| `utils/config.py` | `load_config()`, `get()` (le `config/project_config.yml`) |

## Camadas de Dados

| Camada | Tabela | Conteudo |
|---|---|---|
| Bronze | `{catalog}.bronze.{batch_type}_transaction_raw` | Dados brutos de transacao (CSV -> Delta) |
| Bronze | `{catalog}.bronze.{batch_type}_identity_raw` | Dados brutos de identidade |
| Bronze | `{catalog}.bronze.monitoring_raw_events` | Eventos brutos de scoring (latency_ms, status, endpoint) |
| Silver | `{catalog}.silver.{batch_type}_data_clean` | Join transaction + identity, DQ null > 80%, feature selection |
| Gold | `{catalog}.gold.fraud_features` | Feature table (152 features + TransactionID + TransactionDT + target + split) |
| Gold | `{catalog}.gold.fraud_features_prd` | Features de producao (feature engineering aplicado) |
| Gold | `{catalog}.gold.model_predictions` | Predicoes + features (append, para drift e dashboard) |

O parametro `batch_type` (train/test/val/prod) torna as camadas escalaveis para qualquer novo batch.

## Pipeline de Notebooks

### 01_bronze_ingest

Carrega CSVs do volume para tabelas Bronze. Parametrizado por `batch_type` (widget). Adiciona `ingestion_timestamp` e `dataset_type`. ZORDER por `TransactionID`.

### 02_silver_clean

Join bronze (transaction + identity) por `TransactionID`. Renomeia colunas `id-XX` para `id_XX` (hifen -> underline). Renomeia `isFraud` -> `target`. Transforma `id_34` (regex extract). Deduplicacao via `.dropDuplicates()`. ZORDER por `TransactionID`.

### 03_gold_feature_engineering

Para `batch_type=train`: split temporal (60/20/20) e `apply_feature_engineering_spark()`. Publica feature table no Unity Catalog com `FeatureEngineeringClient.create_table()`.

Para `batch_type=prd`: aplica feature engineering e salva em `gold.fraud_features_prd` (Delta table, overwrite).

### 04_scoring

Carrega dados de `gold.fraud_features_prd`, envia features ao endpoint REST `model_fraud` (1000 registros por batch), mede latencia com `time.perf_counter()`, e salva:
- **Bronze** (`bronze.monitoring_raw_events`): eventos brutos com `latency_ms`, `status`, `endpoint_name`, `n_records`, `event_timestamp`
- **Gold** (`gold.model_predictions`): 152 features + `probability_fraud` + `is_fraud` + `prediction_timestamp` (append)

### 05_monitoring

Dashboard MLOps que le de `gold.fraud_features` (treino), `gold.model_predictions` (producao) e `bronze.monitoring_raw_events` (latencia):
- **Latencia do endpoint**: avg, p50, p95, p99 (ms)
- **Throughput**: predicoes por batch, predicoes/min
- **Error rate**: % de falhas
- **Data drift**: KS test (numericas) + PSI (categoricas)
- **Prediction drift**: distribuicao de `probability_fraud` treino vs producao
- **Alertas**: drift > 10% (atencao), > 20% (retreinar)

Detalhes em [monitoring.md](monitoring.md).

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

Loader: `databricks_fraud_model.utils.load_config()`. Todos os notebooks usam `cfg = load_config()` em vez de hardcoded.

## Testes

| Tipo | Arquivo | Cobertura |
|---|---|---|
| Unit | `tests/unit/test_config.py` | load_config, required keys, catalog, get helper |
| Unit | `tests/unit/test_features.py` | 152 features, no duplicates, MODEL_KEEP vs DROP, FE columns |
| Unit | `tests/unit/test_scoring.py` | _clean_records (NaN, n, selection), score_batch (mock), save_to_bronze (mock) |
| Integration | `tests/integration/test_scoring_e2e.py` | endpoint READY, 152 features in prod, score_batch real, gold+bronze writes |
| Integration | `tests/integration/test_pipeline_etl.py` | bronze/silver/gold tables exist, split column, model_predictions schema |

18 testes unitarios passando, 3 skipped (Spark-dependent). Testes de integracao requerem workspace ativo.

## Jobs (DABs)

Declarative Automation Bundle (`databricks.yml`) com 2 jobs:

| Job | Arquivo | Schedule | Tasks |
|---|---|---|---|
| ETL Pipeline | `resources/jobs/etl_pipeline.job.yml` | Manual | 01_bronze -> 02_silver -> 03_gold (batch_type=prd) |
| Inference & Monitoring | `resources/jobs/inference_monitoring.job.yml` | Diario 8AM | 04_scoring -> 05_monitoring |
| Retraining | `resources/jobs/retraining.job.yml` | Manual | 01_prepare -> 02_fe -> 03_train -> 04_evaluate -> 05_ab_test -> 06_promote |

Targets: `dev` (default, `fraud_detection_dev`) e `prod` (`fraud_detection_prod`). Variaveis: `catalog`, `schema`, `batch_type`.

## Split Temporal

Dados particionados por `TransactionDT` (timestamp em segundos):

| Split | Percentil | Linhas | Funcao |
|---|---|---|---|
| Train | 0 a 60% | 357.753 | Treino do modelo |
| Test | 60 a 80% | 118.579 | Avaliacao intermediaria |
| Val | 80 a 100% | 114.208 | Validacao final + serving |

Walk-forward dentro do treino: 3 janelas temporais (dias 1-61, 1-76, 1-91) para cross-validation.

## Ambientes

| Ambiente | Catalogo | Funcao |
|---|---|---|
| Desenvolvimento (dev) | `fraud_detection_dev` | Desenvolvimento, EDA, treino, HPO, scoring, monitoring |
| Homologacao (hml) | `fraud_detection_hml` | Testes de staging, validacao de modelo e pipeline antes de producao |
| Producao (prd) | `fraud_detection_prd` | Serving em tempo real, modelo champion, dados de producao |

## Tecnologias

* Databricks (Serverless compute, Unity Catalog)
* Delta Lake (Bronze, Silver, Gold)
* Feature Store (`databricks-feature-engineering`)
* MLflow (tracking, model registry, alias @champion)
* XGBoost, Optuna, scikit-learn
* Model Serving (REST endpoint `model_fraud`)
* Declarative Automation Bundles (DABs)
* pytest (unit + integration tests)