# Arquitetura

## Visao Geral

Pipeline de deteccao de fraude em transacoes financeiras com medallion architecture (Bronze, Silver, Gold), Feature Store no Unity Catalog, modelo XGBoost otimizado via Optuna, e serving em tempo real via Online Feature Store.

```
Bronze (raw) -> Silver (clean) -> Gold (features) -> Feature Store -> Modelo -> Serving
```

## Camadas de Dados

| Camada | Tabela | Conteudo |
|---|---|---|
| Bronze | `fraud_detection_dev.bronze.transaction_data` | Dados brutos de transacao (IEEE-CIS Fraud Detection) |
| Bronze | `fraud_detection_dev.bronze.identity_data` | Dados brutos de identidade (device, IP, browser) |
| Silver | `fraud_detection_dev.silver.fraud_data_clean` | Join transaction + identity, limpeza, tipos |
| Gold | `fraud_detection_dev.gold.fraud_features` | Feature table com 156 colunas, 590.540 linhas |
| Gold (serving) | `fraud_detection_dev.gold.fraud_features_serving` | Projeccao compacta para Online Store (114k linhas) |

## Pipeline de Ingestao

### Bronze

Notebooks: `pipeline/bronze/ingest_transaction_data`, `pipeline/bronze/ingest_identity_data`

Carregamento dos datasets brutos do IEEE-CIS Fraud Detection para tabelas Delta no Unity Catalog. Sem transformacoes.

### Silver

Notebook: `pipeline/silver/ingest_data_complete`

Join entre transaction e identity data por `TransactionID`. Conversao de tipos, remocao de colunas administrativas (`ingestion_timestamp`, `dataset_type`). Resultado: 590.540 linhas com 434 colunas.

### Gold (Feature Engineering)

**Produção:** `pipeline/gold/feature_engineering` (notebook estável usado em jobs automatizados)  
**Experimentação:** `notebooks/02_FE_PRE_PROCE` (testes de novas features, diferentes transformações)

Feature selection em 3 etapas (qualidade, evidência univariada, redundância) seguida de feature engineering. Resultado: 152 features + 4 colunas de controle (TransactionID, TransactionDT, target, split). Publicado como feature table no Unity Catalog com `FeatureEngineeringClient.create_table()`.

**Fluxo de desenvolvimento:**
1. Experimentar novas features em `notebooks/02_FE_PRE_PROCE`
2. Features validadas são promovidas para `pipeline/gold/feature_engineering`
3. Pipeline de produção usa apenas `pipeline/gold/` para garantir estabilidade

## Estrutura do Projeto

```
databricks-fraud-detection/
  pipeline/
    bronze/          # Ingestao de dados brutos
    silver/          # Limpeza e join
    gold/            # Feature engineering (producao)
  notebooks/
    01_EDA           # Analise exploratoria
    02_FE_PRE_PROCE  # Feature engineering + feature store
    03_MODEL_TRAINING # Treino, HPO, avaliacao
    04_DEPLOYED_MODEL # Deploy para serving
  retraining/
    01_prepare_data
    02_feature_engineering
    03_train_challenger
    04_evaluate_challenger
    05_ab_test
    06_promote_model
  config/
    project_config.yml  # Configuracao central
  src/
    config.py           # Loader de config
  resources/
    jobs/               # Definicoes de jobs (DABs)
    serving/            # Definicoes de serving (DABs)
  docs/
    architecture.md
    data.md
    modeling.md
    mlops.md
    serving.md
    monitoring.md
```

## Configuracao Central

Arquivo: `config/project_config.yml`

```yaml
project:
  name: databricks-fraud-detection

feature_store:
  table: fraud_detection_dev.gold.fraud_features
  split_methodology: temporal  # walk-forward, nao aleatorio

experiments:
  hpo: .../experiments/fraud_detection_walkforward
  deploy: .../experiments/fraud_detection_deployment

model:
  registered_name: fraud_detection_dev.models.xgboost_fraud_model
  optimal_threshold: 0.83
  max_bins: 1500
  seed: 42

preprocessing:
  exclude_cols: [TransactionID, TransactionDT, target, class_weight]
  imputer_strategy: median
  handle_invalid: keep
```

## Split Temporal

Dados particionados por `TransactionDT` (timestamp em segundos):

| Split | Percentil | Linhas | Funcao |
|---|---|---|---|
| Train | 0 a 60% | 357.753 | Treino do modelo |
| Test | 60 a 80% | 118.579 | Avaliacao intermediaria |
| Val | 80 a 100% | 114.208 | Validacao final + serving |

Walk-forward dentro do treino: 3 janelas temporais (dias 1-61, 1-76, 1-91) para cross-validation.

## Experimentos MLflow

| Experimento | ID | Link (leitura) |
|---|---|---|
| Model Selection (walk-forward) | 4081575290465878 | [ml/experiments/4081575290465878](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/4081575290465878) |
| HPO (Optuna) | 986719377495176 | [ml/experiments/986719377495176](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/986719377495176) |
| Deployment | 28212970340608 | [ml/experiments/28212970340608](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/28212970340608) |

## Ambientes

O projeto utiliza 3 ambientes (catalogos no Unity Catalog) para promocao controlada de codigo, dados e modelos:

| Ambiente | Catalogo | Funcao |
|---|---|---|
| Desenvolvimento (dev) | `fraud_detection_dev` | Desenvolvimento, EDA, feature engineering, treino, HPO, experimentacao |
| Homologacao (hml) | `fraud_detection_hml` | Validacao do pipeline completo antes de producao, testes integrados |
| Producao (prd) | `fraud_detection_prd` | Serving em tempo real, modelo champion, dados de producao |

### Promocao entre ambientes

```
dev (experimentacao e treino)
  -> hml (validacao do pipeline, testes integrados)
  -> prd (serving, promocao do alias @champion)
```

O fluxo Champion/Challenger (retreino, A/B test, promocao) esta planejado nos notebooks da pasta `retraining/` mas ainda nao foi executado. Detalhes em [mlops.md](mlops.md).

Todos os recursos (tabelas, feature tables, modelos, online stores) seguem o padrao `<catalogo>.<schema>.<objeto>`. O catalogo atual do projeto e `fraud_detection_dev`.

## Tecnologias

* Databricks (Serverless compute, Unity Catalog)
* Delta Lake (Bronze, Silver, Gold)
* Feature Store (`databricks-feature-engineering`)
* MLflow (tracking, model registry)
* XGBoost, Optuna, scikit-learn
* Online Feature Store (CU_8) para serving real-time
* Model Serving (REST endpoint)