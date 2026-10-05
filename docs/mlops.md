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

Cada versao do modelo registra `pip_requirements` com versoes exatas:

```python
pip_requirements=[
    "scikit-learn==1.7.2",
    "xgboost==3.4.1",
    "cloudpickle==3.1.1",
    "pandas",
    "numpy",
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
* Configuracao central em `config/project_config.yml` carregada via `src/config.py`
* `requirements.txt` e `pyproject.toml` para dependencias do projeto
* Declarative Automation Bundle (`databricks.yml`) para deploy de recursos

## Configuracao Central

Arquivo: `config/project_config.yml`

```yaml
project:
  name: databricks-fraud-detection

feature_store:
  table: fraud_detection_dev.gold.fraud_features
  split_methodology: temporal

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

Loader: `src/config.py` com funcao `load_config()`.

## Metricas de Producao

Monitoramento do endpoint nao implementado ainda. Planejado:
* Drift detection (PSI nas features de serving)
* Latencia do endpoint
* Taxa de fraude detectada vs esperada
* Alertas sobre degradacao de PR-AUC