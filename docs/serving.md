# Serving em Tempo Real

## Arquitetura Atual: Direct Features (v13+)

O modelo v13 utiliza direct features - o caller envia todas as 152 features diretamente no payload REST, sem FeatureLookup ou Online Store. Esta arquitetura elimina a dependencia de componentes externos e reduz a latencia de serving.

## Features de Serving

Das 156 colunas da Gold, 152 sao features de serving. As colunas excluidas:

| Coluna | Motivo da exclusao |
|---|---|
| `TransactionID` | Primary key (rastreabilidade, nao feature) |
| `TransactionDT` | Timeseries key (rastreabilidade, nao feature) |
| `target` | Label predita pelo modelo |
| `split` | Rastreabilidade de treino |
| `class_weight` | Configuracao de treino |

As 152 features estao definidas em `src/databricks_fraud_model/features/features.py` (`MODEL_FEATURES`) como single source of truth.

## Pinning de Dependencias

Os `pip_requirements` do modelo devem fixar as versoes exatas das bibliotecas usadas no treino. O serving environment instala os pacotes listados; se as versoes nao forem fixadas, pode instalar versoes diferentes, causando falha na desserializacao de objetos pickle (o ColumnTransformer embute classes internas da versao usada).

**Versoes salvas no modelo (requirements.txt):**

```
mlflow==3.8.1
xgboost
scikit-learn==1.6.1
pandas
numpy
cloudpickle==3.1.1
```

**⚠️ Atencao:** `xgboost`, `pandas` e `numpy` estao sem pin de versao. O serving endpoint instalara a versao mais recente disponivel, que pode ser incompativel com o modelo treinado. No proximo registro do modelo, fixar todas as versoes:

```python
pip_requirements=[
    "scikit-learn==1.6.1",
    "xgboost==<versao_do_treino>",
    "cloudpickle==3.1.1",
    "pandas==<versao_do_treino>",
    "numpy==<versao_do_treino>",
]
```

## Arquitetura de Serving

```
Request (152 features em dataframe_records)
  -> POST /serving-endpoints/model_fraud/invocations
  -> pyfunc: model_input[feature_names] -> preprocessor.transform()
  -> xgboost.predict_proba() -> threshold 0.83
  -> Response: {probability_fraud: float, is_fraud: int}
```

| Componente | Nome |
|---|---|
| Feature Table (treino) | `fraud_detection_dev.gold.fraud_features` |
| Feature Table (producao) | `fraud_detection_dev.gold.fraud_features_prd` |
| Modelo UC | `fraud_detection_dev.models.lgbm_fraud_model` |
| Alias | `@champion` (v1) |
| Endpoint | `model_fraud` |
| Compute | CPU Small, scale-to-zero |
| Threshold | 0.83 (aplicado dentro do pyfunc) |

## Scoring via REST (Producao)

O notebook `04_scoring` envia as 152 features diretamente ao endpoint REST sem FeatureLookup, usando `score_batch()` da package:

```python
from databricks_fraud_model import score_batch, save_predictions

dados = spark.sql(f"SELECT * FROM {catalog}.gold.fraud_features_prd").toPandas()
preds = score_batch(dados, n=1000)
save_predictions(dados, preds)
```

O `score_batch()` envia `dataframe_records` ao endpoint `/serving-endpoints/model_fraud/invocations` via `WorkspaceClient`. A latencia e medida com `time.perf_counter()` e salva em `bronze.monitoring_raw_events`.

Detalhes do monitoramento em [monitoring.md](monitoring.md).

## Depreciacao do Online Store

Em Out/2026, o Online Store foi completamente descontinuado e removido:

- Online store `fraud-detection-online-store` deletado via `fe.delete_online_store()`
- Tabelas `fraud_features_serving` e `event_log_*` (MANAGED) removidas
- Tabelas `fraud_features_online_v4` e `fraud_features_serving_online_v5` (FOREIGN) ficaram orfas
- Modelo re-registrado como v13 com `serving_mode=direct_features` - sem FeatureLookup

Motivo: Direct features eliminam a complexidade de sincronizacao Online Store e permitem controle total das features no payload REST.

## Notas

1. Direct features: o caller e responsavel por enviar todas as 152 features no payload.
2. Fixar dependencias: sempre usar versoes exatas nos `pip_requirements`.
3. NaN para None: valores NaN devem ser convertidos para None antes do envio REST (conformidade JSON).
4. Scale-to-zero: o endpoint escala para zero apos inatividade; primeira requisicao pode ter cold start (~37s).