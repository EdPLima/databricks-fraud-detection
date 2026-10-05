# Serving em Tempo Real

## Feature Store Online

A Feature Store do projeto opera em duas camadas:

**Offline Store** (`fraud_detection_dev.gold.fraud_features`): tabela Delta no Unity Catalog com 590.540 linhas (train + test + val) e 156 colunas. Usada para treino, validacao e batch scoring. Sem requisito de baixa latencia.

**Online Store** (`fraud-detection-online-store`, CU_8): armazenamento chave-valor de baixa latencia (~512 MB). O serving endpoint consulta este store em milissegundos para obter as features da transacao em tempo real.

## Features de Serving

Das 156 colunas da Gold, 152 sao features de serving. As 4 excluidas sao definidas em `config/project_config.yml` (`preprocessing.exclude_cols`) mais `split`:

| Coluna | Motivo da exclusao |
|---|---|
| `TransactionID` | Primary key (lookup, nao feature) |
| `TransactionDT` | Timeseries key (lookup, nao feature) |
| `target` | Label predita pelo modelo |
| `split` | Rastreabilidade de treino |
| `class_weight` | Configuracao de treino |

## Tabela Serving Compacta

A Gold completa (574 MB) excede o limite do CU_8 (~512 MB). A tabela `fraud_detection_dev.gold.fraud_features_serving` foi criada como projecao otimizada para o Online Store.

**Dimensoes:** 114.208 linhas (split=val) com 154 colunas (2 PKs + 152 features), ~134 MB.

**Decisoes de design:**

1. Apenas `split = val`: transacoes mais recentes. Treino e teste nao sao consultados em producao. Reducao de 80% das linhas.
2. Sem `target` e `split`: nao sao inputs do modelo. Reducao de 156 para 154 colunas.
3. Mesmas PKs (`TransactionID` + `TransactionDT` timeseries): compatibilidade com o FeatureLookup.
4. Todas as 152 features preservadas: nenhuma feature removida para nao degradar performance.

Criada com `FeatureEngineeringClient.create_table()`. Change Data Feed habilitado para sincronizacao continua.

## FeatureLookup

O modelo foi registrado com `FeatureEngineeringClient.log_model()` e `FeatureLookup` apontando para `fraud_detection_dev.gold.fraud_features_serving`. O FeatureSpec empacotado no modelo contem:

- `table_name`: `fraud_detection_dev.gold.fraud_features_serving`
- `lookup_key`: `TransactionID`
- `timestamp_lookup_key`: `TransactionDT`
- `feature_names`: 152 features

Em producao, o request recebe apenas `TransactionID` + `TransactionDT`. O endpoint busca as 152 features no Online Store via FeatureSpec, aplica o preprocessor (OrdinalEncoder + StandardScaler + SimpleImputer), executa o XGBoost e retorna `probability_fraud` + `is_fraud`.

## Pinning de Dependencias

Os `pip_requirements` do modelo devem fixar as versoes exatas das bibliotecas usadas no treino. O serving environment instala os pacotes listados; se as versoes nao forem fixadas, pode instalar versoes diferentes, causando falha na desserializacao de objetos pickle (o ColumnTransformer embute classes internas da versao usada).

```python
pip_requirements=[
    "scikit-learn==1.7.2",
    "xgboost==3.4.1",
    "cloudpickle==3.1.1",
    "pandas",
    "numpy",
]
```

## Arquitetura

```
Request (TransactionID, TransactionDT)
  -> Online Store lookup (fraud_features_serving_online, CU_8)
  -> 152 features retornadas
  -> pyfunc FraudDetectionModel: preprocessor.transform() -> xgboost.predict_proba()
  -> threshold 0.83 aplicado
  -> Response: {probability_fraud: float, is_fraud: int}
```

| Componente | Nome |
|---|---|
| Feature Table (offline) | `fraud_detection_dev.gold.fraud_features` |
| Feature Table (serving) | `fraud_detection_dev.gold.fraud_features_serving` |
| Online Store | `fraud-detection-online-store` (CU_8) |
| Online Table | `fraud_detection_dev.gold.fraud_features_serving_online` |
| Modelo UC | `fraud_detection_dev.models.xgboost_fraud_model` |
| Endpoint | `model_fraud_xgb` |

## Pipeline de Sincronizacao

Apos `fe.publish_table()`, um pipeline continua e criado automaticamente. Monitora o Change Data Feed da Delta table e replica mudancas para o Online Store em tempo real.

## Notas

1. Separar treino de serving: Gold para treino, serving table para producao.
2. Fixar dependencias: sempre usar versoes exatas nos `pip_requirements`.
3. Validar FeatureLookup: modelo e Online Store devem referenciar a mesma tabela.
4. Habilitar Change Data Feed: necessario para o pipeline de sincronizacao.