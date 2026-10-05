# Modelagem

## Walk-Forward Cross-Validation

Notebook: `03_MODEL_TRAINING`. Experimento: [fraud_detection_walkforward](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/986719377495176).

Split temporal dentro do treino (dias 1-102). 3 janelas com expanding window:

| Janela | Treino (dias) | Teste (dias) |
|---|---|---|
| 1 | 1-61 | 62-76 |
| 2 | 1-76 | 77-91 |
| 3 | 1-91 | 92-102 |

Class weight aplicado: `weight_fraud = 1 / fraud_ratio`, `weight_legit = 1 / (1 - fraud_ratio)`.

## Selecao de Modelos

4 modelos comparados no walk-forward CV. Experimento: [selection_model_timeseries_split](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/4081575290465878).

| Modelo | PR-AUC (mean) | ROC-AUC (mean) | F1 (mean) | KS (mean) |
|---|---|---|---|---|
| **XGBoost** | **0.4856** | **0.8803** | **0.3116** | **0.6055** |
| LightGBM | 0.4698 | 0.8760 | 0.3126 | 0.6011 |
| LogisticRegression | 0.3450 | 0.8342 | 0.2167 | 0.5238 |
| RandomForest | 0.3127 | 0.8269 | 0.2110 | 0.5035 |

XGBoost selecionado por PR-AUC superior. PR-AUC e a metrica prioritaria por ser robusta em classes desbalanceadas (3,5% de fraudes).

### Configuracao XGBoost (baseline)

```python
XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=50,
    gamma=0.001,
    max_bin=1500,
    eval_metric="aucpr",
    n_jobs=-1,
)
```

## HPO com Optuna

Notebook: `03_MODEL_TRAINING`, celula `Optuna HPO v2`. Experimento: [fraud_detection_walkforward](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/986719377495176).

* 20 trials com nested MLflow runs (parent + child)
* Otimizacao para maximizar PR-AUC na validacao
* Espaco de busca: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_lambda, reg_alpha
* Run pai (melhor modelo): `25178e58be204eae80693a7849d634ed`

## Avaliacao

### Metricas no conjunto de teste

| Metrica | Valor |
|---|---|
| PR-AUC | 0.3919 |
| ROC-AUC | 0.8674 |
| KS | 0.5741 |
| F1 (threshold 0.5) | 0.2521 |
| Precision | 0.1545 |
| Recall | 0.6841 |
| Brier Score | 0.111 |

### Otimizacao de Threshold

Threshold otimizado na **validacao** (nao no teste, para evitar data leak). Metrica: maximizacao de F1-score.

Para cada threshold entre 0.01 e 0.99, calculado: TP, FP, TN, FN, precision, recall, F1, FPR, Recall$ (valor recuperado), FP$ (custo falsos positivos), Net$ (retorno liquido).

**Threshold otimo: 0.83** (configurado em `project_config.yml`).

### Faixas de Risco

Score do modelo segmentado em 4 faixas:

| Faixa | Score | Taxa de fraude |
|---|---|---|
| Baixo | < percentil 50 | ~0,3% |
| Medio | percentil 50-80 | ~2% |
| Alto | percentil 80-95 | ~10% |
| Critico | > percentil 95 | ~40% |

Teste Chi-Quadrado confirma que a segmentacao e estatisticamente significativa (p < 0.001). A faixa Critico concentra a maioria das fraudes.

### Lift Acumulado

Top 20% dos scores capturam aproximadamente 70% das fraudes (lift ~3.5x vs aleatorio).

## Interpretabilidade

### Feature Importance

XGBoost feature importance (gain). Top features:
* Variaveis V com alto IV (V258, V257, V265)
* `card1_amt_mean` / `card1_amt_std` (agregacoes por cartao)
* `TransactionAmt` (valor da transacao)
* `DeviceInfo` (informacao do dispositivo)
* Features temporais (`transaction_hour`)

### SHAP

SHAP analysis com 2.000 amostras do conjunto de teste. SHAP summary plot confirma que variaveis V e agregacoes por cartao sao as maiores contribuidoras para a deteccao de fraude.

## Deploy

Notebook: `04_DEPLOYED_MODEL`. Experimento: [fraud_detection_deployment](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/28212970340608).

Modelo registrado no Unity Catalog como `fraud_detection_dev.models.xgboost_fraud_model` com alias `@champion`.

Deploy como pyfunc com FeatureLookup para a tabela serving compacta. Detalhes da arquitetura de serving em [serving.md](serving.md).