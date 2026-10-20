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

21: | Modelo        | PR-AUC | ROC-AUC | F1   | Precision | Recall | KS    | Brier Score |
22: |--------------|--------|---------|------|-----------|--------|-------|-------------|
23: | LightGBM     | 0.5043 | 0.8858  | 0.3797 | 0.8606   | 0.2435 | 0.6171 | 0.0269      |
24: | XGBoost      | 0.5009 | 0.8854  | 0.3862 | 0.8516   | 0.2497 | 0.6163 | 0.0268      |
25: | LogisticRegression | 0.3102 | 0.8326 | 0.1917 | 0.8121 | 0.1917 | 0.5129 | 0.0302 |
26: | RandomForest | 0.2159 | 0.8345 | 0.1231 | 0.8782 | 0.1231 | 0.5006 | 0.0311 |
27: 
28: LightGBM e XGBoost apresentaram resultados próximos, mas LightGBM foi escolhido para produção pela eficiência computacional, escalabilidade e resultados equivalentes. PR-AUC é a métrica prioritária por ser robusta em classes desbalanceadas.

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

99: LightGBM feature importance (gain). Top features:
100: * Variáveis V com alto IV (V258, V257, V265)
101: * `card1_amt_mean` / `card1_amt_std` (agregações por cartão)
102: * `TransactionAmt` (valor da transação)
103: * `DeviceInfo` (informação do dispositivo)
104: * Features temporais (`transaction_hour`)

### SHAP

108: SHAP analysis com 2.000 amostras do conjunto de teste. SHAP summary plot confirma que variáveis V e agregações por cartão são as maiores contribuidoras para a detecção de fraude.

## Deploy

Notebook: `04_DEPLOYED_MODEL`. Experimento: [fraud_detection_deployment](https://dbc-6181e49f-87e5.cloud.databricks.com/ml/experiments/28212970340608).

113: 
114: Modelo LightGBM registrado no Unity Catalog como `fraud_detection_dev.models.lgbm_fraud_model` com alias `@champion`.
115: 
116: Deploy como pyfunc com FeatureLookup para a tabela serving compacta. Detalhes da arquitetura de serving em [serving.md](serving.md).