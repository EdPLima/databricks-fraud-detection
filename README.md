# Pipeline ML de Detecção de Fraude

Pipeline de machine learning para detectar transações fraudulentas no Databricks com Feature Store, serving em tempo real, monitoramento MLOps e pipeline de retreino Champion/Challenger.
<p align="center">
  <img src="docs/images/Arquitetura MLOps para Detecção de Fraudes.png" alt="Arquitetura MLOps para Detecção de Fraudes" width="700"/>
</p>



## Perguntas de Negócio

- **Taxa de fraudes:** 3,5% das transações.
- **Relações mais fortes:** Variáveis V e agregações de cartão (foco em V258, V257, V265, C4, C8) têm maior poder preditivo, forte correlação e redundância identificada.
- **Testes estatísticos:**
    - KS (Kolmogorov-Smirnov): Selecionou C4, C8, V303, V304, V302 como features numéricas com maior poder discriminante.
    - Chi-Quadrado/V de Cramer: Card1, DeviceInfo, id_31, id_30 destacaram-se em associação com fraude.
- **Impacto financeiro:** $3,08M — fraudes concentradas em poucos emissores (card1).
- **Dispositivos e identidades:** Combinações de DeviceType, DeviceInfo e id_31 aumentam substancialmente o risco de fraude.
- **Valores das fraudes:** Fraudes têm valores e variâncias muito superiores às legítimas; variáveis como V200 chegam a 200x mais variância em fraudes.
- **Variáveis anônimas:** 19 variáveis V têm IV > 0,5 — grande potencial preditivo e importância no modelo.
- **Limiar ideal:** Threshold = 0.83 (F1 ótimo). Top 5% dos scores concentra ~40% das fraudes; top 20% captura ~70% dos casos.


## Próximos Passos

- **Job dedicado de geração de dados de inferência:** Automatizar a preparação/atualização dos dados a serem enviados para pontuação pelo modelo (por exemplo, extração/transf. diária de lotes a serem inferidos).
- **Aprimorar dashboard de monitoramento:** Incluir gráficos e métricas mais robustos: drift por feature, evolução do volume de inferências, fatias de risco, latências e erros.
- **Ajustar triggers e alarmes operacionais:** Configurar alertas pró-ativos para drift, quedas de recall, volumes abaixo do esperado e erros de serving.



## CI/CD

O projeto usa GitHub Actions para CI/CD com 3 branches:

- **`dev`** - desenvolvimento ativo (CI: lint + testes + bundle validate)
- **`staging`** - homologacao (CI + CD deploy para hml)
- **`main`** - producao (CI + CD deploy para prod)

Retreino agendado automaticamente todo dia 10 de cada mes.

## Quick Start

```bash
# Instalar dependências
pip install -e .

# Deploy para desenvolvimento
databricks bundle deploy -t dev

# Executar ETL pipeline (bronze -> silver -> gold)
databricks bundle run -t dev etl_pipeline

# Executar inference & monitoring (scoring -> dashboard)
databricks bundle run -t dev inference_monitoring
```

## Dataset

Dataset IEEE-CIS Fraud Detection do Kaggle com transações reais de e-commerce.

| Split | Registros | Taxa de Fraude |
|-------|---------|------------|
| Train | 590.540 | 3.50% (20.663 fraudes) |
| Test | 506.691 | - |

**Características principais**
* 394 features de transação (valor, cartão, padrões temporais)
* 41 features de identidade (device fingerprinting, dados de browser)
* Desbalanceamento de classes requerendo técnicas especializadas
* Cobertura parcial de identidade (24% das transações)

**Ambientes:** Dados organizados no Unity Catalog através de `fraud_detection_dev`, `fraud_detection_hml` e `fraud_detection_prod` com arquitetura medallion (bronze/silver/gold).

## Descobertas da EDA

Notebook: `01_EDA`. Fonte: `fraud_detection_dev.silver.fraud_data_clean` (590.540 linhas, 434 colunas).

### Desbalanceamento de classes

| target | count | percentage | valor total |
|---|---|---|---|
| 0 (legítima) | 569.877 | 96,5% | $76,66M |
| 1 (fraude) | 20.663 | 3,5% | $3,08M |

Dataset altamente desbalanceado (ratio 27:1). O impacto financeiro das fraudes representa **$3,08M** — 3,87% do valor total transacionado.

### Outliers e fraude (Isolation Forest)

Isolation Forest (`contamination=0,05`, `n_estimators=300`) marcou 5% das transações (29.527) como anomalias.

**Insight principal:** **17,45% das anomalias são fraudulentas** vs 3,5% na população geral — 5x mais concentrado. Os outliers não foram removidos pois representam comportamentos naturais do fenômeno de fraude.

### Testes estatísticos

**KS (Kolmogorov-Smirnov):** Principais features numéricas discriminantes:

| Feature | KS    | Interpretação              |
|---------|-------|---------------------------|
| C4      | 0,331 | Alto poder discriminante   |
| C8      | 0,316 | Alto poder discriminante   |
| V303    | 0,316 | Alto poder discriminante   |
| V304    | 0,314 | Alto poder discriminante   |
| V302    | 0,312 | Alto poder discriminante   |

**Chi-Quadrado + V de Cramer:** Principais features categóricas associadas à fraude:

| Feature      | Chi2   | V de Cramer | Associação |
|--------------|--------|-------------|------------|
| card1        | 72.289 | 0,350       | Forte      |
| DeviceInfo   | 44.514 | 0,275       | Moderada   |
| id_31        | 20.808 | 0,188       | Moderada   |
| id_30        | 19.758 | 0,183       | Moderada   |

### Variáveis anônimas (V1-V339)

Information Value (IV) calculado para todas as 339 variáveis V. **19 variáveis com IV > 0,5** (poder preditivo muito forte).

**Insight de variância:** 18 das 19 variáveis com alto IV apresentam razão de variância > 3 entre fraudes e legítimas. Transações fraudulentas têm variância muito maior. Exemplo extremo: **V200 tem 200x mais variância em fraudes**. Isso indica que o modelo pode usar a variabilidade dessas features como sinal de fraude.

**Multicolinearidade:** 17 pares de variáveis V com |correlação| > 0,8. Isso exige seleção cuidadosa para evitar redundância.

### Impacto financeiro por segmento

Análise do valor monetário de fraudes por `ProductCD`, `card4`, `card6`, `P_emaildomain` e `card1`. **Concentração de valor fraudado em poucos emissores de cartão** — poucos valores de card1 concentram a maior parte do valor fraudado.

### Dispositivo e identidade

Taxa de fraude por `DeviceType`: desktop e mobile com taxas diferentes. Combinações de `DeviceInfo` + `id_31` (browser) com maior concentração de fraude.

## Redução de Dimensionalidade

De **434 colunas** na Silver para **152 features** no modelo — uma redução de **65%**.

### Etapa 1: Qualidade → 286 features

Removidas 71 features com problemas estruturais:
* Variância quase zero (var < 0,01)
* Cardinalidade problemática (constante ou ID-like: razão único/não-nulo > 95%)
* Quase constante (valor mais frequente em > 95% das linhas)

### Etapa 2: Evidência Univariada → ranking por poder preditivo

* Numéricas: Information Value (IV) + KS
* Categóricas: Chi-Quadrado + V de Cramer
* Score unificado para ranking (escala 0-1)

Top features por evidência: V258 (0,45), D5 (0,44), DeviceInfo (0,42), V257 (0,42), V265 (0,42).

### Etapa 3: Redundância → 152 features

Remoção de features altamente correlacionadas:
* Numéricas: correlação de Pearson |r| > 0,85
* Categóricas: V de Cramer entre pares > 0,85
* Entre pares redundantes, a de menor evidência univariada foi removida
* Exceção: `DeviceType` restaurada por fornecer informação de origem (mobile/desktop)

### Features engenheiradas (+20)

20 features criadas e combinadas com as 132 selecionadas:

| Tipo | Features |
|---|---|
| Temporais | `transaction_hour`, `transaction_day`, `is_morning`, `is_afternoon`, `is_night`, `is_weekend` |
| Valor | `TransactionAmt_log`, `TransactionAmt_round` |
| Email | `email_match`, `P_email_free`, `R_email_free` |
| Agregações cartão | `card1_tx_count`, `card1_amt_mean`, `card1_amt_std`, `TransactionAmt_dev_card`, `TransactionAmt_zscore_card` |
| Agregações endereço | `addr1_tx_count`, `addr1_amt_mean` |
| Dispositivo | `DeviceType_isnull`, `device_mismatch` |

**Total final: 152 features** (20 categóricas + 132 numéricas), versionadas em `MODEL_FEATURES` com hash SHA-256 para garantia de reprodutibilidade.

## Resultados do Modelo

### Seleção de modelos (walk-forward CV)

4 modelos comparados com validação temporal (3 janelas expanding window):

| Modelo        | PR-AUC | ROC-AUC | F1   | Precision | Recall | KS    | Brier Score |
|--------------|--------|---------|------|-----------|--------|-------|-------------|
| LightGBM     | 0.5043 | 0.8858  | 0.3797 | 0.8606   | 0.2435 | 0.6171 | 0.0269      |
| XGBoost      | 0.5009 | 0.8854  | 0.3862 | 0.8516   | 0.2497 | 0.6163 | 0.0268      |
| LogisticRegression | 0.3102 | 0.8326 | 0.1917 | 0.8121 | 0.1917 | 0.5129 | 0.0302 |
| RandomForest | 0.2159 | 0.8345 | 0.1231 | 0.8782 | 0.1231 | 0.5006 | 0.0311 |

LightGBM selecionado para HPO. PR-AUC é a métrica prioritária por ser robusta em classes desbalanceadas.

### HPO com Optuna

30 trials otimizando PR-AUC na validação. Espaço de busca: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_lambda, reg_alpha.

### Métricas no teste

| Métrica | Valor |
|---|---|
| PR-AUC | 0,3919 |
| ROC-AUC | 0,8674 |
| KS | 0,5741 |
| F1 (threshold 0,5) | 0,2521 |
| Precision | 0,1545 |
| Recall | 0,6841 |
| Brier Score | 0,111 |

**Threshold otimizado: 0,83** (maximiza F1 na validação, sem data leak no teste).

### Faixas de risco

| Faixa | Score | Taxa de fraude |
|---|---|---|
| Baixo | < percentil 50 | ~0,3% |
| Médio | percentil 50-80 | ~2% |
| Alto | percentil 80-95 | ~10% |
| Crítico | > percentil 95 | ~40% |

Chi-Quadrado confirma que a segmentação é estatisticamente significativa (p < 0,001).

### Lift acumulado

Top 20% dos scores capturam **~70% das fraudes** (lift ~3,5x vs aleatório).

### Interpretabilidade

* **Feature importance (gain):** variáveis V com alto IV (V258, V257, V265), agregações por cartão (`card1_amt_mean`, `card1_amt_std`), `TransactionAmt`, `DeviceInfo`, features temporais
* **SHAP:** análise com 2.000 amostras confirma que variáveis V e agregações por cartão são as maiores contribuidoras

## Impacto no Negócio

### Cenário atual

* **$3,08M** em transações fraudulentas entre 20.663 casos
* Taxa de fraude base: 3,5% das transações
* Concentração de valor fraudado em poucos emissores de cartão

### Capacidade do modelo

* **Recall de 68,4%**: o modelo identifica 68,4% de todas as fraudes
* Se aplicado para prevenção, pode evitar **~$2,1M** em fraudes (68,4% de $3,08M)
* **Top 20% dos scores capturam ~70% das fraudes**: revisar apenas 20% do volume detecta a maioria dos casos

### Segmentação de risco para revisão manual

A segmentação em 4 faixas permite priorização de revisão:

* **Faixa Crítica (5% do volume, 40% de fraude):** concentra 11x mais fraudes que a baseline — revisar este grupo primeiro maximiza o retorno por transação revisada
* **Faixa Alto (15% do volume, 10% de fraude):** 3x mais fraudes que a baseline
* Faixas Baixo + Médio (80% do volume, <2% de fraude): podem passar com mínima supervisão

### Benefícios operacionais

* Redução de custo de revisão: focar equipe em 5-20% das transações em vez de 100%
* Detecção automatizada com serving em tempo real via endpoint REST (`model_fraud`)
* Monitoramento contínuo de drift (KS + PSI), latência e qualidade das predições
* Retreino automatizado Champion/Challenger com A/B test e gates de promoção

## Estrutura do Projeto

```
├── pipeline/
│   ├── bronze/         # Ingestão de dados brutos
│   ├── silver/         # Limpeza e transformação
│   ├── gold/           # Feature engineering
│   ├── scoring/        # Batch scoring via endpoint REST
│   └── monitoring/     # Dashboard MLOps (drift, latência, alertas)
├── retraining/         # Pipeline de retreino Champion/Challenger (6 notebooks)
├── src/                # Package Python (features, scoring, config)
├── tests/              # Testes unitários e de integração
├── notebooks/          # EDA e experimentação
├── resources/jobs/     # Definições de jobs (DABs)
├── docs/               # Documentação técnica
└── databricks.yml      # Declarative Automation Bundle
```

## Arquitetura

### Workflow MLOps

O projeto segue as melhores práticas do [Databricks MLOps Stacks](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-stacks/) com ambientes isolados, orquestração via DABs e padrão Champion/Challenger (planejado).

![MLOps Architecture](docs/images/mlops-architecture.png)

**Componentes principais:**
* **Git Provider:** Versionamento de código no Databricks Repo
* **Development Workspace:** Experimentação, EDA, feature engineering, treino, scoring, monitoramento
* **Staging/Production Workspaces:** Planejados (catálogos hml/prd no UC)
* **Unity Catalog:** Catálogo `fraud_detection_dev` com tabelas, feature tables e modelos
* **MLflow:** Tracking, model registry, alias @champion

**Status atual vs planejado:**

| Componente | Status |
|------------|--------|
| Development workspace + MLflow tracking | ✅ Implementado |
| Feature Store (Unity Catalog) | ✅ Implementado |
| Model Registry + Serving endpoint | ✅ Implementado |
| Git source control | ✅ Implementado |
| Testes automatizados (unit + integration) | ✅ Implementado |
| Batch inference job | ✅ Implementado |
| Monitoramento & drift detection | ✅ Implementado |
| Staging/Production workspaces | ✅ Catálogos hml/prod no UC |
| CI/CD (GitHub Actions) | ✅ Implementado (ci.yml + cd.yml) |
| Retraining automatizado (Champion/Challenger) | ✅ Implementado (6 notebooks) |

### Pipeline de Dados

**Bronze (raw) → Silver (features) → Gold (aggregates)**

**Pipeline ML:** Treino via walk-forward CV + Optuna HPO, serving via REST endpoint, monitoramento com drift detection
* Modelo LightGBM otimizado, registrado no UC com alias @champion
* Batch scoring via endpoint REST `model_fraud`
* Monitoramento MLOps: data drift (KS + PSI), prediction drift, latência, alertas
* Retreino automatizado Champion/Challenger com A/B test e gates de promoção

**Stack Tecnológico**
* Plataforma: Databricks (serverless compute)
* Armazenamento: Delta Lake (ACID, time travel)
* ML: Scikit-learn, LightGBM, XGBoost, Optuna
* Orquestração: Databricks Jobs com DABs
* Governança: Unity Catalog

## Executando o Pipeline

**Desenvolvimento**
```bash
databricks bundle deploy -t dev
databricks bundle run -t dev etl_pipeline
databricks bundle run -t dev inference_monitoring
```

**Produção**
```bash
databricks bundle deploy -t prod
# Jobs criados no workspace, agendados via DABs
```

## Documentação

Veja `/docs` para guias detalhados:
* [Arquitetura](architecture.md) - visão geral e estrutura do projeto
* [Dados](data.md) - dataset, EDA, preprocessamento, feature engineering
* [Modelagem](modeling.md) - walk-forward CV, HPO, avaliação, threshold
* [Serving](serving.md) - Feature Store online, endpoint REST, scoring
* [Monitoramento](monitoring.md) - drift, latência, alertas
* [MLOps](mlops.md) - MLflow, model registry, jobs, testes

