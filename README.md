# Pipeline ML de Detecção de Fraude

Pipeline de machine learning para detectar transações fraudulentas no Databricks com Feature Store, serving em tempo real, monitoramento MLOps e pipeline de retreino Champion/Challenger (planejado).

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

**Ambientes:** Dados organizados no Unity Catalog através de `fraud_detection_dev`, `fraud_detection_hml` e `fraud_detection_prd` com arquitetura medallion (bronze/silver/gold).

## Estrutura do Projeto

```
├── pipeline/
│   ├── bronze/         # Ingestão de dados brutos
│   ├── silver/         # Limpeza e transformação
│   ├── gold/           # Feature engineering
│   ├── scoring/        # Batch scoring via endpoint REST
│   └── monitoring/     # Dashboard MLOps (drift, latência, alertas)
├── retraining/         # Pipeline de retreino Champion/Challenger (planejado)
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
| Staging/Production workspaces | 🔄 Planejado (catálogos hml/prd) |
| CI/CD (GitHub Actions) | 🔄 Planejado |
| Retraining automatizado (Champion/Challenger) | 🔄 Planejado (notebooks em branco) |

### Pipeline de Dados

**Bronze (raw) → Silver (features) → Gold (aggregates)**

**Pipeline ML:** Treino via walk-forward CV + Optuna HPO, serving via REST endpoint, monitoramento com drift detection
* Modelo XGBoost otimizado, registrado no UC com alias @champion
* Batch scoring via endpoint REST `model_fraud`
* Monitoramento MLOps: data drift (KS + PSI), prediction drift, latência, alertas
* Retreino automatizado Champion/Challenger: planejado

**Stack Tecnológico**
* Plataforma: Databricks (serverless compute)
* Armazenamento: Delta Lake (ACID, time travel)
* ML: Scikit-learn, XGBoost, Optuna
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
* [Arquitetura](architecture.md) — visão geral e estrutura do projeto
* [Dados](data.md) — dataset, EDA, preprocessamento, feature engineering
* [Modelagem](modeling.md) — walk-forward CV, HPO, avaliação, threshold
* [Serving](serving.md) — Feature Store online, endpoint REST, scoring
* [Monitoramento](monitoring.md) — drift, latência, alertas
* [MLOps](mlops.md) — MLflow, model registry, jobs, testes

