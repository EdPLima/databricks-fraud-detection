# Pipeline ML de Detecção de Fraude

Pipeline de machine learning para detectar transações fraudulentas no Databricks com retreino automatizado e deployment Champion/Challenger.

## Quick Start

```bash
# Instalar dependências
pip install -e .

# Deploy para desenvolvimento
databricks bundle deploy -t dev

# Executar pipeline de retreino
databricks bundle run -t dev retraining_job
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
│   └── gold/           # Agregações de negócio
├── retraining/         # Pipeline ML (6 estágios)
│   ├── 01_prepare_data
│   ├── 02_feature_engineering
│   ├── 03_train_challenger
│   ├── 04_evaluate_challenger
│   ├── 05_ab_test
│   └── 06_promote_model
├── src/                # Módulos Python reutilizáveis
├── tests/              # Testes unitários e de integração
├── notebooks/          # Análise exploratória
└── databricks.yml      # Configuração DABs
```

## Arquitetura

### Workflow MLOps

O projeto segue as melhores práticas do [Databricks MLOps Stacks](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-stacks/) com 3 ambientes isolados, CI/CD automatizado e padrão Champion/Challenger.

![MLOps Architecture](docs/images/mlops-architecture.png)

**Componentes principais:**
* **Git Provider:** Versionamento de código com CI/CD (testes unitários em PRs, testes de integração em staging)
* **Development Workspace:** Experimentação, EDA, feature engineering, treino inicial
* **Staging Workspace:** Validação do pipeline, treino do challenger, avaliação vs champion
* **Production Workspace:** Serving em tempo real, batch inference, A/B testing, monitoramento
* **Unity Catalog:** Catálogos separados por ambiente com tabelas, feature tables e modelos
* **MLflow:** Tracking, model registry, aliases @champion/@challenger

**Status atual vs planejado:**

| Componente | Status |
|------------|--------|
| Development workspace + MLflow tracking | ✅ Implementado |
| Feature Store (Unity Catalog) | ✅ Implementado |
| Model Registry + Serving endpoint | ✅ Implementado |
| Git source control | ✅ Implementado |
| Staging/Production workspaces | 🔄 Planejado (catálogos hml/prd) |
| CI/CD (GitHub Actions) | 🔄 Planejado |
| Testes automatizados | 🔄 Planejado |
| Batch inference job | 🔄 Planejado |
| Monitoramento & drift detection | 🔄 Planejado |

### Pipeline de Dados

**Bronze (raw) → Silver (features) → Gold (aggregates)**

**Pipeline ML:** Retreino automatizado com padrão Champion/Challenger
* Modelo Champion serve tráfego de produção
* Challenger treinado e avaliado contra o Champion
* Promoção automatizada baseada em testes estatísticos e métricas de negócio
* MLflow tracking para experimentos e model registry

**Stack Tecnológico**
* Plataforma: Databricks (serverless compute)
* Armazenamento: Delta Lake (ACID, time travel)
* ML: Scikit-learn, XGBoost, LightGBM
* Orquestração: Databricks Jobs com DABs
* Governança: Unity Catalog

## Executando o Pipeline

**Desenvolvimento**
```bash
databricks bundle deploy -t dev
databricks bundle run -t dev retraining_job
```

**Produção**
```bash
databricks bundle deploy -t prd
# Configurar execuções agendadas via Databricks Jobs
```

## Documentação

Veja `/docs` para guias detalhados:
* Arquitetura e padrões de design
* Especificações de feature engineering
* Critérios de avaliação de modelo
* Procedimentos de deployment

