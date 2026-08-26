# Fraud Detection System

End-to-end ML system for detecting fraudulent transactions on Databricks with automated retraining and Champion/Challenger deployment.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Deploy to development
databricks bundle deploy -t dev

# Run retraining pipeline
databricks bundle run -t dev retraining_job
```

## Dataset

IEEE-CIS Fraud Detection dataset from Kaggle with real-world e-commerce transactions.

| Split | Records | Fraud Rate |
|-------|---------|------------|
| Train | 590,540 | 3.50% (20,663 frauds) |
| Test | 506,691 | - |

**Key Characteristics**
* 394 transaction features (amount, card info, temporal patterns)
* 41 identity features (device fingerprinting, browser data)
* Class imbalance requiring specialized techniques
* Partial identity coverage (24% of transactions)

**Environments:** Data organized in Unity Catalog across `fraud_detection_dev`, `fraud_detection_hml`, and `fraud_detection_prd` with medallion architecture (bronze/silver/gold).

## Project Structure

```
├── pipeline/
│   ├── bronze/         # Raw data ingestion
│   ├── silver/         # Cleaned & transformed
│   └── gold/           # Business aggregations
├── retraining/         # ML pipeline (6 stages)
│   ├── 01_prepare_data
│   ├── 02_feature_engineering
│   ├── 03_train_challenger
│   ├── 04_evaluate_challenger
│   ├── 05_ab_test
│   └── 06_promote_model
├── src/                # Reusable Python modules
├── tests/              # Unit & integration tests
├── notebooks/          # Exploratory analysis
└── databricks.yml      # DABs configuration
```

## Architecture

**Data Pipeline:** Bronze (raw) → Silver (features) → Gold (aggregates)

**ML Pipeline:** Automated retraining with Champion/Challenger pattern
* Champion model serves production traffic
* Challenger trained and evaluated against Champion
* Automated promotion based on statistical tests and business metrics
* MLflow tracking for experiments and model registry

**Tech Stack**
* Platform: Databricks (serverless compute)
* Storage: Delta Lake (ACID, time travel)
* ML: Scikit-learn, XGBoost, LightGBM
* Orchestration: Databricks Jobs with DABs
* Governance: Unity Catalog

## Running the Pipeline

**Development**
```bash
databricks bundle deploy -t dev
databricks bundle run -t dev retraining_job
```

**Production**
```bash
databricks bundle deploy -t prd
# Configure scheduled runs via Databricks Jobs
```

## Documentation

See `/docs` for detailed guides:
* Architecture and design patterns
* Feature engineering specifications
* Model evaluation criteria
* Deployment procedures

