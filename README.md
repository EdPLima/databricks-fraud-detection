# Fraud Detection System

End-to-end ML system for detecting fraudulent transactions on Databricks, featuring automated retraining and Champion/Challenger model deployment.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run development notebooks in sequence
01_EDA_FE → 02_PRE_PROCE → 03_MODEL_TRAINING_EVALUATION

# Deploy production pipeline
databricks bundle deploy
```

## Project Structure

```
├── notebooks/              # Development & exploration
├── pipeline/retraining/    # Production MLOps pipeline (6 stages)
├── src/                    # Reusable Python modules
├── tests/                  # Unit and integration tests
└── databricks.yml          # Bundle configuration
```

## Business Questions Addressed

1. What characteristics are associated with fraud?
2. Are there high-risk segments (product, device, card type)?
3. How does fraud differ from legitimate transactions?
4. Which features have the strongest discriminative power?
5. Are there temporal or behavioral patterns?
6. What is the fraud vs false positive trade-off?
7. Does the model reduce financial losses?
8. Is Challenger better than Champion for business metrics?

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources (Unity Catalog)                 │
│                    Raw Transaction Data Tables                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Development Phase (notebooks/)                 │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │   01_EDA_FE  │──▶│ 02_PRE_PROCE │──▶│ 03_MODEL_TRAINING  │  │
│  │              │   │              │   │   _EVALUATION      │  │
│  └──────────────┘   └──────────────┘   └────────────────────┘  │
│       Research          Transform            Baseline Model      │
└─────────────────────────────┬───────────────────────────────────┘
                              │ Insights & Patterns
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Production Pipeline (pipeline/retraining/)          │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │01_prepare   │──▶│02_feature    │──▶│03_train          │    │
│  │   _data     │   │  _engineering│   │   _challenger    │    │
│  └─────────────┘   └──────────────┘   └────────┬─────────┘    │
│                                                  │               │
│  ┌─────────────┐   ┌──────────────┐            │               │
│  │06_promote   │◀──│05_ab_test    │◀───────────┘               │
│  │   _model    │   │              │   ┌──────────────────┐    │
│  └──────┬──────┘   └──────────────┘   │04_evaluate       │    │
│         │                          ◀───│   _challenger    │    │
│         │                              └──────────────────┘    │
└─────────┼──────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLflow Model Registry                         │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │  Champion Model  │              │ Challenger Model │         │
│  │   (Production)   │              │  (Evaluation)    │         │
│  └──────────────────┘              └──────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

**1. Data Layer (Unity Catalog)**
* Raw transaction data storage
* Feature tables (train/test/validation splits)
* Preprocessed datasets with version control
* Governed access with lineage tracking

**2. Development Environment**
* `01_EDA_FE`: Statistical analysis, feature importance, segmentation
* `02_PRE_PROCE`: Data cleaning, outlier treatment, train/test split
* `03_MODEL_TRAINING_EVALUATION`: Baseline model development and metrics

**3. Production Pipeline**
* `01_prepare_data`: Data validation, quality checks, schema enforcement
* `02_feature_engineering`: Consistent feature transformation, encoding
* `03_train_challenger`: New model training with hyperparameter tuning
* `04_evaluate_challenger`: Performance metrics, drift detection, validation
* `05_ab_test`: Statistical significance testing (Champion vs Challenger)
* `06_promote_model`: Automated promotion based on business rules

**4. Model Registry (MLflow)**
* Version control for all models
* Champion/Challenger model tracking
* Model lineage and experiment metadata
* Deployment stage management (Staging/Production)

**5. Orchestration Layer (Databricks Jobs)**
* Scheduled pipeline execution
* Event-triggered retraining
* Failure handling and alerting
* Parameter management

### Design Patterns

**Champion/Challenger Pattern**
* Champion: Current production model serving predictions
* Challenger: New candidate model evaluated against Champion
* Promotion criteria: Statistical significance + business metrics improvement
* Rollback capability if Challenger underperforms

**Feature Store Pattern**
* Centralized feature definitions
* Consistent feature computation (training/serving)
* Point-in-time correctness for temporal features
* Feature reuse across models

**Data Versioning**
* Immutable training datasets with timestamps
* Reproducible model training
* Delta Lake time travel for historical analysis

### Technology Decisions

| Component | Technology | Justification |
|-----------|------------|---------------|
| Compute | Databricks Serverless | Auto-scaling, cost efficiency |
| Storage | Delta Lake | ACID transactions, time travel |
| ML Framework | Scikit-learn/XGBoost | Production-ready, interpretable |
| Experiment Tracking | MLflow | Native Databricks integration |
| Orchestration | Databricks Jobs | Unified platform, no external deps |
| Governance | Unity Catalog | Centralized access control |

## Tech Stack

* Platform: Databricks (serverless)
* Languages: Python, SQL
* ML: Scikit-learn, XGBoost, LightGBM
* MLflow: Experiment tracking and model registry
* Unity Catalog: Data and model governance

## Key Features

* Automated retraining pipeline
* Champion/Challenger framework with A/B testing
* Feature Store integration
* Model performance monitoring
* Spark-based scalable processing

## Prerequisites

* Databricks workspace
* Unity Catalog access
* MLflow Model Registry permissions

## Running the Pipeline

**Manual Execution:**
```bash
# Run pipeline stages sequentially (01-06)
```

**Automated Scheduling:**
* Configure as Databricks Job
* Set triggers (schedule, data arrival, manual)

## Documentation

Detailed documentation available in `/docs`:
* Business requirements and metrics
* Feature engineering guide
* Model evaluation criteria
* Deployment procedures

## Contributing

Follow project conventions:
* Add tests for new features
* Update documentation
* Follow existing code style

## License

[Specify your license]