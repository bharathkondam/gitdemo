# Cloud-Native Customer Churn Analytics Platform

This repo delivers a Prefect-powered, cloud-ready data science application that predicts churn for subscription-based telco customers. The solution combines automated data engineering, multi-model machine learning, and an API surface that exposes operational telemetry and predictive insights.

## Business Problem

Recurring-revenue telcos lose profit when subscribers discontinue service. Industry studies show that retaining existing customers is far cheaper than acquiring new ones, but proactive retention requires identifying risk early. The open **Telco Customer Churn** dataset (IBM sample; 7,043 records, 21 features) captures demographics, service usage, account tenure, billing preferences, and churn labels. We ingest and analyze that dataset to build churn propensity models for prioritizing retention campaigns.

## Solution Overview

- **Data Pipeline (Prefect)**  
  - Ingests the Telco dataset from a public GitHub mirror.  
  - Cleans, imputes, type-casts, scales numeric columns, and encodes categoricals.  
  - Generates summary statistics, missing-value audits, schema snapshots, and correlation/histogram visualizations.  
  - Logs structured JSON events, captures metadata in SQLite, and provides a Streamlit dashboard.  
  - Deployed via Prefect deployments scheduled every two minutes (modifiable via UI).

- **Machine Learning Pipeline**  
  - Loads processed data, splits 70/30 with 20% of training reserved for validation.  
  - Trains two algorithms: Logistic Regression and Gradient Boosting.  
  - Evaluates accuracy, ROC AUC, precision, recall, and F1; captures confusion matrices, ROC curves, SHAP summaries, and feature importance.  
  - Selects the champion model by ROC AUC and persists artifacts to `artifacts/ml_pipeline`.

- **API Layer (FastAPI)**  
  - Health check, pipeline status, deployment metadata, and metrics endpoints.  
  - Prediction endpoint serving online scoring with preprocessing assets from the data pipeline.  
  - Endpoint to list at least four key application details (latest run, metrics, model path, deployment info).

- **Monitoring**  
  - Structured logs written to `logs/pipeline.log`.  
  - Streamlit dashboard surfaces run history, metrics, and tail logs.  
  - Prefect Cloud recommended for hosted orchestration and observability.

## Repository Layout

```
├── data/                 # Raw, interim, processed data (gitignored except .gitkeep)
├── logs/                 # JSON logs
├── reports/              # Auto-generated EDA & model evaluation artifacts
├── artifacts/            # Persisted models and metadata
├── src/
│   ├── api/              # FastAPI app and services
│   ├── common/           # Config, logging, paths
│   ├── data_pipeline/    # Prefect data tasks & flow
│   ├── ml_pipeline/      # Prefect ML tasks & flow
│   └── monitoring/       # Streamlit dashboard
├── deployments/          # Prefect deployment YAMLs (2-minute schedule)
├── scripts/              # Convenience runners for flows
├── tests/                # Light unit tests
└── requirements.txt
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are inside an existing Conda environment, install dependencies with `conda install --file requirements.txt` or `pip install -r requirements.txt` after ensuring packages like `cmake` are available (needed for `pyarrow`).

## Prefect Cloud Integration

```bash
prefect cloud login --key <API_KEY> --workspace <ACCOUNT>/<WORKSPACE>
prefect deployment apply deployments/data_pipeline-deployment.yaml
prefect deployment apply deployments/ml_pipeline-deployment.yaml
prefect worker start --pool default-agent-pool
```

The deployments are scheduled every two minutes; adjust cadence in Prefect Cloud if required.

## Local Pipeline Execution

```bash
prefect server start      # or use Prefect Cloud
prefect worker start --pool default-agent-pool

python scripts/run_data_pipeline.py
python scripts/run_ml_pipeline.py
```

## API Service

```bash
uvicorn src.api.main:app --reload
```

Interactive docs: `http://localhost:8000/docs`

## Monitoring Dashboard

```bash
streamlit run src/monitoring/dashboard.py
```

## Cloud Deployment Notes

- **Containerization**: Use the provided `Dockerfile`; push to a registry for Prefect agents or managed compute (Cloud Run, ECS Fargate, Azure Container Apps).  
- **Storage**: Swap local paths with cloud buckets via environment variables (`DATA_DIR`, `ARTIFACTS_DIR`).  
- **Logging**: Forward JSON logs to tools like CloudWatch, Stackdriver, or Grafana Loki.  
- **Security**: Store Prefect API keys, database URIs, and model secrets with Prefect blocks or managed secret stores (AWS Secrets Manager, GCP Secret Manager).

