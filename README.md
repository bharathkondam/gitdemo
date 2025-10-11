# Customer Churn Analytics Platform (Prefect)

This repository implements the complete assignment using **Prefect 3**, **scikit-learn**, **FastAPI**, and **Streamlit**. The codebase provides:

- A Prefect data pipeline scheduled every 2 minutes (Sub-objective 1)
- A Prefect machine-learning pipeline that trains two models and logs four metrics (Sub-objective 2)
- An API and dashboard exposing application metadata (Sub-objective 3)

The solution is entirely local — Prefect pulls code from the same workspace. Deployments are defined in `prefect.yaml`.

---

## Sub-Objective 1 – Data Pipeline

| Activity | Implementation |
|----------|----------------|
| **1.1 Business Understanding** | README documents churn problem using the Telco dataset (7,043 customers). |
| **1.2 Data Ingestion** | `flows/data_pipeline_flow.py` → `download_dataset_task()` downloads CSV from IBM GitHub. |
| **1.3 Data Pre-processing** | `preprocess_dataset_task()` (module `src/pipeline/preprocessing.py`) produces summary statistics, missing values, dtypes, imputation, scaling, and saves a parquet dataset. |
| **1.4 EDA** | `run_eda_task()` (module `src/pipeline/eda.py`) creates correlation heatmap, tenure histogram, categorical churn plots, and RandomForest feature importance. Artifacts saved in `reports/figures/`. |
| **1.5 DataOps** | Prefect deployment `data-pipeline/interval` (cron `*/2 * * * *`) logs to `logs/pipeline.log`, writes metadata to `reports/run_metadata.json`, and the Streamlit dashboard (`dashboard/app.py`) visualises the results. |

## Sub-Objective 2 – Machine Learning Pipeline

| Activity | Implementation |
|----------|----------------|
| **2.1 Model Preparation** | `src/ml/modeling.py` trains Logistic Regression & Gradient Boosting. |
| **2.2 Model Training** | Prefect deployment `ml-pipeline/interval` consumes the processed dataset and fits both models (70/30 train-test). |
| **2.3 Model Evaluation** | Accuracy, precision, recall, F1 logged in `reports/metrics.json`. |
| **2.4 MLOps** | Metrics and model artifact paths appended to `reports/model_run_metadata.json`. Streamlit and FastAPI expose the latest numbers. |

## Sub-Objective 3 – API Access

| Activity | Implementation |
|----------|----------------|
| **3.1 Retrieve details** | `src/api/main.py` reads run metadata & metrics via Prefect-generated files. |
| **3.2 Display details** | `/api/details`, `/api/metrics`, `/api/model-runs`, `/api/artifacts` return at least four key application values. |

---

## Project Structure

```
├── airflow/                     # (optional) legacy; can be ignored
├── artifacts/models/            # persisted models (.joblib)
├── data/raw | data/processed    # datasets
├── dashboard/app.py             # Streamlit dashboard
├── flows/
│   ├── data_pipeline_flow.py    # Prefect data pipeline
│   └── ml_pipeline_flow.py      # Prefect ML pipeline
├── logs/pipeline.log            # structured logs
├── reports/                     # summary tables, figures, metrics
├── src/
│   ├── api/main.py              # FastAPI service
│   ├── ml/modeling.py           # ML training & logging
│   └── pipeline/                # ingestion, preprocessing, eda, config, logging
├── tests/                       # smoke tests
├── prefect.yaml                 # Prefect deployments (2-minute cron)
└── requirements.txt
```

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Prefect Setup & Deployments

1. Initialise Prefect (creates local profile):
   ```bash
   prefect profile create local
   prefect config view
   ```
2. Deploy the flows:
   ```bash
   prefect deploy --prefect-file prefect.yaml --all
   ```
3. Start a worker (new terminal):
   ```bash
   prefect worker start --pool default-agent-pool
   ```
4. Trigger manual runs:
   ```bash
   prefect deployment run data-pipeline/interval
   prefect deployment run ml-pipeline/interval
   ```

The `prefect.yaml` schedules both deployments every two minutes. View runs in Prefect UI or CLI (`prefect deployment ls`).

## Monitoring & API

- **Dashboard:** `streamlit run dashboard/app.py`
- **REST API:** `uvicorn src.api.main:app --reload`

Useful endpoints:
- `GET /api/health`
- `GET /api/details`
- `GET /api/metrics`
- `GET /api/model-runs`
- `GET /api/artifacts`

## Testing

```bash
pytest
```

Simple smoke tests confirm directories exist and preprocessing runs without errors (extend as needed).

---

## Notes

- To change the dataset, update `PipelineConfig.dataset_url` & rerun deployments.
- Prefect Cloud integration is optional; everything runs locally. To use Cloud, login (`prefect cloud login`) and run `prefect deploy --prefect-file prefect.yaml --all` there.
- Logs, metadata, and metrics are capped to the last 20 entries to avoid uncontrolled growth.
