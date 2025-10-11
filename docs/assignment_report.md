# Customer Churn Analytics Platform – Assignment Report

## 1. Executive Summary
This document describes the end-to-end implementation of the churn analytics platform using Prefect, scikit-learn, FastAPI, and Streamlit. The solution satisfies each activity across the three sub-objectives in the assignment:
- **Data pipeline** orchestrated by Prefect with automated ingestion, preprocessing, and exploratory data analysis every 2 minutes.
- **Machine-learning pipeline** that trains two algorithms (Logistic Regression & Gradient Boosting) and logs key metrics.
- **API access** and dashboard components exposing processed information and run metadata.

All code resides in the repository `~/Projects/ApiDriven`. Prefect deployments are defined in `prefect.yaml`; supporting modules live under `src/`.

---

## 2. Business Understanding (1.1)
- **Problem:** Telecom providers need to identify customers likely to churn to intervene early.
- **Dataset:** IBM’s Telco Customer Churn (7,043 entries, 21 features) covering demographics, services, billing, and a churn label.
- **Decision Drivers:** Understanding relationships between features (e.g., contract type, tenure, payment method) and churn probability.

---

## 3. Data Pipeline (Sub-Objective 1)

### 3.1 Ingestion (1.2)
- Module `src/pipeline/ingestion.py` downloads CSV from GitHub (Dataset URL specified in `PipelineConfig`).
- Prefect Task: `download_dataset_task` in `flows/data_pipeline_flow.py`.
- Raw files stored under `data/raw/telco_customer_churn.csv`.

### 3.2 Preprocessing (1.3)
- Module `src/pipeline/preprocessing.py`:
  - Generates summary statistics, missing value counts, and data types.
  - Cleans numeric columns (converts `TotalCharges`), imputes missing values, scales numerics, and encodes categoricals.
  - Saves the processed dataset to `data/processed/telco_churn_processed.parquet`.
- Prefect Task: `preprocess_dataset_task`.

### 3.3 Exploratory Data Analysis (1.4)
- Module `src/pipeline/eda.py`:
  - Correlation heatmap, tenure histogram, categorical churn distributions.
  - RandomForest feature importance; accuracy report stored in `reports/`.
- Prefect Task: `run_eda_task`.

### 3.4 DataOps Automation (1.5)
- Flow `data_pipeline_flow` orchestrates the tasks.
- Deployment `data-pipeline/interval` in `prefect.yaml` schedules the flow every **2 minutes** (cron `*/2 * * * *`).
- Structured logs in `logs/pipeline.log`.
- Metadata appended to `reports/run_metadata.json` for dashboard/API access.

---

## 4. Machine Learning Pipeline (Sub-Objective 2)

### 4.1 Model Preparation (2.1)
- Module `src/ml/modeling.py` selects **Logistic Regression** and **Gradient Boosting Classifier**.

### 4.2 Training (2.2)
- Flow `ml_pipeline_flow` reads the processed parquet from the data pipeline, splits dataset **70/30** (stratified), and trains both models.
- Artifacts persisted in `artifacts/models/*.joblib`.

### 4.3 Evaluation (2.3)
- Metrics computed per model: accuracy, precision, recall, F1-score.
- Stored in `reports/metrics.json`.

### 4.4 MLOps Logging (2.4)
- Run metadata, metric summaries, and model paths appended to `reports/model_run_metadata.json`.
- Streamlit dashboard and FastAPI endpoints surface metrics for monitoring.
- Deployment `ml-pipeline/interval` (cron `*/2 * * * *`).

---

## 5. API & Dashboard (Sub-Objective 3)

### 5.1 FastAPI (`src/api/main.py`)
- Endpoints:
  - `GET /api/health` – service heartbeat.
  - `GET /api/details` – last pipeline run timestamp, processed rows/columns, artifacts.
  - `GET /api/metrics` – latest model metrics.
  - `GET /api/model-runs` – history of model runs.
  - `GET /api/artifacts` – lists available figures and model files.

### 5.2 Streamlit (`dashboard/app.py`)
- Displays pipeline logs, recent metadata, latest metrics, EDA figures, and model history.
- `sys.path` bootstrap ensures `src/` modules resolve when running `streamlit` from any directory.

### 5.3 Requirements
- `requirements.txt` pins Prefect 3.4.0 along with required libraries (pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, fastapi, uvicorn, etc.).

---

## 6. Running the Solution

### 6.1 Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 6.2 Prefect Deployments
1. Deploy flows:
   ```bash
   prefect deploy --prefect-file prefect.yaml --all
   ```
2. Start worker:
   ```bash
   prefect worker start --pool default-agent-pool
   ```
3. Optional manual run:
   ```bash
   prefect deployment run data-pipeline/interval
   prefect deployment run ml-pipeline/interval
   ```

### 6.3 Dashboard & API
```bash
streamlit run dashboard/app.py
uvicorn src.api.main:app --reload
```

### 6.4 Testing
```bash
pytest
```

---

## 7. File Artifacts
- **Data:** `data/raw/`, `data/processed/`
- **Logs:** `logs/pipeline.log`
- **Reports:** `reports/summary_statistics.csv`, `reports/figures/*.png`, `reports/metrics.json`
- **Models:** `artifacts/models/*.joblib`

---

## 8. Notes for Submission
- Ensure Prefect server (local or cloud) is configured before scheduling flows.
- If using Prefect Cloud, run `prefect cloud login` and reapply deployments.
- Include `docs/assignment_report.md` with the submission to outline deliverables.
- Provide screenshots (optional) of the dashboard and Prefect UI showing flow runs.

