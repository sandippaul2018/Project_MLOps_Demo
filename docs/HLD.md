# BAT OptiCure MLOps - High Level Design

## 1. Objectives
- Provide an end-to-end MLOps demo for tobacco curing quality prediction (Premium vs Standard).
- Automate data generation, training, evaluation, and artifact publishing on every push to `main`.
- Offer a lightweight UI (Streamlit) for batch quality simulation and metric visibility.
- Keep the stack simple (Python + scikit-learn) to emphasize pipeline reliability, not model complexity.

## 2. Scope / Non-Goals
- In scope: synthetic data pipeline, model comparison and selection, artifact persistence, Streamlit UI, GitHub Actions CI/CD.
- Out of scope: multi-tenant auth, feature store, model registry, canary/blue-green deploys, GPU/large-model training.

## 3. Architecture (logical)
- Data layer: synthetic generator writes `curing_data.csv` + `curing_data_metadata.json` to repo root.
- Training layer: `train.py` trains 4 classifiers (LR, RF, GB, SVM), picks best by accuracy, stores `bat_curing_pipeline.pkl` + `model_metrics.csv`.
- Serving layer: Streamlit app (`src/app.py`) loads the pickle, runs inference, and renders metrics.
- CI/CD layer: GitHub Actions workflow `.github/workflows/mlops.yml` triggers on push/manual; runs generate → train → commit artifacts.
- Storage: git-tracked artifacts (CSV, pickle, metrics) committed back to repo; no external blob store.

## 4. Component Responsibilities
- `src/generate_data.py`: create 500 synthetic records, enforce business rule (temp >78C or humidity <45% => Standard), write CSV + metadata.
- `src/train.py`: split data, train LR/RF/GB/SVM, compute Accuracy/Precision/Recall/AUC, persist best model + metrics CSV.
- `src/app.py`: Streamlit UI with sliders for temp/humidity/leaf moisture, invokes loaded model for prediction, shows confidence, temp deviation, and metrics table.
- `.github/workflows/mlops.yml`: CI job on `ubuntu-latest`, Python 3.9, installs deps, runs data gen + train, auto-commits artifacts when changed.

## 5. Data Flow
1) Generate: `generate_data.py` → `curing_data.csv`, `curing_data_metadata.json`.
2) Train: `train.py` reads CSV → split → train 4 models → pick best → save `bat_curing_pipeline.pkl`, `model_metrics.csv`.
3) Serve: Streamlit loads pickle + metrics → user inputs sliders → inference → display grade + confidence; metrics expander renders CSV.
4) CI/CD: on push, workflow executes steps 1-2 and commits updated artifacts back to `main`.

## 6. Models & Metrics
- Models: Logistic Regression, RandomForest (100 trees), Gradient Boosting (100 estimators), SVM (RBF, prob=True).
- Selection: highest Accuracy on stratified 80/20 split (seed 123).
- Metrics logged: Accuracy, Precision (weighted), Recall (weighted), AUC (binary/OVA).
- Artifacts: `bat_curing_pipeline.pkl`, `model_metrics.csv`.

## 7. Environments
- Local dev: Python 3.12 venv, Streamlit UI, manual runs of generate/train.
- CI: GitHub Actions (Python 3.9) to maximize compatibility; headless training.
- No separate staging/prod deployments; Streamlit intended for local demo.

## 8. CI/CD Pipeline (GitHub Actions)
- Triggers: push to `main`, manual dispatch.
- Steps: checkout → setup Python 3.9 → `pip install -r requirements.txt` → run generate → run train → commit/push artifacts (pickle, metrics, CSV).
- Idempotency: commit only when diffs exist to avoid empty commits.

## 9. Observability & Logging
- Console logs in data generation and training scripts for traceability.
- Streamlit shows inline errors if model load or inference fails.
- No centralized log/metric store; acceptable for demo scope.

## 10. Security & Compliance
- No external data; all synthetic.
- GitHub permissions: workflow uses `contents: write` to push artifacts.
- No secrets required; if later adding Docker pushes, configure GH secrets for registry creds.

## 11. Scaling & Performance
- Dataset small (500 rows); models lightweight; fits single CPU runner.
- Streamlit single-instance; acceptable for demo; not load-balanced.
- If scaling needed: move artifacts to object storage, add model registry, containerize app, add autoscaling infra.

## 12. Risks / Limitations
- Model overfits synthetic rules; not production-grade.
- Artifacts committed to git can bloat history over time.
- Workflow runs on every push to `main` (could be noisy); mitigate by restricting triggers or adding caching.

## 13. Ops Runbook (minimal)
- Generate + Train locally: `python src/generate_data.py && python src/train.py`
- Run UI: `streamlit run src/app.py` (ensure `bat_curing_pipeline.pkl` exists)
- Rerun pipeline via CI: push to `main` or trigger `workflow_dispatch`
- Debug model load errors: confirm `bat_curing_pipeline.pkl` present in repo root; retrain if missing.
