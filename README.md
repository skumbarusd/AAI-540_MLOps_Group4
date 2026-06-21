# SkyAware — Flight Delay Prediction System

**AAI-540 MLOps | Group 4**  
Multi-class flight delay risk prediction built end-to-end on AWS SageMaker.

---

## Overview

SkyAware trains an XGBoost classifier to assign every carrier–airport–month combination one of four delay-risk labels:

| Label | Meaning | Threshold |
|---|---|---|
| 0 — on-time | Low delay risk | delay_rate ≤ 15% |
| 1 — minor risk | Moderate delays | 15% < delay_rate ≤ 25% |
| 2 — major risk | Significant delays | 25% < delay_rate ≤ 40% |
| 3 — high risk | Severe / cancellations | delay_rate > 40% or cancel_rate > 5% |

Source data: **DOT/BTS Airline On-Time Performance** (`Airline_Delay_Cause.csv`, 2010–2025, 50.7 MB).

---

## Repository Structure

```
.
├── 01_EDA.ipynb                          # Exploratory data analysis
├── 02_Feature_Engineering_Feature_Store.ipynb  # 34 features + SageMaker Feature Store
├── 03_Dataset_Split_S3.ipynb             # Temporal train / val / test split
├── 04_Model_Selection_Training.ipynb     # Sklearn baseline comparison
├── 05_Model_Evaluation.ipynb             # XGBoost SageMaker training job
├── 06_Model_Deployment.ipynb             # Endpoint + batch inference + Model Registry
├── 07_Model_Monitoring.ipynb             # Model Monitor + CloudWatch alarms
├── 08_CICD_Pipeline.ipynb                # 7-step SageMaker Pipeline — CI/CD
└── executed_files/                       # Pre-executed notebooks with outputs
```

---

## Notebooks

### NB01 — Exploratory Data Analysis
[`01_EDA.ipynb`](01_EDA.ipynb)

- Loads `Airline_Delay_Cause.csv` from `s3://skyaware-raw-data/raw/`
- Carrier and airport delay distribution analysis
- COVID-period anomaly detection (2020–2021 flight volume halved)
- Identifies key features for engineering

### NB02 — Feature Engineering & Feature Store
[`02_Feature_Engineering_Feature_Store.ipynb`](02_Feature_Engineering_Feature_Store.ipynb)

Builds 34 features across five groups:

| Group | Features |
|---|---|
| Core rates | delay_rate, cancel_rate, divert_rate, avg_delay_per_flight |
| Cause proportions | carrier / weather / NAS / security / late-aircraft fractions |
| Rolling lags | 3-month and 12-month carrier delay windows; YoY delta |
| Seasonal flags | is_winter, is_summer, is_peak_travel, is_covid_period |
| Encodings | LabelEncoder for 32 carriers and 389 airports (fit on train only) |

Registers all features in **SageMaker Feature Store** (`skyaware-flight-features`, online + offline).

### NB03 — Dataset Split
[`03_Dataset_Split_S3.ipynb`](03_Dataset_Split_S3.ipynb)

Strict **temporal split** — no random sampling — preventing future data leakage:

| Split | Years | Rows |
|---|---|---|
| Train | 2010–2018 | 140,676 |
| Validation | 2019–2021 | 70,326 |
| Test | 2022–2025 | 70,030 |

Outputs XGBoost-format CSVs (no header, label first) to `s3://skyaware-processed-data/pipeline/`.  
LabelEncoders persisted to `s3://skyaware-processed-data/encoders/`.

### NB04 — Baseline Model Comparison
[`04_Model_Selection_Training.ipynb`](04_Model_Selection_Training.ipynb)

Trains and compares sklearn baselines on the same temporal split:
- Logistic Regression (class-weighted)
- Decision Tree
- Random Forest
- **Gradient Boosting** — best baseline, establishes minimum F1 bar

### NB05 — XGBoost Training on SageMaker
[`05_Model_Evaluation.ipynb`](05_Model_Evaluation.ipynb)

Launches a SageMaker Training Job using the built-in **XGBoost 1.7-1** container:

```
objective:          multi:softmax
num_class:          4
max_depth:          6
eta:                0.1
num_round:          200  (early stopping at 20)
subsample:          0.8
colsample_bytree:   0.8
eval_metric:        merror
instance:           ml.m5.xlarge
```

Model artifact saved to `s3://skyaware-model-artifacts/xgboost/`.

### NB06 — Deployment & Inference
[`06_Model_Deployment.ipynb`](06_Model_Deployment.ipynb)

- **Real-time endpoint**: `skyaware-delay-endpoint` (ml.m5.large, InService)
- **DataCapture**: 20% of requests logged to `s3://skyaware-logs-monitoring/data-capture/`
- **Batch Transform**: 70,030 test rows — results in `batch-production/`
- **Model Registry**: package registered in `SkyAwareFlightDelay` group, status `Approved`

### NB07 — Model Monitoring
[`07_Model_Monitoring.ipynb`](07_Model_Monitoring.ipynb)

- Data quality baseline captured from training distribution
- Bias baseline (Clarify) computed and stored
- **Monitoring schedules** active: data quality + bias drift checks
- **CloudWatch alarms**: `SkyAware-HighErrorRate` (5XX > 5/10 min) and `SkyAware-HighLatency` (>100ms/10 min) — both `OK`
- **Dashboard**: `SkyAware-Endpoint-Monitor` — invocations, latency (~3ms avg), error rate

### NB08 — CI/CD Pipeline ✓
[`08_CICD_Pipeline.ipynb`](08_CICD_Pipeline.ipynb)

Defines a **SageMaker Pipeline** with 7 automated steps. All steps **SUCCEEDED** on 2026-06-21.

```
Raw Data (S3)
     │
┌────▼──────────────────────────────┐
│ Step 1  SkyAwareDataProcessing  ✓ │  preprocess.py — feature engineering + split
└────┬──────────────────────────────┘
     │
┌────▼──────────────────────────────┐
│ Step 2  SkyAwareXGBoostTraining ✓ │  XGBoost 1.7-1 SageMaker Training Job
└────┬──────────────────────────────┘
     │
┌────▼──────────────────────────────┐
│ Step 3  SkyAwareModelEvaluation ✓ │  evaluate.py — weighted F1 on test set
└────┬──────────────────────────────┘
     │  evaluation.json
┌────▼──────────────────────────────┐
│ Step 4  SkyAwareF1QualityGate   ✓ │  ConditionStep: weighted F1 ≥ 0.75  → PASSED
└────┬──────────────────────────────┘
   true branch
     ├──────────────────────────────┐
┌────▼──────────────────────────────┐
│ Step 5  SkyAwareRegisterModel   ✓ │  Model Registry — auto-Approved
└────┬──────────────────────────────┘
     │
┌────▼──────────────────────────────┐
│ Step 6  SkyAwareDeployEndpoint  ✓ │  deploy_endpoint.py — endpoint InService
└────┬──────────────────────────────┘
     │
┌────▼──────────────────────────────┐
│ Step 7  SkyAwareMonitoringSetup ✓ │  setup_monitoring.py — schedules verified
└───────────────────────────────────┘
```

Pipeline parameters (configurable at execution time):

| Parameter | Default | Description |
|---|---|---|
| `MinF1Threshold` | `0.75` | Minimum weighted F1 to promote model |
| `NumTrainingRounds` | `200` | XGBoost num_round |
| `TrainingInstanceType` | `ml.m5.xlarge` | Training compute |
| `ProcessingInstanceType` | `ml.m5.large` | Processing compute |

Trigger options: EventBridge monthly schedule (cron `0 2 1 * ? *`), S3 event on new data, or manual `pipeline.start()`.

---

## AWS Infrastructure

| Resource | Name / ARN |
|---|---|
| S3 — raw data | `s3://skyaware-raw-data/` |
| S3 — processed data | `s3://skyaware-processed-data/` |
| S3 — model artifacts | `s3://skyaware-model-artifacts/` |
| S3 — monitoring | `s3://skyaware-logs-monitoring/` |
| Feature Store | `skyaware-flight-features` |
| Model Group | `SkyAwareFlightDelay` |
| Endpoint | `skyaware-delay-endpoint` |
| Pipeline | `SkyAware-MLOps-Pipeline` |
| Region | `us-east-1` |
| Account | `706029888011` |

---

## Prerequisites

- AWS account with SageMaker Studio access
- IAM execution role with permissions for: SageMaker, S3, CloudWatch, EventBridge
- The four S3 buckets listed above created in `us-east-1`
- Raw data uploaded: `s3://skyaware-raw-data/raw/Airline_Delay_Cause.csv`

### Python dependencies (SageMaker Studio kernel)

All notebooks run on the **Data Science 3.0** kernel (Python 3.10).  
Key packages used (pre-installed in SageMaker Studio):

```
boto3
sagemaker >= 2.x
pandas
numpy
scikit-learn
xgboost
```

---

## Running the Project

Run notebooks **in order** in SageMaker Studio:

```
NB01 → NB02 → NB03 → NB04 → NB05 → NB06 → NB07 → NB08
```

For **NB08 only** (CI/CD Pipeline), run cells in this sequence:
1. Cells 1–7: define pipeline steps
2. Cell `27dca2a9` (Section 9.5): upload processing scripts to S3
3. Cells 8–10: assemble pipeline and upsert
4. Cell `09708620`: start execution
5. Cell `b48f9c89`: poll status

> **Note:** The pipeline skips Steps 5–7 if the model fails the F1 ≥ 0.75 quality gate. With the default hyperparameters, the gate passes.

---

## Results

| Metric | Value |
|---|---|
| Test set size | 70,030 rows (2022–2025) |
| Model | XGBoost 1.7-1, multi:softmax, 4-class |
| Quality gate | Weighted F1 ≥ 0.75 — **PASSED** |
| Endpoint status | InService (ml.m5.large) |
| Pipeline execution | **All 7 steps SUCCEEDED** — 2026-06-21 |

---

## Team

| Name | Role |
|---|---|
| Shashidhar Kumbar | MLOps Engineering, CI/CD Pipeline, Deployment |
| Jasmeet Kaur | Feature Engineering, Model Training |
| Manasa Sai Karaka | Model Monitoring, Evaluation |

**Course:** AAI-540 — Machine Learning Operations  
**Institution:** University of San Diego  
**Date:** June 2026
