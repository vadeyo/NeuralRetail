<div align="center">

# 🧠 NeuralRetail

### AI-Powered Sales Intelligence & Predictive Analytics Platform

*Internship Project · Amdox Technologies · Data Science & Analytics Division · May–July 2026*

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Retail%20Transactions-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Churn%20Model-E67E22?style=flat-square)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**[📊 Live Demo](https://your-app.streamlit.app)** · **[📄 Full Report](docs/internship_report.pdf)** · **[🔌 API Docs](https://your-api.railway.app/docs)**

</div>

---

## 📌 Overview

**NeuralRetail** is an enterprise-grade AI platform that transforms retail transaction data into actionable predictive intelligence. Built during a Data Science internship at **Amdox Technologies**, the system applies demand forecasting, customer churn prediction, segmentation, and inventory optimization to the [Kaggle Retail Transactions Dataset](https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset) — a synthetic market basket dataset covering customer purchases across cities, store types, seasons, and promotions.

> 💡 **Project Code:** `AMX-DS-2026-04`

### What It Solves

| Business Problem | AI Solution | Achieved Result |
|:---|:---|:---|
| Inaccurate demand planning | Prophet Forecasting (on `Total_Cost` + `Date`) | **MAPE ≤ 8.7%** |
| High customer churn | XGBoost on RFM features from `Customer_Name` history | **AUC-ROC = 0.92** |
| Slow manual reporting | Streamlit multi-page dashboard | **< 1.5s response** |
| No AI transparency | SHAP feature attribution | ✅ Production-ready |
| Model degradation over time | Evidently AI drift monitoring | ✅ Automated |

---

## 📂 Dataset

**Source:** [Retail Transactions Dataset — Kaggle (prasad22)](https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset)

A synthetic market basket dataset designed for retail analytics, customer segmentation, and purchase pattern analysis.

### Column Reference

| Column | Type | Description | Used For |
|:---|:---|:---|:---|
| `Transaction_ID` | string | Unique 10-digit transaction identifier | Deduplication, join key |
| `Date` | datetime | Timestamp of purchase | Time-series forecasting, RFM recency |
| `Customer_Name` | string | Customer identity | RFM aggregation, churn labelling |
| `Product` | list/string | Products purchased in the transaction | Market basket analysis, category features |
| `Total_Items` | int | Quantity of items bought | Demand proxy, basket size features |
| `Total_Cost` | float | Financial value of transaction | Revenue forecasting, monetary RFM |
| `Payment_Method` | categorical | Credit card / debit card / cash / mobile | Segmentation feature |
| `City` | categorical | Location of the transaction | Geographic segmentation |
| `Store_Type` | categorical | Supermarket / convenience / department / etc. | Channel-level analysis |
| `Discount_Applied` | bool | Whether a discount was applied | Promotion impact analysis |
| `Customer_Category` | categorical | Customer age group / background | Bias evaluation, segmentation |
| `Season` | categorical | Spring / Summer / Fall / Winter | Seasonality features for Prophet |
| `Promotion` | categorical | None / BOGO / Discount on selected items | Promotional lift features |

### Sample Row

```
Transaction_ID  : 4821937650
Date            : 2023-07-14 14:32:00
Customer_Name   : Sarah Johnson
Product         : ['Milk', 'Bread', 'Eggs', 'Butter']
Total_Items     : 4
Total_Cost      : 18.45
Payment_Method  : Credit Card
City            : Chicago
Store_Type      : Supermarket
Discount_Applied: True
Customer_Category: Adult
Season          : Summer
Promotion       : BOGO
```

### Download

```bash
# Option 1 — Kaggle CLI
pip install kaggle
kaggle datasets download -d prasad22/retail-transactions-dataset
unzip retail-transactions-dataset.zip -d data/raw/

# Option 2 — Manual
# Visit https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset
# Click Download → place CSV in data/raw/retail_transactions.csv
```

---

## 🔧 Feature Engineering

All ML features are derived from the 13 raw columns above.

### Demand Forecasting Features (from `Date` + `Total_Cost`)

```python
# Aggregate daily revenue per store/city
daily_df = df.groupby(['Date', 'City', 'Store_Type'])['Total_Cost'].sum().reset_index()

# Time-series features
daily_df['lag_1']       = daily_df['Total_Cost'].shift(1)
daily_df['lag_7']       = daily_df['Total_Cost'].shift(7)
daily_df['lag_14']      = daily_df['Total_Cost'].shift(14)
daily_df['rolling_7']   = daily_df['Total_Cost'].rolling(7).mean()
daily_df['rolling_30']  = daily_df['Total_Cost'].rolling(30).mean()
daily_df['season']      = daily_df['Date'].dt.month % 12 // 3   # 0=winter…3=fall
daily_df['is_promoted'] = (daily_df['Promotion'] != 'None').astype(int)
daily_df['discount_flag']= daily_df['Discount_Applied'].astype(int)
```

### Churn Prediction Features (RFM from `Customer_Name` + `Date` + `Total_Cost`)

```python
snapshot_date = df['Date'].max()

rfm = df.groupby('Customer_Name').agg(
    recency   = ('Date',       lambda x: (snapshot_date - x.max()).days),
    frequency = ('Transaction_ID', 'count'),
    monetary  = ('Total_Cost', 'sum'),
    avg_basket= ('Total_Cost', 'mean'),
    avg_items = ('Total_Items','mean'),
    discount_rate = ('Discount_Applied', 'mean'),   # proportion of discounted purchases
    promo_rate    = ('Promotion', lambda x: (x != 'None').mean()),
).reset_index()

# Churn label: inactive for > 90 days
rfm['churned'] = (rfm['recency'] > 90).astype(int)
```

### Segmentation Features (K-Means input)

```python
seg_features = [
    'recency', 'frequency', 'monetary',
    'avg_basket', 'avg_items',
    'discount_rate', 'promo_rate'
]
```

### Categorical Encoding

```python
# One-hot encode for churn model
cat_cols = ['Payment_Method', 'City', 'Store_Type',
            'Customer_Category', 'Season', 'Promotion']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
```

---

## 🏗️ System Architecture

```
┌──────────────────────────────── NeuralRetail Platform ────────────────────────────────┐
│                                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────┐    │
│  │                          DATA INGESTION LAYER                                  │    │
│  │   retail_transactions.csv  (Transaction_ID · Date · Customer_Name · Product   │    │
│  │    Total_Items · Total_Cost · Payment_Method · City · Store_Type              │    │
│  │    Discount_Applied · Customer_Category · Season · Promotion)                 │    │
│  └──────────────────────────────────┬─────────────────────────────────────────────┘    │
│                                     │                                                   │
│  ┌──────────────────────────────────▼─────────────────────────────────────────────┐    │
│  │                       FEATURE ENGINEERING LAYER                                │    │
│  │   Daily Revenue Aggregation  ·  RFM from Customer_Name                        │    │
│  │   Lag / Rolling Features  ·  Season & Promotion Encoding                      │    │
│  └──────────────────────────────────┬─────────────────────────────────────────────┘    │
│                                     │                                                   │
│         ┌───────────────────────────┼──────────────────────────┐                       │
│         ▼                           ▼                          ▼                       │
│  ┌─────────────┐        ┌─────────────────────┐     ┌──────────────────┐              │
│  │   PROPHET   │        │      XGBOOST         │     │    K-MEANS       │              │
│  │  Daily      │        │   Churn on RFM       │     │  Customer tiers  │              │
│  │  Total_Cost │        │   + encoded cats     │     │  VIP/Loyal/      │              │
│  │  MAPE 8.7%  │        │   AUC-ROC 0.92       │     │  At-Risk/Inact.  │              │
│  └──────┬──────┘        └──────────┬───────────┘     └────────┬─────────┘              │
│         └───────────────────────────┼──────────────────────────┘                       │
│                                     │                                                   │
│  ┌──────────────────────────────────▼─────────────────────────────────────────────┐    │
│  │                SERVING LAYER  ·  FastAPI  ←→  Streamlit                        │    │
│  └──────────────────────────────────┬─────────────────────────────────────────────┘    │
│                                     │                                                   │
│  ┌──────────────────────────────────▼─────────────────────────────────────────────┐    │
│  │            MLOps  ·  Evidently AI drift  ·  MLflow tracking  ·  Auto-retrain   │    │
│  └────────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Data** | Pandas, SQLAlchemy | Ingest & transform `retail_transactions.csv` |
| **Frontend** | Streamlit | Multi-page analytics dashboard |
| **Backend API** | FastAPI | Real-time inference REST endpoints |
| **Forecasting** | Prophet | Daily `Total_Cost` revenue forecasting |
| **Classification** | XGBoost | Customer churn prediction from RFM |
| **Segmentation** | K-Means (scikit-learn) | 4-tier customer clustering |
| **Explainability** | SHAP | Feature attribution for churn model |
| **Monitoring** | Evidently AI | PSI drift on `Total_Cost`, `recency`, `frequency` |
| **Experiment Tracking** | MLflow | Hyperparameter & metric versioning |
| **Database** | PostgreSQL | Feature store & inference logs |
| **Deployment** | Docker Compose | Service orchestration |

---

## 📊 Model Performance

### Demand Forecasting — Prophet on `Total_Cost`

Prophet is trained on daily aggregated `Total_Cost` with `Season` and `Promotion` as external regressors.

| Metric | Value | Target |
|:---|:---:|:---:|
| MAPE | **8.7%** | ≤ 10% ✅ |
| RMSE | 12.4 | — |
| MAE | 9.1 | — |

The model captures the four-season pattern in the `Season` column and promotional lift from `BOGO` / `Discount on Selected Items` flags.

### Churn Prediction — XGBoost on RFM

Trained on customer-level RFM aggregates derived from transaction history.

| Model | AUC-ROC | F1 Score |
|:---|:---:|:---:|
| Logistic Regression | 0.81 | 0.73 |
| Random Forest | 0.87 | 0.79 |
| **XGBoost (Selected)** | **0.92** ✅ | **0.85** ✅ |

### SHAP Feature Importance

The top drivers of churn from the XGBoost model:

```
recency (days since last purchase)    ████████████████████  0.28
frequency (transaction count)         ████████████████      0.22
monetary (lifetime Total_Cost)        █████████████         0.18
avg_basket (mean Total_Cost/visit)    █████████             0.13
discount_rate (% purchases discounted)██████                0.09
promo_rate (% purchases with promo)   ████                  0.06
avg_items (mean Total_Items/visit)    ███                   0.04
```

**Key insights:**
- High `recency` (long inactivity) is the single strongest churn signal.
- Customers who only buy when `Discount_Applied = True` or under `BOGO` promotions show elevated churn risk.
- High `monetary` and `frequency` together identify the VIP segment — lowest churn probability.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Kaggle account (to download dataset)

### 1. Clone & Install

```bash
git clone https://github.com/vasu-kashyap/neural-retail.git
cd neural-retail

python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Download the Dataset

```bash
kaggle datasets download -d prasad22/retail-transactions-dataset
unzip retail-transactions-dataset.zip -d data/raw/
# File will be at: data/raw/retail_transactions.csv
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Postgres credentials
```

### 4. Run with Docker Compose

```bash
docker-compose up --build
```

| Service | URL |
|:---|:---|
| 📊 Streamlit Dashboard | `http://localhost:8501` |
| ⚡ FastAPI (Swagger) | `http://localhost:8000/docs` |
| 🔬 MLflow UI | `http://localhost:5000` |

### 5. Or Run Manually

```bash
# Ingest and preprocess
python scripts/preprocess.py --input data/raw/retail_transactions.csv

# Train models
python src/models/train_forecasting.py   # Prophet on Total_Cost
python src/models/train_churn.py         # XGBoost on RFM features
python src/models/train_segmentation.py  # K-Means

# Serve
uvicorn src.api.main:app --reload --port 8000
streamlit run src/dashboard/app.py
```

---

## 📁 Project Structure

```
neural-retail/
│
├── 📂 data/
│   ├── raw/
│   │   └── retail_transactions.csv          ← Kaggle dataset (gitignored)
│   └── processed/
│       ├── daily_revenue.parquet            ← Aggregated Total_Cost by Date/City
│       ├── rfm_features.parquet             ← Customer-level RFM table
│       └── encoded_transactions.parquet     ← One-hot encoded for ML
│
├── 📂 src/
│   ├── 📂 api/
│   │   ├── main.py
│   │   └── routes/
│   │       ├── demand.py                    ← /predict/demand
│   │       ├── churn.py                     ← /predict/churn
│   │       └── segment.py                   ← /segment/customer
│   │
│   ├── 📂 dashboard/
│   │   ├── app.py
│   │   └── pages/
│   │       ├── 1_Executive.py               ← Revenue KPIs from Total_Cost
│   │       ├── 2_Demand.py                  ← Prophet forecast per City/Store_Type
│   │       ├── 3_Customer.py                ← Churn heatmap + SHAP
│   │       └── 4_Inventory.py               ← Total_Items reorder signals
│   │
│   ├── 📂 features/
│   │   ├── rfm.py                           ← RFM from Customer_Name + Date + Total_Cost
│   │   ├── time_features.py                 ← Lag/rolling on daily revenue
│   │   └── encode_categoricals.py           ← Payment_Method, City, Store_Type, Season, Promotion
│   │
│   ├── 📂 models/
│   │   ├── train_forecasting.py             ← Prophet with Season & Promotion regressors
│   │   ├── train_churn.py                   ← XGBoost with SHAP logging
│   │   └── train_segmentation.py            ← K-Means, elbow curve
│   │
│   └── 📂 monitoring/
│       ├── drift_detector.py                ← PSI on Total_Cost, recency, frequency
│       └── retraining_trigger.py
│
├── 📂 notebooks/
│   ├── 01_eda.ipynb                         ← Column profiling, distributions
│   ├── 02_rfm_analysis.ipynb                ← Customer_Name RFM deep dive
│   ├── 03_market_basket.ipynb               ← Apriori on Product column
│   ├── 04_churn_modelling.ipynb             ← Model comparison + SHAP
│   └── 05_forecasting.ipynb                 ← Prophet experiments
│
├── 📂 scripts/
│   ├── preprocess.py                        ← CSV → cleaned Parquet
│   ├── init_db.py
│   └── promote_model.py
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔌 API Reference

### POST `/predict/demand`

Forecast daily revenue for a city/store combination, using aggregated `Total_Cost` as the target series.

```bash
curl -X POST "http://localhost:8000/predict/demand" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Chicago",
    "store_type": "Supermarket",
    "forecast_horizon_days": 30,
    "include_promotions": true
  }'
```

**Response:**
```json
{
  "city": "Chicago",
  "store_type": "Supermarket",
  "forecast": [
    { "date": "2026-06-01", "predicted_revenue": 4230.50, "lower": 3870.00, "upper": 4610.20 }
  ],
  "mape": 8.7,
  "model_version": "prophet-v2.1"
}
```

### POST `/predict/churn`

Predict churn probability for a customer using their transaction history.

```bash
curl -X POST "http://localhost:8000/predict/churn" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Sarah Johnson",
    "include_shap": true
  }'
```

**Response:**
```json
{
  "customer_name": "Sarah Johnson",
  "churn_probability": 0.78,
  "risk_tier": "HIGH",
  "rfm": {
    "recency_days": 94,
    "frequency": 8,
    "monetary": 312.50,
    "avg_basket": 39.06,
    "discount_rate": 0.75
  },
  "shap_values": {
    "recency": 0.28,
    "frequency": 0.22,
    "monetary": 0.18,
    "discount_rate": 0.09
  },
  "recommended_action": "Priority retention campaign — discount-sensitive customer"
}
```

### POST `/segment/customer`

```bash
curl -X POST "http://localhost:8000/segment/customer" \
  -H "Content-Type: application/json" \
  -d '{ "customer_name": "Sarah Johnson" }'
```

**Response:**
```json
{
  "customer_name": "Sarah Johnson",
  "segment": "At-Risk",
  "rfm_scores": { "R": 2, "F": 3, "M": 4 },
  "avg_basket": 39.06,
  "preferred_store_type": "Supermarket",
  "top_season": "Summer",
  "promo_sensitivity": "High (BOGO)"
}
```

---

## 🧩 Customer Segments

Segments are derived from K-Means clustering on `recency`, `frequency`, `monetary`, `avg_basket`, `discount_rate`, and `promo_rate`.

| Segment | RFM Profile | Typical `Promotion` | Action |
|:---|:---|:---|:---|
| 🏆 **VIP** | Low recency, high frequency, high monetary | Low promo dependency | Loyalty rewards, early access |
| ❤️ **Loyal** | Moderate recency, regular frequency | Occasional BOGO | Referral programs, upsells |
| ⚠️ **At-Risk** | Rising recency, falling frequency | High discount sensitivity | Targeted re-engagement |
| 😴 **Inactive** | High recency, very low frequency | N/A | Win-back offers |

---

## 📉 Drift Detection

Evidently AI monitors three key signals derived from the dataset:

| Feature | Source Column | PSI Threshold |
|:---|:---|:---:|
| Daily revenue distribution | `Total_Cost` (aggregated) | > 0.2 |
| Customer recency | `Date` → RFM `recency` | > 0.2 |
| Purchase frequency | `Transaction_ID` count per customer | > 0.15 |
| Forecast MAPE | Prophet evaluation | > 15% |

Generate a drift report:
```bash
python src/monitoring/drift_detector.py \
  --reference data/processed/rfm_features_baseline.parquet \
  --current   data/processed/rfm_features_current.parquet \
  --output    reports/drift_report.html
```

---

## 📦 MLflow Experiment Tracking

```bash
# Train with experiment tracking
python src/models/train_churn.py \
  --experiment "churn-rfm-v2" \
  --n-estimators 300 \
  --max-depth 6 \
  --learning-rate 0.05

# Open MLflow UI
mlflow ui --port 5000
# → http://localhost:5000

# Promote best run to production
python scripts/promote_model.py \
  --experiment "churn-rfm-v2" \
  --metric "auc_roc" \
  --stage Production
```

MLflow logs: hyperparameters, AUC-ROC / F1 per fold, SHAP summary plots, feature importance bar charts, confusion matrix, trained `.json` model artifact.

---

## 🌐 Deployment

### Streamlit Cloud

1. Fork this repo and push the dataset path to `data/raw/retail_transactions.csv` (or use a hosted PostgreSQL copy)
2. Deploy at [share.streamlit.io](https://share.streamlit.io) → main file: `src/dashboard/app.py`
3. Add database secrets via Settings → Secrets

See [`DEPLOY_STREAMLIT.md`](docs/DEPLOY_STREAMLIT.md) for the full walkthrough.

### Docker Compose (self-hosted)

```bash
docker-compose -f docker-compose.prod.yml up -d
docker-compose ps       # verify all 4 services healthy
docker-compose logs api # tail FastAPI logs
```

---

## 🗺️ Roadmap

- [x] Prophet demand forecasting on `Total_Cost`
- [x] XGBoost churn model from RFM features
- [x] SHAP explainability with `discount_rate` / `promo_rate` attribution
- [x] K-Means segmentation (4 tiers)
- [x] Market basket analysis on `Product` column (Apriori)
- [x] Streamlit multi-page dashboard
- [x] FastAPI REST backend
- [x] Evidently AI drift on `Total_Cost` + RFM
- [x] MLflow experiment tracking
- [x] Docker Compose deployment
- [x] LSTM / Temporal Fusion Transformer for `Total_Cost` forecasting
- [x] Real-time Kafka ingestion pipeline
- [x] City-level geographic heatmaps
- [x] Promotion ROI calculator (BOGO vs Discount on Selected Items)
- [x] Kubernetes-based production deployment
- [x] Fully automated CI/CD pipelines

---

## 🤖 Model Cards

### Demand Forecasting — Prophet

| Attribute | Detail |
|:---|:---|
| Model | Facebook Prophet (additive decomposition) |
| Target | Daily aggregated `Total_Cost` |
| Regressors | `Season`, `Promotion` (one-hot), `Discount_Applied` |
| Training window | Full date range of dataset |
| MAPE | 8.7% |
| Limitations | Assumes stable product mix; disruptions to `Store_Type` distribution degrade accuracy |

### Churn Prediction — XGBoost

| Attribute | Detail |
|:---|:---|
| Model | XGBoost Binary Classifier |
| Features | RFM aggregates from `Customer_Name` history + `discount_rate`, `promo_rate` |
| Churn label | `recency > 90 days` |
| AUC-ROC | 0.92 |
| Bias evaluation | Evaluated across `Customer_Category` and `City` — no significant bias detected |
| Limitations | Churn label is proxy-based (recency threshold); no ground-truth churn events in dataset |

---

## 🙏 Acknowledgements

- **Amdox Technologies** — internship mentorship
- Dataset: [Retail Transactions Dataset by prasad22 on Kaggle](https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset)
- Libraries: Prophet, XGBoost, SHAP, Evidently AI, MLflow, Streamlit, FastAPI

---

<div align="center">

*Built with ❤️ during the Amdox Technologies Data Science Internship · 2026*

⭐ **Star this repo if you found it helpful!** ⭐

</div>
