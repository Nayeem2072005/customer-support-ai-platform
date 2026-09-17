# 🎫 AI-Powered Customer Support Intelligence Platform

> **GUVI × HCL Master Data Science Capstone Project**  
> Domain: SaaS / E-Commerce / Customer Experience / AI Operations

---

## 📌 Project Overview

An end-to-end AI platform that automates three interconnected customer support tasks:

| Task | Approach | Model |
|------|----------|-------|
| **Ticket Type Classification** | NLP — fine-tuned transformer | DistilBERT |
| **Priority Prediction** | Tabular ML | LightGBM (Optuna-tuned) |
| **Resolution Time Estimation** | Regression | XGBoost (Optuna-tuned) |
| **Customer Segmentation** | Unsupervised | K-Means (k=7) |

---

## 🗂️ Project Structure

```
Final Project/
├── notebooks/
│   ├── 02_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 03_Preprocessing.ipynb          # Text cleaning, feature engineering
│   ├── 04_Baseline_Models.ipynb        # LR + Naive Bayes baselines
│   ├── 05_NER_Analysis.ipynb           # Named Entity Recognition
│   ├── 06_Ensemble_Models.ipynb        # RF, XGBoost, LightGBM + Optuna
│   ├── 07_BiLSTM.ipynb                 # BiLSTM with GloVe embeddings
│   ├── 08_DistilBERT_FineTuning.ipynb  # DistilBERT fine-tuning
│   ├── 09_Regression_Clustering.ipynb  # Regression + K-Means clustering
│   └── 10_Explainability_Evaluation.ipynb # SHAP + final evaluation
│
├── models/
│   ├── distilbert_ticket_type/         # Fine-tuned DistilBERT
│   ├── resolution_time_xgboost.joblib  # XGBoost regressor
│   ├── kmeans_segmentation.joblib      # K-Means model
│   ├── kmeans_scaler.joblib            # Feature scaler
│   ├── label_encoders.joblib           # Categorical encoders
│   └── tfidf_vectorizer_ensemble.joblib # TF-IDF vectorizer
│
├── api/
│   └── main.py                         # FastAPI inference endpoint
│
├── capstone_app/
│   ├── app.py                          # Streamlit dashboard entry point
│   └── pages/
│       ├── home.py
│       ├── eda.py
│       ├── classifier.py
│       ├── model_comparison.py
│       ├── explainability.py
│       ├── clustering.py
│       └── mlflow_results.py
│
├── data_raw/
│   └── customer_support_tickets.csv    # Original Kaggle dataset
│
├── data_processed/
│   ├── tickets_with_features.csv       # Engineered features
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── reports/
│   ├── mlflow_all_runs_consolidated.csv
│   ├── shap_summary_plot.png
│   └── shap_waterfall_plot.png
│
├── mlruns/                             # MLflow experiment tracking
├── Dockerfile                          # Docker containerization
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🚀 Quick Setup

### 1. Clone the repository
```bash
git clone https://github.com/Nayeem2072005/customer-support-ai-platform.git
cd customer-support-ai-platform
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Place `customer_support_tickets.csv` from [Kaggle](https://drive.google.com/file/d/1Xr-7TlAunE4uYI07sxhTOXiIcs0wZboz/view) into `data_raw/`

---

## 🖥️ Running the Streamlit Dashboard

```bash
cd capstone_app
python -m streamlit run app.py
```

Opens at `http://localhost:8501`

---

## ⚡ Running the FastAPI Server

```bash
cd "Final Project"
uvicorn api.main:app --reload --port 8000
```

Opens at `http://localhost:8000`  
Swagger UI at `http://localhost:8000/docs`

### Example API Call

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "ticket_subject": "Cannot login to my account",
       "ticket_description": "I have been trying to access my account for 2 days.",
       "customer_age": 35,
       "ticket_channel": "Email",
       "product_purchased": "GadgetX Pro",
       "customer_gender": "Male"
     }'
```

### Expected Response
```json
{
  "ticket_type": "Account Management",
  "ticket_type_method": "Rule-based NLP classifier",
  "priority": "Medium",
  "estimated_resolution_hours": 13.0,
  "estimated_resolution_days": 0.54,
  "confidence": "Medium",
  "model_version": "1.0.0"
}
```

---

## 🐳 Running with Docker

```bash
# Build image
docker build -t customer-support-ai .

# Run container
docker run -p 8000:8000 customer-support-ai

# Test health
curl http://localhost:8000/health
```

---

## 📊 Model Results

### Classification (Ticket Type)
| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| Naive Bayes (baseline) | ~21% | ~0.21 |
| Logistic Regression | ~21% | ~0.20 |
| XGBoost (Optuna) | ~24% | ~0.23 |
| BiLSTM (GloVe) | ~25% | ~0.24 |
| **DistilBERT (fine-tuned)** | **~28%** | **~0.27** |

> ⚠️ Near-random accuracy is expected — this is a synthetic dataset with weak label-text correlation. Documented thoroughly in notebooks.

### Regression (Time to Resolution)
| Model | CV-RMSE | Notes |
|-------|---------|-------|
| Mean Baseline | 5.88 | |
| LightGBM (Optuna) | 5.67 | |
| **XGBoost (Optuna)** | **5.60** | **Best** |

### Clustering (Customer Segmentation)
| Metric | Value |
|--------|-------|
| Silhouette Score | 0.2140 |
| Davies-Bouldin Index | 1.2367 |
| Optimal k | 7 |

---

## 🧪 MLflow Experiment Tracking

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000
# Open http://localhost:5000
```

**21 experiment runs** across 5 experiments:
- `customer_support_baselines`
- `customer_support_ensemble_models`
- `customer_support_clustering`
- `customer_support_regression`
- `customer_support_deep_learning`

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.10 |
| NLP | HuggingFace Transformers, DistilBERT, TF-IDF, GloVe |
| ML | Scikit-learn, XGBoost, LightGBM, PyTorch |
| Tuning | Optuna |
| Explainability | SHAP |
| Tracking | MLflow |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, Uvicorn |
| Deployment | Docker |
| Data | Pandas, NumPy, SciPy |

---

## 📋 Project Guidelines Followed

- ✅ PEP8 coding standards
- ✅ Random seed fixed (`random_state=42`)
- ✅ Git version control with meaningful commits
- ✅ All model runs logged to MLflow
- ✅ Markdown cells documenting every notebook step
- ✅ Train/val/test stratified split
- ✅ NULL checks and data validation
- ✅ Requirements.txt with pinned versions

---

## 👤 Author

**M Nayeem Mohammed**  
BCA (Artificial Intelligence) — Jain University  
GUVI × HCL Master Data Science Program  
GitHub: [github.com/Nayeem2072005](https://github.com/Nayeem2072005)
