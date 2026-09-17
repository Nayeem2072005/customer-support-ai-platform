"""pages/home.py"""
import streamlit as st, pandas as pd, os
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def show():
    st.title("🎫 AI-Powered Customer Support Intelligence Platform")
    st.markdown("### GUVI × HCL Master Data Science Capstone")
    st.divider()
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,val,lbl in zip([c1,c2,c3,c4,c5],
        ["8,469","DistilBERT","6","10+","3"],
        ["Total Tickets","Best NLP Model","ML Models Built","MLflow Runs","Customer Segments"]):
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                     f'<div class="metric-lbl">{lbl}</div></div>',unsafe_allow_html=True)
    st.divider()
    col1,col2 = st.columns([3,2])
    with col1:
        st.markdown('<div class="section-header">🎯 Problem Statement</div>',unsafe_allow_html=True)
        st.markdown("""
Customer support teams receive thousands of tickets daily with no automated way to
classify, prioritize, or predict resolution time. This platform solves that with:

- **Task 1 — Ticket Type Classification (NLP):** DistilBERT fine-tuned on ticket text → Billing / Technical / Account / Product Inquiry
- **Task 2 — Priority Prediction (Tabular ML):** XGBoost / LightGBM → Low / Medium / High / Critical
- **Task 3 — Resolution Time Regression:** XGBoost regressor → estimated hours to resolve
- **Task 4 — Customer Segmentation:** K-Means → 3 behavioural customer segments
        """)
        st.markdown('<div class="section-header">📂 Dataset Preview</div>',unsafe_allow_html=True)
        raw = os.path.join(BASE,"data_raw","customer_support_tickets.csv")
        if os.path.exists(raw):
            df = pd.read_csv(raw)
            st.dataframe(df.head(8),use_container_width=True,hide_index=True)
            st.caption(f"Kaggle Customer Support Ticket Dataset — {len(df):,} rows × {len(df.columns)} columns")
        else:
            st.info("Place dataset at `data_raw/customer_support_tickets.csv`")
    with col2:
        st.markdown('<div class="section-header">🏗️ Architecture</div>',unsafe_allow_html=True)
        st.code("""Raw Tickets (CSV)
      │
      ▼
EDA & Preprocessing
      │
 ┌────┴──────┐
 ▼           ▼
NLP       Tabular ML
────      ──────────
DistilBERT  XGBoost
BiLSTM    LightGBM
TF-IDF    RandomForest
      │
      ▼
Regression + K-Means
      │
      ▼
SHAP Explainability
      │
      ▼
MLflow Tracking
      │
      ▼
Streamlit Dashboard""",language="text")
    st.divider()
    st.markdown('<div class="section-header">🛠️ Tech Stack</div>',unsafe_allow_html=True)
    tags=["Python 3.10","PyTorch 2.11","HuggingFace Transformers","DistilBERT","XGBoost",
          "LightGBM","Scikit-learn","Optuna","SHAP","MLflow","Streamlit","Pandas","NumPy",
          "K-Means","PCA","t-SNE","FastAPI","Docker","Matplotlib","Seaborn","Plotly"]
    st.markdown(" ".join(f'<span class="tag">{t}</span>' for t in tags),unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-header">📋 Notebook Pipeline</div>',unsafe_allow_html=True)
    for nb,desc in [
        ("02_EDA.ipynb","EDA — distributions, word clouds, class imbalance, satisfaction analysis"),
        ("03_Preprocessing.ipynb","Text cleaning, TF-IDF, Word2Vec, feature engineering, stratified split"),
        ("04_Baseline_Models.ipynb","Logistic Regression + Naive Bayes baselines with MLflow logging"),
        ("05_NER_Analysis.ipynb","Named Entity Recognition on ticket text using spaCy"),
        ("06_Ensemble_Models.ipynb","RF, XGBoost (Optuna 20 trials), LightGBM (Optuna 20 trials) — tabular + TF-IDF"),
        ("07_BiLSTM.ipynb","PyTorch BiLSTM with GloVe 100d embeddings for ticket classification"),
        ("08_DistilBERT_FineTuning.ipynb","DistilBERT fine-tuned on ticket subject + description — core NLP deliverable"),
        ("09_Regression_Clustering.ipynb","XGBoost/LightGBM regressor (Optuna 25 trials) + K-Means segmentation"),
        ("10_Explainability_Evaluation.ipynb","SHAP summary/waterfall/dependence plots + final model comparison"),
    ]:
        st.markdown(f"✅ **`{nb}`** — {desc}")
