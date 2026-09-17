"""
app.py — AI-Powered Customer Support Intelligence Platform
GUVI × HCL Capstone | Run: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Customer Support AI Platform",
    page_icon="🎫", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f172a}
[data-testid="stSidebar"] .stRadio label{color:#e2e8f0!important;font-size:.95rem}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span{color:#94a3b8!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2{color:#38bdf8!important}
.metric-card{background:linear-gradient(135deg,#1e3a5f,#0f172a);border-radius:12px;
  padding:20px 16px;border:1px solid #334155;text-align:center;margin-bottom:8px}
.metric-val{font-size:1.9rem;font-weight:700;color:#38bdf8}
.metric-lbl{font-size:.82rem;color:#94a3b8;margin-top:4px}
.section-header{font-size:1.25rem;font-weight:700;color:#38bdf8;
  border-bottom:2px solid #1e3a5f;padding-bottom:6px;margin:20px 0 12px 0}
.tag{display:inline-block;background:#1e3a5f;color:#38bdf8;border-radius:20px;
  padding:3px 12px;margin:3px;font-size:.8rem;border:1px solid #334155}
.stButton>button{background:linear-gradient(135deg,#0ea5e9,#6366f1);color:white;
  border:none;border-radius:8px;font-weight:600}
.result-box{background:#0f172a;border:1px solid #334155;border-radius:10px;padding:20px;margin-top:12px}
.pred-label{font-size:1.5rem;font-weight:700;color:#38bdf8}
.pred-sub{font-size:.9rem;color:#94a3b8;margin-top:4px}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎫 Support AI Platform")
    st.markdown("*GUVI × HCL Capstone*")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home", "📊 EDA & Insights", "🤖 Live Ticket Classifier",
        "📈 Model Comparison", "🔍 SHAP Explainability",
        "📦 Customer Segmentation", "🧪 MLflow Results",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("Python · Streamlit · DistilBERT · XGBoost · MLflow · SHAP")

if   page == "🏠 Home":                  from pages.home import show;            show()
elif page == "📊 EDA & Insights":        from pages.eda import show;             show()
elif page == "🤖 Live Ticket Classifier":from pages.classifier import show;      show()
elif page == "📈 Model Comparison":      from pages.model_comparison import show; show()
elif page == "🔍 SHAP Explainability":   from pages.explainability import show;  show()
elif page == "📦 Customer Segmentation": from pages.clustering import show;      show()
elif page == "🧪 MLflow Results":        from pages.mlflow_results import show;  show()
