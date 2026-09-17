"""pages/classifier.py — Live Ticket Classifier"""
import streamlit as st
import os, joblib, numpy as np
from scipy.sparse import hstack, csr_matrix

BASE   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(BASE, "models")

TICKET_TYPES    = ["Billing Inquiry", "Technical Issue", "Account Management", "Product Inquiry", "Cancellation Request"]
PRIORITY_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}

@st.cache_resource
def load_tabular_models():
    try:
        tfidf = joblib.load(os.path.join(MODELS, "tfidf_vectorizer_ensemble.joblib"))
        le    = joblib.load(os.path.join(MODELS, "label_encoders.joblib"))
        reg   = joblib.load(os.path.join(MODELS, "resolution_time_xgboost.joblib"))
        return tfidf, le, reg, True
    except Exception as e:
        return None, None, None, False

def safe_encode(le_dict, col, val):
    if col in le_dict:
        le_obj = le_dict[col]
        if val in le_obj.classes_:
            return int(le_obj.transform([val])[0])
    return 0

def build_X(tfidf, le, reg, text, age, channel, product, gender, desc):
    text_vec = tfidf.transform([text])
    ch_enc   = safe_encode(le, "Ticket Channel", channel)
    pr_enc   = safe_encode(le, "Product Purchased", product)
    gen_enc  = safe_encode(le, "Customer Gender", gender)
    tab      = csr_matrix([[age, ch_enc, pr_enc, gen_enc, 0, 0, len(desc), 0]])
    X        = hstack([text_vec, tab]).toarray()
    # Fix shape mismatch
    expected = reg.n_features_in_
    if X.shape[1] < expected:
        X = np.hstack([X, np.zeros((1, expected - X.shape[1]))])
    elif X.shape[1] > expected:
        X = X[:, :expected]
    return X

def rule_based_type(text):
    t = text.lower()
    if any(w in t for w in ["billing","payment","charge","invoice","refund","money"]):
        return "Billing Inquiry"
    elif any(w in t for w in ["cancel","cancellation","unsubscribe","terminate"]):
        return "Cancellation Request"
    elif any(w in t for w in ["error","crash","bug","not working","issue","broken","fail"]):
        return "Technical Issue"
    elif any(w in t for w in ["account","login","password","access","username","sign in"]):
        return "Account Management"
    else:
        return "Product Inquiry"

def show():
    st.title("🤖 Live Ticket Classifier")
    st.markdown("Type or paste a support ticket — get instant AI predictions")
    st.divider()

    tfidf, le, reg, tab_ok = load_tabular_models()

    c1, c2, c3 = st.columns(3)
    c1.warning("⚠️ DistilBERT not loaded")
    c2.success("✅ Tabular models loaded") if tab_ok else c2.warning("⚠️ Tabular models not loaded")
    c3.info("ℹ️ Running in CPU mode")

    st.divider()
    col_input, col_features = st.columns([2, 1])

    with col_input:
        st.markdown("#### 📝 Ticket Input")
        subject     = st.text_input("Ticket Subject", placeholder="e.g. Unable to login to my account")
        description = st.text_area("Ticket Description", height=150,
            placeholder="e.g. I have been trying to access my account for the past two days...")
        combined_text = f"{subject} {description}".strip()

    with col_features:
        st.markdown("#### ⚙️ Customer Features")
        age     = st.slider("Customer Age", 18, 80, 35)
        channel = st.selectbox("Ticket Channel", ["Email", "Chat", "Phone", "Social Media"])
        product = st.selectbox("Product", ["GadgetX Pro", "HomeSmartHub", "FitTrack Band",
                                           "SoundWave Speaker", "VisionCam 4K"])
        gender  = st.selectbox("Gender", ["Male", "Female", "Other"])

    predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)

    if predict_btn and combined_text:
        st.divider()
        st.markdown("### 🎯 Prediction Results")
        col1, col2, col3 = st.columns(3)

        # ── Ticket Type (rule-based) ────────────────────────
        with col1:
            st.markdown("#### 🏷️ Ticket Type")
            pred = rule_based_type(combined_text)
            st.markdown(f'<div class="result-box"><div class="pred-label">🏷️ {pred}</div>'
                        f'<div class="pred-sub">Rule-based NLP classifier</div></div>',
                        unsafe_allow_html=True)

        # ── Priority + Resolution ───────────────────────────
        if tab_ok:
            try:
                X = build_X(tfidf, le, reg, combined_text, age, channel, product, gender, description)
                hours = float(reg.predict(X)[0])

                with col2:
                    st.markdown("#### 🚨 Priority Level")
                    if hours < 5:    priority = "Low"
                    elif hours < 15: priority = "Medium"
                    elif hours < 30: priority = "High"
                    else:            priority = "Critical"
                    icon = PRIORITY_COLORS[priority]
                    st.markdown(f'<div class="result-box"><div class="pred-label">{icon} {priority}</div>'
                                f'<div class="pred-sub">Based on XGBoost regressor</div></div>',
                                unsafe_allow_html=True)

                with col3:
                    st.markdown("#### ⏱️ Resolution Time")
                    days = hours / 24
                    st.markdown(f'<div class="result-box"><div class="pred-label">⏱️ ~{abs(hours):.1f}h</div>'
                                f'<div class="pred-sub">≈ {abs(days):.1f} days estimated</div></div>',
                                unsafe_allow_html=True)
                    st.caption("⚠️ Dataset timestamps are synthetic — model demonstrates pipeline.")

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            with col2: st.warning("Tabular models not loaded")
            with col3: st.warning("Tabular models not loaded")

    elif predict_btn:
        st.warning("Please enter a ticket description first.")

    st.divider()
    st.markdown("#### 💡 Try These Example Tickets")
    examples = [
        ("Billing Issue", "I was charged twice for my subscription",
         "I noticed two charges of $49.99 on my credit card for the same month. Please refund the duplicate."),
        ("Tech Issue", "App keeps crashing on startup",
         "Every time I open the mobile app it crashes within 3 seconds. Tried reinstalling but problem persists."),
        ("Account", "Cannot reset my password",
         "I clicked forgot password but the reset email never arrives. Checked spam folder too."),
    ]
    cols = st.columns(3)
    for col, (title, subj, desc) in zip(cols, examples):
        with col:
            with st.expander(f"📋 {title}"):
                st.markdown(f"**Subject:** {subj}")
                st.markdown(f"**Description:** {desc}")
                