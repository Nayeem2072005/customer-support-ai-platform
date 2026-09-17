"""pages/classifier.py — Live Ticket Classifier using DistilBERT + XGBoost"""
import streamlit as st
import os, joblib, numpy as np, pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(BASE, "models")

TICKET_TYPES     = ["Billing Inquiry", "Technical Issue", "Account Management", "Product Inquiry"]
PRIORITY_LEVELS  = ["Low", "Medium", "High", "Critical"]
PRIORITY_COLORS  = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}

@st.cache_resource
def load_distilbert():
    try:
        from transformers import pipeline
        model_path = os.path.join(MODELS, "distilbert_ticket_type")
        if os.path.exists(model_path):
            clf = pipeline("text-classification", model=model_path,
                           tokenizer=model_path, return_all_scores=True)
            return clf, True
        return None, False
    except Exception as e:
        return None, False

@st.cache_resource
def load_tabular_models():
    try:
        tfidf    = joblib.load(os.path.join(MODELS, "tfidf_vectorizer_ensemble.joblib"))
        le       = joblib.load(os.path.join(MODELS, "label_encoders.joblib"))
        reg      = joblib.load(os.path.join(MODELS, "resolution_time_xgboost.joblib"))
        return tfidf, le, reg, True
    except Exception as e:
        return None, None, None, False

def show():
    st.title("🤖 Live Ticket Classifier")
    st.markdown("Type or paste a support ticket — get instant AI predictions")
    st.divider()

    distilbert_clf, bert_ok  = load_distilbert()
    tfidf, le, reg, tab_ok   = load_tabular_models()

    # ── Status bar ──────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.success("✅ DistilBERT loaded") if bert_ok else c1.warning("⚠️ DistilBERT not loaded")
    c2.success("✅ Tabular models loaded") if tab_ok else c2.warning("⚠️ Tabular models not loaded")
    c3.info("ℹ️ GPU: " + ("Available" if __import__("torch").cuda.is_available() else "CPU mode"))

    st.divider()
    col_input, col_features = st.columns([2, 1])

    with col_input:
        st.markdown("#### 📝 Ticket Input")
        subject = st.text_input("Ticket Subject", placeholder="e.g. Unable to login to my account")
        description = st.text_area("Ticket Description", height=150,
            placeholder="e.g. I have been trying to access my account for the past two days but keep getting an error message saying my password is incorrect. I have tried resetting my password but the reset email is not arriving...")
        combined_text = f"{subject} {description}".strip()

    with col_features:
        st.markdown("#### ⚙️ Customer Features")
        age         = st.slider("Customer Age", 18, 80, 35)
        channel     = st.selectbox("Ticket Channel", ["Email", "Chat", "Phone", "Social Media"])
        product     = st.selectbox("Product", ["GadgetX Pro", "HomeSmartHub", "FitTrack Band",
                                               "SoundWave Speaker", "VisionCam 4K"])
        gender      = st.selectbox("Gender", ["Male", "Female", "Other"])

    predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)

    if predict_btn and combined_text:
        st.divider()
        st.markdown("### 🎯 Prediction Results")

        col1, col2, col3 = st.columns(3)

        # ── DistilBERT — Ticket Type ────────────────────────
        with col1:
            st.markdown("#### 🏷️ Ticket Type (DistilBERT)")
            if bert_ok and distilbert_clf:
                with st.spinner("Running DistilBERT..."):
                    try:
                        results = distilbert_clf(combined_text[:512])[0]
                        # Sort by score
                        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
                        top = results_sorted[0]
                        label = top["label"]
                        # Try to map LABEL_0 etc to readable names
                        if label.startswith("LABEL_"):
                            idx = int(label.split("_")[1])
                            label = TICKET_TYPES[idx] if idx < len(TICKET_TYPES) else label

                        st.markdown(f'<div class="result-box">'
                                    f'<div class="pred-label">🏷️ {label}</div>'
                                    f'<div class="pred-sub">Confidence: {top["score"]*100:.1f}%</div>'
                                    f'</div>', unsafe_allow_html=True)

                        # Confidence bars
                        st.markdown("**Confidence scores:**")
                        for r in results_sorted:
                            lbl = r["label"]
                            if lbl.startswith("LABEL_"):
                                idx = int(lbl.split("_")[1])
                                lbl = TICKET_TYPES[idx] if idx < len(TICKET_TYPES) else lbl
                            st.progress(r["score"], text=f"{lbl}: {r['score']*100:.1f}%")
                    except Exception as e:
                        st.error(f"DistilBERT error: {e}")
            else:
                # Fallback — rule-based demo
                keywords = combined_text.lower()
                if any(w in keywords for w in ["billing","payment","charge","invoice","refund"]):
                    pred = "Billing Inquiry"
                elif any(w in keywords for w in ["error","crash","bug","not working","issue","problem"]):
                    pred = "Technical Issue"
                elif any(w in keywords for w in ["account","login","password","access","username"]):
                    pred = "Account Management"
                else:
                    pred = "Product Inquiry"
                st.markdown(f'<div class="result-box">'
                            f'<div class="pred-label">🏷️ {pred}</div>'
                            f'<div class="pred-sub">Rule-based fallback (load DistilBERT model)</div>'
                            f'</div>', unsafe_allow_html=True)

        # ── Priority Prediction ─────────────────────────────
        with col2:
            st.markdown("#### 🚨 Priority Level")
            if tab_ok:
                try:
                    from scipy.sparse import hstack, csr_matrix
                    # Build features matching notebook structure
                    text_vec = tfidf.transform([combined_text])
                    # Encode categoricals using saved label encoders
                    def safe_encode(le_dict, col, val):
                        if col in le_dict:
                            le_obj = le_dict[col]
                            if val in le_obj.classes_:
                                return int(le_obj.transform([val])[0])
                        return 0
                    ch_enc  = safe_encode(le, "Ticket Channel", channel)
                    pr_enc  = safe_encode(le, "Product Purchased", product)
                    gen_enc = safe_encode(le, "Customer Gender", gender)
                    tab_feat = csr_matrix([[age, ch_enc, pr_enc, gen_enc, 0, 0, len(description), 0]])
                    X = hstack([text_vec, tab_feat])
                    # Use regression model score to derive priority bracket
                    hours = float(reg.predict(X)[0])
                    if hours < 5:    priority = "Low"
                    elif hours < 15: priority = "Medium"
                    elif hours < 30: priority = "High"
                    else:            priority = "Critical"
                    icon = PRIORITY_COLORS[priority]
                    st.markdown(f'<div class="result-box">'
                                f'<div class="pred-label">{icon} {priority}</div>'
                                f'<div class="pred-sub">Based on XGBoost regressor output</div>'
                                f'</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Priority error: {e}")
            else:
                st.warning("Load tabular models to predict priority")

        # ── Resolution Time ─────────────────────────────────
        with col3:
            st.markdown("#### ⏱️ Resolution Time")
            if tab_ok:
                try:
                    from scipy.sparse import hstack, csr_matrix
                    text_vec = tfidf.transform([combined_text])
                    def safe_encode(le_dict, col, val):
                        if col in le_dict:
                            le_obj = le_dict[col]
                            if val in le_obj.classes_:
                                return int(le_obj.transform([val])[0])
                        return 0
                    ch_enc  = safe_encode(le, "Ticket Channel", channel)
                    pr_enc  = safe_encode(le, "Product Purchased", product)
                    gen_enc = safe_encode(le, "Customer Gender", gender)
                    tab_feat = csr_matrix([[age, ch_enc, pr_enc, gen_enc, 0, 0, len(description), 0]])
                    X = hstack([text_vec, tab_feat])
                    hours = float(reg.predict(X)[0])
                    days  = hours / 24
                    st.markdown(f'<div class="result-box">'
                                f'<div class="pred-label">⏱️ ~{hours:.1f}h</div>'
                                f'<div class="pred-sub">≈ {days:.1f} days to resolution</div>'
                                f'</div>', unsafe_allow_html=True)
                    st.caption("⚠️ Note: Dataset timestamps are synthetic — regression target is noisy. "
                               "Model demonstrates pipeline, not production accuracy.")
                except Exception as e:
                    st.error(f"Regression error: {e}")
            else:
                st.warning("Load tabular models to predict resolution time")

    elif predict_btn:
        st.warning("Please enter a ticket subject or description first.")

    # ── Example Tickets ──────────────────────────────────────
    st.divider()
    st.markdown("#### 💡 Try These Example Tickets")
    examples = [
        ("Billing Issue", "I was charged twice for my subscription last month",
         "I noticed two charges of $49.99 on my credit card statement for the same month. Please refund the duplicate charge immediately."),
        ("Tech Issue", "App keeps crashing on startup",
         "Every time I open the mobile app it crashes within 3 seconds. I have tried reinstalling but the problem persists. Running iOS 17."),
        ("Account", "Cannot reset my password",
         "I clicked forgot password but the reset email never arrives. I have checked spam folder. My email is correct as I can see old emails from you."),
    ]
    cols = st.columns(3)
    for col, (title, subj, desc) in zip(cols, examples):
        with col:
            with st.expander(f"📋 {title}"):
                st.markdown(f"**Subject:** {subj}")
                st.markdown(f"**Description:** {desc[:100]}...")
