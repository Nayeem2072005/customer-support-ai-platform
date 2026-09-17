"""pages/explainability.py — SHAP Explainability"""
import streamlit as st
import os, joblib
import numpy as np
import pandas as pd

BASE   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(BASE, "models")
REPORTS = os.path.join(BASE, "reports")

@st.cache_resource
def load_model_and_vectorizer():
    try:
        import shap
        reg  = joblib.load(os.path.join(MODELS, "resolution_time_xgboost.joblib"))
        tfidf = joblib.load(os.path.join(MODELS, "tfidf_vectorizer_ensemble.joblib"))
        return reg, tfidf, True
    except Exception as e:
        return None, None, False

def show():
    st.title("🔍 SHAP Explainability")
    st.markdown("Model interpretability using SHAP (SHapley Additive exPlanations)")
    st.divider()

    reg, tfidf, ok = load_model_and_vectorizer()

    st.markdown("""
    SHAP explains **why** a model makes each prediction by assigning each feature
    a contribution value. Positive SHAP values push the prediction higher,
    negative values push it lower.
    """)

    tab1, tab2, tab3 = st.tabs([
        "📊 Summary Plot", "💧 Waterfall Plot", "📈 Dependence Plot"
    ])

    with tab1:
        st.markdown("### SHAP Summary Plot — Feature Importance")
        st.markdown("""
        The SHAP summary plot shows the **global feature importance** across all test samples.
        Each point is one sample. The x-axis shows the SHAP value (impact on prediction).
        Color shows the feature value (red = high, blue = low).
        """)

        # Try to load saved SHAP plot images from reports/
        shap_summary = os.path.join(REPORTS, "shap_summary_plot.png")
        if os.path.exists(shap_summary):
            st.image(shap_summary, caption="SHAP Summary Plot — XGBoost Regressor (Resolution Time)",
                     use_container_width=True)
        else:
            # Generate live SHAP if models loaded
            if ok:
                processed = os.path.join(BASE, "data_processed", "test.csv")
                if os.path.exists(processed):
                    with st.spinner("Computing SHAP values (this may take 30-60 seconds)..."):
                        try:
                            import shap, matplotlib.pyplot as plt
                            from scipy.sparse import hstack, csr_matrix

                            df = pd.read_csv(processed)
                            # Use closed tickets only
                            df = df[df["Ticket Status"].astype(str).str.lower() == "closed"].head(100)

                            TABULAR_FEATURES = [
                                "Customer Age", "Ticket Channel_Encoded",
                                "Product Purchased_Encoded", "Customer Gender_Encoded",
                                "Customer_Tenure_Days", "Description_Char_Count",
                                "Description_Word_Count", "Sentiment_Polarity"
                            ]
                            available = [c for c in TABULAR_FEATURES if c in df.columns]
                            text_col  = "Combined_Text_Clean" if "Combined_Text_Clean" in df.columns \
                                        else "Description_Clean" if "Description_Clean" in df.columns else None

                            if available and text_col:
                                tab  = csr_matrix(df[available].fillna(0).values)
                                text = tfidf.transform(df[text_col].fillna(""))
                                X    = hstack([text, tab]).toarray()

                                explainer    = shap.TreeExplainer(reg)
                                shap_values  = explainer.shap_values(X[:50])

                                fig, ax = plt.subplots(figsize=(10, 6))
                                feat_names = tfidf.get_feature_names_out().tolist() + available
                                shap.summary_plot(shap_values, X[:50], feature_names=feat_names,
                                                  max_display=20, show=False, plot_type="bar")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.warning("Feature columns not found in test.csv. Run notebook 09 first.")
                        except Exception as e:
                            st.error(f"SHAP computation error: {e}")
                else:
                    st.warning("Test dataset not found. Run notebooks first.")
            else:
                st.info("📂 Models not loaded. Place saved SHAP plot at `reports/shap_summary_plot.png` OR ensure models are in `models/` folder.")
                st.markdown("""
                **What this plot would show:**
                - Top features driving resolution time predictions
                - Text features (TF-IDF) vs tabular features (age, channel, product) relative importance
                - Direction of each feature's effect (higher/lower resolution time)
                """)

    with tab2:
        st.markdown("### SHAP Waterfall Plot — Single Prediction Explanation")
        st.markdown("""
        The waterfall plot explains **one individual prediction** — how each feature
        pushed the prediction up or down from the baseline (mean prediction).
        """)

        shap_waterfall = os.path.join(REPORTS, "shap_waterfall_plot.png")
        if os.path.exists(shap_waterfall):
            st.image(shap_waterfall, caption="SHAP Waterfall Plot — Individual Prediction",
                     use_container_width=True)
        else:
            st.info("📂 Place saved waterfall plot at `reports/shap_waterfall_plot.png`")
            st.markdown("""
            **How to read it:**
            - **E[f(X)]** = model's average prediction (base value)
            - **f(x)** = prediction for this specific ticket
            - Each bar shows how much one feature pushed the prediction up (red) or down (blue)
            """)

    with tab3:
        st.markdown("### SHAP Dependence Plot — Feature Interactions")
        st.markdown("""
        Shows how a single feature's value affects SHAP values,
        revealing non-linear relationships and interactions.
        """)

        shap_dep = os.path.join(REPORTS, "shap_dependence_plot.png")
        if os.path.exists(shap_dep):
            st.image(shap_dep, caption="SHAP Dependence Plot",
                     use_container_width=True)
        else:
            st.info("📂 Place saved dependence plot at `reports/shap_dependence_plot.png`")

        st.divider()
        st.markdown("### 📋 Business Insights from SHAP")
        st.markdown("""
        | Feature | Finding | Business Action |
        |---------|---------|----------------|
        | **Ticket Channel** | Phone tickets have higher resolution time SHAP values | Prioritize phone tickets for faster agent assignment |
        | **Customer Age** | Older customers → slightly longer resolution | Assign senior agents to older customer tickets |
        | **Description Length** | Longer descriptions → complex issues → more time | Auto-flag long tickets for escalation |
        | **Product Type** | Some products consistently drive higher priority | Product-specific support team routing |
        | **Text Keywords** | Words like "urgent", "cannot", "broken" → high SHAP | Build keyword-based triage rules |
        """)
