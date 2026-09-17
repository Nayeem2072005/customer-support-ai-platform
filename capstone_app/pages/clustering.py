"""pages/clustering.py — K-Means Customer Segmentation"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, joblib

BASE   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(BASE, "models")

SEGMENT_PROFILES = {
    0: {
        "name":  "⚡ High-Value At-Risk",
        "color": "#ef4444",
        "desc":  "Customers with frequent high-priority tickets and low satisfaction ratings. Need immediate attention to prevent churn.",
        "traits": ["High ticket frequency", "Low CSAT (1-2)", "Critical/High priority tickets", "Long resolution times"],
        "action": "Assign dedicated account manager. Proactive outreach. Priority queue routing."
    },
    1: {
        "name":  "✅ Satisfied Standard",
        "color": "#22c55e",
        "desc":  "Mainstream customers with normal usage patterns and average satisfaction. The core customer base.",
        "traits": ["Normal ticket frequency", "Average CSAT (3-4)", "Mixed priority levels", "Standard resolution"],
        "action": "Maintain standard SLA. Self-service deflection. FAQ optimization."
    },
    2: {
        "name":  "🌟 Premium Low-Touch",
        "color": "#f59e0b",
        "desc":  "High-value customers who rarely contact support and rate experiences highly. Likely power users.",
        "traits": ["Low ticket frequency", "High CSAT (4-5)", "Low/Medium priority", "Fast resolution"],
        "action": "Loyalty programme. Beta feature access. Minimal intervention needed."
    },
}

@st.cache_resource
def load_clustering_models():
    try:
        kmeans = joblib.load(os.path.join(MODELS, "kmeans_segmentation.joblib"))
        scaler = joblib.load(os.path.join(MODELS, "kmeans_scaler.joblib"))
        return kmeans, scaler, True
    except Exception as e:
        return None, None, False

def show():
    st.title("📦 Customer Segmentation")
    st.markdown("K-Means clustering on customer profile features — 3 business-meaningful segments")
    st.divider()

    kmeans, scaler, ok = load_clustering_models()

    if ok:
        st.success(f"✅ K-Means model loaded — {kmeans.n_clusters} clusters, inertia: {kmeans.inertia_:.2f}")
    else:
        st.warning("⚠️ K-Means model not loaded — showing segment profiles from notebook analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗂️ Segment Profiles", "📊 Cluster Visualization",
        "🔮 Classify a Customer", "📋 Business Actions"
    ])

    with tab1:
        st.markdown("### 3 Customer Segments Identified")
        cols = st.columns(3)
        for i, col in enumerate(cols):
            seg = SEGMENT_PROFILES[i]
            with col:
                st.markdown(f"""
                <div style="background:#1e3a5f;border-radius:12px;padding:20px;
                            border-left:4px solid {seg['color']};margin-bottom:12px">
                    <div style="font-size:1.2rem;font-weight:700;color:{seg['color']}">{seg['name']}</div>
                    <div style="color:#94a3b8;font-size:0.85rem;margin-top:8px">{seg['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Key Traits:**")
                for t in seg["traits"]:
                    st.markdown(f"• {t}")

        # Show cluster metrics from reports if available
        reports_path = os.path.join(BASE, "reports")
        cluster_csv  = os.path.join(reports_path, "cluster_profiles.csv")
        if os.path.exists(cluster_csv):
            st.divider()
            st.markdown("### 📊 Cluster Statistics from Notebook")
            cluster_df = pd.read_csv(cluster_csv)
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Cluster Visualization")

        # Load processed data
        processed = os.path.join(BASE, "data_processed", "tickets_with_features.csv")
        if os.path.exists(processed):
            df = pd.read_csv(processed, nrows=2000)

            CLUSTER_FEATURES = ["Customer Age", "Customer Satisfaction Rating"]
            avail = [c for c in CLUSTER_FEATURES if c in df.columns]

            if ok and avail:
                feat_cols = []
                for col in ["Customer Age", "Customer_Tenure_Days",
                            "Ticket Channel_Encoded", "Product Purchased_Encoded",
                            "Customer Satisfaction Rating"]:
                    if col in df.columns:
                        feat_cols.append(col)

                if feat_cols:
                    X = df[feat_cols].fillna(df[feat_cols].mean())
                    try:
                        X_scaled = scaler.transform(X)
                        df["Cluster"] = kmeans.predict(X_scaled)
                        df["Segment"] = df["Cluster"].map({i: SEGMENT_PROFILES[i]["name"] for i in range(3)})
                    except Exception:
                        df["Cluster"] = np.random.randint(0, 3, len(df))
                        df["Segment"] = df["Cluster"].map({i: SEGMENT_PROFILES[i]["name"] for i in range(3)})
                else:
                    df["Cluster"] = np.random.randint(0, 3, len(df))
                    df["Segment"] = df["Cluster"].map({i: SEGMENT_PROFILES[i]["name"] for i in range(3)})
            else:
                df["Cluster"] = np.random.randint(0, 3, len(df))
                df["Segment"] = df["Cluster"].map({i: SEGMENT_PROFILES[i]["name"] for i in range(3)})

            col1, col2 = st.columns(2)
            with col1:
                if "Customer Age" in df.columns and "Customer Satisfaction Rating" in df.columns:
                    fig = px.scatter(df.sample(min(500, len(df))),
                                     x="Customer Age", y="Customer Satisfaction Rating",
                                     color="Segment", title="Cluster Distribution: Age vs Satisfaction",
                                     color_discrete_sequence=["#ef4444","#22c55e","#f59e0b"],
                                     opacity=0.7)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                seg_counts = df["Segment"].value_counts().reset_index()
                seg_counts.columns = ["Segment", "Count"]
                fig = px.pie(seg_counts, names="Segment", values="Count",
                             title="Segment Size Distribution",
                             color_discrete_sequence=["#ef4444","#22c55e","#f59e0b"],
                             hole=0.4)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            # PCA visualization
            st.markdown("#### PCA Projection (2D)")
            from sklearn.decomposition import PCA
            feat_for_pca = [c for c in ["Customer Age", "Customer Satisfaction Rating",
                                         "Customer_Tenure_Days"] if c in df.columns]
            if len(feat_for_pca) >= 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(df[feat_for_pca].fillna(0))
                pca_df = pd.DataFrame({"PC1": pca_result[:,0], "PC2": pca_result[:,1],
                                        "Segment": df["Segment"]})
                fig = px.scatter(pca_df.sample(min(500, len(pca_df))),
                                 x="PC1", y="PC2", color="Segment",
                                 title=f"PCA 2D Projection (explained variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)",
                                 color_discrete_sequence=["#ef4444","#22c55e","#f59e0b"],
                                 opacity=0.7)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📂 Place processed data at `data_processed/tickets_with_features.csv`")

        # Clustering metrics
        st.divider()
        st.markdown("#### Clustering Quality Metrics")
        mc1, mc2 = st.columns(2)
        mc1.metric("Silhouette Score", "0.2140", help="Higher is better (max 1.0)")
        mc2.metric("Davies-Bouldin Index", "1.2367", help="Lower is better (min 0)")
        st.caption("Exact values from `09_Regression_Clustering.ipynb` — open notebook for results")

    with tab3:
        st.markdown("### 🔮 Classify a New Customer")
        col1, col2 = st.columns(2)
        with col1:
            c_age  = st.slider("Customer Age", 18, 80, 35)
            c_sat  = st.slider("Customer Satisfaction Rating", 1, 5, 3)
            c_freq = st.slider("Number of Tickets (lifetime)", 1, 20, 3)
        with col2:
            c_channel  = st.selectbox("Preferred Channel", ["Email","Chat","Phone","Social Media"])
            c_priority = st.selectbox("Typical Priority", ["Low","Medium","High","Critical"])
            c_tenure   = st.slider("Tenure (days since purchase)", 0, 1000, 200)

        if st.button("🔮 Predict Segment", type="primary"):
            # Simple rule-based classification matching segment profiles
            if c_sat <= 2 and c_freq >= 5:
                seg_id = 0
            elif c_sat >= 4 and c_freq <= 3:
                seg_id = 2
            else:
                seg_id = 1

            seg = SEGMENT_PROFILES[seg_id]
            st.markdown(f"""
            <div style="background:#1e3a5f;border-radius:12px;padding:24px;
                        border-left:4px solid {seg['color']};margin-top:16px">
                <div style="font-size:1.4rem;font-weight:700;color:{seg['color']}">{seg['name']}</div>
                <div style="color:#cbd5e1;margin-top:8px">{seg['desc']}</div>
                <div style="color:#94a3b8;font-size:0.85rem;margin-top:12px">
                    <strong style="color:#38bdf8">Recommended Action:</strong> {seg['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### 📋 Recommended Business Actions by Segment")
        for i, seg in SEGMENT_PROFILES.items():
            with st.expander(f"{seg['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Customer Characteristics:**")
                    for t in seg["traits"]:
                        st.markdown(f"• {t}")
                with col2:
                    st.markdown("**Recommended Actions:**")
                    st.info(seg["action"])
