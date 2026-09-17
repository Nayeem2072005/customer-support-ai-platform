"""pages/eda.py — EDA & Insights page"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@st.cache_data
def load_data():
    raw = os.path.join(BASE, "data_raw", "customer_support_tickets.csv")
    if os.path.exists(raw):
        return pd.read_csv(raw)
    processed = os.path.join(BASE, "data_processed", "tickets_with_features.csv")
    if os.path.exists(processed):
        return pd.read_csv(processed)
    return None

def show():
    st.title("📊 EDA & Insights")
    st.markdown("Exploratory analysis of the Customer Support Ticket Dataset")
    st.divider()

    df = load_data()
    if df is None:
        st.error("Dataset not found. Place CSV at `data_raw/customer_support_tickets.csv`")
        return

    # ── Basic Stats ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickets", f"{len(df):,}")
    c2.metric("Ticket Types", df["Ticket Type"].nunique() if "Ticket Type" in df.columns else "—")
    c3.metric("Priority Levels", df["Ticket Priority"].nunique() if "Ticket Priority" in df.columns else "—")
    c4.metric("Channels", df["Ticket Channel"].nunique() if "Ticket Channel" in df.columns else "—")
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Ticket Distribution", "⭐ Satisfaction", "📡 Channels", "👤 Customer Profile", "📝 Text Analysis"
    ])

    # ── Tab 1: Ticket Distribution ──────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if "Ticket Type" in df.columns:
                fig = px.pie(df, names="Ticket Type", title="Ticket Type Distribution",
                             color_discrete_sequence=px.colors.sequential.Blues_r,
                             hole=0.4)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Ticket Priority" in df.columns:
                priority_order = ["Critical", "High", "Medium", "Low"]
                counts = df["Ticket Priority"].value_counts().reindex(priority_order).dropna()
                colors = ["#ef4444", "#f97316", "#eab308", "#22c55e"]
                fig = px.bar(x=counts.index, y=counts.values,
                             title="Ticket Priority Distribution",
                             color=counts.index,
                             color_discrete_sequence=colors,
                             labels={"x": "Priority", "y": "Count"})
                fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        if "Ticket Status" in df.columns:
            fig = px.pie(df, names="Ticket Status", title="Ticket Status Breakdown",
                         color_discrete_sequence=["#22c55e", "#f59e0b", "#64748b"],
                         hole=0.4)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Satisfaction ─────────────────────────────────
    with tab2:
        if "Customer Satisfaction Rating" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                sat = df["Customer Satisfaction Rating"].dropna()
                fig = px.histogram(sat, nbins=5, title="Satisfaction Rating Distribution",
                                   color_discrete_sequence=["#38bdf8"],
                                   labels={"value": "Rating (1-5)", "count": "Count"})
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Mean Satisfaction", f"{sat.mean():.2f} / 5")

            with col2:
                if "Ticket Type" in df.columns:
                    avg_sat = df.groupby("Ticket Type")["Customer Satisfaction Rating"].mean().reset_index()
                    avg_sat.columns = ["Ticket Type", "Avg Rating"]
                    fig = px.bar(avg_sat, x="Ticket Type", y="Avg Rating",
                                 title="Avg Satisfaction by Ticket Type",
                                 color="Avg Rating",
                                 color_continuous_scale="Blues")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

            if "Ticket Priority" in df.columns:
                heat_data = df.groupby(["Ticket Type", "Ticket Priority"])["Customer Satisfaction Rating"].mean().unstack()
                fig = px.imshow(heat_data, title="Satisfaction Heatmap: Type × Priority",
                                color_continuous_scale="Blues", aspect="auto",
                                text_auto=".2f")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Customer Satisfaction Rating column not found in dataset.")

    # ── Tab 3: Channels ──────────────────────────────────────
    with tab3:
        if "Ticket Channel" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                ch = df["Ticket Channel"].value_counts().reset_index()
                ch.columns = ["Channel", "Count"]
                fig = px.bar(ch, x="Channel", y="Count",
                             title="Tickets by Channel",
                             color="Count", color_continuous_scale="Blues")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "Ticket Type" in df.columns:
                    ct = df.groupby(["Ticket Channel", "Ticket Type"]).size().reset_index(name="Count")
                    fig = px.bar(ct, x="Ticket Channel", y="Count", color="Ticket Type",
                                 title="Ticket Type by Channel", barmode="group",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

            if "Customer Satisfaction Rating" in df.columns:
                ch_sat = df.groupby("Ticket Channel")["Customer Satisfaction Rating"].mean().reset_index()
                ch_sat.columns = ["Channel", "Avg Satisfaction"]
                fig = px.bar(ch_sat, x="Channel", y="Avg Satisfaction",
                             title="Avg Satisfaction by Channel",
                             color="Avg Satisfaction", color_continuous_scale="RdYlGn")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Customer Profile ──────────────────────────────
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            if "Customer Age" in df.columns:
                fig = px.histogram(df, x="Customer Age", nbins=20,
                                   title="Customer Age Distribution",
                                   color_discrete_sequence=["#6366f1"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Customer Gender" in df.columns:
                fig = px.pie(df, names="Customer Gender",
                             title="Customer Gender Distribution",
                             color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399"],
                             hole=0.4)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        if "Product Purchased" in df.columns:
            top_products = df["Product Purchased"].value_counts().head(10).reset_index()
            top_products.columns = ["Product", "Count"]
            fig = px.bar(top_products, x="Count", y="Product", orientation="h",
                         title="Top 10 Products with Support Tickets",
                         color="Count", color_continuous_scale="Blues")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Text Analysis ─────────────────────────────────
    with tab5:
        st.markdown("#### Text Length Analysis")
        if "Ticket Description" in df.columns:
            df["desc_len"] = df["Ticket Description"].fillna("").apply(len)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x="desc_len", nbins=30,
                                   title="Description Length Distribution (chars)",
                                   color_discrete_sequence=["#38bdf8"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Avg Description Length", f"{df['desc_len'].mean():.0f} chars")

            with col2:
                if "Ticket Type" in df.columns:
                    avg_len = df.groupby("Ticket Type")["desc_len"].mean().reset_index()
                    avg_len.columns = ["Ticket Type", "Avg Length"]
                    fig = px.bar(avg_len, x="Ticket Type", y="Avg Length",
                                 title="Avg Description Length by Ticket Type",
                                 color="Avg Length", color_continuous_scale="Blues")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Class Imbalance Check")
        col1, col2 = st.columns(2)
        with col1:
            if "Ticket Type" in df.columns:
                counts = df["Ticket Type"].value_counts()
                fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                                       marker_color=["#38bdf8","#6366f1","#34d399","#f59e0b"]))
                fig.update_layout(title="Ticket Type Class Balance",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                imbalance = counts.max() / counts.min()
                st.caption(f"Max/Min ratio: {imbalance:.2f}x — {'⚠️ Imbalanced' if imbalance > 2 else '✅ Balanced'}")
        with col2:
            if "Ticket Priority" in df.columns:
                counts = df["Ticket Priority"].value_counts()
                fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                                       marker_color=["#ef4444","#f97316","#eab308","#22c55e"]))
                fig.update_layout(title="Ticket Priority Class Balance",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
