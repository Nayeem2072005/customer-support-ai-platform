"""pages/mlflow_results.py — MLflow Experiment Tracking Results"""
import streamlit as st
import pandas as pd
import plotly.express as px
import os

BASE    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS = os.path.join(BASE, "reports")

@st.cache_data
def load_mlflow_csv():
    path = os.path.join(REPORTS, "mlflow_all_runs_consolidated.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def show():
    st.title("🧪 MLflow Experiment Tracking")
    st.markdown("All model runs logged with full hyperparameters, metrics and artifacts")
    st.divider()

    df = load_mlflow_csv()

    if df is not None:
        st.success(f"✅ Loaded {len(df)} experiment runs from `reports/mlflow_all_runs_consolidated.csv`")
        st.divider()

        # ── Summary metrics ──────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs", len(df))
        c2.metric("Experiments", df["experiment_name"].nunique() if "experiment_name" in df.columns else "—")

        metric_cols = [c for c in df.columns if any(m in c.lower() for m in ["f1","accuracy","rmse","auc","r2"])]
        if metric_cols:
            best_col = metric_cols[0]
            c3.metric(f"Best {best_col}", f"{df[best_col].max():.4f}" if df[best_col].dtype != object else "—")
        c4.metric("Models Compared", df["model_type"].nunique() if "model_type" in df.columns else len(df))

        st.divider()

        # ── Tabs ─────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["📋 All Runs", "📊 Metric Comparison", "🏆 Best Models"])

        with tab1:
            st.markdown("### All MLflow Runs")
            # Filters
            cols_to_show = st.multiselect("Columns to display", df.columns.tolist(),
                                           default=df.columns[:10].tolist())
            st.dataframe(df[cols_to_show] if cols_to_show else df,
                         use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv,
                               file_name="mlflow_runs.csv", mime="text/csv")

        with tab2:
            st.markdown("### Metric Comparison Across Runs")
            if metric_cols:
                selected_metric = st.selectbox("Select metric", metric_cols)
                group_by = st.selectbox("Group by", [c for c in ["model_type","experiment_name","run_name"]
                                                       if c in df.columns])
                if group_by in df.columns and selected_metric in df.columns:
                    plot_df = df[[group_by, selected_metric]].dropna()
                    fig = px.bar(plot_df, x=group_by, y=selected_metric,
                                 title=f"{selected_metric} by {group_by}",
                                 color=selected_metric,
                                 color_continuous_scale="Blues",
                                 text=plot_df[selected_metric].round(4))
                    fig.update_layout(xaxis_tickangle=-30,
                                      paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                    # Line chart — runs over time
                    if "start_time" in df.columns or "run_id" in df.columns:
                        st.markdown("#### Metric Progression Across Runs")
                        fig2 = px.line(df.reset_index(), x="index", y=selected_metric,
                                       color=group_by if group_by in df.columns else None,
                                       title=f"{selected_metric} across all runs",
                                       markers=True)
                        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                           plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No standard metric columns (f1, accuracy, rmse, auc, r2) found in CSV")

        with tab3:
            st.markdown("### 🏆 Best Run per Experiment")
            if metric_cols and "experiment_name" in df.columns:
                best_col = st.selectbox("Rank by metric", metric_cols, key="best_metric")
                ascending = st.checkbox("Lower is better (e.g. RMSE)", value=False)
                best_runs = df.sort_values(best_col, ascending=ascending)\
                              .groupby("experiment_name").first().reset_index()
                st.dataframe(best_runs, use_container_width=True, hide_index=True)
            else:
                st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    else:
        # ── No CSV found — show instructions ─────────────────
        st.info("📂 MLflow CSV not found at `reports/mlflow_all_runs_consolidated.csv`")
        st.divider()
        st.markdown("### 🚀 How to View Your MLflow Dashboard")

        st.markdown("#### Option 1: Launch MLflow UI (Recommended)")
        st.code("""
# In your project terminal (with venv activated):
cd "C:\\Users\\CHARU\\OneDrive\\Desktop\\Final Project"
mlflow ui --backend-store-uri ./mlruns --port 5000

# Then open: http://localhost:5000
        """, language="bash")

        st.markdown("#### Option 2: Export runs to CSV")
        st.code("""
import mlflow
import pandas as pd

client = mlflow.tracking.MlflowClient(tracking_uri="./mlruns")
experiments = client.search_experiments()

all_runs = []
for exp in experiments:
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    for run in runs:
        row = {"experiment": exp.name, "run_id": run.info.run_id,
               "status": run.info.status, **run.data.params, **run.data.metrics}
        all_runs.append(row)

df = pd.DataFrame(all_runs)
df.to_csv("reports/mlflow_all_runs_consolidated.csv", index=False)
print(f"Exported {len(df)} runs")
        """, language="python")

        st.markdown("#### What MLflow Tracked in Your Project:")
        st.markdown("""
        | Experiment | Models Logged | Key Metrics |
        |------------|--------------|-------------|
        | `customer_support_ensemble_models` | Random Forest, XGBoost (Optuna), LightGBM (Optuna) | F1-macro, Accuracy, ROC-AUC |
        | `regression_clustering` | XGBoost Regressor, LightGBM Regressor | RMSE, MAE, R² |
        | `baseline_models` | Logistic Regression, Naive Bayes | F1-macro, Accuracy |
        | `bilstm_classification` | BiLSTM (GloVe) | F1-macro, Accuracy |
        | `distilbert_finetuning` | DistilBERT | F1-macro, Accuracy |
        """)

        st.markdown("#### 📸 MLflow Screenshot")
        st.markdown("Run `mlflow ui` and take a screenshot of your experiment dashboard to show the evaluator.")

        # Show the path to mlruns
        mlruns_path = os.path.join(BASE, "mlruns")
        if os.path.exists(mlruns_path):
            st.success(f"✅ MLflow tracking directory found at `{mlruns_path}`")
            # Count experiments
            exps = [d for d in os.listdir(mlruns_path)
                    if os.path.isdir(os.path.join(mlruns_path, d))]
            st.metric("MLflow Experiments Found", len(exps))
        else:
            st.warning("MLflow `mlruns/` directory not found in project root")
