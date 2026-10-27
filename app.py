import os
import sys
import io
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure workspace root is always at the head of Python search path
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from src.data_processor import DataProcessor
from src.utils import evaluate_risk_level, generate_retention_strategy
from src.dynamic_engine import DynamicDataEngine, create_sample_ecommerce_dataset

# -----------------------------------------------------------------------------
# Streamlit App Configuration & Modern Glassmorphism Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation & Churn Intelligence Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 6px;
    }

    /* Glass Panels */
    .glass-panel {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
    }

    /* Strategy Recommendation Card */
    .strategy-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #6366F1;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .strategy-title {
        font-weight: 600;
        color: #E2E8F0;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .strategy-desc {
        color: #94A3B8;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    /* Badge Pills */
    .risk-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Streamlit Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Pre-Trained Artifact Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_pretrained_artifacts():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    meta_path = os.path.join(models_dir, "pipeline_metadata.json")

    if not os.path.exists(meta_path):
        from src.model_pipeline import ModelPipeline
        pipeline = ModelPipeline()
        pipeline.run_training_pipeline()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    supervised_models = joblib.load(os.path.join(models_dir, "supervised_models.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    kmeans = joblib.load(os.path.join(models_dir, "kmeans.joblib"))
    iso_forest = joblib.load(os.path.join(models_dir, "isolation_forest.joblib"))
    pca_2d = joblib.load(os.path.join(models_dir, "pca_2d.joblib"))
    pca_3d = joblib.load(os.path.join(models_dir, "pca_3d.joblib"))

    return {
        "metadata": meta,
        "models": supervised_models,
        "scaler": scaler,
        "kmeans": kmeans,
        "iso_forest": iso_forest,
        "pca_2d": pca_2d,
        "pca_3d": pca_3d
    }


@st.cache_data
def load_telco_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "TelcoCustomerChurn.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        processor = DataProcessor(df)
        df = processor.rename_columns()
        df, _ = processor.clean_total_charges()
        df = processor.convert_data_types()
        return df
    return None


# -----------------------------------------------------------------------------
# Sidebar Navigation & Mode Controller
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style='text-align: center; padding: 6px 0 16px 0;'>
    <div style='font-size: 2.5rem;'>🔮</div>
    <h2 style='margin: 4px 0 0 0; color: #F8FAFC; font-size: 1.4rem;'>Intelligence Studio</h2>
    <p style='color: #94A3B8; font-size: 0.8rem; margin: 0;'>Dynamic ML & Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Select Intelligence Mode",
    options=["📤 Upload Custom Dataset / Database", "🔮 Pre-Trained Telco Churn Demo"],
    index=0,
    help="Upload your own tabular database for dynamic prediction, violin plots, and clustering, or explore the pre-trained benchmark."
)

st.sidebar.markdown("---")

# Session state initialization for dynamic mode
if "dynamic_raw_df" not in st.session_state:
    st.session_state["dynamic_raw_df"] = None
if "dynamic_engine" not in st.session_state:
    st.session_state["dynamic_engine"] = None
if "dynamic_ready" not in st.session_state:
    st.session_state["dynamic_ready"] = False


# =============================================================================
# MODE 1: DYNAMIC AUTO-ML & CUSTOM DATASET STUDIO
# =============================================================================
if mode == "📤 Upload Custom Dataset / Database":
    st.sidebar.subheader("📁 Ingest Custom Dataset")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Tabular File (CSV / Excel / TSV)",
        type=["csv", "xlsx", "tsv", "txt"],
        help="Upload any customer database or tabular file with target and feature columns."
    )

    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("🎲 Sample Data", use_container_width=True, help="Load synthetic 400-row E-Commerce customer portfolio"):
            st.session_state["dynamic_raw_df"] = create_sample_ecommerce_dataset(n_samples=450)
            st.session_state["dynamic_engine"] = None
            st.session_state["dynamic_ready"] = False
            st.rerun()

    with col_btn2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state["dynamic_raw_df"] = None
            st.session_state["dynamic_engine"] = None
            st.session_state["dynamic_ready"] = False
            st.rerun()

    # Process uploaded file
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df_in = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".tsv"):
                df_in = pd.read_csv(uploaded_file, sep="\t")
            else:
                df_in = pd.read_csv(uploaded_file)
            st.session_state["dynamic_raw_df"] = df_in
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

    # Header Section
    col_logo, col_head = st.columns([1, 8])
    with col_logo:
        st.markdown("<div style='font-size: 3.2rem; text-align: center;'>⚡</div>", unsafe_allow_html=True)
    with col_head:
        st.markdown("""
            <h1 style='margin-bottom: 2px; background: linear-gradient(90deg, #6366F1, #38BDF8, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Dynamic AutoML, Visual Analytics & Clustering Engine
            </h1>
            <p style='color: #94A3B8; font-size: 1.05rem; margin-top: 0;'>
                Upload any database to instantly train 6 classifiers, compute Precision, Recall, F1 score leaderboards, generate Violin distribution plots, and perform K-Means 3D clustering.
            </p>
        """, unsafe_allow_html=True)

    if st.session_state["dynamic_raw_df"] is None:
        # Welcome & Instructions Banner
        st.markdown("""
            <div class="glass-panel" style="text-align: center; padding: 48px 24px; border: 1px dashed rgba(99, 102, 241, 0.4);">
                <div style="font-size: 3.5rem; margin-bottom: 12px;">📂</div>
                <h2 style="color: #F8FAFC; margin-bottom: 8px;">Upload Any Tabular Dataset to Begin</h2>
                <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 20px auto; font-size: 0.95rem;">
                    Upload your customer churn dataset, sales transactions, or credit records in CSV or Excel format.
                    The platform auto-detects features, cleans data, trains 6 ML models, and builds interactive violin plots and cluster visualizations.
                </p>
                <div style="color: #64748B; font-size: 0.85rem;">
                    💡 Tip: Click <b>'🎲 Sample Data'</b> in the sidebar to test instantly with a synthetic E-Commerce portfolio.
                </div>
            </div>
        """, unsafe_allow_html=True)

    else:
        df_raw = st.session_state["dynamic_raw_df"]
        engine_preview = DynamicDataEngine(df_raw)
        detection = engine_preview.auto_detect_columns()

        # Configuration Accordion
        with st.expander("⚙️ Dataset Configuration & Column Mapping", expanded=not st.session_state["dynamic_ready"]):
            st.markdown(f"**Loaded Dataset Shape:** `{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns`")
            
            c_cfg1, c_cfg2, c_cfg3 = st.columns(3)
            with c_cfg1:
                all_cols = list(df_raw.columns)
                default_target_idx = all_cols.index(detection["suggested_target"]) if detection["suggested_target"] in all_cols else len(all_cols) - 1
                selected_target = st.selectbox(
                    "Target Column to Predict",
                    options=all_cols,
                    index=default_target_idx,
                    help="Select the column representing churn, conversion, default, or class."
                )

            with c_cfg2:
                # Target unique values
                target_uniques = df_raw[selected_target].dropna().astype(str).str.strip().unique().tolist()
                pos_default = target_uniques[1] if len(target_uniques) > 1 else target_uniques[0]
                # Pick sensible default if available
                for cand in ["Yes", "1", "True", "Churn", "default"]:
                    for u in target_uniques:
                        if u.lower() == cand.lower():
                            pos_default = u
                            break
                selected_pos_val = st.selectbox(
                    "Positive Class Label (Event of Interest)",
                    options=target_uniques,
                    index=target_uniques.index(pos_default) if pos_default in target_uniques else 0,
                    help="Which value represents the target positive event (e.g. Churn = 'Yes')."
                )

            with c_cfg3:
                id_defaults = [c for c in detection["id_candidates"] if c in all_cols and c != selected_target]
                selected_ids = st.multiselect(
                    "ID / Key Columns to Drop",
                    options=[c for c in all_cols if c != selected_target],
                    default=id_defaults,
                    help="Unique identifiers that should not be used as predictive features."
                )

            # Hyperparameters row
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                test_size = st.slider("Test Split Ratio", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
            with p_col2:
                n_clusters = st.slider("K-Means Clusters (K)", min_value=2, max_value=8, value=3, step=1)
            with p_col3:
                contamination = st.slider("Anomaly Contamination (%)", min_value=1.0, max_value=10.0, value=3.0, step=0.5) / 100.0
            with p_col4:
                cap_outliers = st.checkbox("IQR Continuous Outlier Capping", value=True)

            btn_train = st.button("🚀 Train Models & Run Intelligence Pipeline", type="primary", use_container_width=True)

            if btn_train:
                with st.spinner("Processing dataset, encoding features, training 6 models, and computing clusters..."):
                    engine = DynamicDataEngine(df_raw)
                    try:
                        engine.sanitize_and_prepare(
                            target_col=selected_target,
                            id_cols=selected_ids,
                            positive_value=selected_pos_val,
                            cap_outliers=cap_outliers
                        )
                        engine.train_and_benchmark_models(test_size=test_size)
                        engine.compute_clustering_and_anomalies(n_clusters=n_clusters, contamination=contamination)
                        st.session_state["dynamic_engine"] = engine
                        st.session_state["dynamic_ready"] = True
                        st.success("✅ Dynamic Pipeline execution completed successfully!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Pipeline Execution Failed: {err}")

        # Render Results Dashboard when pipeline has executed
        if st.session_state["dynamic_ready"] and st.session_state["dynamic_engine"] is not None:
            engine = st.session_state["dynamic_engine"]
            eval_results = engine.evaluation_results
            top_model = eval_results[0] if eval_results else {"Model": "N/A", "F1_Score": 0, "Accuracy": 0}
            total_records = len(engine.clean_df)
            pos_count = int(engine.y_clean.sum())
            pos_rate = (pos_count / total_records) * 100 if total_records else 0.0
            num_features = len(engine.encoded_feature_names)
            anom_count = int((engine.scored_df["Is_Anomaly"] == "Anomaly (Outlier)").sum()) if engine.scored_df is not None else 0

            # Top KPI Metric Cards
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Records</div>
                        <div class="metric-value">{total_records:,}</div>
                        <div class="metric-sub">Processed accounts</div>
                    </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Encoded Features</div>
                        <div class="metric-value" style="color: #38BDF8;">{num_features}</div>
                        <div class="metric-sub">{len(engine.numeric_cols)} num • {len(engine.categorical_cols)} cat</div>
                    </div>
                """, unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{engine.inverse_label_mapping.get(1, 'Target')} Rate</div>
                        <div class="metric-value" style="color: #EF4444;">{pos_rate:.1f}%</div>
                        <div class="metric-sub">{pos_count:,} positive events</div>
                    </div>
                """, unsafe_allow_html=True)
            with k4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Champion Classifier</div>
                        <div class="metric-value" style="font-size: 1.3rem; color: #10B981;">{top_model['Model'][:16]}</div>
                        <div class="metric-sub">F1: {top_model['F1_Score']:.4f} • Acc: {top_model['Accuracy']:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            with k5:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Detected Outliers</div>
                        <div class="metric-value" style="color: #F59E0B;">{anom_count:,}</div>
                        <div class="metric-sub">Isolation Forest (3%)</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            # Dynamic Main Navigation Tabs
            d_tab_violin, d_tab_leaderboard, d_tab_cluster, d_tab_anomaly, d_tab_predict = st.tabs([
                "🎻 Violin & Distribution Plots",
                "📈 Precision, Recall, F1 Leaderboard",
                "👥 K-Means Clustering & 2D/3D PCA",
                "🕵️ Anomaly Account Detection",
                "🎯 Live Inference & Scored Export"
            ])

            # -----------------------------------------------------------------
            # DYNAMIC TAB 1: VIOLIN & DISTRIBUTION PLOTS (Requested Feature)
            # -----------------------------------------------------------------
            with d_tab_violin:
                st.markdown("### 🎻 Multi-Dimensional Feature Violin & Density Distribution")
                st.write(
                    "Violin plots combine kernel density estimation (KDE) with box plots to visualize multimodal distributions, "
                    "interquartile ranges, and outlier spread across target classes and clusters."
                )

                if engine.numeric_cols:
                    v_col1, v_col2 = st.columns([1, 1])
                    with v_col1:
                        sel_violin_feat = st.selectbox(
                            "Select Continuous Feature for Target Violin Plot",
                            options=engine.numeric_cols,
                            index=0
                        )
                        # Plotly Violin by Target
                        target_col_to_plot = engine.target_col if engine.target_col and engine.target_col in engine.clean_df.columns else selected_target
                        fig_violin_target = px.violin(
                            engine.clean_df,
                            y=sel_violin_feat,
                            x=target_col_to_plot,
                            color=target_col_to_plot,
                            box=True,
                            points="outliers",
                            color_discrete_sequence=["#10B981", "#EF4444", "#38BDF8", "#F59E0B"]
                        )
                        fig_violin_target.update_layout(
                            title=f"Feature Distribution: {sel_violin_feat} by {target_col_to_plot}",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#94A3B8"),
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_violin_target, use_container_width=True)

                    with v_col2:
                        sel_violin_cluster_feat = st.selectbox(
                            "Select Feature for Cluster Violin Plot",
                            options=engine.numeric_cols,
                            index=min(1, len(engine.numeric_cols) - 1)
                        )
                        # Plotly Violin by Cluster
                        fig_violin_cluster = px.violin(
                            engine.scored_df,
                            y=sel_violin_cluster_feat,
                            x="Assigned_Cluster",
                            color="Assigned_Cluster",
                            box=True,
                            points="outliers",
                            color_discrete_sequence=["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#38BDF8"]
                        )
                        fig_violin_cluster.update_layout(
                            title=f"Feature Distribution: {sel_violin_cluster_feat} across K-Means Clusters",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#94A3B8"),
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_violin_cluster, use_container_width=True)

                    # Correlation Heatmap for continuous features
                    if len(engine.numeric_cols) >= 2:
                        st.markdown("#### 🌡️ Continuous Feature Correlation Heatmap")
                        corr_sub = engine.clean_df[engine.numeric_cols].corr()
                        fig_corr = px.imshow(
                            corr_sub,
                            text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            aspect="auto"
                        )
                        fig_corr.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#94A3B8"),
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=380
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                else:
                    st.info("No continuous numeric columns available for violin plot visualization.")

            # -----------------------------------------------------------------
            # DYNAMIC TAB 2: MODEL BENCHMARK LEADERBOARD (Precision, Recall, F1)
            # -----------------------------------------------------------------
            with d_tab_leaderboard:
                st.markdown("### 📈 Supervised Model Benchmark & Evaluation Leaderboard")
                st.write(
                    "Direct head-to-head comparison of Accuracy, Precision, Recall, F1 Score, and ROC-AUC evaluated on the holdout test set."
                )

                eval_df = pd.DataFrame(eval_results)
                # Formatted Leaderboard
                st.dataframe(
                    eval_df.style.highlight_max(
                        axis=0,
                        subset=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
                        color="#6366F155"
                    ).format({
                        "Accuracy": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1_Score": "{:.4f}",
                        "ROC_AUC": lambda v: f"{v:.4f}" if v is not None and not pd.isna(v) else "N/A"
                    }),
                    use_container_width=True
                )

                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

                # Grouped Bar Comparison Chart
                melted_df = eval_df.melt(
                    id_vars=["Model"],
                    value_vars=["Accuracy", "Precision", "Recall", "F1_Score"],
                    var_name="Metric",
                    value_name="Score"
                )
                fig_bar = px.bar(
                    melted_df,
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    color_discrete_map={
                        "Accuracy": "#38BDF8",
                        "Precision": "#10B981",
                        "Recall": "#F59E0B",
                        "F1_Score": "#6366F1"
                    }
                )
                fig_bar.update_layout(
                    title="Metric Comparison (Accuracy vs Precision vs Recall vs F1 Score)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94A3B8"),
                    yaxis=dict(range=[0, 1.05]),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # ROC Curves and Confusion Matrix Row
                roc_col, cm_col = st.columns([1, 1])
                with roc_col:
                    st.markdown("#### 🎯 Interactive ROC Curves")
                    fig_roc = go.Figure()
                    for m_name, r_data in engine.roc_data.items():
                        fig_roc.add_trace(go.Scatter(
                            x=r_data["fpr"], y=r_data["tpr"],
                            mode="lines",
                            name=f"{m_name} (AUC = {r_data['auc']:.2f})"
                        ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode="lines", line=dict(dash="dash", color="#64748B"),
                        name="Random Baseline (AUC = 0.50)"
                    ))
                    fig_roc.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94A3B8"),
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        legend=dict(font=dict(size=9)),
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                with cm_col:
                    st.markdown("#### 🔲 Confusion Matrix Inspector")
                    sel_cm_model = st.selectbox(
                        "Model to Inspect",
                        options=list(engine.models.keys()),
                        index=0
                    )
                    cm_matrix = engine.cm_data.get(sel_cm_model, [[0, 0], [0, 0]])
                    fig_cm = px.imshow(
                        cm_matrix,
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=[f"Negative ({engine.inverse_label_mapping.get(0, '0')})", f"Positive ({engine.inverse_label_mapping.get(1, '1')})"],
                        y=[f"Negative ({engine.inverse_label_mapping.get(0, '0')})", f"Positive ({engine.inverse_label_mapping.get(1, '1')})"],
                        color_continuous_scale="Blues"
                    )
                    fig_cm.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94A3B8"),
                        coloraxis_showscale=False,
                        height=320,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                # Feature Importance
                if engine.feature_importances:
                    st.markdown("#### 🌳 Top Feature Importances (Random Forest)")
                    top_feats = engine.feature_importances[:14]
                    fi_df = pd.DataFrame(top_feats).sort_values(by="importance", ascending=True)
                    fig_fi = px.bar(
                        fi_df, x="importance", y="feature", orientation="h",
                        color="importance", color_continuous_scale="Viridis",
                        labels={"importance": "Relative Importance", "feature": "Feature"}
                    )
                    fig_fi.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94A3B8"),
                        coloraxis_showscale=False,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

            # -----------------------------------------------------------------
            # DYNAMIC TAB 3: CLUSTERING & 2D/3D PCA PROJECTIONS
            # -----------------------------------------------------------------
            with d_tab_cluster:
                st.markdown("### 👥 Unsupervised K-Means Behavioral Clustering & Dimensionality Reduction")
                st.write(
                    "Customer segmentation assigns accounts into distinct behavioral clusters based on similarity in normalized feature space."
                )

                # Cluster Cards
                cluster_cols = st.columns(len(engine.cluster_profiles))
                for idx, (c_id, c_info) in enumerate(engine.cluster_profiles.items()):
                    with cluster_cols[idx]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">{c_info['label']}</div>
                                <div class="metric-value">{c_info['size']:,}</div>
                                <div class="metric-sub">{c_info['pct']}% of dataset • {c_info['target_positive_rate']}% event rate</div>
                            </div>
                        """, unsafe_allow_html=True)

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                p3d_col, p2d_col = st.columns([1, 1])

                with p3d_col:
                    st.markdown("#### 🌐 Interactive 3D PCA Projection")
                    if engine.scored_df is not None and "PCA_Dim3" in engine.scored_df.columns:
                        fig_pca3 = px.scatter_3d(
                            engine.scored_df.head(1500),
                            x="PCA_Dim1", y="PCA_Dim2", z="PCA_Dim3",
                            color="Assigned_Cluster",
                            hover_data=[selected_target] if selected_target in engine.scored_df.columns else None,
                            color_discrete_sequence=["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#38BDF8"]
                        )
                        fig_pca3.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#94A3B8"),
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=480
                        )
                        st.plotly_chart(fig_pca3, use_container_width=True)
                    else:
                        st.info("3D PCA requires at least 3 feature dimensions.")

                with p2d_col:
                    st.markdown("#### 🗺️ 2D PCA Cluster Distribution")
                    if engine.scored_df is not None and "PCA_Dim2" in engine.scored_df.columns:
                        fig_pca2 = px.scatter(
                            engine.scored_df.head(1500),
                            x="PCA_Dim1", y="PCA_Dim2",
                            color="Assigned_Cluster",
                            hover_data=[selected_target] if selected_target in engine.scored_df.columns else None,
                            color_discrete_sequence=["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#38BDF8"]
                        )
                        fig_pca2.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#94A3B8"),
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=480
                        )
                        st.plotly_chart(fig_pca2, use_container_width=True)

                # Cluster Centroid Comparison Table
                st.markdown("#### 📋 Cluster Behavioral Comparison Profile")
                cluster_table_rows = []
                for c_id, c_info in engine.cluster_profiles.items():
                    row = {
                        "Cluster": c_info["label"],
                        "Size": c_info["size"],
                        "Pct": f"{c_info['pct']}%",
                        f"{engine.inverse_label_mapping.get(1, 'Target')} Rate": f"{c_info['target_positive_rate']}%"
                    }
                    row.update(c_info["numeric_means"])
                    cluster_table_rows.append(row)
                st.dataframe(pd.DataFrame(cluster_table_rows), use_container_width=True)

            # -----------------------------------------------------------------
            # DYNAMIC TAB 4: ANOMALY ACCOUNT DETECTION (Isolation Forest)
            # -----------------------------------------------------------------
            with d_tab_anomaly:
                st.markdown("### 🕵️ Anomaly Account Detection (Isolation Forest)")
                st.write(
                    "Isolation Forest isolates outliers by randomly partitioning feature trees. Accounts requiring few splits are flagged as anomalies."
                )

                scored = engine.scored_df
                if scored is not None:
                    total_pts = len(scored)
                    anom_mask = (scored["Is_Anomaly"] == "Anomaly (Outlier)")
                    anom_pts = int(anom_mask.sum())
                    anom_pct = (anom_pts / total_pts) * 100 if total_pts else 0

                    a_col1, a_col2 = st.columns([1, 2])
                    with a_col1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Flagged Anomalies</div>
                                <div class="metric-value" style="color: #EF4444;">{anom_pts:,}</div>
                                <div class="metric-sub">{anom_pct:.1f}% contamination rate</div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Normal Portfolio Baseline</div>
                                <div class="metric-value" style="color: #10B981;">{total_pts - anom_pts:,}</div>
                                <div class="metric-sub">{100 - anom_pct:.1f}% regular account behavior</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with a_col2:
                        fig_anom = px.scatter(
                            scored.head(1500),
                            x="PCA_Dim1", y="PCA_Dim2",
                            color="Is_Anomaly",
                            color_discrete_map={"Normal": "#38BDF8", "Anomaly (Outlier)": "#EF4444"},
                            title="PCA Projection: Normal Accounts vs Anomalous Outliers"
                        )
                        fig_anom.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#94A3B8"),
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=360
                        )
                        st.plotly_chart(fig_anom, use_container_width=True)

                    st.markdown("#### 🚨 Flagged Anomaly Records")
                    st.dataframe(scored[anom_mask].head(100), use_container_width=True)

            # -----------------------------------------------------------------
            # DYNAMIC TAB 5: LIVE PREDICTOR & FULL SCORED DATASET EXPORT
            # -----------------------------------------------------------------
            with d_tab_predict:
                st.markdown("### 🎯 Single-Account Prediction Simulator & Portfolio Batch Export")
                st.write(
                    "Simulate predictions on interactive profile inputs or export the entire scored dataset with predicted labels, probabilities, and clusters."
                )

                # Real-Time Predictor Form
                st.markdown("#### ⚡ Real-Time Customer Profile Simulator")
                with st.form("dynamic_single_predict_form"):
                    sim_cols = st.columns(min(4, max(1, len(engine.numeric_cols) + len(engine.categorical_cols))))
                    sim_inputs = {}
                    
                    # Numeric controls
                    col_idx = 0
                    for ncol in engine.numeric_cols[:6]:
                        with sim_cols[col_idx % len(sim_cols)]:
                            min_val = float(engine.clean_df[ncol].min())
                            max_val = float(engine.clean_df[ncol].max())
                            mean_val = float(engine.clean_df[ncol].median())
                            if min_val >= max_val:
                                max_val = min_val + 1.0
                            mean_val = min(max_val, max(min_val, mean_val))
                            sim_inputs[ncol] = st.slider(f"{ncol}", min_value=min_val, max_value=max_val, value=mean_val)
                        col_idx += 1

                    # Categorical controls
                    for ccol in engine.categorical_cols[:4]:
                        with sim_cols[col_idx % len(sim_cols)]:
                            opts = list(engine.clean_df[ccol].unique())
                            sim_inputs[ccol] = st.selectbox(f"{ccol}", options=opts, index=0)
                        col_idx += 1

                    submit_sim = st.form_submit_button("⚡ Predict Individual Outcome", type="primary")

                if submit_sim:
                    # Construct single-row DataFrame
                    sim_row = pd.DataFrame([sim_inputs])
                    # Encode matching engine feature space with drop_first=False
                    if engine.categorical_cols:
                        sim_encoded = pd.get_dummies(sim_row, columns=[c for c in engine.categorical_cols if c in sim_row.columns], drop_first=False, dtype=int)
                    else:
                        sim_encoded = sim_row.copy()
                    
                    # Align columns
                    aligned = pd.DataFrame(0, index=[0], columns=engine.encoded_feature_names)
                    for col in engine.encoded_feature_names:
                        if col in sim_encoded.columns:
                            aligned[col] = sim_encoded[col]
                        elif col in sim_row.columns:
                            aligned[col] = pd.to_numeric(sim_row[col], errors='coerce').fillna(0)
                        else:
                            aligned[col] = 0

                    # Scale
                    aligned_scaled = engine.scaler.transform(aligned)
                    best_m = engine.models.get(top_model["Model"])
                    
                    if best_m is not None:
                        pred_code = best_m.predict(aligned_scaled)[0]
                        pred_label = engine.inverse_label_mapping.get(pred_code, str(pred_code))
                        pred_prob = best_m.predict_proba(aligned_scaled)[0, 1] if hasattr(best_m, "predict_proba") else 0.5
                        
                        risk_info = evaluate_risk_level(pred_prob)
                        
                        r_col1, r_col2 = st.columns([1, 2])
                        with r_col1:
                            st.markdown(f"""
                                <div class="metric-card" style="border-left: 4px solid {risk_info['badge_color']};">
                                    <div class="metric-title">Predicted Outcome</div>
                                    <div class="metric-value" style="color: {risk_info['badge_color']};">
                                        {pred_label} ({pred_prob:.1%})
                                    </div>
                                    <div class="metric-sub">{risk_info['tier']} • {risk_info['icon']}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with r_col2:
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=pred_prob * 100,
                                number={'suffix': "%"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': risk_info['badge_color']},
                                    'steps': [
                                        {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.2)"},
                                        {'range': [35, 65], 'color': "rgba(245, 158, 11, 0.2)"},
                                        {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                                    ]
                                }
                            ))
                            fig_gauge.update_layout(height=180, margin=dict(l=20, r=20, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"))
                            st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("---")

                # Full Scored Dataset Export
                st.markdown("#### 📥 Full Scored Dataset & CSV Download")
                st.write("All records scored with Predicted Target, Probability, Cluster assignments, and Anomaly flags:")
                st.dataframe(engine.scored_df.head(250), use_container_width=True)

                csv_buffer = io.StringIO()
                engine.scored_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Complete Scored Dataset (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="dynamic_scored_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )


# =============================================================================
# MODE 2: PRE-TRAINED IBM TELCO CHURN BENCHMARK (With Added Violin Plots)
# =============================================================================
else:
    artifacts = load_pretrained_artifacts()
    meta = artifacts["metadata"]
    dataset_df = load_telco_dataset()

    # Header Section
    col_logo, col_head = st.columns([1, 8])
    with col_logo:
        st.markdown("<div style='font-size: 3.2rem; text-align: center;'>🔮</div>", unsafe_allow_html=True)
    with col_head:
        st.markdown("""
            <h1 style='margin-bottom: 2px; background: linear-gradient(90deg, #6366F1, #A855F7, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Customer Segmentation & Churn Intelligence Engine
            </h1>
            <p style='color: #94A3B8; font-size: 1.05rem; margin-top: 0;'>
                Production-grade pre-trained machine learning platform for customer churn risk prediction, behavioral K-Means clustering, and anomaly detection.
            </p>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Main Navigation Tabs (includes Violin Plots)
    tab_overview, tab_violin, tab_predict, tab_clustering, tab_anomaly, tab_benchmark = st.tabs([
        "📊 Executive KPI Dashboard",
        "🎻 Violin & Distribution Analysis",
        "🎯 Churn Prediction Engine",
        "👥 Customer Segmentation (K-Means)",
        "🕵️ Anomaly Account Detection",
        "📈 Model Benchmark Leaderboard"
    ])

    # -------------------------------------------------------------------------
    # TELCO TAB 1: EXECUTIVE KPI DASHBOARD
    # -------------------------------------------------------------------------
    with tab_overview:
        total_customers = meta.get("total_samples", 7035)
        churn_count = meta.get("churn_count", 1869)
        churn_rate = (churn_count / total_customers) * 100 if total_customers else 26.6
        non_churn = total_customers - churn_count

        avg_monthly = dataset_df['MonthlyCharges'].mean() if dataset_df is not None else 64.76
        avg_tenure = dataset_df['tenure'].mean() if dataset_df is not None else 32.4
        at_risk_mrr = churn_count * avg_monthly

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Customers</div>
                    <div class="metric-value">{total_customers:,}</div>
                    <div class="metric-sub">Active tracked accounts</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Churn Rate</div>
                    <div class="metric-value" style="color: #EF4444;">{churn_rate:.1f}%</div>
                    <div class="metric-sub">{churn_count:,} churned customers</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Retained Ratio</div>
                    <div class="metric-value" style="color: #10B981;">{100 - churn_rate:.1f}%</div>
                    <div class="metric-sub">{non_churn:,} active loyal users</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">At-Risk MRR</div>
                    <div class="metric-value" style="color: #F59E0B;">${at_risk_mrr:,.0f}</div>
                    <div class="metric-sub">Monthly revenue exposed</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi5:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Tenure</div>
                    <div class="metric-value">{avg_tenure:.1f} mo</div>
                    <div class="metric-sub">Avg spend: ${avg_monthly:.2f}/mo</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        if dataset_df is not None:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("### 🏷️ Churn Rate by Contract Duration")
                contract_churn = dataset_df.groupby('Contract')['Churn'].apply(
                    lambda s: (s.str.lower() == 'yes').mean() * 100
                ).reset_index(name='ChurnRate')
                fig_contract = px.bar(
                    contract_churn, x='Contract', y='ChurnRate',
                    text=contract_churn['ChurnRate'].apply(lambda v: f"{v:.1f}%"),
                    color='ChurnRate',
                    color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
                    labels={'ChurnRate': 'Churn Rate (%)'}
                )
                fig_contract.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94A3B8'),
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_contract, use_container_width=True)

            with c2:
                st.markdown("### 🌐 Churn by Internet Service Type")
                internet_churn = dataset_df.groupby('InternetService')['Churn'].apply(
                    lambda s: (s.str.lower() == 'yes').mean() * 100
                ).reset_index(name='ChurnRate')
                fig_internet = px.bar(
                    internet_churn, x='InternetService', y='ChurnRate',
                    text=internet_churn['ChurnRate'].apply(lambda v: f"{v:.1f}%"),
                    color='ChurnRate',
                    color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
                    labels={'ChurnRate': 'Churn Rate (%)'}
                )
                fig_internet.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94A3B8'),
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_internet, use_container_width=True)

    # -------------------------------------------------------------------------
    # TELCO TAB 2: VIOLIN & DISTRIBUTION ANALYSIS (Requested Feature)
    # -------------------------------------------------------------------------
    with tab_violin:
        st.markdown("### 🎻 Violin Distribution Analytics (IBM Telco Customer Portfolio)")
        st.write(
            "Violin plots visualize feature probability density curves together with embedded box plots to reveal distribution skewness between churned and retained subscribers."
        )

        if dataset_df is not None:
            v_col1, v_col2 = st.columns([1, 1])
            with v_col1:
                sel_telco_v = st.selectbox(
                    "Feature for Churn Violin Plot",
                    options=["tenure", "MonthlyCharges", "TotalCharges"],
                    index=0
                )
                fig_v_telco = px.violin(
                    dataset_df,
                    y=sel_telco_v,
                    x="Churn",
                    color="Churn",
                    box=True,
                    points="outliers",
                    color_discrete_map={"No": "#10B981", "Yes": "#EF4444"}
                )
                fig_v_telco.update_layout(
                    title=f"Distribution of {sel_telco_v} by Churn Status",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94A3B8"),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_v_telco, use_container_width=True)

            with v_col2:
                sel_telco_v2 = st.selectbox(
                    "Feature for Contract Violin Plot",
                    options=["MonthlyCharges", "tenure", "TotalCharges"],
                    index=0
                )
                fig_v_contract = px.violin(
                    dataset_df,
                    y=sel_telco_v2,
                    x="Contract",
                    color="Contract",
                    box=True,
                    points="outliers",
                    color_discrete_sequence=["#6366F1", "#EC4899", "#38BDF8"]
                )
                fig_v_contract.update_layout(
                    title=f"Distribution of {sel_telco_v2} by Contract Duration",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94A3B8"),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_v_contract, use_container_width=True)

    # -------------------------------------------------------------------------
    # TELCO TAB 3: CHURN PREDICTION ENGINE
    # -------------------------------------------------------------------------
    with tab_predict:
        st.markdown("### 🎯 Real-Time Individual Customer Risk Scoring & Prescriptive Retention")
        st.write("Configure the customer profile below or pick an instant preset to run inference and view customized retention tactics.")

        p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 2])
        with p_col1:
            load_high_risk = st.button("🚨 Load High-Risk Profile", use_container_width=True)
        with p_col2:
            load_loyal = st.button("🛡️ Load Loyal Customer", use_container_width=True)
        with p_col3:
            load_avg = st.button("⚖️ Load Typical Customer", use_container_width=True)
        with p_col4:
            selected_model_name = st.selectbox(
                "Prediction Classifier",
                options=list(artifacts["models"].keys()),
                index=0
            )

        if "demo_profile" not in st.session_state or load_avg:
            st.session_state["demo_profile"] = {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
                "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 45.0, "TotalCharges": 1080.0
            }
        elif load_high_risk:
            st.session_state["demo_profile"] = {
                "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 98.5, "TotalCharges": 197.0
            }
        elif load_loyal:
            st.session_state["demo_profile"] = {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
                "tenure": 68, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
                "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
                "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Two year", "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0, "TotalCharges": 4420.0
            }

        prof = st.session_state["demo_profile"]

        with st.form("telco_inference_form"):
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                in_tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=int(prof["tenure"]))
                in_monthly = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=float(prof["MonthlyCharges"]))
            with r2:
                in_contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(prof["Contract"]))
                in_payment = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(prof["PaymentMethod"]))
            with r3:
                in_internet = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(prof["InternetService"]))
                in_techsupport = st.selectbox("Tech Support", options=["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(prof["TechSupport"]))
            with r4:
                in_security = st.selectbox("Online Security", options=["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(prof["OnlineSecurity"]))
                in_paperless = st.selectbox("Paperless Billing", options=["Yes", "No"], index=["Yes", "No"].index(prof["PaperlessBilling"]))

            submit_pred = st.form_submit_button("⚡ Run Multi-Model Churn Scoring", type="primary", use_container_width=True)

        # Inference computation
        input_data = {
            "gender": prof.get("gender", "Female"),
            "SeniorCitizen": prof.get("SeniorCitizen", 0),
            "Partner": prof.get("Partner", "Yes"),
            "Dependents": prof.get("Dependents", "No"),
            "tenure": in_tenure,
            "PhoneService": prof.get("PhoneService", "Yes"),
            "MultipleLines": prof.get("MultipleLines", "No"),
            "InternetService": in_internet,
            "OnlineSecurity": in_security,
            "OnlineBackup": prof.get("OnlineBackup", "No"),
            "DeviceProtection": prof.get("DeviceProtection", "No"),
            "TechSupport": in_techsupport,
            "StreamingTV": prof.get("StreamingTV", "No"),
            "StreamingMovies": prof.get("StreamingMovies", "No"),
            "Contract": in_contract,
            "PaperlessBilling": in_paperless,
            "PaymentMethod": in_payment,
            "MonthlyCharges": in_monthly,
            "TotalCharges": in_monthly * in_tenure
        }

        feat_df = DataProcessor.prepare_inference_features(input_data, outlier_bounds=meta.get("outlier_bounds"))
        scaled_feat = artifacts["scaler"].transform(feat_df)
        model_obj = artifacts["models"][selected_model_name]
        prob = model_obj.predict_proba(scaled_feat)[0, 1] if hasattr(model_obj, "predict_proba") else 0.5
        risk_info = evaluate_risk_level(prob)

        res1, res2 = st.columns([1, 1])
        with res1:
            st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {risk_info['badge_color']};">
                    <div class="metric-title">Predicted Risk Assessment ({selected_model_name})</div>
                    <div class="metric-value" style="color: {risk_info['badge_color']};">
                        {risk_info['tier']} • {prob * 100:.1f}% Churn Risk
                    </div>
                    <div class="metric-sub">{risk_info['icon']} {risk_info['action_urgency']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_info['badge_color']},
                    'steps': [
                        {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [35, 65], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ]
                }
            ))
            fig_g.update_layout(height=180, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"))
            st.plotly_chart(fig_g, use_container_width=True)

        with res2:
            st.markdown("#### 📋 Prescriptive Retention Playbook")
            tactics = generate_retention_strategy(input_data, prob)
            for t in tactics:
                st.markdown(f"""
                    <div class="strategy-card">
                        <div class="strategy-title">{t['category']}: {t['title']}</div>
                        <div class="strategy-desc">{t['detail']}</div>
                    </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TELCO TAB 4: CUSTOMER SEGMENTATION (K-MEANS)
    # -------------------------------------------------------------------------
    with tab_clustering:
        st.markdown("### 👥 Customer Segmentation (K-Means Clustering)")
        clusters = meta.get("cluster_profiles", {})
        
        c_cols = st.columns(3)
        for idx in range(3):
            c_data = clusters.get(str(idx), clusters.get(idx, {}))
            with c_cols[idx]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Cluster {idx}: {c_data.get('label', 'Segment')}</div>
                        <div class="metric-value">{c_data.get('count', 0):,}</div>
                        <div class="metric-sub">{c_data.get('pct', 0)}% of portfolio • {c_data.get('churn_rate', 0)}% churn rate</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        viz = meta.get("viz_data", {})
        if viz and "pc1" in viz:
            viz_df = pd.DataFrame(viz)
            viz_df["Cluster_Name"] = viz_df["cluster"].map(lambda c: f"Cluster {c}")
            
            p3, p2 = st.columns([1, 1])
            with p3:
                st.markdown("#### 🌐 3D PCA Projection")
                fig_3d = px.scatter_3d(
                    viz_df, x="pc1", y="pc2", z="pc3",
                    color="Cluster_Name",
                    hover_data=["tenure", "monthly_charges"],
                    color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444"]
                )
                fig_3d.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"), height=460)
                st.plotly_chart(fig_3d, use_container_width=True)

            with p2:
                st.markdown("#### 🗺️ 2D PCA Cluster Scatter")
                fig_2d = px.scatter(
                    viz_df, x="pc1", y="pc2",
                    color="Cluster_Name",
                    hover_data=["tenure", "monthly_charges"],
                    color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444"]
                )
                fig_2d.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"), height=460)
                st.plotly_chart(fig_2d, use_container_width=True)

    # -------------------------------------------------------------------------
    # TELCO TAB 5: ANOMALY ACCOUNT DETECTION (Isolation Forest)
    # -------------------------------------------------------------------------
    with tab_anomaly:
        st.markdown("### 🕵️ Anomaly Account Detection (Isolation Forest)")
        anom_count = meta.get("anomaly_count", 211)
        total_s = meta.get("total_samples", 7035)

        st.markdown(f"""
            <div class="metric-card" style="max-width: 400px;">
                <div class="metric-title">Flagged Outlier Accounts</div>
                <div class="metric-value" style="color: #EF4444;">{anom_count:,}</div>
                <div class="metric-sub">{anom_count / total_s * 100:.1f}% contamination threshold</div>
            </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TELCO TAB 6: MODEL BENCHMARK LEADERBOARD
    # -------------------------------------------------------------------------
    with tab_benchmark:
        st.markdown("### 📈 Supervised Machine Learning Benchmark Leaderboard")
        eval_df = pd.DataFrame(meta.get("evaluation_results", []))
        st.dataframe(
            eval_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'], color="#6366F144"),
            use_container_width=True
        )

        b1, b2 = st.columns([1, 1])
        with b1:
            st.markdown("#### 🎯 ROC Curve Comparison")
            fig_roc = go.Figure()
            for m_name, r_data in meta.get("roc_data", {}).items():
                fig_roc.add_trace(go.Scatter(x=r_data["fpr"], y=r_data["tpr"], mode='lines', name=f"{m_name} (AUC={r_data['auc']:.2f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#64748B'), name="Random Baseline"))
            fig_roc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'))
            st.plotly_chart(fig_roc, use_container_width=True)

        with b2:
            st.markdown("#### 🌳 Top Feature Importances (Random Forest)")
            feat_df = pd.DataFrame(meta.get("feature_importances", [])[:12]).sort_values(by='importance', ascending=True)
            fig_imp = px.bar(feat_df, x='importance', y='feature', orientation='h', color='importance', color_continuous_scale='Viridis')
            fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'), coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)


# -----------------------------------------------------------------------------
# Global Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 12px 0;'>
    Customer Segmentation & Churn Prediction Intelligence Platform • Built with Python, Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
