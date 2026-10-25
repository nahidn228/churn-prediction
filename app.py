import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_processor import DataProcessor
from src.utils import evaluate_risk_level, generate_retention_strategy

# -----------------------------------------------------------------------------
# Streamlit App Configuration & Modern Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation & Churn Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
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
        padding: 20px 24px;
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
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 6px;
    }

    /* Glass Panels */
    .glass-panel {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
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
# Data & Artifact Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    meta_path = os.path.join(models_dir, "pipeline_metadata.json")

    # If artifacts do not exist, run pipeline
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
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "TelcoCustomerChurn.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        processor = DataProcessor(df)
        df = processor.rename_columns()
        df, _ = processor.clean_total_charges()
        df = processor.convert_data_types()
        return df
    return None


artifacts = load_artifacts()
meta = artifacts["metadata"]
dataset_df = load_dataset()

# -----------------------------------------------------------------------------
# Header Section
# -----------------------------------------------------------------------------
col_logo, col_head = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size: 3.2rem; text-align: center;'>🔮</div>", unsafe_allow_html=True)
with col_head:
    st.markdown("""
        <h1 style='margin-bottom: 2px; background: linear-gradient(90deg, #6366F1, #A855F7, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Customer Segmentation & Churn Intelligence Engine
        </h1>
        <p style='color: #94A3B8; font-size: 1.05rem; margin-top: 0;'>
            Production-grade machine learning platform for customer churn risk prediction, behavioral K-Means clustering, and anomaly detection.
        </p>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_predict, tab_clustering, tab_anomaly, tab_benchmark = st.tabs([
    "📊 Executive KPI Dashboard",
    "🎯 Churn Prediction Engine",
    "👥 Customer Segmentation (K-Means)",
    "🕵️ Anomaly Account Detection",
    "📈 Model Benchmark Leaderboard"
])


# =============================================================================
# TAB 1: EXECUTIVE KPI DASHBOARD
# =============================================================================
with tab_overview:
    total_customers = meta.get("total_samples", 7035)
    churn_count = meta.get("churn_count", 1869)
    churn_rate = (churn_count / total_customers) * 100 if total_customers else 26.6
    non_churn = total_customers - churn_count

    avg_monthly = dataset_df['MonthlyCharges'].mean() if dataset_df is not None else 64.76
    avg_tenure = dataset_df['tenure'].mean() if dataset_df is not None else 32.4
    at_risk_mrr = churn_count * avg_monthly

    # Top KPI Metrics Cards
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

    # Visualizations Row 1
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

        # Visualizations Row 2
        c3, c4 = st.columns([1, 1])
        with c3:
            st.markdown("### 💳 Churn Rate by Payment Method")
            pay_churn = dataset_df.groupby('PaymentMethod')['Churn'].apply(
                lambda s: (s.str.lower() == 'yes').mean() * 100
            ).reset_index(name='ChurnRate').sort_values(by='ChurnRate', ascending=True)
            fig_pay = px.bar(
                pay_churn, y='PaymentMethod', x='ChurnRate', orientation='h',
                text=pay_churn['ChurnRate'].apply(lambda v: f"{v:.1f}%"),
                color='ChurnRate',
                color_continuous_scale=['#10B981', '#EF4444'],
                labels={'ChurnRate': 'Churn Rate (%)'}
            )
            fig_pay.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8'),
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_pay, use_container_width=True)

        with c4:
            st.markdown("### ⏳ Tenure Distribution by Churn Status")
            fig_tenure = px.histogram(
                dataset_df, x='tenure', color='Churn', barmode='overlay',
                color_discrete_map={'Yes': '#EF4444', 'No': '#3B82F6'},
                nbins=30, labels={'tenure': 'Tenure (Months)'}
            )
            fig_tenure.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8'),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_tenure, use_container_width=True)

        # Interactive Dataset Explorer
        st.markdown("### 🔍 Customer Dataset Explorer & Filtering")
        with st.expander("Filter & Explore Raw Customer Records", expanded=False):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                sel_contract = st.multiselect("Contract Type", options=dataset_df['Contract'].unique().tolist())
            with f_col2:
                sel_internet = st.multiselect("Internet Service", options=dataset_df['InternetService'].unique().tolist())
            with f_col3:
                sel_churn = st.multiselect("Churn Status", options=dataset_df['Churn'].unique().tolist())

            filtered_df = dataset_df.copy()
            if sel_contract:
                filtered_df = filtered_df[filtered_df['Contract'].isin(sel_contract)]
            if sel_internet:
                filtered_df = filtered_df[filtered_df['InternetService'].isin(sel_internet)]
            if sel_churn:
                filtered_df = filtered_df[filtered_df['Churn'].isin(sel_churn)]

            st.dataframe(filtered_df.head(250), use_container_width=True)
            st.caption(f"Showing {len(filtered_df):,} matching customer accounts.")


# =============================================================================
# TAB 2: CHURN PREDICTION ENGINE
# =============================================================================
with tab_predict:
    st.markdown("### 🎯 Real-Time Individual Customer Risk Scoring & Prescriptive Retention")
    st.write("Configure the customer profile below or pick an instant preset to run multi-model inference and view customized retention tactics.")

    # Preset profiles
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
            index=0,
            help="Logistic Regression delivers top ROC-AUC (84.4%) with well-calibrated probabilities."
        )

    # Default values based on preset
    if load_high_risk:
        def_tenure = 2
        def_monthly = 89.50
        def_total = 179.00
        def_contract = "Month-to-month"
        def_payment = "Electronic check"
        def_internet = "Fiber optic"
        def_tech = "No"
        def_security = "No"
        def_senior = "No"
        def_partner = "No"
        def_dependents = "No"
        def_paperless = "Yes"
    elif load_loyal:
        def_tenure = 60
        def_monthly = 65.00
        def_total = 3900.00
        def_contract = "Two year"
        def_payment = "Credit card (automatic)"
        def_internet = "DSL"
        def_tech = "Yes"
        def_security = "Yes"
        def_senior = "No"
        def_partner = "Yes"
        def_dependents = "Yes"
        def_paperless = "No"
    else:
        def_tenure = 18
        def_monthly = 60.00
        def_total = 1080.00
        def_contract = "Month-to-month"
        def_payment = "Mailed check"
        def_internet = "DSL"
        def_tech = "No"
        def_security = "No"
        def_senior = "No"
        def_partner = "No"
        def_dependents = "No"
        def_paperless = "Yes"

    # Input Form
    with st.form("churn_prediction_form"):
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            st.markdown("##### 👤 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=1 if def_senior == "Yes" else 0)
            partner = st.selectbox("Partner", ["No", "Yes"], index=1 if def_partner == "Yes" else 0)
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=1 if def_dependents == "Yes" else 0)

            st.markdown("##### 📞 Telephony")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        with col_f2:
            st.markdown("##### 🌐 Internet & Services")
            internet_service = st.selectbox(
                "Internet Service", ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(def_internet)
            )
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=1 if def_security == "Yes" else 0)
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=1 if def_tech == "Yes" else 0)
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col_f3:
            st.markdown("##### 💳 Contract & Billing")
            contract = st.selectbox(
                "Contract Plan", ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(def_contract)
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if def_paperless == "Yes" else 1)
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(def_payment)
            )
            tenure = st.slider("Tenure with Company (Months)", min_value=1, max_value=72, value=def_tenure)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=float(def_monthly), step=1.0)
            # Auto-calculate default total charges estimate
            calculated_total = max(monthly_charges, monthly_charges * tenure)
            total_charges = st.number_input("Total Charges to Date ($)", min_value=10.0, max_value=12000.0, value=float(def_total if def_total else calculated_total), step=10.0)

        submit_btn = st.form_submit_button("⚡ Predict Customer Churn Risk", use_container_width=True)

    # Inference Execution
    customer_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Prepare features
    inference_features = DataProcessor.prepare_inference_features(
        customer_input,
        expected_columns=meta["feature_columns"],
        outlier_bounds=meta["outlier_bounds"]
    )
    scaled_features = artifacts["scaler"].transform(inference_features)

    # Predict with chosen model
    chosen_model = artifacts["models"][selected_model_name]
    predicted_class = int(chosen_model.predict(scaled_features)[0])
    if hasattr(chosen_model, "predict_proba"):
        churn_prob = float(chosen_model.predict_proba(scaled_features)[0][1])
    else:
        churn_prob = 1.0 if predicted_class == 1 else 0.0

    # Risk tier evaluation
    risk_info = evaluate_risk_level(churn_prob)

    # Display Prediction Results
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown("#### 📊 Risk Assessment Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(churn_prob * 100, 1),
            number={'suffix': "%", 'font': {'size': 44, 'family': 'Outfit'}},
            title={'text': f"<b>{risk_info['tier']}</b><br><span style='font-size:0.8em;color:#94A3B8'>{risk_info['action_urgency']}</span>", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': risk_info['badge_color'], 'thickness': 0.28},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [35, 65], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': round(churn_prob * 100, 1)
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC'),
            height=320,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown("#### 💡 Prescriptive Retention Action Plan")
        strategies = generate_retention_strategy(customer_input, churn_prob)
        for strat in strategies:
            st.markdown(f"""
                <div class="strategy-card">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #818CF8; font-weight: 700;">{strat['category']}</div>
                    <div class="strategy-title">{strat['title']}</div>
                    <div class="strategy-desc">{strat['detail']}</div>
                </div>
            """, unsafe_allow_html=True)

    # Batch Prediction Sub-Section
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📁 Batch CSV Scoring")
    uploaded_file = st.file_uploader("Upload customer CSV file to score entire portfolio at once", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df):,} records.")
        if st.button("🚀 Run Batch Churn Scoring", use_container_width=True):
            batch_prep = DataProcessor.prepare_inference_features(
                batch_df,
                expected_columns=meta["feature_columns"],
                outlier_bounds=meta["outlier_bounds"]
            )
            batch_scaled = artifacts["scaler"].transform(batch_prep)
            batch_preds = chosen_model.predict(batch_scaled)
            batch_probs = (
                chosen_model.predict_proba(batch_scaled)[:, 1]
                if hasattr(chosen_model, "predict_proba")
                else batch_preds.astype(float)
            )
            scored_df = batch_df.copy()
            scored_df['Churn_Probability'] = np.round(batch_probs * 100, 2)
            scored_df['Predicted_Churn'] = np.where(batch_preds == 1, "Yes", "No")
            scored_df['Risk_Tier'] = [evaluate_risk_level(p)['tier'] for p in batch_probs]

            st.dataframe(scored_df.head(100), use_container_width=True)
            csv_data = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Scored Batch Predictions CSV",
                data=csv_data,
                file_name="telco_scored_churn_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )


# =============================================================================
# TAB 3: CUSTOMER SEGMENTATION (K-MEANS)
# =============================================================================
with tab_clustering:
    st.markdown("### 👥 Unsupervised Customer Segmentation (K-Means Clustering)")
    st.write(
        "K-Means clustering categorizes the customer base into 3 behavioral segments based on tenure, service depth, "
        "and payment behavior. Dimensionality reduction via Principal Component Analysis (PCA) maps the 30-dimensional space into 2D and 3D."
    )

    # Segment Cards Row
    cluster_profiles = meta.get("cluster_profiles", {})
    cols_clusters = st.columns(3)
    for c_id, profile in cluster_profiles.items():
        with cols_clusters[int(c_id)]:
            risk_color = "#10B981" if profile["risk"] == "Low" else ("#F59E0B" if profile["risk"] == "Medium" else "#EF4444")
            st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {risk_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #F8FAFC; font-size: 1.1rem;">Cluster {c_id}: {profile['label']}</span>
                        <span class="risk-badge" style="background: {risk_color}22; color: {risk_color};">{profile['risk']} Risk</span>
                    </div>
                    <div style="margin: 12px 0; color: #94A3B8; font-size: 0.85rem;">{profile['desc']}</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.8rem; color: #CBD5E1;">
                        <div>👥 Population: <b>{profile['count']:,} ({profile['pct']}%)</b></div>
                        <div>📉 Churn Rate: <b>{profile['churn_rate']}%</b></div>
                        <div>⏳ Avg Tenure: <b>{profile['avg_tenure']} mos</b></div>
                        <div>💵 Avg Monthly: <b>${profile['avg_monthly_charges']}</b></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # PCA Visualizations
    viz_meta = meta.get("viz_data", {})
    if viz_meta:
        viz_df = pd.DataFrame(viz_meta)
        viz_df['Cluster_Label'] = viz_df['cluster'].map(
            lambda c: f"Cluster {c}: {cluster_profiles.get(str(c), {}).get('label', '')}"
        )

        c_v1, c_v2 = st.columns([1, 1])
        with c_v1:
            st.markdown("#### 🗺️ 2D PCA Cluster Distribution")
            fig_pca2d = px.scatter(
                viz_df, x='pc1', y='pc2', color='Cluster_Label',
                color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'],
                hover_data=['tenure', 'monthly_charges'],
                labels={'pc1': 'Principal Component 1', 'pc2': 'Principal Component 2'}
            )
            fig_pca2d.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=20, b=50)
            )
            st.plotly_chart(fig_pca2d, use_container_width=True)

        with c_v2:
            st.markdown("#### 🌐 3D PCA Interactive Cluster Projection")
            fig_pca3d = px.scatter_3d(
                viz_df, x='pc1', y='pc2', z='pc3', color='Cluster_Label',
                color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'],
                hover_data=['tenure', 'monthly_charges'],
                labels={'pc1': 'PC 1', 'pc2': 'PC 2', 'pc3': 'PC 3'}
            )
            fig_pca3d.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=10, r=10, t=10, b=10),
                scene=dict(
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)")
                )
            )
            st.plotly_chart(fig_pca3d, use_container_width=True)


# =============================================================================
# TAB 4: ANOMALY ACCOUNT DETECTION (ISOLATION FOREST)
# =============================================================================
with tab_anomaly:
    st.markdown("### 🕵️ Unsupervised Anomaly Detection (Isolation Forest)")
    st.write(
        "Using Isolation Forest with 3% contamination threshold to detect unusual customer accounts. "
        "These accounts represent atypical combinations of billing amounts, tenure, or unusual service configurations."
    )

    total_anomalies = meta.get("anomaly_count", 212)
    a_col1, a_col2 = st.columns([1, 2])

    with a_col1:
        st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #EC4899;">
                <div class="metric-title">Flagged Outlier Accounts</div>
                <div class="metric-value" style="color: #EC4899;">{total_anomalies:,}</div>
                <div class="metric-sub">Contamination rate: 3.0%</div>
                <div style="margin-top: 14px; font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
                    Accounts with anomalous isolation scores exhibit extreme feature combinations, such as ultra-high charges with minimal tenure, or divergent service bundles.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with a_col2:
        if viz_meta:
            viz_df = pd.DataFrame(viz_meta)
            viz_df['Anomaly_Status'] = viz_df['anomaly'].map({-1: "Flagged Anomaly", 1: "Normal Account"})
            fig_anomaly = px.scatter(
                viz_df, x='pc1', y='pc2', color='Anomaly_Status',
                color_discrete_map={"Flagged Anomaly": "#EF4444", "Normal Account": "#3B82F6"},
                hover_data=['tenure', 'monthly_charges'],
                labels={'pc1': 'Principal Component 1', 'pc2': 'Principal Component 2'}
            )
            fig_anomaly.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8'),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)


# =============================================================================
# TAB 5: MODEL BENCHMARK LEADERBOARD
# =============================================================================
with tab_benchmark:
    st.markdown("### 📈 Supervised Machine Learning Benchmark Leaderboard")
    st.write(
        "Direct comparison of 6 classifiers trained on 80/20 stratified split with continuous outlier capping and standard scaling."
    )

    eval_results = meta.get("evaluation_results", [])
    eval_df = pd.DataFrame(eval_results)

    # Leaderboard Table
    st.dataframe(
        eval_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'], color="#6366F144"),
        use_container_width=True
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    b_col1, b_col2 = st.columns([1, 1])

    with b_col1:
        st.markdown("#### 🎯 ROC Curve Comparison")
        roc_dict = meta.get("roc_data", {})
        fig_roc = go.Figure()
        for m_name, r_data in roc_dict.items():
            fig_roc.add_trace(go.Scatter(
                x=r_data["fpr"], y=r_data["tpr"],
                mode='lines',
                name=f"{m_name} (AUC = {r_data['auc']:.2f})"
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', line=dict(dash='dash', color='#64748B'),
            name="Random Baseline (AUC = 0.50)"
        ))
        fig_roc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8'),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(font=dict(size=10)),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with b_col2:
        st.markdown("#### 🌳 Top Feature Importances (Random Forest)")
        feat_imps = meta.get("feature_importances", [])[:12]
        feat_df = pd.DataFrame(feat_imps).sort_values(by='importance', ascending=True)
        fig_imp = px.bar(
            feat_df, x='importance', y='feature', orientation='h',
            color='importance', color_continuous_scale='Viridis',
            labels={'importance': 'Gini Importance', 'feature': 'Feature'}
        )
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8'),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Confusion Matrix Heatmap viewer
    st.markdown("#### 🔲 Confusion Matrix Inspector")
    cm_model_choice = st.selectbox("Inspect Confusion Matrix for Model", options=list(artifacts["models"].keys()), index=0)
    cm_matrix = meta.get("cm_data", {}).get(cm_model_choice, [[0, 0], [0, 0]])
    fig_cm = px.imshow(
        cm_matrix,
        text_auto=True,
        labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
        x=["Retained (0)", "Churned (1)"],
        y=["Retained (0)", "Churned (1)"],
        color_continuous_scale="Blues"
    )
    fig_cm.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94A3B8'),
        coloraxis_showscale=False,
        height=320,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 12px 0;'>
    E-Commerce Customer Segmentation & Churn Prediction Intelligence App • Built with Python, Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
