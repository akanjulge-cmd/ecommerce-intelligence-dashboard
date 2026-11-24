"""
E-Commerce Intelligence Dashboard
Global Sales Analytics and Machine Learning Prediction System
Introduction to Data Science (IDS) Academic Project
"""

import datetime
import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Local modules
from models.model_pipeline import predict_single_order
from utils.data_loader import compute_kpis, load_and_process_data
from utils.visualizations import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_monthly_sales_trend,
    plot_order_status_distribution,
    plot_quantity_by_category,
    plot_sales_by_category,
    plot_sales_by_country,
    plot_yearly_sales,
)

# ---------------------------------------------------------
# Page Configuration & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1E293B;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    /* Gradient Header */
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #7C3AED 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #E0E7FF;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Modern KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px -3px rgba(37, 99, 235, 0.1);
        border-color: #BFDBFE;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #1E3A8A;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #10B981;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Explanation Boxes */
    .chart-insight {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 18px;
        border-radius: 0 10px 10px 0;
        font-size: 0.9rem;
        color: #334155;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .chart-insight strong {
        color: #1E3A8A;
    }

    /* Academic Disclosure Box */
    .disclosure-box {
        background: #FEF3C7;
        border: 1px solid #F59E0B;
        border-left: 5px solid #D97706;
        padding: 16px 20px;
        border-radius: 10px;
        color: #78350F;
        margin: 18px 0;
    }
    .disclosure-box strong {
        color: #92400E;
    }

    /* Prediction Hero Card */
    .prediction-hero {
        background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 100%);
        border: 2px solid #818CF8;
        border-radius: 16px;
        padding: 26px;
        text-align: center;
        box-shadow: 0 8px 24px -4px rgba(99, 102, 241, 0.15);
        margin: 20px 0;
    }
    .prediction-title {
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #4F46E5;
    }
    .prediction-amount {
        font-size: 3rem;
        font-weight: 900;
        color: #1E1B4B;
        margin: 10px 0;
    }

    /* Workflow Step Badge */
    .workflow-pill {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        margin: 4px;
        border: 1px solid #C7D2FE;
    }

    /* Status Badges */
    .badge-completed { background: #D1FAE5; color: #065F46; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
    .badge-pending { background: #FEF3C7; color: #92400E; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
    .badge-cancelled { background: #FEE2E2; color: #991B1B; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# State & Resource Initialization
# ---------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None


@st.cache_resource
def load_ml_assets():
    """Loads the trained model and associated metadata."""
    model_path = os.path.join("models", "rf_model.joblib")
    meta_path = os.path.join("models", "model_metadata.json")
    comp_path = os.path.join("models", "test_predictions.json")

    # If assets are missing, build them automatically
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        from models.model_pipeline import build_and_train_model
        build_and_train_model()

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    test_preds = []
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            test_preds = json.load(f)

    return model, metadata, test_preds


# Load Data & Model
try:
    df_raw, df_clean, data_stats = load_and_process_data()
    kpis = compute_kpis(df_clean)
    rf_model, model_meta, test_comparison = load_ml_assets()
except Exception as e:
    st.error(f"Error initializing system assets: {e}")
    st.stop()


# ---------------------------------------------------------
# Sidebar Navigation & System Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shopping-bag-full.png",
        width=64
    )
    st.markdown("### **Navigation**")

    page = st.radio(
        "Select Page:",
        [
            "🏠 Home Dashboard",
            "🔍 Data Explorer",
            "📊 Sales Analysis",
            "🤖 Machine Learning Model",
            "🔮 New Sales Prediction",
            "📜 Prediction History",
            "ℹ️ About Project"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### **Project Quick Facts**")
    st.markdown(f"• **Dataset Size:** {data_stats['clean_shape'][0]:,} rows")
    st.markdown(f"• **Model:** Random Forest (200 Trees)")
    st.markdown(f"• **Model $R^2$ Score:** {model_meta['r2_score']:.4f}")
    st.markdown(f"• **Model RMSE:** ${model_meta['rmse']:.2f}")

    st.markdown("---")
    st.caption("Introduction to Data Science (IDS) Project\nCollege Faculty Evaluation Version")


# ---------------------------------------------------------
# Global Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>E-Commerce Intelligence Dashboard</h1>
        <p>Global Sales Analytics and Machine Learning Prediction System • Random Forest Regressor</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PAGE 1: HOME DASHBOARD
# =========================================================
if page == "🏠 Home Dashboard":
    st.markdown("### 📈 Executive Sales Overview")
    st.markdown(
        "High-level key performance metrics calculated directly from the cleaned e-commerce dataset "
        f"({data_stats['clean_shape'][0]:,} transactions across {kpis['num_countries']} nations)."
    )

    # 6 Attractive KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-value">${kpis['total_revenue']:,.0f}</div>
                <div class="kpi-sub">Across All Markets</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Orders</div>
                <div class="kpi-value">{kpis['total_orders']:,}</div>
                <div class="kpi-sub">Unique Transactions</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Quantity Sold</div>
                <div class="kpi-value">{kpis['total_quantity']:,.0f}</div>
                <div class="kpi-sub">Units Dispatched</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg Order Value</div>
                <div class="kpi-value">${kpis['avg_order_value']:.2f}</div>
                <div class="kpi-sub">Per Customer Order</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Countries</div>
                <div class="kpi-value">{kpis['num_countries']}</div>
                <div class="kpi-sub">Global Footprint</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Categories</div>
                <div class="kpi-value">{kpis['num_categories']}</div>
                <div class="kpi-sub">Catalog Segments</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visual Row 1: Revenue Trend & Top Countries
    r1_col1, r1_col2 = st.columns([1.2, 1])

    with r1_col1:
        st.plotly_chart(plot_monthly_sales_trend(df_clean), use_container_width=True)
        st.markdown(
            """
            <div class="chart-insight">
                <strong>💡 Revenue Trend Insight:</strong> Monthly sales display healthy cyclical demand. The trajectory highlights peaks during promotional quarters, establishing strong volume for predictive modeling.
            </div>
            """,
            unsafe_allow_html=True
        )

    with r1_col2:
        st.plotly_chart(plot_sales_by_country(df_clean, top_n=6), use_container_width=True)
        st.markdown(
            """
            <div class="chart-insight">
                <strong>💡 Market Leadership:</strong> The United States, United Kingdom, and China lead revenue generation, contributing to the majority of total sales.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Main Visual Row 2: Categories & Status
    r2_col1, r2_col2 = st.columns([1.1, 0.9])

    with r2_col1:
        st.plotly_chart(plot_sales_by_category(df_clean), use_container_width=True)
        st.markdown(
            """
            <div class="chart-insight">
                <strong>💡 Category Performance:</strong> High-ticket categories like Electronics and Apparel dominate revenue, while high-frequency categories drive steady order velocity.
            </div>
            """,
            unsafe_allow_html=True
        )

    with r2_col2:
        st.plotly_chart(plot_order_status_distribution(df_clean), use_container_width=True)
        st.markdown(
            """
            <div class="chart-insight">
                <strong>💡 Fulfillment Health:</strong> Over 50% of orders are marked Completed, with minimal cancellation rates, reflecting steady customer delivery.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📋 Recent Order Transactions")
    recent_cols = [
        "Order ID", "Order Date", "Country", "Product Category",
        "Quantity Sold", "Unit Price", "Total Sales", "Order Status"
    ]
    st.dataframe(
        df_clean[recent_cols].tail(10).sort_values(by="Order Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 2: DATA EXPLORER
# =========================================================
elif page == "🔍 Data Explorer":
    st.markdown("### 🔍 Interactive Dataset Explorer & Quality Audit")
    st.markdown(
        "Explore raw and cleaned records, inspect missing value handling, filter orders across dimensions, and export data subsets."
    )

    # Data Quality Metrics Banner
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    with q_col1:
        st.metric(
            "Raw Records",
            f"{data_stats['raw_shape'][0]:,} rows",
            f"{data_stats['summary_rows_removed']} summary row removed"
        )
    with q_col2:
        st.metric(
            "Cleaned Records",
            f"{data_stats['clean_shape'][0]:,} rows",
            f"{data_stats['duplicates_removed']} duplicates purged"
        )
    with q_col3:
        st.metric(
            "Recovered Fields",
            f"{data_stats['recovered_sales'] + data_stats['recovered_price'] + data_stats['recovered_qty']} items",
            "Cross-calculated 3 financial fields"
        )
    with q_col4:
        st.metric(
            "Unit Price Outliers",
            f"{data_stats['outliers_capped']} capped",
            f"IQR Upper: ${data_stats['iqr_upper']:.2f}"
        )

    st.markdown("---")

    # Interactive Filters Card
    with st.expander("🎯 Filter & Search Options", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            selected_countries = st.multiselect(
                "Filter by Country:",
                options=sorted(df_clean["Country"].unique().tolist()),
                default=[]
            )

        with f_col2:
            selected_categories = st.multiselect(
                "Filter by Product Category:",
                options=sorted(df_clean["Product Category"].unique().tolist()),
                default=[]
            )

        with f_col3:
            selected_statuses = st.multiselect(
                "Filter by Order Status:",
                options=sorted(df_clean["Order Status"].unique().tolist()),
                default=[]
            )

        with f_col4:
            search_query = st.text_input("Search (Order ID / Keywords):", "")

        # Date Range Filter
        min_date = df_clean["Order Date"].min().date()
        max_date = df_clean["Order Date"].max().date()
        date_range = st.date_input(
            "Select Order Date Range:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

    # Apply Filters
    filtered_df = df_clean.copy()

    if selected_countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df["Product Category"].isin(selected_categories)]
    if selected_statuses:
        filtered_df = filtered_df[filtered_df["Order Status"].isin(selected_statuses)]
    if search_query:
        sq = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["Order ID"].str.lower().str.contains(sq) |
            filtered_df["Country"].str.lower().str.contains(sq) |
            filtered_df["Product Category"].str.lower().str.contains(sq)
        ]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range[0], date_range[1]
        filtered_df = filtered_df[
            (filtered_df["Order Date"].dt.date >= start_d) &
            (filtered_df["Order Date"].dt.date <= end_d)
        ]

    # Filter Result Summary & Download
    action_col1, action_col2 = st.columns([3, 1])
    with action_col1:
        st.markdown(
            f"**Displaying {len(filtered_df):,} of {len(df_clean):,} records** "
            f"| Total Filtered Sales: **${filtered_df['Total Sales'].sum():,.2f}**"
        )
    with action_col2:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered CSV",
            data=csv_data,
            file_name="filtered_ecommerce_sales.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Data Table View
    st.dataframe(
        filtered_df[[
            "Order ID", "Order Date", "Country", "Product Category",
            "Quantity Sold", "Unit Price", "Total Sales", "Order Status"
        ]],
        use_container_width=True,
        hide_index=True
    )

    # Data Quality Audit Section
    with st.expander("📊 Dataset Schema & Quality Diagnostics"):
        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            st.markdown("##### 🔍 Missing Values (Before vs After Cleaning)")
            null_compare = pd.DataFrame({
                "Raw Missing Values": data_stats["raw_nulls"],
                "Cleaned Missing Values": data_stats["clean_nulls"]
            })
            st.dataframe(null_compare, use_container_width=True)

        with diag_col2:
            st.markdown("##### 📐 Data Types & Preprocessing Actions")
            dtypes_df = pd.DataFrame({
                "Column": df_clean.columns,
                "Cleaned Dtype": [str(t) for t in df_clean.dtypes],
                "Cleaning Action": [
                    "Deduplicated by ID",
                    "Parsed to datetime64[ns]",
                    "Imputed 'Unknown' for nulls",
                    "Imputed 'Unspecified' for nulls",
                    "Numeric cast & financial recovery",
                    "Numeric cast & IQR clipped",
                    "Strip commas & financial recovery",
                    "Imputed 'Unknown' for nulls",
                    "Extracted Year",
                    "Extracted Month",
                    "Extracted Period YYYY-MM",
                    "Extracted Day Name"
                ]
            })
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)


# =========================================================
# PAGE 3: SALES ANALYSIS
# =========================================================
elif page == "📊 Sales Analysis":
    st.markdown("### 📊 Comprehensive E-Commerce Sales Analysis")
    st.markdown(
        "Interactive exploratory visualizations uncovering global sales distribution, category dynamics, temporal trends, and operational fulfillment."
    )

    tab1, tab2, tab3 = st.tabs(["🌍 Geographic & Category", "📈 Temporal Trends", "📦 Volume & Fulfillment"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_sales_by_country(df_clean), use_container_width=True)
            st.markdown(
                """
                <div class="chart-insight">
                    <strong>1. Total Sales by Country:</strong>
                    Displays cumulative revenue per country. Tier-1 developed e-commerce markets (United States, United Kingdom) yield top revenue, demonstrating regional market depth.
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.plotly_chart(plot_sales_by_category(df_clean), use_container_width=True)
            st.markdown(
                """
                <div class="chart-insight">
                    <strong>2. Sales by Product Category:</strong>
                    Compares sales across categories. Electronics and Apparel lead total dollar volume due to higher average prices per unit.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_sales_by_country(df_clean, top_n=10), use_container_width=True)
            st.markdown(
                """
                <div class="chart-insight">
                    <strong>7. Top 10 Countries by Sales:</strong>
                    Focuses exclusively on the ten highest grossing territories to facilitate strategic inventory and marketing budget allocation.
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:
            cat_top10 = (
                df_clean.groupby("Product Category", as_index=False)["Total Sales"]
                .sum()
                .sort_values(by="Total Sales", ascending=False)
            )
            fig_top_cat = plot_sales_by_category(df_clean)
            fig_top_cat.update_layout(title="Top Product Categories Ranked by Sales")
            st.plotly_chart(fig_top_cat, use_container_width=True)
            st.markdown(
                """
                <div class="chart-insight">
                    <strong>8. Top Categories by Sales:</strong>
                    Ranks merchandise categories from highest to lowest sales, identifying the core revenue engines of the enterprise.
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab2:
        t1, t2 = st.columns([1.3, 0.7])
        with t1:
            st.plotly_chart(plot_monthly_sales_trend(df_clean), use_container_width=True)
            st.markdown(
                """
                <div class="chart-insight">
                    <strong>3. Monthly Sales Trends:</strong>
                    Shows monthly aggregated sales across the full timeline. Seasonal cycles and promotional periods create observable surges in order volume.
                </div>
                """,
                unsafe_allow_html=True
            )

        with t2:
            st.plotly_chart(plot_yearly_sales(df_clean), use_container_width=True)
            st.markdown(
                """
                <div class="chart-insight">
                    <strong>4. Yearly Sales Trends:</strong>
                    Illustrates the yearly distribution of order revenue. Highlights active calendar years recorded in the transactional log.
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab3:
        v1, v2 = st.columns(2)
        with v1:
            st.plotly_chart(plot_quantity_by_category(df_clean), use_container_width=True)
            st.markdown(
                """
                <div class="chart-insight">
                    <strong>5. Quantity Sold by Category:</strong>
                    Measures physical item movement. Categories with lower unit prices frequently exhibit substantial unit velocity, highlighting distinct volume vs margin dynamics.
                </div>
                """,
                unsafe_allow_html=True
            )

        with v2:
            st.plotly_chart(plot_order_status_distribution(df_clean), use_container_width=True)
            st.markdown(
                """
                <div class="chart-insight">
                    <strong>6. Order Status Distribution:</strong>
                    Breaks down order fulfillment. Completed orders represent the revenue recognized, while Pending and Processing orders represent work-in-progress pipeline.
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# PAGE 4: MACHINE LEARNING MODEL
# =========================================================
elif page == "🤖 Machine Learning Model":
    st.markdown("### 🤖 Machine Learning Model Architecture & Performance")
    st.markdown(
        "Detailed breakdown of the **Random Forest Regressor** trained in the IDS project notebook, "
        "including hyperparameter configuration, evaluation metrics, feature importances, and academic disclosures."
    )

    # Model Workflow Representation
    st.markdown("##### 🔄 Machine Learning Pipeline Workflow")
    workflow_html = """
    <div style="margin: 12px 0 24px 0;">
        <span class="workflow-pill">1. Raw Dataset (2,001 rows)</span> ➔
        <span class="workflow-pill">2. Cleaning & Cross-Recovery</span> ➔
        <span class="workflow-pill">3. IQR Outlier Capping</span> ➔
        <span class="workflow-pill">4. Feature Engineering</span> ➔
        <span class="workflow-pill">5. 80/20 Train-Test Split</span> ➔
        <span class="workflow-pill">6. Random Forest (200 Trees)</span> ➔
        <span class="workflow-pill">7. Model Evaluation</span>
    </div>
    """
    st.markdown(workflow_html, unsafe_allow_html=True)

    # Core Metrics Badges
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Algorithm", "Random Forest", "200 Estimators")
    with m_col2:
        st.metric("R² Score", f"{model_meta['r2_score']:.4f}", "Variance Explained")
    with m_col3:
        st.metric("RMSE", f"${model_meta['rmse']:.2f}", "Root Mean Sq Error")
    with m_col4:
        st.metric("MAE", f"${model_meta['mae']:.2f}", "Mean Absolute Error")
    with m_col5:
        st.metric("Feature Space", f"{model_meta['features_count']} cols", "31 Encoded Dummies")

    # Simple Algorithm Explanation
    st.info(
        "📖 **Algorithm Explanation:** Random Forest Regression is an ensemble learning method that builds "
        "multiple decision trees during training and combines their individual predictions to output a robust, "
        "generalized sales estimate while mitigating individual tree overfitting."
    )

    # ACADEMIC DISCLOSURE - Target Leakage / Proxy Discussion
    st.markdown(
        """
        <div class="disclosure-box">
            <h4>⚠️ Academic Disclosure & Target Leakage Analysis (Faculty Review Note)</h4>
            <p>
                In e-commerce transactions, <strong>Total Sales = Quantity Sold × Unit Price</strong> by definition.
                In the project notebook, the feature <code>Qty_x_Price</code> was explicitly computed and included in the feature set.
            </p>
            <ul>
                <li><strong>Why is R² = 0.8808 and not 1.0?</strong> During preprocessing, <code>Unit Price</code> was clipped using IQR
                to a maximum ceiling of <strong>$367.37</strong>. However, the ground-truth target <code>Total Sales</code> was preserved without clipping.
                For high-value items (such as high-end Electronics with prices above $700), <code>Qty_x_Price</code> uses the clipped price,
                forcing the Random Forest to learn non-linear price corrections using Country, Category, and Date features.</li>
                <li><strong>Academic Integrity:</strong> We clearly disclose that <code>Qty_x_Price</code> serves as a strong mathematical proxy.
                The model is an order sales value estimator rather than an unconstrained multi-year demand forecasting model.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Visualizations: Actual vs Predicted & Feature Importances
    p_col1, p_col2 = st.columns([1.1, 0.9])

    with p_col1:
        if test_comparison:
            test_df = pd.DataFrame(test_comparison)
            st.plotly_chart(
                plot_actual_vs_predicted(test_df["Actual"], test_df["Predicted"]),
                use_container_width=True
            )
        else:
            st.info("Test comparisons are being refreshed.")

    with p_col2:
        st.plotly_chart(
            plot_feature_importance(model_meta["top_features"]),
            use_container_width=True
        )

    # Test Comparison Table (First 10-15 test records from notebook)
    st.markdown("### 🔬 Test Set Predictions Sample (Actual vs Predicted)")
    if test_comparison:
        comp_display = pd.DataFrame(test_comparison)
        st.dataframe(comp_display, use_container_width=True)


# =========================================================
# PAGE 5: NEW SALES PREDICTION
# =========================================================
elif page == "🔮 New Sales Prediction":
    st.markdown("### 🔮 Interactive Sales Value Estimator")
    st.markdown(
        "Enter new transactional order details below. The input is transformed through the exact feature pipeline "
        "and evaluated by the trained **Random Forest Regressor** to estimate the order's total sales value."
    )

    # Academic Notice Banner
    st.caption(
        "ℹ️ *Note: Uses IQR outlier clipping (max price $367.37) and one-hot encoded country/category vectors matching the trained model.*"
    )

    with st.form("prediction_form"):
        st.markdown("##### 📝 Order Parameters")
        f1, f2, f3 = st.columns(3)

        with f1:
            input_country = st.selectbox(
                "Country:",
                options=model_meta["countries"],
                index=model_meta["countries"].index("United States") if "United States" in model_meta["countries"] else 0
            )
            input_quantity = st.number_input(
                "Quantity Sold (Units):",
                min_value=1,
                max_value=500,
                value=4,
                step=1
            )

        with f2:
            input_category = st.selectbox(
                "Product Category:",
                options=model_meta["categories"],
                index=model_meta["categories"].index("Electronics") if "Electronics" in model_meta["categories"] else 0
            )
            input_unit_price = st.number_input(
                "Unit Price ($ USD):",
                min_value=0.50,
                max_value=5000.0,
                value=85.00,
                step=5.00,
                format="%.2f"
            )

        with f3:
            input_status = st.selectbox(
                "Order Status:",
                options=model_meta["statuses"],
                index=model_meta["statuses"].index("Completed") if "Completed" in model_meta["statuses"] else 0
            )
            input_date = st.date_input(
                "Order Date:",
                value=datetime.date.today()
            )

        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            submitted = st.form_submit_button("🔮 Predict Sales", use_container_width=True, type="primary")

    # When form is submitted
    if submitted:
        with st.spinner("Generating sales estimate with Random Forest..."):
            order_data = {
                "Country": input_country,
                "Product Category": input_category,
                "Order Status": input_status,
                "Quantity Sold": input_quantity,
                "Unit Price": input_unit_price,
                "Order Date": str(input_date)
            }

            result = predict_single_order(rf_model, model_meta, order_data)

            # Store in session state
            st.session_state["last_prediction"] = result

            # Append to history
            history_record = {
                "ID": f"PRED-{len(st.session_state['prediction_history']) + 1001}",
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Country": input_country,
                "Category": input_category,
                "Quantity": input_quantity,
                "Unit Price ($)": f"${input_unit_price:.2f}",
                "Estimated Sales ($)": f"${result['predicted_sales']:.2f}",
                "Base Calculation ($)": f"${result['baseline_sales']:.2f}",
                "Variance ($)": f"${result['variance']:.2f}",
                "Outlier Capped": "Yes" if result["is_outlier"] else "No"
            }
            st.session_state["prediction_history"].append(history_record)

    # Render Last Prediction
    if st.session_state["last_prediction"] is not None:
        res = st.session_state["last_prediction"]

        st.markdown(
            f"""
            <div class="prediction-hero">
                <div class="prediction-title">Random Forest Estimated Total Sales</div>
                <div class="prediction-amount">${res['predicted_sales']:,.2f}</div>
                <p style="color: #4B5563; font-size: 1rem; margin: 0;">
                    Base Calculation (Qty × Price): <strong>${res['baseline_sales']:,.2f}</strong> | Model Variance: <strong>${res['variance']:,.2f}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### 📋 Order Specification & Pipeline Trace")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"• **Country:** {res['country']}")
            st.markdown(f"• **Product Category:** {res['product_category']}")
            st.markdown(f"• **Order Status:** {res['order_status']}")
        with c2:
            st.markdown(f"• **Quantity Sold:** {res['quantity']} units")
            st.markdown(f"• **Entered Unit Price:** ${res['raw_unit_price']:.2f}")
            st.markdown(f"• **Clipped Unit Price (IQR):** ${res['clipped_unit_price']:.2f}")
        with c3:
            st.markdown(f"• **Order Date:** {res['order_date']}")
            st.markdown(f"• **Outlier Status:** {'⚠️ Capped by IQR' if res['is_outlier'] else '✅ In Normal Range'}")
            st.markdown(f"• **Baseline Model Feature (Qty × Price):** ${res['clipped_baseline_sales']:.2f}")

        # Detailed Result Explanation
        st.markdown("<br>", unsafe_allow_html=True)
        if res["is_outlier"]:
            st.warning(
                f"⚠️ **Outlier Notice:** The entered unit price (${res['raw_unit_price']:.2f}) exceeds the model's IQR training ceiling "
                f"(${model_meta['iqr_upper_bound']:.2f}). Following the notebook's preprocessing pipeline, the price was capped at "
                f"${res['clipped_unit_price']:.2f}. The model prediction (${res['predicted_sales']:.2f}) accounts for this adjustment."
            )
        else:
            st.success(
                f"✅ **Prediction Explanation:** The entered order parameters fall within the primary distribution of training data. "
                f"The Random Forest model estimated ${res['predicted_sales']:.2f}, closely aligning with the theoretical base value "
                f"of ${res['baseline_sales']:.2f} with minor adjustments for country ({res['country']}) and category ({res['product_category']})."
            )

        if st.button("🔄 Reset / Clear Prediction"):
            st.session_state["last_prediction"] = None
            st.rerun()


# =========================================================
# PAGE 6: PREDICTION HISTORY
# =========================================================
elif page == "📜 Prediction History":
    st.markdown("### 📜 Session Prediction History")
    st.markdown(
        "Review and compare predictions generated during the current application session. "
        "Download session records as CSV or clear the log."
    )

    history = st.session_state["prediction_history"]

    if len(history) == 0:
        st.info("No predictions generated yet in this session. Go to **🔮 New Sales Prediction** to test sample orders.")
    else:
        hist_df = pd.DataFrame(history)

        top_col1, top_col2 = st.columns([3, 1])
        with top_col1:
            st.markdown(f"**Total Session Predictions:** {len(hist_df)}")
        with top_col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["prediction_history"] = []
                st.session_state["last_prediction"] = None
                st.rerun()

        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # Download Button
        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prediction History as CSV",
            data=csv_hist,
            file_name="session_prediction_history.csv",
            mime="text/csv"
        )


# =========================================================
# PAGE 7: ABOUT PROJECT
# =========================================================
elif page == "ℹ️ About Project":
    st.markdown("### ℹ️ Academic Project Documentation & Faculty Presentation Guide")

    # Problem Statement & Objectives
    st.markdown("#### 🎯 Problem Statement & Objectives")
    st.markdown(
        """
        **Problem Statement:**
        Global e-commerce enterprises process high volumes of transactional records characterized by missing financial attributes,
        outlier price variations, and diverse regional behaviors. The objective of this project is to clean, analyze,
        and build a predictive model to estimate order total sales values based on transactional and catalog attributes.

        **Core Objectives:**
        1. **Exploratory Data Analysis (EDA):** Perform cross-national revenue analysis and product category performance assessment.
        2. **Data Cleaning & Recovery:** Recover missing financial variables through tri-field cross-calculation and handle outliers using IQR.
        3. **Machine Learning Development:** Implement a Random Forest Regressor ($n=200$) to predict order total sales values.
        4. **Interactive Software Prototype:** Provide faculty and stakeholders with an interactive Streamlit prototype for real-time demonstration.
        """
    )

    # 8-Step Data Science Workflow
    st.markdown("#### 🔄 8-Step Project Workflow")
    workflow_steps = [
        ("1. Data Collection", "Ingestion of 2,001 raw transactional records from Global_Ecommerce_Sales.csv."),
        ("2. Data Understanding", "Profiling column data types, identifying non-standard 'Total' summary row, and inspecting distributions."),
        ("3. Data Preprocessing", "Removing summary row, parsing strings with commas to numeric floats, and handling duplicates."),
        ("4. Financial Field Recovery", "Three-way cross recovery: if Total Sales is null, recover via Quantity × Unit Price (and vice versa)."),
        ("5. Outlier Capping (IQR)", "Detecting 194 unit price outliers using Q1-1.5*IQR and capping at upper bound of $367.37."),
        ("6. Feature Engineering", "Deriving Year, Month, DayOfWeek, and Qty_x_Price interaction feature; applying one-hot encoding."),
        ("7. Model Training & Evaluation", "Training Random Forest Regressor (200 trees, random_state=42), achieving R² = 0.8808 and RMSE = $261.70."),
        ("8. Interactive Web Prototype", "Building an interactive Streamlit analytics and prediction application for live faculty evaluation.")
    ]
    for title, desc in workflow_steps:
        st.markdown(f"• **{title}:** {desc}")

    # Faculty Presentation Demonstration Scenario
    st.markdown("#### 🎓 Sample Test Scenario for Faculty Demonstration")
    st.markdown(
        """
        To present this project effectively to your professor or viva examiner, use the following structured demo script:

        1. **Step 1: Introduction (Home Dashboard):**
           - Showcase executive KPIs: Total Revenue (~$1.09M), Total Orders (1,758 cleaned), and 12 distinct countries.
           - Point out the monthly sales curve showing steady business growth.
        2. **Step 2: Data Engineering Rigor (Data Explorer):**
           - Show the missing values before and after cleaning (from 154 missing sales down to 0).
           - Explain the tri-field recovery logic: `Sales = Quantity × Price`.
        3. **Step 3: Machine Learning Model Page:**
           - Highlight the **Random Forest Regressor** metrics: **$R^2 = 0.8808$** and **$RMSE = 261.70$**.
           - Demonstrate the **Feature Importance** chart showing `Qty_x_Price` as the dominant feature.
           - Explain the **Academic Disclosure**: explain why $R^2$ is 0.8808 (due to the IQR clipping of Unit Price to $367.37 on items like high-end laptops).
        4. **Step 4: Live Prediction (New Sales Prediction Page):**
           - **Test Case A (Normal Range):** Country: *United States*, Category: *Apparel & Accessories*, Quantity: *4*, Unit Price: *$50.00*.
             ➔ *Result:* Estimated Sales ~**$200.00**, exactly matching standard catalog pricing.
           - **Test Case B (High-Value Outlier):** Country: *China*, Category: *Electronics*, Quantity: *4*, Unit Price: *$995.00*.
             ➔ *Result:* System flags the price as an IQR outlier capped at $367.37, showing the model's learned compensation!
        """
    )

    # Technology Stack
    st.markdown("#### 🛠️ Technology Stack & Libraries")
    tech_cols = st.columns(4)
    with tech_cols[0]:
        st.markdown("**Core Language**\n• Python 3.10+")
    with tech_cols[1]:
        st.markdown("**Web Application**\n• Streamlit\n• HTML5 / CSS3")
    with tech_cols[2]:
        st.markdown("**Machine Learning**\n• Scikit-learn\n• Random Forest\n• Joblib")
    with tech_cols[3]:
        st.markdown("**Analytics & Graphics**\n• Pandas & NumPy\n• Plotly Express\n• Matplotlib & Seaborn")

