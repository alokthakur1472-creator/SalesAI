import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SalesAI | Sales Intelligence Dashboard",
    page_icon="SA",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #0b1220;
    }

    /* Main container */
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Headings */
    h1 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    h2 {
        color: #f8fafc !important;
        font-weight: 650 !important;
    }

    h3 {
        color: #e5e7eb !important;
        font-weight: 600 !important;
    }

    /* Normal text */
    p {
        color: #cbd5e1;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 18px;
        min-height: 115px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-weight: 700;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #111827;
        border-color: #334155;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #263244;
        border-radius: 10px;
    }

    /* Divider */
    hr {
        border-color: #263244;
    }

    /* Buttons */
    button {
        border-radius: 8px !important;
    }

    /* Footer */
    .developer-footer {
        text-align: center;
        margin-top: 45px;
        padding-top: 18px;
        padding-bottom: 10px;
        border-top: 1px solid #263244;
        color: #64748b;
        font-size: 13px;
        letter-spacing: 0.4px;
    }

    .developer-name {
        color: #94a3b8;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "clean_sales_data.csv"
)

FUTURE_FORECAST_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "forecasts",
    "future_forecast.csv"
)

OVERALL_FORECAST_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "forecasts",
    "overall_future_forecast.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "forecasts",
    "model_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    sales = pd.read_csv(PROCESSED_FILE)

    forecast = pd.read_csv(
        FUTURE_FORECAST_FILE
    )

    overall_forecast = pd.read_csv(
        OVERALL_FORECAST_FILE
    )

    model_results = pd.read_csv(
        MODEL_FILE
    )

    # Convert dates
    if "Date" in sales.columns:
        sales["Date"] = pd.to_datetime(
            sales["Date"]
        )

    if "Date" in forecast.columns:
        forecast["Date"] = pd.to_datetime(
            forecast["Date"]
        )

    if "Date" in overall_forecast.columns:
        overall_forecast["Date"] = pd.to_datetime(
            overall_forecast["Date"]
        )

    return (
        sales,
        forecast,
        overall_forecast,
        model_results
    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    PROCESSED_FILE,
    FUTURE_FORECAST_FILE,
    OVERALL_FORECAST_FILE,
    MODEL_FILE
]

missing_files = []

for file_path in required_files:

    if not os.path.exists(file_path):
        missing_files.append(file_path)


if missing_files:

    st.error(
        "SalesAI cannot start because some required files are missing."
    )

    st.write("Missing files:")

    for file_path in missing_files:
        st.code(file_path)

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

sales, forecast, overall_forecast, model_results = load_data()


# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = [
    "Store",
    "Date",
    "Weekly_Sales",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment"
]

missing_columns = [
    column
    for column in required_columns
    if column not in sales.columns
]

if missing_columns:

    st.error(
        "The processed dataset does not contain the required columns."
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("SalesAI")

    st.caption(
        "AI-Powered Sales Intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Store Analysis",
            "Forecast & AI Insights",
            "Model Performance"
        ]
    )

    st.divider()

    st.subheader("Store Filter")

    stores = sorted(
        sales["Store"].unique()
    )

    selected_store = st.selectbox(
        "Select Store",
        ["All Stores"] + stores
    )

    st.divider()

    st.caption(
        "SalesAI Analytics Platform"
    )


# ============================================================
# GLOBAL CALCULATIONS
# ============================================================

total_sales = sales[
    "Weekly_Sales"
].sum()

average_weekly_sales = sales[
    "Weekly_Sales"
].mean()

store_sales = (
    sales
    .groupby("Store")["Weekly_Sales"]
    .sum()
    .sort_values(
        ascending=False
    )
)

top_store = store_sales.index[0]

top_store_sales = store_sales.iloc[0]

forecast_total = forecast[
    "Forecasted_Sales"
].sum()

best_model_row = (
    model_results
    .sort_values("MAPE")
    .iloc[0]
)

best_model = best_model_row["Model"]

best_model_mape = best_model_row["MAPE"]


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.title(
        "Sales Intelligence Dashboard"
    )

    st.write(
        "Historical performance, store analytics and "
        "machine-learning sales forecasting."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI SECTION
    # --------------------------------------------------------

    st.subheader("Business Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Historical Sales",
            f"${total_sales / 1_000_000_000:.2f}B"
        )

    with col2:

        st.metric(
            "Average Weekly Sales",
            f"${average_weekly_sales / 1_000_000:.2f}M"
        )

    with col3:

        st.metric(
            "Top Store",
            f"Store {int(top_store)}"
        )

    with col4:

        st.metric(
            "12-Week Forecast",
            f"${forecast_total / 1_000_000:.2f}M"
        )

    with col5:

        st.metric(
            "Best Model MAPE",
            f"{best_model_mape:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # MONTHLY SALES TREND
    # --------------------------------------------------------

    st.subheader(
        "Historical Sales Trend"
    )

    monthly_sales = (
        sales
        .groupby(
            sales["Date"].dt.to_period("M")
        )["Weekly_Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales["Date"] = (
        monthly_sales["Date"]
        .dt
        .to_timestamp()
    )

    fig = px.line(
        monthly_sales,
        x="Date",
        y="Weekly_Sales",
        markers=True,
        title="Monthly Sales Performance"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(
            color="#e5e7eb"
        ),
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # STORE PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Store Performance"
    )

    store_summary = (
        sales
        .groupby("Store")["Weekly_Sales"]
        .sum()
        .reset_index()
        .sort_values(
            "Weekly_Sales",
            ascending=False
        )
    )

    fig_store = px.bar(
        store_summary,
        x="Store",
        y="Weekly_Sales",
        title="Total Sales by Store"
    )

    fig_store.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(
            color="#e5e7eb"
        ),
        xaxis_title="Store",
        yaxis_title="Total Sales"
    )

    st.plotly_chart(
        fig_store,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # HOLIDAY IMPACT
    # --------------------------------------------------------

    st.subheader(
        "Holiday Sales Impact"
    )

    holiday_summary = (
        sales
        .groupby("Holiday_Flag")["Weekly_Sales"]
        .mean()
        .reset_index()
    )

    holiday_summary["Period"] = (
        holiday_summary["Holiday_Flag"]
        .map(
            {
                0: "Non-Holiday",
                1: "Holiday"
            }
        )
    )

    fig_holiday = px.bar(
        holiday_summary,
        x="Period",
        y="Weekly_Sales",
        title="Average Weekly Sales: Holiday vs Non-Holiday"
    )

    fig_holiday.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(
            color="#e5e7eb"
        ),
        xaxis_title="Period",
        yaxis_title="Average Weekly Sales"
    )

    st.plotly_chart(
        fig_holiday,
        use_container_width=True
    )


# ============================================================
# STORE ANALYSIS
# ============================================================

elif page == "Store Analysis":

    st.title(
        "Store Analysis"
    )

    st.write(
        "Analyze historical sales performance "
        "for individual stores."
    )

    st.divider()

    if selected_store == "All Stores":

        analysis_data = sales.copy()

    else:

        analysis_data = sales[
            sales["Store"] == selected_store
        ].copy()

    store_total = (
        analysis_data["Weekly_Sales"]
        .sum()
    )

    store_average = (
        analysis_data["Weekly_Sales"]
        .mean()
    )

    store_max = (
        analysis_data["Weekly_Sales"]
        .max()
    )

    store_min = (
        analysis_data["Weekly_Sales"]
        .min()
    )

    # --------------------------------------------------------
    # STORE KPIs
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Sales",
            f"${store_total / 1_000_000:.2f}M"
        )

    with col2:

        st.metric(
            "Average Weekly Sales",
            f"${store_average:,.0f}"
        )

    with col3:

        st.metric(
            "Highest Week",
            f"${store_max:,.0f}"
        )

    with col4:

        st.metric(
            "Lowest Week",
            f"${store_min:,.0f}"
        )

    st.divider()

    # --------------------------------------------------------
    # STORE SALES TREND
    # --------------------------------------------------------

    st.subheader(
        "Weekly Sales Trend"
    )

    fig = px.line(
        analysis_data,
        x="Date",
        y="Weekly_Sales",
        title="Weekly Sales"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(
            color="#e5e7eb"
        ),
        xaxis_title="Date",
        yaxis_title="Weekly Sales",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # STORE RANKING
    # --------------------------------------------------------

    st.subheader(
        "Store Ranking"
    )

    ranking = (
        sales
        .groupby("Store")
        .agg(
            Total_Sales=(
                "Weekly_Sales",
                "sum"
            ),
            Average_Weekly_Sales=(
                "Weekly_Sales",
                "mean"
            ),
            Maximum_Weekly_Sales=(
                "Weekly_Sales",
                "max"
            )
        )
        .reset_index()
        .sort_values(
            "Total_Sales",
            ascending=False
        )
    )

    ranking[
        "Total_Sales"
    ] = ranking[
        "Total_Sales"
    ].round(2)

    ranking[
        "Average_Weekly_Sales"
    ] = ranking[
        "Average_Weekly_Sales"
    ].round(2)

    ranking[
        "Maximum_Weekly_Sales"
    ] = ranking[
        "Maximum_Weekly_Sales"
    ].round(2)

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FORECAST & AI INSIGHTS
# ============================================================

elif page == "Forecast & AI Insights":

    st.title(
        "Forecast & AI Insights"
    )

    st.write(
        "Machine-learning based future sales predictions "
        "using the best-performing XGBoost model."
    )

    st.divider()

    # --------------------------------------------------------
    # FORECAST KPIs
    # --------------------------------------------------------

    forecast_average = (
        forecast["Forecasted_Sales"]
        .mean()
    )

    forecast_max = (
        forecast["Forecasted_Sales"]
        .max()
    )

    forecast_min = (
        forecast["Forecasted_Sales"]
        .min()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Forecasted Sales",
            f"${forecast_total / 1_000_000:.2f}M"
        )

    with col2:

        st.metric(
            "Average Store / Week",
            f"${forecast_average:,.0f}"
        )

    with col3:

        st.metric(
            "Maximum Forecast",
            f"${forecast_max:,.0f}"
        )

    with col4:

        st.metric(
            "Minimum Forecast",
            f"${forecast_min:,.0f}"
        )

    st.divider()

    # --------------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------------

    st.subheader(
        "Future Sales Forecast"
    )

    forecast_chart_data = (
        forecast
        .groupby("Date")["Forecasted_Sales"]
        .sum()
        .reset_index()
    )

    fig_forecast = px.line(
        forecast_chart_data,
        x="Date",
        y="Forecasted_Sales",
        markers=True,
        title="12-Week Future Sales Forecast"
    )

    fig_forecast.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(
            color="#e5e7eb"
        ),
        xaxis_title="Forecast Date",
        yaxis_title="Forecasted Sales",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------

    st.subheader(
        "Business Insights"
    )

    holiday_average = (
        sales[
            sales["Holiday_Flag"] == 1
        ]["Weekly_Sales"]
        .mean()
    )

    nonholiday_average = (
        sales[
            sales["Holiday_Flag"] == 0
        ]["Weekly_Sales"]
        .mean()
    )

    holiday_difference = (
        (
            holiday_average
            - nonholiday_average
        )
        / nonholiday_average
    ) * 100

    st.info(
        f"Holiday weeks generated approximately "
        f"{holiday_difference:.1f}% higher average sales "
        f"than non-holiday weeks."
    )

    st.info(
        f"Store {int(top_store)} is the strongest historical "
        f"performer with approximately "
        f"${top_store_sales / 1_000_000:.2f}M "
        f"in total sales."
    )

    st.info(
        f"The {best_model} model achieved a test MAPE of "
        f"{best_model_mape:.2f}%."
    )

    st.info(
        "The forecasting system uses historical sales patterns, "
        "calendar features, lag variables and rolling sales "
        "averages to generate future predictions."
    )

    st.divider()

    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------

    st.subheader(
        "Forecast Data"
    )

    st.dataframe(
        forecast,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "Model Performance"
    )

    st.write(
        "Comparison of forecasting models evaluated "
        "on the historical test period."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    st.subheader(
        "Model Comparison"
    )

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    if "MAPE" in model_results.columns:

        st.subheader(
            "MAPE Comparison"
        )

        mape_data = (
            model_results
            .sort_values("MAPE")
        )

        fig_mape = px.bar(
            mape_data,
            x="Model",
            y="MAPE",
            title="Mean Absolute Percentage Error"
        )

        fig_mape.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(
                color="#e5e7eb"
            ),
            xaxis_title="Model",
            yaxis_title="MAPE (%)"
        )

        st.plotly_chart(
            fig_mape,
            use_container_width=True
        )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    if "MAE" in model_results.columns:

        st.subheader(
            "MAE Comparison"
        )

        mae_data = (
            model_results
            .sort_values("MAE")
        )

        fig_mae = px.bar(
            mae_data,
            x="Model",
            y="MAE",
            title="Mean Absolute Error"
        )

        fig_mae.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(
                color="#e5e7eb"
            ),
            xaxis_title="Model",
            yaxis_title="MAE"
        )

        st.plotly_chart(
            fig_mae,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # SELECTED MODEL
    # --------------------------------------------------------

    st.subheader(
        "Selected Production Model"
    )

    st.success(
        f"{best_model} was selected as the best-performing "
        f"model based on the lowest MAPE."
    )

    st.write(
        f"Test MAPE: {best_model_mape:.2f}%"
    )


# ============================================================
# PROFESSIONAL DEVELOPER FOOTER
# ============================================================

st.markdown(
    """
    <div class="developer-footer">
        SalesAI — AI-Powered Sales Intelligence
        <br>
        Developed by <span class="developer-name">Alok Thakur</span>
    </div>
    """,
    unsafe_allow_html=True
)