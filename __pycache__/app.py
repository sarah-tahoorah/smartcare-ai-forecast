import pandas as pd
import streamlit as st

from data_processing import load_or_generate_data
from feature_engineering import add_features
from forecast_models import forecast
from evaluation import evaluate_models
from utils import apply_theme, kpi_card, indicator_gauge, line_chart, heatmap_calendar, bar_chart, forecast_chart

st.set_page_config(page_title="SMARTCARE AI", layout="wide")
apply_theme()

st.sidebar.markdown("## SMARTCARE AI")
st.sidebar.markdown("Predictive Forecasting of Care Load & Placement Demand")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Insights", "Forecast Dashboard", "Model Comparison", "Early Warning System"],
)

# Load data
@st.cache_data
def get_data():
    return load_or_generate_data()

raw_df = get_data()

# Date filter
min_date = raw_df["date"].min()
max_date = raw_df["date"].max()

start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(max_date - pd.Timedelta(days=365), max_date),
    min_value=min_date,
    max_value=max_date,
)

filtered = raw_df[(raw_df["date"] >= pd.to_datetime(start_date)) & (raw_df["date"] <= pd.to_datetime(end_date))]

# Features for forecasting
feat_df = add_features(raw_df)
feature_cols = [
    "net_pressure",
    "care_lag_1",
    "care_lag_7",
    "care_lag_14",
    "pressure_lag_1",
    "pressure_lag_7",
    "pressure_lag_14",
    "care_roll_7",
    "care_roll_14",
    "care_roll_std_14",
    "day_of_week",
    "month",
]

# Sidebar controls
model_choice = st.sidebar.selectbox(
    "Forecast model",
    ["Naive", "Moving Average", "ARIMA", "SARIMA", "Exponential Smoothing", "Random Forest", "Gradient Boosting"],
)

horizon = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14, step=1)
threshold = st.sidebar.slider("Warning threshold (children in care)", min_value=9000, max_value=18000, value=15000, step=100)

if page == "Home":
    st.markdown("# SMARTCARE AI")
    st.markdown("### Predictive Forecasting of Care Load & Placement Demand")
    st.markdown(
        "A simulated analytics environment for the U.S. Department of Health and Human Services to monitor placement demand, "
        "forecast care load, and detect early warning signals."
    )

    latest = raw_df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Current Children in HHS Care", f"{int(latest['children_in_care']):,}", "CC")
    with c2:
        kpi_card("Daily Transfers", f"{int(latest['transfers']):,}", "TR")
    with c3:
        kpi_card("Daily Discharges", f"{int(latest['discharges']):,}", "DS")
    with c4:
        net_pressure = int(latest["transfers"] - latest["discharges"])
        kpi_card("Net Pressure Indicator", f"{net_pressure:+,}", "NP")

    st.markdown("---")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(line_chart(filtered, "date", ["children_in_care"], "Care Load Trend"), use_container_width=True)
    with right:
        st.plotly_chart(indicator_gauge(net_pressure, "Net Pressure Gauge"), use_container_width=True)

elif page == "Data Insights":
    st.markdown("# Data Insights")
    st.markdown("Understanding trends, seasonality, and flow dynamics.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(line_chart(filtered, "date", ["children_in_care"], "Trend of Children in HHS Care"), use_container_width=True)
    with c2:
        seasonality = (
            filtered.assign(month=filtered["date"].dt.month)
            .groupby("month")["children_in_care"]
            .mean()
            .reset_index()
        )
        st.plotly_chart(line_chart(seasonality, "month", ["children_in_care"], "Seasonality Analysis"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(line_chart(filtered, "date", ["transfers", "discharges"], "Transfers vs Discharges"), use_container_width=True)
    with c4:
        heat_df = filtered.copy()
        heat_df["dow"] = heat_df["date"].dt.day_name().str[:3]
        heat_df["month"] = heat_df["date"].dt.month_name().str[:3]
        pivot = heat_df.pivot_table(index="dow", columns="month", values="children_in_care", aggfunc="mean")
        st.plotly_chart(heatmap_calendar(pivot, "Daily Patterns Heatmap"), use_container_width=True)

elif page == "Forecast Dashboard":
    st.markdown("# Forecast Dashboard")
    st.markdown("Forecast future care load with confidence intervals.")

    fc = forecast(model_choice, feat_df, horizon, feature_cols)
    future_dates = pd.date_range(raw_df["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_df = pd.DataFrame({"date": future_dates, "pred": fc["pred"], "lo": fc["lo"], "hi": fc["hi"]})

    st.plotly_chart(forecast_chart(raw_df, forecast_df), use_container_width=True)

    st.markdown("### Forecast Summary")
    st.dataframe(forecast_df.assign(pred=forecast_df["pred"].round(0)))

elif page == "Model Comparison":
    st.markdown("# Model Comparison")
    st.markdown("Compare model performance using MAE, RMSE, and MAPE.")

    metrics = evaluate_models(feat_df, horizon, feature_cols)
    best_model = metrics.iloc[0]["Model"]

    st.markdown(f"**Best Performing Model:** {best_model}")
    st.dataframe(metrics.style.highlight_min(axis=0, subset=["MAE", "RMSE", "MAPE"], color="#22d3ee"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(bar_chart(metrics, "Model", "MAE", "MAE by Model"), use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(metrics, "Model", "RMSE", "RMSE by Model"), use_container_width=True)
    with c3:
        st.plotly_chart(bar_chart(metrics, "Model", "MAPE", "MAPE by Model"), use_container_width=True)

elif page == "Early Warning System":
    st.markdown("# Early Warning System")
    st.markdown("Automated risk alerts based on projected care load.")

    fc = forecast(model_choice, feat_df, horizon, feature_cols)
    future_dates = pd.date_range(raw_df["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_df = pd.DataFrame({"date": future_dates, "pred": fc["pred"], "lo": fc["lo"], "hi": fc["hi"]})

    max_pred = forecast_df["pred"].max()

    if max_pred > threshold:
        st.markdown(
            """
            <div class="alert-box">
              <div class="alert-title">Capacity Stress Risk</div>
              <div class="muted">Forecasted care load exceeds the configured threshold. Initiate surge planning.</div>
            </div>
            <div class="alert-box">
              <div class="alert-title">Incoming Surge</div>
              <div class="muted">Projected upward trend suggests a sustained increase in placement demand.</div>
            </div>
            <div class="alert-box">
              <div class="alert-title">Placement Bottleneck</div>
              <div class="muted">Potential mismatch between transfers and discharges; review staffing capacity.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.success("No high-risk alerts detected for the selected horizon.")

    st.plotly_chart(forecast_chart(raw_df, forecast_df), use_container_width=True)
