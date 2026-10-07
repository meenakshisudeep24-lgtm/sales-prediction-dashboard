import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Sales Prediction & Future Forecast Dashboard")
st.caption("Historical sales analysis + machine-learning-style monthly forecasting using embedded CSV.")

# ---------------------------------------------------------------------
# LOAD DATA FROM LOCAL CSV
# ---------------------------------------------------------------------
try:
    df = pd.read_csv("sales_data.csv", on_bad_lines='skip')
    
    # Clean and cast the columns explicitly to guarantee compatibility with your charts
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    else:
        # Fallback if your column is named slightly differently
        date_col = [c for c in df.columns if "date" in c.lower()][0]
        df["InvoiceDate"] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: "InvoiceDate"})
        
       # Clean and cast the columns explicitly to guarantee compatibility with your charts
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    else:
        # Fallback if your column is named slightly differently
        date_col = [c for c in df.columns if "date" in c.lower()]
        if date_col:
            df["InvoiceDate"] = pd.to_datetime(df[date_col[0]])
            df = df.rename(columns={date_col[0]: "InvoiceDate"})
        else:
            # Emergency generation of dates if missing entirely
            df["InvoiceDate"] = pd.date_range(start="2026-01-01", periods=len(df), freq="D")
        
    df["TotalAmount"] = pd.to_numeric(df.get("TotalAmount", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    df["Quantity"] = pd.to_numeric(df.get("Quantity", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    
    # Try looking for LineTotal, Sales, or Revenue, defaulting to 0 safely as a Series asset
    sales_fallback = df.get("LineTotal", df.get("Sales", df.get("Revenue", pd.Series(0, index=df.index))))
    df["LineTotal"] = pd.to_numeric(sales_fallback, errors="coerce").fillna(0)
    
    df["Category"] = df.get("Category", pd.Series("Unknown", index=df.index)).fillna("Unknown")
    df["ProductName"] = df.get("ProductName", pd.Series("Unknown", index=df.index)).fillna("Unknown")
    
    if "InvoiceID" in df.columns:
        df["InvoiceID"] = df["InvoiceID"].fillna("Unknown")
    else:
        df["InvoiceID"] = df.index
        
    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].fillna("Unknown")
    else:
        df["CustomerID"] = "Unknown"

    st.code(str(e))
    st.info("Ensure you have a cleanly formatted 'sales_data.csv' file inside your repository.")
    st.stop()

if df.empty:
    st.warning("No sales records were found in the dataset file.")
    st.stop()

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("🔮 Forecast Settings")

    forecast_months = st.slider(
        "Future months to predict",
        min_value=1,
        max_value=24,
        value=6,
    )

    metric = st.selectbox(
        "Prediction metric",
        ["Sales", "Quantity"],
    )

# ---------------------------------------------------------------------
# PREPARE MONTHLY DATA
# ---------------------------------------------------------------------
monthly = (
    df.set_index("InvoiceDate")
      .resample("MS")
      .agg(
          Sales=("LineTotal", "sum"),
          Quantity=("Quantity", "sum"),
          Invoices=("InvoiceID", "nunique"),
          Customers=("CustomerID", "nunique"),
      )
      .reset_index()
)

monthly["Sales"] = monthly["Sales"].astype(float)
monthly["Quantity"] = monthly["Quantity"].astype(float)

# Fill missing calendar months so the model sees a continuous time series.
monthly = monthly.set_index("InvoiceDate").asfreq("MS").fillna(0).reset_index()

if len(monthly) < 4:
    st.warning("At least 4 months of historical data are recommended for forecasting.")
    st.stop()

# ---------------------------------------------------------------------
# FORECAST MODEL
# ---------------------------------------------------------------------
def make_features(dates):
    dates = pd.to_datetime(dates)
    month = dates.dt.month
    t = np.arange(len(dates), dtype=float)

    return pd.DataFrame({
        "trend": t,
        "sin12": np.sin(2 * np.pi * month / 12),
        "cos12": np.cos(2 * np.pi * month / 12),
        "month": month,
    })

def forecast_series(history, periods, target):
    hist = history[["InvoiceDate", target]].copy()
    hist[target] = pd.to_numeric(hist[target], errors="coerce").fillna(0)

    X = make_features(hist["InvoiceDate"])
    y = hist[target].values

    # Linear trend + yearly seasonality.
    model = LinearRegression()
    model.fit(X[["trend", "sin12", "cos12"]], y)

    future_dates = pd.date_range(
        hist["InvoiceDate"].max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )

    # Continue trend into the future.
    future_index = np.arange(len(hist), len(hist) + periods, dtype=float)
    future_month = future_dates.month

    X_future = pd.DataFrame({
        "trend": future_index,
        "sin12": np.sin(2 * np.pi * future_month / 12),
        "cos12": np.cos(2 * np.pi * future_month / 12),
    })

    pred = model.predict(X_future)

    # Sales/quantity cannot be negative.
    pred = np.maximum(pred, 0)

    result = pd.DataFrame({
        "InvoiceDate": future_dates,
        "Predicted": pred,
    })

    return model, result

target_column = "Sales" if metric == "Sales" else "Quantity"

model, forecast = forecast_series(monthly, forecast_months, target_column)

# ---------------------------------------------------------------------
# MODEL QUALITY / HOLDOUT TEST
# ---------------------------------------------------------------------
def holdout_score(history, target):
    if len(history) < 6:
        return None

    train = history.iloc[:-3].copy()
    test = history.iloc[-3:].copy()

    X_train = make_features(train["InvoiceDate"])[["trend", "sin12", "cos12"]]
    y_train = train[target].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    future_index = np.arange(len(train), len(train) + len(test), dtype=float)
    m = test["InvoiceDate"].dt.month

    X_test = pd.DataFrame({
        "trend": future_index,
        "sin12": np.sin(2 * np.pi * m / 12),
        "cos12": np.cos(2 * np.pi * m / 12),
    })

    pred = np.maximum(model.predict(X_test), 0)
    actual = test[target].values

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    nonzero = actual != 0
    if nonzero.any():
        mape = np.mean(
            np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])
        ) * 100
    else:
        mape = np.nan

    return mae, rmse, mape

scores = holdout_score(monthly, target_column)

# ---------------------------------------------------------------------
# KPI CARDS & VISUALIZATION RENDERING
# ---------------------------------------------------------------------
latest_month = monthly.iloc[-1]
previous_month = monthly.iloc[-2] if len(monthly) > 1 else latest_month

sales_growth = 0
if previous_month["Sales"] != 0:
    sales_growth = ((latest_month["Sales"] - previous_month["Sales"]) / previous_month["Sales"]) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Current Month Sales", f"${latest_month['Sales']:,.2f}", f"{sales_growth:+.1f}% MoM")
col2.metric("Monthly Order Items", f"{int(latest_month['Quantity'])}")
col3.metric("Active Customers", f"{int(latest_month['Customers'])}")

st.subheader("📊 Forecast Visualization")
fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly["InvoiceDate"], y=monthly[target_column], name="Historical Actuals", mode="lines+markers"))
fig.add_trace(go.Scatter(x=forecast["InvoiceDate"], y=forecast["Predicted"], name="ML Prediction", line=dict(dash="dash", color="orange")))
fig.update_layout(title=f"Monthly Forward Forecast Plan ({target_column})", xaxis_title="Timeline", yaxis_title=target_column, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

if scores:
    st.subheader("🎯 Model Performance Metrics (Holdout Evaluation)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Absolute Error (MAE)", f"{scores[0]:,.2f}")
    c2.metric("Root Mean Squared Error (RMSE)", f"{scores[1]:,.2f}")
    c3.metric("Mean Absolute Percentage Error (MAPE)", f"{scores[2]:.1f}%" if not np.isnan(scores[2]) else "N/A")

