
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from urllib.parse import quote_plus
df = pd.read_csv("sales_data.csv")

st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Sales Prediction & Future Forecast Dashboard")
st.caption("Historical sales analysis + machine-learning-style monthly forecasting using your SalesDB.")

# ---------------------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_sales(server, database, username, password, driver):
    if username:
        conn_str = (
            f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
            f"UID={username};PWD={password};TrustServerCertificate=yes;"
        )
    else:
        conn_str = (
            f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
            f"Trusted_Connection=yes;TrustServerCertificate=yes;"
        )

    conn = pyodbc.connect(conn_str)

    query = """
    SELECT
        i.InvoiceID,
        CAST(i.InvoiceDate AS date) AS InvoiceDate,
        i.CustomerID,
        i.SalesPersonID,
        i.TotalAmount,
        i.PaymentStatus,
        d.ProductID,
        p.ProductName,
        p.Category,
        d.Quantity,
        d.UnitPrice,
        d.DiscountPct,
        d.LineTotal
    FROM dbo.Invoice i
    INNER JOIN dbo.InvoiceDetails d
        ON i.InvoiceID = d.InvoiceID
    LEFT JOIN dbo.ProductMaster p
        ON d.ProductID = p.ProductID
    ORDER BY i.InvoiceDate;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce").fillna(0)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["LineTotal"] = pd.to_numeric(df["LineTotal"], errors="coerce").fillna(0)
    df["Category"] = df["Category"].fillna("Unknown")
    df["ProductName"] = df["ProductName"].fillna("Unknown")
    return df


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Database Settings")

    server = st.text_input("SQL Server", r"localhost\SQLEXPRESS")
    database = st.text_input("Database", "SalesDB")
    driver = st.text_input("ODBC Driver", "ODBC Driver 17 for SQL Server")

    auth = st.radio("Authentication", ["Windows", "SQL Server"], index=0)

    username = ""
    password = ""

    if auth == "SQL Server":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

    st.divider()
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

    st.divider()
    st.caption("Tip: if your SQL Server is local, try `localhost\\SQLEXPRESS` or `.`.")


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
try:
    df = load_sales(server, database, username, password, driver)
except Exception as e:
    st.error("Could not connect to SQL Server.")
    st.code(str(e))
    st.info(
        "Check the SQL Server name, database name, ODBC driver, and authentication. "
        "Your uploaded script creates the database as SalesDB."
    )
    st.stop()

if df.empty:
    st.warning("No sales records were found.")
    st.stop()


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

    # Keep the original timeline index for the held-out period.
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
# KPI CARDS
# ---------------------------------------------------------------------
latest_month = monthly.iloc[-1]
previous_month = monthly.iloc[-2]

sales_growth = 0
if previous_month["Sales"] != 0:
    sales_growth = (
        (latest_month["Sales"] - previous_month["Sales"])
        / abs(previous_month["Sales"])
    ) * 100

forecast_total = forecast["Predicted"].sum()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Sales", f"{monthly['Sales'].sum():,.2f}")

with c2:
    st.metric("Latest Month Sales", f"{latest_month['Sales']:,.2f}", f"{sales_growth:+.1f}%")

with c3:
    st.metric("Historical Quantity", f"{monthly['Quantity'].sum():,.0f}")

with c4:
    label = "Forecast Sales" if metric == "Sales" else "Forecast Quantity"
    st.metric(label, f"{forecast_total:,.0f}")


# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🔮 Future Forecast",
    "📈 Prediction Analysis",
    "📋 Data",
])


# ---------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Monthly Sales Trend")

    fig_sales = px.line(
        monthly,
        x="InvoiceDate",
        y="Sales",
        markers=True,
        title="Historical Monthly Sales",
        labels={"InvoiceDate": "Month", "Sales": "Sales"},
    )
    fig_sales.update_layout(hovermode="x unified")
    st.plotly_chart(fig_sales, use_container_width=True, key="historical_sales_chart")

    col1, col2 = st.columns(2)

    with col1:
        category = (
            df.groupby("Category", as_index=False)["LineTotal"]
              .sum()
              .sort_values("LineTotal", ascending=False)
        )

        fig_category = px.bar(
            category,
            x="Category",
            y="LineTotal",
            title="Sales by Category",
            labels={"LineTotal": "Sales"},
        )
        st.plotly_chart(
            fig_category,
            use_container_width=True,
            key="category_sales_chart",
        )

    with col2:
        product = (
            df.groupby("ProductName", as_index=False)["LineTotal"]
              .sum()
              .sort_values("LineTotal", ascending=False)
              .head(10)
        )

        fig_product = px.bar(
            product,
            x="LineTotal",
            y="ProductName",
            orientation="h",
            title="Top 10 Products by Sales",
            labels={"LineTotal": "Sales", "ProductName": "Product"},
        )
        st.plotly_chart(
            fig_product,
            use_container_width=True,
            key="top_products_chart",
        )


# ---------------------------------------------------------------------
# FUTURE FORECAST
# ---------------------------------------------------------------------
with tab2:
    st.subheader(f"🔮 Next {forecast_months} Months — {metric} Forecast")

    # Combined historical + forecast chart
    hist_plot = monthly[["InvoiceDate", target_column]].copy()
    hist_plot["Type"] = "Historical"
    hist_plot = hist_plot.rename(columns={target_column: "Value"})

    forecast_plot = forecast.copy()
    forecast_plot["Type"] = "Predicted"
    forecast_plot = forecast_plot.rename(columns={"Predicted": "Value"})

    combined = pd.concat([hist_plot, forecast_plot], ignore_index=True)

    fig_forecast = go.Figure()

    fig_forecast.add_trace(
        go.Scatter(
            x=hist_plot["InvoiceDate"],
            y=hist_plot["Value"],
            mode="lines+markers",
            name="Historical",
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_plot["InvoiceDate"],
            y=forecast_plot["Value"],
            mode="lines+markers",
            name="Forecast",
            line=dict(dash="dash"),
        )
    )

    fig_forecast.update_layout(
        title=f"Historical vs Future {metric}",
        xaxis_title="Month",
        yaxis_title=metric,
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True,
        key="future_forecast_chart",
    )

    st.subheader("Predicted Future Months")

    display_forecast = forecast.copy()
    display_forecast["Month"] = display_forecast["InvoiceDate"].dt.strftime("%B %Y")
    display_forecast[f"Predicted {metric}"] = display_forecast["Predicted"].round(2)

    st.dataframe(
        display_forecast[["Month", f"Predicted {metric}"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    csv = display_forecast[["Month", f"Predicted {metric}"]].to_csv(index=False)
    st.download_button(
        "⬇️ Download Forecast CSV",
        csv,
        file_name="future_sales_forecast.csv",
        mime="text/csv",
        key="download_forecast",
    )


# ---------------------------------------------------------------------
# PREDICTION ANALYSIS
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Prediction Model Analysis")

    if scores is not None:
        mae, rmse, mape = scores

        a, b, c = st.columns(3)
        with a:
            st.metric("MAE", f"{mae:,.2f}")
        with b:
            st.metric("RMSE", f"{rmse:,.2f}")
        with c:
            st.metric(
                "MAPE",
                "N/A" if np.isnan(mape) else f"{mape:.2f}%"
            )

        st.caption(
            "Metrics are calculated by training on all but the last 3 historical months "
            "and testing on those 3 months. They are diagnostic, not a guarantee of future accuracy."
        )
    else:
        st.info("More historical months are needed for the holdout accuracy test.")

    # Actual vs fitted values
    X_hist = make_features(monthly["InvoiceDate"])[["trend", "sin12", "cos12"]]
    fitted = np.maximum(model.predict(X_hist), 0)

    prediction_check = pd.DataFrame({
        "InvoiceDate": monthly["InvoiceDate"],
        "Actual": monthly[target_column],
        "Model": fitted,
    })

    fig_model = go.Figure()

    fig_model.add_trace(
        go.Scatter(
            x=prediction_check["InvoiceDate"],
            y=prediction_check["Actual"],
            mode="lines+markers",
            name="Actual",
        )
    )

    fig_model.add_trace(
        go.Scatter(
            x=prediction_check["InvoiceDate"],
            y=prediction_check["Model"],
            mode="lines",
            name="Model",
        )
    )

    fig_model.update_layout(
        title=f"Actual vs Modelled Monthly {metric}",
        xaxis_title="Month",
        yaxis_title=metric,
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True,
        key="actual_vs_model_chart",
    )

    st.subheader("Monthly Growth")

    growth = monthly.copy()
    growth["Growth %"] = growth[target_column].pct_change() * 100

    fig_growth = px.bar(
        growth,
        x="InvoiceDate",
        y="Growth %",
        title=f"Monthly {metric} Growth",
        labels={"InvoiceDate": "Month"},
    )
    st.plotly_chart(
        fig_growth,
        use_container_width=True,
        key="monthly_growth_chart",
    )


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
with tab4:
    st.subheader("Sales Data")

    search = st.text_input("Search product/category/customer", key="data_search")

    view = df.copy()

    if search:
        mask = (
            view["ProductName"].astype(str).str.contains(search, case=False, na=False)
            | view["Category"].astype(str).str.contains(search, case=False, na=False)
            | view["CustomerID"].astype(str).str.contains(search, case=False, na=False)
        )
        view = view[mask]

    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Sales Data",
        view.to_csv(index=False),
        file_name="sales_dashboard_data.csv",
        mime="text/csv",
        key="download_sales_data",
    )
