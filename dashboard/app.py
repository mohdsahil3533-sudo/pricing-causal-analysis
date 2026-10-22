from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"

# Page configuration
st.set_page_config(
    page_title="Pricing Causal Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Pricing Causal Analysis Dashboard")
st.markdown(
    "Analyze pricing changes, market revenue, Synthetic Control "
    "and Double Machine Learning results."
)

# Load dataset
data_file = DATA_DIR / "pricing_data_v2.csv"

if not data_file.exists():
    data_file = DATA_DIR / "pricing_data.csv"

if not data_file.exists():
    st.error("Dataset nahi mili. Please data folder check karein.")
    st.stop()

df = pd.read_csv(data_file, parse_dates=["date"])

# Sidebar filters
st.sidebar.header("Dashboard Filters")

markets = sorted(df["market"].unique().tolist())
selected_markets = st.sidebar.multiselect(
    "Select markets",
    options=markets,
    default=markets
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.info("Please select both start and end dates.")
    st.stop()

start_date, end_date = date_range

filtered_df = df[
    df["market"].isin(selected_markets)
    & (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("Selected filters ke liye data nahi mila.")
    st.stop()

# KPI metrics
total_revenue = filtered_df["revenue"].sum()
total_units = filtered_df["units_sold"].sum()
average_price = filtered_df["price"].mean()
market_count = filtered_df["market"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
col2.metric("Units Sold", f"{total_units:,.0f}")
col3.metric("Average Price", f"₹{average_price:,.2f}")
col4.metric("Markets Selected", market_count)

st.divider()

# Daily revenue chart
st.subheader("📈 Daily Revenue by Market")

daily_revenue = (
    filtered_df.groupby(["date", "market"], as_index=False)["revenue"]
    .sum()
)

fig_revenue = px.line(
    daily_revenue,
    x="date",
    y="revenue",
    color="market",
    title="Daily Revenue Trends",
    labels={
        "date": "Date",
        "revenue": "Revenue (₹)",
        "market": "Market"
    }
)

st.plotly_chart(fig_revenue, use_container_width=True)

# Market comparison
st.subheader("🏙️ Market Performance Comparison")

market_summary = (
    filtered_df.groupby("market", as_index=False)
    .agg(
        total_revenue=("revenue", "sum"),
        total_units=("units_sold", "sum"),
        average_price=("price", "mean")
    )
    .sort_values("total_revenue", ascending=False)
)

fig_market = px.bar(
    market_summary,
    x="market",
    y="total_revenue",
    color="market",
    title="Total Revenue by Market",
    labels={
        "market": "Market",
        "total_revenue": "Total Revenue (₹)"
    }
)

st.plotly_chart(fig_market, use_container_width=True)
st.dataframe(market_summary, use_container_width=True)

# Price change analysis
st.subheader("🎯 Delhi Price Change Analysis")

treatment_date = pd.Timestamp("2025-05-01")

delhi_df = df[df["market"] == "Delhi"].copy()

if not delhi_df.empty:
    delhi_daily = (
        delhi_df.groupby("date", as_index=False)["revenue"].sum()
    )

    fig_delhi = px.line(
        delhi_daily,
        x="date",
        y="revenue",
        title="Delhi Revenue Before and After Price Change",
        labels={"date": "Date", "revenue": "Revenue (₹)"}
    )

    fig_delhi.add_vline(
        x=treatment_date,
        line_dash="dash",
        annotation_text="Price change"
    )

    st.plotly_chart(fig_delhi, use_container_width=True)

    pre = delhi_df[delhi_df["date"] < treatment_date]
    post = delhi_df[delhi_df["date"] >= treatment_date]

    a, b = st.columns(2)

    if not pre.empty:
        a.metric("Pre-change Avg. Revenue", f"₹{pre['revenue'].mean():,.0f}")

    if not post.empty:
        b.metric("Post-change Avg. Revenue", f"₹{post['revenue'].mean():,.0f}")

st.caption(
    "Note: This dashboard uses synthetic data. "
    "Before/after comparisons alone do not establish causality."
)


# Synthetic Control Analysis
st.divider()
st.subheader("🧪 Synthetic Control: Delhi vs Synthetic Delhi")

try:
    from scipy.optimize import minimize

    # Use daily revenue for each market
    panel = (
        df.pivot_table(
            index="date",
            columns="market",
            values="revenue",
            aggfunc="sum"
        )
        .sort_index()
    )

    treatment_date = pd.Timestamp("2025-05-01")
    target = "Delhi"
    donors = [m for m in panel.columns if m != target]

    pre = panel.index < treatment_date
    post = panel.index >= treatment_date

    if not pre.any() or not post.any():
        st.warning("Pre-treatment ya post-treatment period missing hai.")
    elif len(donors) < 2:
        st.warning("Synthetic Control ke liye aur control markets chahiye.")
    else:
        y_pre = panel.loc[pre, target].to_numpy(dtype=float)
        X_pre = panel.loc[pre, donors].to_numpy(dtype=float)

        # Scale data to improve numerical stability
        scale = max(float(y_pre.std()), 1.0)
        y_scaled = y_pre / scale
        X_scaled = X_pre / scale

        def objective(weights):
            errors = y_scaled - X_scaled @ weights
            return float(np.mean(errors ** 2))

        result = minimize(
            objective,
            x0=[1 / len(donors)] * len(donors),
            method="SLSQP",
            bounds=[(0, 1)] * len(donors),
            constraints=[
                {"type": "eq", "fun": lambda w: w.sum() - 1}
            ],
            options={"maxiter": 2000, "ftol": 1e-12}
        )

        if not result.success:
            st.warning(
                "Synthetic Control optimizer converge nahi hua: "
                + result.message
            )
        else:
            weights = result.x
            synthetic = panel[donors].to_numpy(dtype=float) @ weights

            comparison = pd.DataFrame({
                "date": panel.index,
                "Actual Delhi": panel[target].to_numpy(dtype=float),
                "Synthetic Delhi": synthetic
            })

            fig_sc = px.line(
                comparison,
                x="date",
                y=["Actual Delhi", "Synthetic Delhi"],
                title="Actual vs Synthetic Delhi Revenue",
                labels={
                    "date": "Date",
                    "value": "Revenue (₹)",
                    "variable": "Series"
                }
            )

            fig_sc.add_vline(
                x=treatment_date,
                line_dash="dash",
                annotation_text="Price change"
            )

            st.plotly_chart(fig_sc, use_container_width=True)

            pre_rmse = (
                (comparison.loc[pre, "Actual Delhi"]
                 - comparison.loc[pre, "Synthetic Delhi"]) ** 2
            ).mean() ** 0.5

            post_gap = (
                comparison.loc[post, "Actual Delhi"]
                - comparison.loc[post, "Synthetic Delhi"]
            ).mean()

            st.markdown("**Synthetic Control Results**")
            c1, c2 = st.columns(2)
            c1.metric("Pre-treatment RMSE", f"₹{pre_rmse:,.2f}")
            c2.metric("Average Post-treatment Gap", f"₹{post_gap:,.2f}/day")

            st.caption(
                "Positive gap means Delhi revenue was higher than its "
                "synthetic comparison after the price change. This is an "
                "estimate, not proof of causality."
            )

            weight_table = pd.DataFrame({
                "Control Market": donors,
                "Synthetic Weight": weights
            }).sort_values("Synthetic Weight", ascending=False)

            st.write("**Control Market Weights**")
            st.dataframe(weight_table, use_container_width=True)

except ImportError:
    st.error(
        "SciPy installed nahi hai. PowerShell mein "
        "`pip install scipy` run karein."
    )


# Double Machine Learning (DML) Analysis
st.divider()
st.subheader("🤖 Double Machine Learning (DML) Results")

try:
    import numpy as np
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    # Prepare the complete dataset for causal analysis
    dml_df = df.copy().sort_values(["date", "market"]).reset_index(drop=True)

    treatment_date = pd.Timestamp("2025-05-01")
    dml_df["day_index"] = (
        dml_df["date"] - dml_df["date"].min()
    ).dt.days

    dml_df["season_sin"] = np.sin(
        2 * np.pi * dml_df["day_index"] / 30
    )
    dml_df["season_cos"] = np.cos(
        2 * np.pi * dml_df["day_index"] / 30
    )

    # Treatment: Delhi after the price change
    dml_df["treatment"] = (
        (dml_df["market"] == "Delhi")
        & (dml_df["date"] >= treatment_date)
    ).astype(int)

    market_dummies = pd.get_dummies(
        dml_df["market"], prefix="market", drop_first=True, dtype=int
    )

    feature_df = pd.concat(
        [
            dml_df[
                [
                    "advertising_spend",
                    "day_index",
                    "season_sin",
                    "season_cos",
                ]
            ],
            market_dummies,
        ],
        axis=1,
    )

    X = feature_df.to_numpy(dtype=float)
    Y = dml_df["revenue"].to_numpy(dtype=float)
    D = dml_df["treatment"].to_numpy(dtype=float)
    groups = dml_df["date"].to_numpy()

    if len(np.unique(groups)) < 5:
        st.warning("DML ke liye kam se kam 5 unique dates chahiye.")
    elif len(np.unique(D)) < 2:
        st.warning("Treatment variable mein treated aur untreated rows dono chahiye.")
    else:
        # Cross-fitting: rows from each date stay in one fold
        splitter = GroupKFold(n_splits=5)
        y_res = np.zeros(len(Y))
        d_res = np.zeros(len(D))

        for train_idx, test_idx in splitter.split(X, Y, groups):
            model_y = make_pipeline(
                StandardScaler(),
                Ridge(alpha=1.0),
            )
            model_d = make_pipeline(
                StandardScaler(),
                Ridge(alpha=1.0),
            )

            model_y.fit(X[train_idx], Y[train_idx])
            model_d.fit(X[train_idx], D[train_idx])

            y_res[test_idx] = Y[test_idx] - model_y.predict(X[test_idx])
            d_res[test_idx] = D[test_idx] - model_d.predict(X[test_idx])

        denominator = np.sum(d_res ** 2)

        if denominator < 1e-10:
            st.warning(
                "Residual treatment variation bahut kam hai; "
                "DML estimate reliable tarike se calculate nahi ho sakta."
            )
        else:
            # Partialling-out treatment effect
            effect = np.sum(d_res * y_res) / denominator

            # Approximate date-clustered standard error
            outcome_residual = y_res - effect * d_res
            score = d_res * outcome_residual

            score_by_date = (
                pd.DataFrame({"date": groups, "score": score})
                .groupby("date")["score"]
                .sum()
            )

            n_clusters = len(score_by_date)

            if n_clusters > 1:
                variance = (
                    n_clusters / (n_clusters - 1)
                    * np.sum(score_by_date.to_numpy() ** 2)
                    / (denominator ** 2)
                )
                standard_error = np.sqrt(max(variance, 0))
                ci_low = effect - 1.96 * standard_error
                ci_high = effect + 1.96 * standard_error

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Estimated DML Effect",
                    f"₹{effect:,.2f}",
                    help="Estimated treatment effect in revenue units per row."
                )
                c2.metric(
                    "Approx. 95% CI",
                    f"₹{ci_low:,.2f} to ₹{ci_high:,.2f}"
                )
                c3.metric(
                    "Clustered Standard Error",
                    f"₹{standard_error:,.2f}"
                )

                st.write(
                    f"**Unique date clusters:** {n_clusters}  "
                    f"| **Residual treatment variance:** "
                    f"{np.var(d_res):.5f}"
                )

                if ci_low > 0 or ci_high < 0:
                    st.info(
                        "The approximate confidence interval excludes zero. "
                        "This does not by itself prove a causal effect."
                    )
                else:
                    st.warning(
                        "The approximate confidence interval includes zero."
                    )

                st.caption(
                    "Caution: This is a model-based estimate on synthetic data. "
                    "There is only one treated market, and date-clustered "
                    "uncertainty may not capture all sources of uncertainty. "
                    "Treat the interval as exploratory, not definitive."
                )

except ImportError as e:
    st.error(
        f"DML ke liye required library missing hai: {e}. "
        "PowerShell mein scikit-learn aur numpy install/check karein."
    )
except Exception as e:
    st.error(f"DML analysis run nahi ho saka: {e}")


# Model Sensitivity Analysis
st.divider()
st.subheader("🔬 DML Model Sensitivity Analysis")

try:
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    import numpy as np

    sensitivity_df = df.copy().sort_values(
        ["date", "market"]
    ).reset_index(drop=True)

    treatment_date = pd.Timestamp("2025-05-01")
    sensitivity_df["day_index"] = (
        sensitivity_df["date"] - sensitivity_df["date"].min()
    ).dt.days

    sensitivity_df["season_sin"] = np.sin(
        2 * np.pi * sensitivity_df["day_index"] / 30
    )
    sensitivity_df["season_cos"] = np.cos(
        2 * np.pi * sensitivity_df["day_index"] / 30
    )
    sensitivity_df["treatment"] = (
        (sensitivity_df["market"] == "Delhi")
        & (sensitivity_df["date"] >= treatment_date)
    ).astype(int)

    market_dummies = pd.get_dummies(
        sensitivity_df["market"],
        prefix="market",
        drop_first=True,
        dtype=int
    )

    feature_sets = {
        "Market + Advertising": ["advertising_spend"],
        "Market + Advertising + Time": [
            "advertising_spend", "day_index"
        ],
        "Market + Advertising + Seasonality": [
            "advertising_spend", "season_sin", "season_cos"
        ],
        "Full Model": [
            "advertising_spend", "day_index",
            "season_sin", "season_cos"
        ],
    }

    Y = sensitivity_df["revenue"].to_numpy(dtype=float)
    D = sensitivity_df["treatment"].to_numpy(dtype=float)
    groups = sensitivity_df["date"].to_numpy()
    splitter = GroupKFold(n_splits=5)

    sensitivity_results = []

    for model_name, features in feature_sets.items():
        numeric_features = sensitivity_df[features].reset_index(drop=True)
        X_df = pd.concat(
            [numeric_features, market_dummies.reset_index(drop=True)],
            axis=1
        )
        X = X_df.to_numpy(dtype=float)

        y_res = np.zeros(len(Y))
        d_res = np.zeros(len(D))

        for train_idx, test_idx in splitter.split(X, Y, groups):
            model_y = make_pipeline(
                StandardScaler(), Ridge(alpha=1.0)
            )
            model_d = make_pipeline(
                StandardScaler(), Ridge(alpha=1.0)
            )

            model_y.fit(X[train_idx], Y[train_idx])
            model_d.fit(X[train_idx], D[train_idx])

            y_res[test_idx] = (
                Y[test_idx] - model_y.predict(X[test_idx])
            )
            d_res[test_idx] = (
                D[test_idx] - model_d.predict(X[test_idx])
            )

        denominator = np.sum(d_res ** 2)

        if denominator > 1e-10:
            effect = np.sum(d_res * y_res) / denominator
            sensitivity_results.append({
                "Model": model_name,
                "Estimated Effect (₹)": effect,
                "Residual Treatment Variance": np.var(d_res)
            })

    if sensitivity_results:
        sensitivity_table = pd.DataFrame(sensitivity_results)

        st.dataframe(
            sensitivity_table.style.format({
                "Estimated Effect (₹)": "₹{:,.2f}",
                "Residual Treatment Variance": "{:.5f}"
            }),
            use_container_width=True
        )

        fig_sensitivity = px.bar(
            sensitivity_table,
            x="Model",
            y="Estimated Effect (₹)",
            title="Estimated Treatment Effect by Model Specification",
            labels={"Estimated Effect (₹)": "Estimated Effect (₹)"}
        )
        fig_sensitivity.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_sensitivity, use_container_width=True)

        st.caption(
            "Sensitivity checks show how estimates change with model "
            "specification. They do not establish causality on their own."
        )
    else:
        st.warning("Sensitivity estimates calculate nahi ho sake.")

except Exception as e:
    st.error(f"Sensitivity analysis error: {e}")


# Data preview and download
with st.expander("🔎 View Dataset"):
    st.dataframe(filtered_df, use_container_width=True)

st.download_button(
    label="⬇️ Download Filtered Data (CSV)",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_pricing_data.csv",
    mime="text/csv"
)
