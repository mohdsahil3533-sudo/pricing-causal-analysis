
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

file_v2 = PROJECT_DIR / "data" / "pricing_data_v2.csv"
file_original = PROJECT_DIR / "data" / "pricing_data.csv"

DATA_FILE = file_v2 if file_v2.exists() else file_original

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values(["date", "market"]).reset_index(drop=True)

# --------------------------------------------------
# 2. Define treatment and outcome
# --------------------------------------------------
intervention_date = pd.Timestamp("2025-05-01")

# D = 1 for Delhi after the price increase
df["treatment"] = (
    (df["market"] == "Delhi")
    & (df["date"] >= intervention_date)
).astype(int)

# Y = revenue
y = df["revenue"].to_numpy(dtype=float)

# --------------------------------------------------
# 3. Prepare observed control variables
# --------------------------------------------------
df["day_index"] = (
    df["date"] - df["date"].min()
).dt.days

df["season_sin"] = np.sin(
    2 * np.pi * df["day_index"] / 30
)

df["season_cos"] = np.cos(
    2 * np.pi * df["day_index"] / 30
)

X = pd.concat(
    [
        df[
            [
                "advertising_spend",
                "day_index",
                "season_sin",
                "season_cos",
            ]
        ],
        pd.get_dummies(
            df["market"],
            prefix="market",
            drop_first=True,
            dtype=int,
        ),
    ],
    axis=1,
).astype(float)

D = df["treatment"].to_numpy(dtype=float)

print("--- Dataset Summary ---")
print("Rows:", len(df))
print("Treatment observations:", int(D.sum()))
print("Control observations:", int((D == 0).sum()))
print("Control variables:", list(X.columns))

print("\n--- Treatment by Market ---")
print(
    df.groupby("market")["treatment"]
    .agg(["sum", "count"])
)

# --------------------------------------------------
# 4. DML with grouped cross-fitting
# --------------------------------------------------

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

model_y = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

model_d = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Keep observations from the same date in the same fold
groups = df["date"].to_numpy()
cv = GroupKFold(n_splits=5)

y_hat = cross_val_predict(
    model_y,
    X,
    y,
    cv=cv,
    groups=groups
)

d_hat = cross_val_predict(
    model_d,
    X,
    D,
    cv=cv,
    groups=groups
)

# Residuals
y_residual = y - y_hat
d_residual = D - d_hat

# DML partialling-out estimate
denominator = np.mean(d_residual ** 2)

if denominator < 1e-8:
    raise ValueError(
        "Treatment residual variance is too small. "
        "The effect may not be identifiable with these controls."
    )

theta = np.mean(d_residual * y_residual) / denominator

# --------------------------------------------------
# 5. Date-clustered standard error
# --------------------------------------------------

# Observation-level influence contribution
influence = (
    d_residual * (y_residual - theta * d_residual)
    / denominator
)

influence_df = pd.DataFrame({
    "date": df["date"].to_numpy(),
    "influence": influence
})

# Sum influence contributions within each date
cluster_scores = (
    influence_df.groupby("date")["influence"].sum().to_numpy()
)

n = len(df)
n_clusters = len(cluster_scores)

# Cluster-robust standard error with finite-cluster correction
variance = (
    n_clusters / (n_clusters - 1)
    * np.sum(cluster_scores ** 2)
    / (n ** 2)
)

standard_error = np.sqrt(variance)

# Approximate 95% confidence interval
ci_lower = theta - 1.96 * standard_error
ci_upper = theta + 1.96 * standard_error

# --------------------------------------------------
# 6. Print results
# --------------------------------------------------

print("\n--- DML Validation Results ---")
print(f"Estimated treatment effect: ₹{theta:,.2f}")
print(f"Clustered standard error: ₹{standard_error:,.2f}")
print(
    "Approximate 95% confidence interval: "
    f"[₹{ci_lower:,.2f}, ₹{ci_upper:,.2f}]"
)
print(f"Number of date clusters: {n_clusters}")
print(f"Residual treatment variance: {denominator:.6f}")
print(
    "Outcome residual RMSE: ₹"
    f"{np.sqrt(np.mean(y_residual ** 2)):,.2f}"
)

if ci_lower <= 0 <= ci_upper:
    print("Interpretation: CI includes zero.")
else:
    print("Interpretation: CI excludes zero.")
