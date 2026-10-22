
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

file_v2 = PROJECT_DIR / "data" / "pricing_data_v2.csv"
file_original = PROJECT_DIR / "data" / "pricing_data.csv"

DATA_FILE = file_v2 if file_v2.exists() else file_original

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values(["date", "market"]).reset_index(drop=True)

intervention_date = pd.Timestamp("2025-05-01")

df["treatment"] = (
    (df["market"] == "Delhi")
    & (df["date"] >= intervention_date)
).astype(int)

df["day_index"] = (
    df["date"] - df["date"].min()
).dt.days

df["season_sin"] = np.sin(
    2 * np.pi * df["day_index"] / 30
)

df["season_cos"] = np.cos(
    2 * np.pi * df["day_index"] / 30
)

# Market indicators
market_features = pd.get_dummies(
    df["market"],
    prefix="market",
    drop_first=True,
    dtype=int,
)

# Outcome and treatment
y = df["revenue"].to_numpy(dtype=float)
D = df["treatment"].to_numpy(dtype=float)

# --------------------------------------------------
# 2. Define different model specifications
# --------------------------------------------------
specifications = {
    "Market + advertising": [
        "advertising_spend",
    ],
    "Market + advertising + time": [
        "advertising_spend",
        "day_index",
    ],
    "Market + advertising + seasonality": [
        "advertising_spend",
        "season_sin",
        "season_cos",
    ],
    "Full model": [
        "advertising_spend",
        "day_index",
        "season_sin",
        "season_cos",
    ],
}

# Same-date observations stay in the same fold
groups = df["date"].to_numpy()
cv = GroupKFold(n_splits=5)

results = []

# --------------------------------------------------
# 3. Estimate DML for each specification
# --------------------------------------------------
for name, numeric_features in specifications.items():

    X = pd.concat(
        [
            df[numeric_features],
            market_features,
        ],
        axis=1,
    ).astype(float)

    model_y = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    model_d = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    y_hat = cross_val_predict(
        model_y,
        X,
        y,
        cv=cv,
        groups=groups,
    )

    d_hat = cross_val_predict(
        model_d,
        X,
        D,
        cv=cv,
        groups=groups,
    )

    y_residual = y - y_hat
    d_residual = D - d_hat

    denominator = np.mean(d_residual ** 2)

    if denominator < 1e-8:
        theta = np.nan
    else:
        theta = (
            np.mean(d_residual * y_residual)
            / denominator
        )

    results.append({
        "specification": name,
        "treatment_effect": theta,
        "treatment_residual_variance": denominator,
        "outcome_residual_rmse": np.sqrt(
            np.mean(y_residual ** 2)
        ),
    })

# --------------------------------------------------
# 4. Display and save results
# --------------------------------------------------
results_df = pd.DataFrame(results)

print("\n--- DML Sensitivity Analysis ---")
print(results_df.round(3).to_string(index=False))

output_file = PROJECT_DIR / "reports" / "dml_sensitivity.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
