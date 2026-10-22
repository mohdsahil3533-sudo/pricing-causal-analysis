from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# File paths
PROJECT_DIR = Path(__file__).resolve().parent

# Use whichever dataset file exists
file_v2 = PROJECT_DIR / "data" / "pricing_data_v2.csv"
file_original = PROJECT_DIR / "data" / "pricing_data.csv"

DATA_FILE = file_v2 if file_v2.exists() else file_original

df = pd.read_csv(DATA_FILE, parse_dates=["date"])

panel = df.pivot(
    index="date",
    columns="market",
    values="revenue"
).sort_index()

intervention_date = pd.Timestamp("2025-05-01")

pre = panel.loc[panel.index < intervention_date]
post = panel.loc[panel.index >= intervention_date]

treated_market = "Delhi"

# Real Delhi result
def synthetic_control_effect(target, donors):
    y_pre = pre[target].to_numpy()
    X_pre = pre[donors].to_numpy()

    # Scale values for numerical stability
    scale = max(np.mean(np.abs(y_pre)), 1.0)
    y_scaled = y_pre / scale
    X_scaled = X_pre / scale

    def objective(weights):
        return np.mean((y_scaled - X_scaled @ weights) ** 2)

    initial_weights = np.full(len(donors), 1 / len(donors))

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(donors),
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0
            }
        ],
        options={"maxiter": 2000, "ftol": 1e-12}
    )

    if not result.success:
        raise RuntimeError(
            f"Optimization failed for {target}: {result.message}"
        )

    weights = result.x

    # Pre-treatment fit
    pre_prediction = X_pre @ weights
    pre_rmse = np.sqrt(np.mean((y_pre - pre_prediction) ** 2))

    # Post-treatment counterfactual and gap
    y_post = post[target].to_numpy()
    X_post = post[donors].to_numpy()
    post_prediction = X_post @ weights

    post_gap = np.mean(y_post - post_prediction)

    # Adjust for any average pre-treatment gap
    pre_gap = np.mean(y_pre - pre_prediction)
    adjusted_effect = post_gap - pre_gap

    return {
        "market": target,
        "donors": ", ".join(donors),
        "pre_rmse": pre_rmse,
        "pre_mean_gap": pre_gap,
        "post_mean_gap": post_gap,
        "adjusted_effect": adjusted_effect
    }

# Estimate the actual Delhi effect using control markets only
control_markets = [
    m for m in panel.columns if m != treated_market
]

results = []

results.append(
    synthetic_control_effect(treated_market, control_markets)
)

# Placebo tests: pretend each control market was treated.
# Exclude Delhi from placebo donors because Delhi was actually treated.
for placebo_market in control_markets:
    donors = [
        m for m in control_markets if m != placebo_market
    ]

    results.append(
        synthetic_control_effect(placebo_market, donors)
    )

results_df = pd.DataFrame(results)
results_df["type"] = np.where(
    results_df["market"] == treated_market,
    "Actual treatment",
    "Placebo"
)

print("\n--- Placebo Test Results ---")
print(
    results_df[
        [
            "market",
            "type",
            "pre_rmse",
            "post_mean_gap",
            "adjusted_effect"
        ]
    ].round(2).to_string(index=False)
)

# Save results
output_file = PROJECT_DIR / "reports" / "placebo_test_results.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")


# Post-treatment gap relative to pre-treatment fit error
results_df["gap_to_rmse_ratio"] = (
    results_df["post_mean_gap"].abs()
    / results_df["pre_rmse"]
)

print("\n--- Placebo Gap-to-RMSE Ratios ---")
print(
    results_df[
        ["market", "type", "pre_rmse", "post_mean_gap",
         "gap_to_rmse_ratio"]
    ].round(3).to_string(index=False)
)
