
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# File paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "pricing_data.csv"
REPORTS_DIR = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE, parse_dates=["date"])

# Daily revenue: one column per market
panel = df.pivot(
    index="date",
    columns="market",
    values="revenue"
).sort_index()

intervention_date = pd.Timestamp("2025-05-01")

# Separate before and after the price change
pre = panel.loc[panel.index < intervention_date]
post = panel.loc[panel.index >= intervention_date]

treated_market = "Delhi"
control_markets = [
    market for market in panel.columns
    if market != treated_market
]

y_pre = pre[treated_market].to_numpy()
X_pre = pre[control_markets].to_numpy()


# Find non-negative weights that sum to 1.
# Normalize revenue to improve numerical stability.

scale = max(np.mean(np.abs(y_pre)), 1.0)

y_scaled = y_pre / scale
X_scaled = X_pre / scale

def objective(weights):
    prediction = X_scaled @ weights
    return np.mean((y_scaled - prediction) ** 2)

initial_weights = np.full(
    len(control_markets),
    1 / len(control_markets)
)

result = minimize(
    objective,
    initial_weights,
    method="SLSQP",
    bounds=[(0.0, 1.0)] * len(control_markets),
    constraints=[
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0
        }
    ],
    options={
        "maxiter": 2000,
        "ftol": 1e-12
    }
)

if not result.success:
    raise RuntimeError(
        f"Weight optimization failed: {result.message}"
    )

weights = result.x

print("\n--- Synthetic Control Weights ---")
for market, weight in zip(control_markets, weights):
    print(f"{market}: {weight:.3f}")

print(f"Total weight: {weights.sum():.3f}")


print("\n--- Synthetic Control Weights ---")
for market, weight in zip(control_markets, weights):
    print(f"{market}: {weight:.3f}")

# Create synthetic Delhi for the full period
synthetic_revenue = (
    panel[control_markets].to_numpy() @ weights
)

results = pd.DataFrame(
    {
        "actual_delhi": panel[treated_market],
        "synthetic_delhi": synthetic_revenue
    },
    index=panel.index
)


# Pre-treatment model accuracy
pre_results = results.loc[
    results.index < intervention_date
]

# RMSE: Root Mean Squared Error
pre_rmse = np.sqrt(
    np.mean(
        (
            pre_results["actual_delhi"]
            - pre_results["synthetic_delhi"]
        ) ** 2
    )
)

# Relative error percentage
pre_mean = pre_results["actual_delhi"].mean()
relative_rmse = (pre_rmse / pre_mean) * 100

print("\n--- Pre-treatment Model Accuracy ---")
print(f"Pre-treatment RMSE: ₹{pre_rmse:,.2f}")
print(f"Relative RMSE: {relative_rmse:.2f}%")


# Estimate post-treatment difference
post_results = results.loc[
    results.index >= intervention_date
]

actual_avg = post_results["actual_delhi"].mean()
synthetic_avg = post_results["synthetic_delhi"].mean()
estimated_effect = actual_avg - synthetic_avg

print("\n--- Post-treatment Results ---")
print(f"Actual Delhi average revenue: ₹{actual_avg:,.2f}")
print(f"Synthetic Delhi average revenue: ₹{synthetic_avg:,.2f}")
print(f"Estimated daily revenue difference: ₹{estimated_effect:,.2f}")

# Plot actual vs synthetic Delhi
plt.figure(figsize=(12, 6))

plt.plot(
    results.index,
    results["actual_delhi"],
    label="Actual Delhi"
)

plt.plot(
    results.index,
    results["synthetic_delhi"],
    label="Synthetic Delhi",
    linestyle="--"
)

plt.axvline(
    intervention_date,
    color="black",
    linestyle=":",
    label="Price change: 1 May 2025"
)

plt.title("Delhi Actual vs Synthetic Revenue")
plt.xlabel("Date")
plt.ylabel("Daily Revenue (₹)")
plt.legend()
plt.tight_layout()

plot_path = REPORTS_DIR / "synthetic_control.png"
plt.savefig(plot_path, dpi=150)
plt.show()

print(f"\nGraph saved to: {plot_path}")


# Actual minus synthetic revenue gap
results["revenue_gap"] = (
    results["actual_delhi"]
    - results["synthetic_delhi"]
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    results.index,
    results["revenue_gap"],
    label="Actual - Synthetic Revenue"
)

ax.axhline(0, linestyle="--", color="black")
ax.axvline(
    intervention_date,
    linestyle=":",
    color="red",
    label="Price change"
)

ax.set_title("Delhi Revenue Gap: Actual vs Synthetic")
ax.set_xlabel("Date")
ax.set_ylabel("Revenue Gap (₹)")
ax.legend()
plt.tight_layout()

gap_path = REPORTS_DIR / "revenue_gap.png"
plt.savefig(gap_path, dpi=150)
plt.show()

print(f"Revenue gap graph saved at: {gap_path}")

print("\n--- Pre-treatment Gap Summary ---")
print(
    results.loc[
        results.index < intervention_date,
        "revenue_gap"
    ].describe().round(2)
)

print("\n--- Post-treatment Gap Summary ---")
print(
    results.loc[
        results.index >= intervention_date,
        "revenue_gap"
    ].describe().round(2)
)


# Compare pre- and post-treatment model errors
pre_gap = results.loc[
    results.index < intervention_date,
    "revenue_gap"
]

post_gap = results.loc[
    results.index >= intervention_date,
    "revenue_gap"
]

pre_rmse = np.sqrt(np.mean(pre_gap ** 2))
post_rmse = np.sqrt(np.mean(post_gap ** 2))

print("\n--- Final Fit Diagnostics ---")
print(f"Pre-treatment mean gap: ₹{pre_gap.mean():,.2f}")
print(f"Post-treatment mean gap: ₹{post_gap.mean():,.2f}")
print(f"Pre-treatment RMSE: ₹{pre_rmse:,.2f}")
print(f"Post-treatment RMSE: ₹{post_rmse:,.2f}")
print(
    "Pre-treatment gap standard deviation: "
    f"₹{pre_gap.std():,.2f}"
)
