
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project ke main folder ka path
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "pricing_data.csv"

# Dataset load karo
df = pd.read_csv(DATA_FILE, parse_dates=["date"])


# Treatment indicator:
# Delhi mein 1 May 2025 se treatment = 1
df["treatment"] = (
    (df["market"] == "Delhi") &
    (df["date"] >= "2025-05-01")
).astype(int)

# Pre-treatment aur post-treatment periods
df["period"] = df["date"].apply(
    lambda date: "Post-treatment"
    if date >= pd.Timestamp("2025-05-01")
    else "Pre-treatment"
)

print("\n--- Treatment Counts ---")
print(df.groupby(["market", "period"])["treatment"].sum())

print("\n--- Average Revenue Before and After ---")
print(
    df.groupby(["market", "period"])["revenue"]
      .mean()
      .round(2)
)

# 1. Dataset ki basic information
# print("\n--- First 5 Rows ---")
# print(df.head())

# print("\n--- Dataset Shape ---")
# print(df.shape)

# print("\n--- Column Information ---")
# df.info()

# 2. Missing values aur duplicate rows
# print("\n--- Missing Values ---")
# print(df.isnull().sum())

# print("\n--- Duplicate Rows ---")
# print(df.duplicated().sum())

# 3. Market-wise summary
# print("\n--- Market-wise Summary ---")
# print(
#     df.groupby("market").agg(
#         total_revenue=("revenue", "sum"),
#         avg_price=("price", "mean"),
#         avg_units_sold=("units_sold", "mean")
#     ).round(2)
# )

# 4. Daily revenue by market
# daily_revenue = df.pivot(
#     index="date",
#     columns="market",
#     values="revenue"
# )

# daily_revenue.plot(figsize=(12, 6))
# plt.axvline(
#     pd.Timestamp("2025-05-01"),
#     linestyle="--",
#     label="Delhi Price Change"
# )
# plt.title("Daily Revenue by Market")
# plt.xlabel("Date")
# plt.ylabel("Revenue")
# plt.legend()
# plt.tight_layout()

# Graph save karo
# OUTPUT_FILE = PROJECT_DIR / "reports" / "daily_revenue.png"
# OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# plt.savefig(OUTPUT_FILE, dpi=150)
# plt.show()

# print(f"\nGraph saved at: {OUTPUT_FILE}")
