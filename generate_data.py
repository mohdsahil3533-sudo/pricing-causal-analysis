from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

markets = ["Delhi", "Mumbai", "Jaipur", "Lucknow", "Pune"]
dates = pd.date_range("2025-01-01", periods=180, freq="D")

# Common demand conditions shared across all markets
common_demand_shock = np.random.normal(0, 4, len(dates))
common_seasonality = (
    20 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
)

# Keep market-specific demand levels stable
base_demand = {
    "Delhi": 260,
    "Mumbai": 190,
    "Jaipur": 245,
    "Lucknow": 265,
    "Pune": 250,
}

data = []

for market in markets:
    for day, date in enumerate(dates):
        advertising = np.random.uniform(800, 1200)

        price = 1000
        if market == "Delhi" and date >= pd.Timestamp("2025-05-01"):
            price = 1100

        demand = (
            base_demand[market]
            + common_seasonality[day]
            + common_demand_shock[day] * 3
            + advertising * 0.025
            - (price - 1000) * 0.15
            + np.random.normal(0, 3)
        )

        units_sold = max(0, int(round(demand)))
        revenue = units_sold * price

        data.append({
            "date": date,
            "market": market,
            "price": price,
            "advertising_spend": round(advertising, 2),
            "units_sold": units_sold,
            "revenue": revenue,
        })

df = pd.DataFrame(data)

output_file = DATA_DIR / "pricing_data.csv"
df.to_csv(output_file, index=False)

print("Improved sample dataset created!")
print("Saved to:", output_file)
print("Dataset shape:", df.shape)
print(df.head())
