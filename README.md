# 📊 Causal Inference & Synthetic Control Engine for Algorithmic Pricing

## 🚀 Live Demo

Explore the interactive dashboard here:

🔗 **[Launch Pricing Causal Analysis Dashboard](https://pricing-causal-analysis.streamlit.app/)**

📂 **[View Source Code on GitHub](https://github.com/mohdsahil3533-sudo/pricing-causal-analysis)**

An end-to-end Python portfolio project exploring a simulated pricing change using Exploratory Data Analysis (EDA), Synthetic Control, placebo checks, Double Machine Learning (DML), sensitivity analysis, and a Streamlit dashboard.

> **Important:** The dataset is synthetic. These results demonstrate an analytical workflow and are not evidence of real-world business impact.

## Highlights
- Generate a reproducible dataset with five markets and 180 days.
- Analyze revenue, units sold, and market-level patterns.
- Compare Delhi's observed revenue with a weighted synthetic comparison.
- Run placebo checks on control markets.
- Estimate a treatment effect with cross-fitted DML.
- Compare alternative model specifications.
- Explore results using an interactive Streamlit dashboard.

## Tech Stack
Python, Pandas, NumPy, SciPy, scikit-learn, Plotly, Streamlit.

## Project Structure
```text
pricing-causal-analysis/
├── data/
├── reports/
├── dashboard/
│   └── app.py
├── generate_data.py
├── analyze_data.py
├── synthetic_control.py
├── placebo_test.py
├── dml_analysis.py
├── dml_sensitivity.py
├── requirements.txt
├── requirements-advanced.txt
├── SETUP.md
└── README.md
```

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-advanced.txt
```

Generate data and run analyses:
```powershell
python generate_data.py
python analyze_data.py
python synthetic_control.py
python placebo_test.py
python dml_analysis.py
python dml_sensitivity.py
```

Launch the dashboard:
```powershell
python -m streamlit run dashboard/app.py
```
Usually opens at `http://localhost:8501`.

## Previously Reported Results
Values below are from prior runs and should be refreshed after rerunning the final scripts:
- Synthetic Control average post-period Delhi–synthetic gap: approximately **₹10,348.60/day**.
- Synthetic Control pre-treatment RMSE: approximately **₹4,980.62**.
- DML estimated effect: approximately **₹10,654.48**.
- Approximate 95% DML confidence interval: **₹7,816.66 to ₹13,492.30**.
- Sensitivity estimates varied across specifications, approximately **₹10,654–₹14,659** in the prior run.

The dashboard may show different values if its implementation, features, folds, or active CSV differ from the prior analysis scripts.

## Methods
### Exploratory Data Analysis
Summarizes revenue, units sold, prices, and market patterns.

### Synthetic Control
Constructs a weighted combination of control markets using pre-intervention revenue to compare against Delhi after the simulated price change on 1 May 2025.

### Placebo Checks
Applies the method to other markets as placebo units. The number of markets is small and some placebo pre-period fits are weak, so these checks are exploratory.

### Double Machine Learning
Uses cross-fitting and residualization to estimate a treatment coefficient conditional on observed covariates. Only one market is treated, limiting causal interpretation.

### Sensitivity Analysis
Compares estimates under different covariate specifications.

## Limitations
- Synthetic data cannot establish actual business impact.
- Delhi is the only treated market; market-specific shocks may be confounded with the intervention.
- Synthetic Control interpretation depends on pre-treatment fit and control-market comparability.
- Placebo inference is limited by the small number of markets.
- The clustered DML interval is approximate and may not capture all uncertainty.
- DML does not automatically remove unobserved confounding.

## Future Improvements
- Add more treated and control markets and longer pre-treatment periods.
- Improve pre-fit diagnostics and placebo summaries.
- Add automated tests and pinned dependencies.
- Export dashboard outputs into a report.
- Validate methods on suitable real-world data.

## Author
**Mohd Sahil**  
Data Analytics | Python | Causal Inference | Interactive Dashboards

When presenting this project in a portfolio, label it clearly as a synthetic-data demonstration and discuss both results and limitations.
