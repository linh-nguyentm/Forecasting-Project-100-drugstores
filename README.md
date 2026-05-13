# Retail Sales Forecasting for 100 Drugstores

This project develops an end-to-end forecasting framework for weekly retail sales across 100 drugstores.

The analysis compares classical statistical forecasting methods with machine learning approaches using exogenous variables and rolling cross-validation. The project was developed as part of the Business Forecasting course at the Technical University of Munich (TUM).

---

## Project Objective

A large European drugstore chain requires reliable short-term sales forecasts to support:

- Inventory planning
- Staffing decisions
- Promotion timing
- Budgeting and logistics optimization

The forecasting task focuses on predicting weekly sales for the next 8 weeks at the individual store level.

---

## Forecasting Framework

The project includes a complete forecasting pipeline:

1. Data preprocessing and weekly aggregation
2. Exploratory Data Analysis (EDA)
3. Statistical baseline forecasting
4. Machine learning forecasting
5. Cross-validation and model evaluation
6. Forecast generation and export

---

## Methods

### Statistical Forecasting Models

- Seasonal Naive
- AutoETS
- AutoARIMA

### Machine Learning Forecasting

- Univariate Random Forest
- Multivariate Random Forest

### Additional Components

- Rolling Cross-Validation
- STL Decomposition
- Feature Engineering
- Hierarchical Forecasting
- Forecast Accuracy Evaluation (MAPE & MSE)

---

## Technologies

- Python
- Pandas
- NumPy
- StatsForecast
- MLForecast
- Scikit-learn
- HierarchicalForecast
- Matplotlib
- Statsmodels

---

## Data Overview

The project uses:

- Historical daily sales
- Customer counts
- Promotion indicators
- Holiday flags
- Store metadata
- Competition distance

The data was aggregated from daily to weekly frequency (`W-SUN`) for forecasting purposes.

Dataset not publicly shared due to course restrictions.

---

## Key Findings

- AutoARIMA achieved the strongest performance among baseline statistical models.
- Multivariate Random Forest substantially improved forecasting accuracy.
- Average MAPE decreased from approximately 16.9% to 6.0% after incorporating exogenous variables.
- Most stores exhibited strong seasonality and relatively low residual variability.
- Promotions and customer volume showed strong relationships with weekly sales.

---

## Repository Structure

```text
Forecasting-Project-100-drugstores/
│
├── src/
│   └── retail_sales_forecasting_pipeline.py
│
├── results/
│   ├── 8wk_weekly_forecasts_per_store.csv
│   ├── total_8wk_sales_forecast_per_store.csv
│   ├── model_selection_summary_per_store.csv
│   ├── outputs.html
│   └── global_8wk_total_sales.txt
│
├── images/
│
├── README.md
├── requirements.txt
└── .gitignore
