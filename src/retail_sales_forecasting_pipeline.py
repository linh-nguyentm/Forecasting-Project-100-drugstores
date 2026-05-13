# %%
# ================================================================================
# 0. PROJECT OVERVIEW
# ================================================================================

# Project: Project 2 – Business Forecasting SoSe 2025 ──────────────────────
# Group Members:
#    - Member 1: Mirella Supo (ID: 03805832)
#    - Member 2: Linh Nguyen (ID: 03805874)
# Date: 27.07.2025
#
# File Structure: ───────────────────────────────────────────────────────────
#   I.   Setup & Configuration
#   II.  Data Loading & Initial Tidy
#   III. Data Filtering
#   IV.  Inspection & Cleaning
#   V.   Weekly Aggregation & Missing Checks
#   VI.  Helper Functions & Plot Wrappers
#   VII. Exploratory Data Analysis (EDA)
#   VIII. Baseline Models & Cross-Validation (Weekly)
#   IX.  Hierarchical Forecasting & Reconciliation (Weekly)
#   X.   Machine-Learning Forecasts (Weekly)
#   XI.  Single-Store Demonstration (Weekly)
#   XII. Final Forecast Generation & Export (Weekly)
#   XIII. Export Results to CSV
#   XIV. Visualize Total Forecast
#
# In this notebook we walk through a full time-series forecasting pipeline:
# from raw data ingestion and cleaning, through statistical and ML benchmarks,
# to final model selection and export of 8-week ahead forecasts.

# Declaration of AI Usage: ─────────────────────────────────────────────
# Throughout this project, we use AI assistance to:
#   • Organize and structure code
#     – Generate and maintain consistent section numbering and headings.
#   • Refactor repetitive logic into functions
#     – Identify repeated code blocks and encapsulate them in reusable functions.
#   • Design complex visualizations
#     – Prototype and iterate on chart types for the most effective communication.
#   • Optimize code readability
#     – Suggest improvements for variable, function, or data naming.
#
# All AI-generated suggestions were reviewed, tested, and adapted by the analysis team
# to ensure correctness, reproducibility, and full alignment with course requirements.
#
# ================================================================================

# %%
# ================================================================================
# I. SETUP & CONFIGURATION
# ================================================================================

# 1. IMPORT REQUIRED LIBRARIES  # ────────────────────────────────────────────────
# Core system and data manipulation
import os                         # Access to operating system functionality (e.g., file paths)
import sys                        # Allows importing local custom modules
import logging                    # For logging progress, warnings, or errors
import warnings                   # To suppress or control warning messages

# Data analysis and visualization
import pandas as pd               # Main data manipulation library (DataFrames, etc.)
import numpy as np                # Numerical computing (arrays, math operations)
import matplotlib.pyplot as plt   # Plotting time series and evaluation results
import statsmodels.api as sm      # For statistical modeling and diagnostics

# Statistical testing
from scipy.stats import ttest_ind       # To compare means (e.g., holiday vs. non-holiday sales)

# Time series forecasting models
from statsforecast import StatsForecast                                  # Main class for time series forecasting
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoARIMA  # Baseline statistical models

# Preprocessing and evaluation tools
from utilsforecast.preprocessing import fill_gaps     # Handle missing time series values, but we won't use it, as we have no gaps
from utilsforecast.evaluation import evaluate         # Compute forecast accuracy metrics
import utilsforecast.losses as ufl                    # Custom loss functions (e.g., MAPE, MSE)

# Hierarchical forecasting support
from hierarchicalforecast.utils import aggregate                     # Aggregate data for hierarchy levels
from hierarchicalforecast.core import HierarchicalReconciliation     # Core reconciliation engine
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut  # Hierarchical strategies

# Machine Learning forecasting
from mlforecast import MLForecast                             # ML forecasting interface
from mlforecast.lag_transforms import ExpandingMean, RollingMean  # Create lagged statistical features
from mlforecast.target_transforms import Differences               # Transform target variable (e.g., differencing)

# Scikit-learn utilities for modeling and interpretation
from sklearn.ensemble import RandomForestRegressor           # ML algorithm used for forecasts
from sklearn.inspection import permutation_importance        # Measure feature importance after training
from sklearn.base import clone                               # Duplicate model object during cross-validation

# Custom STL decomposition module
sys.path.append('decomposition.py')    # Ensure custom module is importable
import decomposition                   # Custom module for seasonal-trend decomposition
from decomposition import STL, decomposition_plot  # Functions for visual STL and seasonality analysis


# %%
# 2. GLOBAL CONSTANTS & HYPER-PARAMETERS  # ──────────────────────────────────────
# These values define our modeling scope, including sampling frequency, 
# number of stores, forecasting window, and cross-validation setup.

FIRST_N_STORES      = 100        # Limit analysis to first 100 stores
WEEKLY_FREQ         = "W-SUN"    # Weekly resampling anchored on Sundays to match retail calendar
FORECAST_HORIZON    = 8          # We forecast 8 weeks ahead (2-month horizon)
INITIAL_TRAIN_WEEKS = 52         # Use 1 year of data for initial training window in cross-validation
CV_STEP_WEEKS       = 8          # Advance the training window every 8 weeks in rolling CV
SEED                = 42         # Ensure reproducibility for models with randomness
CUTOFF_DATE = pd.to_datetime("2015-07-19")  # Split point between training and holdout set

# 3. WARNINGS & LOGGING CONFIGURATION  # ─────────────────────────────────────────
# Silence unnecessary warnings and activate logging to track pipeline execution.
# This helps keep output clean while enabling meaningful runtime tracking.

warnings.simplefilter("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# %%
# ================================================================================
# II. DATA LOADING & INITIAL TIDY
# ================================================================================

# 1. READ RAW CSV FILES  # ───────────────────────────────────────────────────────
# Load raw sales data, future covariates, and store metadata into memory.
# Keeping raw copies untouched for traceability and reproducibility.

sales_csv   = "sales_data.csv"
future_csv  = "future_values.csv"
meta_csv    = "metadata.csv"

sales_raw   = pd.read_csv(sales_csv,  parse_dates=["date"])
future_raw  = pd.read_csv(future_csv, parse_dates=["date"])
meta_raw    = pd.read_csv(meta_csv)


# 2. RENAME COLUMNS TO FORECASTING CONVENTION  # ────────────────────────────────
# We rename columns to match `StatsForecast` and `MLForecast` requirements (unique_id, ds, y).
# This ensures compatibility with time series libraries and consistent formatting across datasets.

sales = (
    sales_raw
    .rename(columns={
        "store_id": "unique_id",
        "date":     "ds",
        "sales":    "y"
    })
)
future = (
    future_raw
    .rename(columns={
        "store_id": "unique_id",
        "date":     "ds"
    })
)
meta = (
    meta_raw
    .rename(columns={"store_id": "unique_id"})
)


# 3. INSPECT DATA STRUCTURE & MISSING VALUES  # ────────────────────────────────
# Quick validation to understand the structure, data types, and any missing values.
# This ensures we catch data quality issues early in the pipeline.

def inspect_df(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} shape: {df.shape} ---")
    print("dtypes:")
    print(df.dtypes)
    print("\nmissing values:")
    print(df.isna().sum())
    print("-" * 40)

inspect_df(sales,  "Sales")
inspect_df(future, "Future covariates")
inspect_df(meta,   "Metadata")

# %%
# ================================================================================
# III. DATA FILTERING
# ================================================================================

# 1. LIMIT TO FIRST N STORES  # ──────────────────────────────────────────────────
# To reduce complexity and runtime, we restrict our dataset to the first N stores.
# This keeps our modeling pipeline computationally feasible while still representative.

FIRST_N_IDS = [f"store_{i}" for i in range(1, FIRST_N_STORES + 1)]
sales = sales[sales.unique_id.isin(FIRST_N_IDS)].copy()
future = future[future.unique_id.isin(FIRST_N_IDS)].copy()
meta = meta[meta.unique_id.isin(FIRST_N_IDS)].copy()


# 2. REMOVE COLUMNS THAT ARE FULLY MISSING IN FUTURE COVARIATES  # ──────────────
# We drop features in the future dataset that are entirely NA,
# as they offer no information and can disrupt model training.

empty_cols = [col for col in future.columns if future[col].isna().all()]
future.drop(columns=empty_cols, inplace=True)


# 3. QUICK CHECK FOR REMAINING MISSING VALUES  # ────────────────────────────────
# After filtering, we double-check for any remaining missing values
# to ensure clean input before proceeding to feature construction.

print("Sales missing:", sales.isna().sum().to_dict())
print("Future missing:", future.isna().sum().to_dict())
print("Meta missing:", meta.isna().sum().to_dict())

print("Meta missing:", meta.isna().sum().to_dict())

# %%
# ================================================================================
# IV. INSPECTION & CLEANING
# ================================================================================

# 1. FREQUENCY CHECK ON A SAMPLE STORE  # ───────────────────────────────────────
# Before any temporal operations, we validate that the time series frequency is consistent.
# We check a sample store to confirm frequency alignment between sales and future covariates.

sample_id = FIRST_N_IDS[-1]
sales_sample = (
    sales[sales.unique_id == sample_id]
    .set_index("ds")
    .sort_index()
)
future_sample = (
    future[future.unique_id == sample_id]
    .set_index("ds")
    .sort_index()
)
print("Inferred sales freq:", pd.infer_freq(sales_sample.index))
print("Inferred future freq:", pd.infer_freq(future_sample.index))

# %%
# 2. DETECT DATE GAPS IN RAW TIME SERIES  # ─────────────────────────────────────
# To check if some time series have gaps (e.g., closed stores or reporting issues).

# REVISA SI USAMOS ESTA FUNCION
def report_gaps(df, name):
    idx = pd.DatetimeIndex(df["ds"].sort_values().unique())
    inferred = pd.infer_freq(idx)
    expected = pd.date_range(idx.min(), idx.max(), freq=inferred)
    missing = expected.difference(idx)
    print(f"{name} expected freq = {inferred}, missing dates count = {len(missing)}")

report_gaps(sales, "Sales")
report_gaps(future, "Future covariates")


# 3. RECODE FLAGS AND CATEGORICAL VARIABLES  # ─────────────────────────────────
# We explicitly define categorical levels and cast flag variables to efficient types.
# This avoids silent errors in modeling and improves memory usage.

holiday_levels     = ["0", "a", "b", "c"]
store_type_levels  = ["a", "b", "c", "d"]
assortment_levels  = ["a", "b", "c"]

for df in (sales, future):
    df["state_holiday"] = pd.Categorical(
        df["state_holiday"].astype(str),
        categories=holiday_levels
    )
    df[["open", "promo", "school_holiday"]] = df[
        ["open", "promo", "school_holiday"]
    ].astype("int8")
    df["had_holiday"] = (df["state_holiday"] != "0").astype("int8")  # custom binary flag


# Recast metadata to clean categories
meta["store_type"] = pd.Categorical(
    meta["store_type"].astype(str),
    categories=store_type_levels
)
meta["assortment"] = pd.Categorical(
    meta["assortment"].astype(str),
    categories=assortment_levels
)


# 4. FINAL TYPE CHECK AND SAMPLE OUTPUT  # ─────────────────────────────────────
# Print to verify that datatypes and value structures are as expected.
# Useful for spotting anomalies before proceeding to feature construction.

print(sales.dtypes)
print(future.dtypes)
print(meta.dtypes)
print(sales.head())
print(future.head())
print(meta.head())

# %%
# ================================================================================
# V.a WEEKLY AGGREGATION & MISSING CHECKS
# ================================================================================

# 1. RESAMPLE DAILY SALES TO WEEKLY LEVEL  # ────────────────────────────────────
# Aggregate key metrics like sales and customers on a weekly basis.
# Fractions are computed for binary flags to retain their average weekly presence (e.g., open_frac).

sales_w = (
    sales
    .set_index("ds")
    .groupby("unique_id")
    .resample(WEEKLY_FREQ)
    .agg(
        y               = ("y", "sum"),
        cust_week       = ("customers", "sum"),
        open_frac       = ("open", "mean"),
        promo_frac      = ("promo", "mean"),
        school_hol_frac = ("school_holiday", "mean"),
        hol_frac        = ("had_holiday", "mean")
    )
    .reset_index()
)

# 2. RESAMPLE FUTURE COVARIATES TO WEEKLY LEVEL  # ──────────────────────────────
# Weekly aggregation ensures covariates align temporally with the sales target.
# Aggregation logic mirrors the structure used in `sales_w`.

future_w = (
    future
    .set_index("ds")
    .groupby("unique_id")
    .resample(WEEKLY_FREQ)
    .agg(
        open_frac       = ("open", "mean"),
        promo_frac      = ("promo", "mean"),
        school_hol_frac = ("school_holiday", "mean"),
        hol_frac        = ("had_holiday", "mean")
    )
    .reset_index()
)

# 3. MERGE WEEKLY SALES, FUTURE COVARIATES & METADATA  # ───────────────────────
# This consolidated dataset becomes our modeling base.
# Duplicate handling ensures one row per unique_id/week, and metadata enriches each store.

merged = (
    pd.concat([sales_w, future_w], ignore_index=True, sort=False)
      .drop_duplicates(subset=["unique_id", "ds"])
      .merge(meta, on="unique_id", how="left")
      .sort_values(["unique_id", "ds"])
      .reset_index(drop=True)
)


# 4. FINAL MISSING-VALUE OVERVIEW (POST-MERGE)  # ──────────────────────────────
# Quick validation to ensure merge didn’t introduce unexpected missingness.

missing_summary = merged.isna().sum()
print(f"Weekly merged shape: {merged.shape}")
print(f"Missing values per column:\n{missing_summary}")


# %%
# ================================================================================
# V.b MISSING VALUES & GAPS CHECK (POST-MERGE, PRE-MODELING)
# ================================================================================

# 1. FILTER TO TRAINING RANGE (BEFORE CUTOFF)  # ────────────────────────────────
# We check data integrity only for the in-sample (historical) period.

df_eval = merged.loc[merged.ds <= CUTOFF_DATE].copy()

# (1.1) CHECK FOR IMPLICIT TEMPORAL GAPS IN WEEKLY SERIES  # ──────────────────────
# Validate that each store's weekly series is continuous (no missing weeks),
# which is crucial for models relying on time-based features.

start_end = (
    df_eval
    .groupby("unique_id")
    .agg(
        observations = ("ds", "count"),
        start_date   = ("ds", "min"),
        end_date     = ("ds", "max")
    )
    .reset_index()
)
start_end["expected_obs"] = (
    ((start_end.end_date - start_end.start_date).dt.days // 7) + 1
)
start_end["has_implicit_gaps"] = (
    start_end.observations != start_end.expected_obs
)

print("Implicit-gap summary (first 5 rows):")
print(start_end.head().to_string(index=False))


# (1.2) COUNT EXPLICIT NAs IN TARGET (Y)  # ───────────────────────────────────────
# Even if weekly dates are aligned, we confirm no actual sales values are missing.

n_explicit = df_eval["y"].isna().sum()
print(f"\nExplicit missing y values up to {CUTOFF_DATE.date()}: {n_explicit}")


# CONCLUSION:
# As it has_implicit_gaps == False and n_explicit == 0,
# We can proceed without any gap‐filling.

# %%
# ================================================================================
# VI. HELPER FUNCTIONS & PLOT WRAPPERS
# ================================================================================

# These utility functions support rapid EDA and STL decomposition inspection.
# Plotting logic is encapsulated here to keep the main notebook flow clean and readable.


def plot_numeric(df: pd.DataFrame,
                 cols: list[str] | None = None,
                 n_cols: int = 2,
                 bins: int = 30) -> None:
    """
    Plot histograms for each numeric column in `df`.
    Supports quick inspection of variable distributions during EDA.
    """
    if cols is None:
        cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=bins)
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def plot_categorical(df: pd.DataFrame,
                     cols: list[str] | None = None,
                     n_cols: int = 2) -> None:
    """
    Plot bar charts for categorical columns in `df`.
    Useful to visualize assortment or store-type distributions.
    """
    if cols is None:
        cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        counts = df[col].value_counts(dropna=False)
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def plot_stl_for_store(store_id: str,
                       period: int = 52,
                       model: str = "additive",
                       lo_frac: float = 0.4,
                       shift_eps: float = 1e-3) -> None:
    """
    Run STL decomposition on the daily series for a selected store.
    Helps interpret seasonality, trend and noise components before modeling.
    """
    df_store = (
        sales
        .loc[lambda d: (d.unique_id == store_id) & (d.open == 1)]
        .loc[lambda d: d.ds <= CUTOFF_DATE, ["ds", "y"]]
        .set_index("ds")
        .sort_index()
    )

    if model == "multiplicative" and (df_store["y"] == 0).any():
        df_store["y"] = df_store["y"] + shift_eps  # avoid zeros in multiplicative STL

    stl = STL(seasonality_period=period, model=model, lo_frac=lo_frac)
    dec = stl.fit(df_store)

    fig = decomposition_plot(
        ts_index  = df_store.index,
        observed  = dec.observed["y"],
        seasonal  = dec.seasonal["y"],
        trend     = dec.trend["y"],
        resid     = dec.resid["y"]
    )
    fig.update_layout(
        title_text=f"{store_id} — {model.capitalize()} STL (period={period})",
        height=600,
        width=800
    )
    fig.show(renderer="notebook")


# %%
# ================================================================================
# VII. EXPLORATORY DATA ANALYSIS (EDA) ON WEEKLY DATA
# ================================================================================

# VII.A DESCRIPTIVE STATISTICS & DISTRIBUTIONS  # ────────────────────────────────
# Quick summary of core weekly metrics to understand central tendency and spread.

def describe_weekly(df_weekly: pd.DataFrame) -> None:
    """Print descriptive statistics for weekly numeric columns."""
    print("\n--- Weekly Numeric Summary ---")
    numeric = df_weekly[
        ['y', 'cust_week', 'open_frac', 'promo_frac', 
         'school_hol_frac', 'hol_frac', 'competition_distance']
    ]
    print(numeric.describe().T)

# VII.B HOLIDAY EFFECT ANALYSIS  # ───────────────────────────────────────────────
# Test whether holidays significantly impact weekly sales overall and by type.

def holiday_ttest(df_weekly: pd.DataFrame) -> None:
    """Compare weekly sales on holiday vs. non-holiday weeks."""
    data    = df_weekly.dropna(subset=['y'])
    no_hol  = data.loc[data.hol_frac == 0, 'y']
    yes_hol = data.loc[data.hol_frac > 0,  'y']
    t_stat, p_val = ttest_ind(yes_hol, no_hol, equal_var=False)
    print(f"Non-holiday mean = {no_hol.mean():.0f}")
    print(f"Holiday-week mean  = {yes_hol.mean():.0f}")
    print(f"t-statistic = {t_stat:.2f}, p-value = {p_val:.3f}")
    plt.figure(figsize=(6,4))
    plt.boxplot([no_hol, yes_hol], labels=["No Holiday","Holiday"])
    plt.title('Weekly Sales: Holiday vs Non-Holiday')
    plt.ylabel('Weekly Sales')
    plt.tight_layout()
    plt.show()

def holiday_type_ttests(df_daily: pd.DataFrame) -> None:
    """Separate t-tests for holiday types A, B, and C on weekly sales."""
    tmp = (
        df_daily[['ds','y','state_holiday']]
        .assign(
            sh_a=lambda d: (d.state_holiday=='a').astype(int),
            sh_b=lambda d: (d.state_holiday=='b').astype(int),
            sh_c=lambda d: (d.state_holiday=='c').astype(int)
        )
        .set_index('ds')
        .resample(WEEKLY_FREQ)
        .agg(
            weekly_sales=('y','sum'),
            sh_a=('sh_a','max'),
            sh_b=('sh_b','max'),
            sh_c=('sh_c','max')
        )
        .dropna(subset=['weekly_sales'])
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12,4), sharey=True)
    for i, h in enumerate(['a','b','c']):
        col = f'sh_{h}'
        no  = tmp.loc[tmp[col]==0, 'weekly_sales']
        yes = tmp.loc[tmp[col]==1, 'weekly_sales']
        t_stat, p_val = ttest_ind(yes, no, equal_var=False)
        print(f"Holiday {h.upper()}: mean_no={no.mean():.0f}, mean_yes={yes.mean():.0f}, p={p_val:.3f}")
        axes[i].boxplot([no, yes], labels=['No','Yes'])
        axes[i].set_title(f"Holiday {h.upper()}")
        axes[i].set_ylabel('Weekly Sales')
    plt.tight_layout()
    plt.show()

# VII.C SEGMENTED TIME-SERIES VISUALIZATIONS  # ─────────────────────────────────
# Visual comparison of sales trends and variability by store type and assortment.

def plot_mean_std_by_store_type(df_weekly: pd.DataFrame) -> None:
    grp    = df_weekly.dropna(subset=['y']).groupby(['ds','store_type'])['y']
    mean_ts = grp.mean().unstack()
    std_ts  = grp.std().unstack()
    plt.figure(figsize=(10,6))
    for st in mean_ts.columns:
        plt.plot(mean_ts.index, mean_ts[st], label=f"Mean {st}")
        plt.fill_between(
            mean_ts.index,
            mean_ts[st] - std_ts[st],
            mean_ts[st] + std_ts[st],
            alpha=0.2
        )
    plt.title('Average Weekly Sales ±1STD by Store Type')
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_sales_by_assortment(df_weekly: pd.DataFrame) -> None:
    df = df_weekly.dropna(subset=['y'])
    levels = df['assortment'].cat.categories.tolist()
    data = [df.loc[df.assortment==lvl, 'y'] for lvl in levels]
    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=levels)
    plt.title('Weekly Sales by Assortment')
    plt.xlabel('Assortment')
    plt.ylabel('Weekly Sales')
    plt.tight_layout()
    plt.show()

def plot_competition_vs_sales(df_weekly: pd.DataFrame) -> None:
    df_clean = df_weekly.dropna(subset=['competition_distance','y'])
    plt.figure(figsize=(8,5))
    plt.scatter(df_clean['competition_distance'], df_clean['y'], alpha=0.5)
    plt.xlabel('Competition Distance')
    plt.ylabel('Weekly Sales')
    plt.title('Weekly Sales vs. Competition Distance')
    plt.tight_layout()
    plt.show()

# VII.D CORRELATION & SEASONALITY METRICS  # ────────────────────────────────────
# Heatmap to inspect feature correlations and STL-based quantifications.

def plot_corr_heatmap(df_weekly: pd.DataFrame, cols: list[str]) -> None:
    corr = df_weekly[cols].corr()
    plt.figure(figsize=(8,6))
    im = plt.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center',
                     color='white' if abs(corr.iloc[i,j])>0.5 else 'black')
    plt.title('Correlation Matrix of Weekly Features', pad=20)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def quantify_fs_rv(df_store: pd.DataFrame, period: int = 52, lo_frac: float = 0.4) -> pd.Series:
    """Compute seasonality strength vs. residual variability via STL."""
    stl = STL(seasonality_period=period, model='additive', lo_frac=lo_frac)
    dec = stl.fit(df_store.dropna(subset=['y']).set_index('ds')[['y']])
    fs = max(0, 1 - np.var(dec.resid['y']) / np.var(dec.seasonal['y'] + dec.resid['y']))
    rv = dec.resid['y'].std() / df_store['y'].mean()
    return pd.Series({'seasonality_strength': round(fs,2),
                      'residual_variability': round(rv,2)})

# %%
# === EXECUTION OF EDA PIPELINE ===

# A. Descriptive stats & distributions
describe_weekly(merged)
plot_numeric(merged)
plot_categorical(merged.drop(columns=['unique_id']))

# B. Holiday effects
holiday_ttest(merged)
holiday_type_ttests(sales)

# C. Segment visuals
plot_mean_std_by_store_type(merged)
plot_sales_by_assortment(merged)
plot_competition_vs_sales(merged)

# D. Correlation & seasonality
numeric_cols = [
    'y', 'cust_week', 'open_frac', 'promo_frac',
    'school_hol_frac', 'hol_frac', 'competition_distance'
]
plot_corr_heatmap(merged, numeric_cols)

fs_rv_summary = merged.groupby('unique_id') \
    .apply(lambda x: quantify_fs_rv(x, period=52, lo_frac=0.4))
print("\n--- Seasonality & Variability per Store (first 5) ---")
print(fs_rv_summary.head())

# %%
# VII.E REPRESENTATIVE STORE SELECTION  # ───────────────────────────────────────
# Identify one store per (store_type × assortment) closest to median sales for deeper inspection.

store_summary = (
    merged
    .groupby('unique_id', as_index=False)
    .agg(
        avg_weekly_sales     = ('y',        'mean'),
        std_weekly_sales     = ('y',        'std'),
        avg_holiday_frac     = ('hol_frac', 'mean'),
        avg_open_frac        = ('open_frac','mean'),
        avg_promo_frac       = ('promo_frac','mean'),
        competition_distance = ('competition_distance','first'),
        store_type           = ('store_type','first'),
        assortment           = ('assortment','first')
    )
)

def pick_representative(df):
    med = df['avg_weekly_sales'].median()
    idx = (df['avg_weekly_sales'] - med).abs().idxmin()
    return df.loc[[idx]]

representatives = (
    store_summary
    .groupby(['store_type','assortment'], group_keys=False)
    .apply(pick_representative)
    .reset_index(drop=True)[
        ['unique_id','store_type','assortment','avg_weekly_sales']
    ]
)

print("\nRepresentative stores by StoreType × Assortment:")
print(representatives.to_string(index=False))

rep_store_ids = representatives["unique_id"].tolist()

# %%
# VII.F STL DECOMPOSITION CHECKS  # ──────────────────────────────────────────────
# Visual STL for each representative store to validate seasonality patterns.

for test_store in rep_store_ids:
    plot_stl_for_store(test_store, period=52, model="additive", lo_frac=0.4)

# %%
# ================================================================================
# VIII. BASELINE MODELS & CROSS-VALIDATION (WEEKLY)
# ================================================================================

# 1. PREPARE TRAINING DATA  # ─────────────────────────────────────────────────────
# Use only historical data up to the cutoff for fitting and evaluating baseline models.

df_weekly_train = merged.loc[merged["ds"] <= CUTOFF_DATE, ["unique_id","ds","y"]].copy()


# 2. DEFINE CLASSIC STATISTICAL MODELS  # ────────────────────────────────────────
# SeasonalNaive, ETS and ARIMA serve as benchmarks for more advanced methods.

baseline_models = [
    SeasonalNaive(season_length=52, alias="SNaive"),
    AutoETS(season_length=52,    alias="AutoETS"),
    AutoARIMA(season_length=52,  alias="AutoARIMA"),
]
sf = StatsForecast(models=baseline_models, freq=WEEKLY_FREQ)


# 3. FIT & FORECAST FULL HISTORICAL PERIOD  # ───────────────────────────────────
# Generate point forecasts over the holdout horizon to compare against later methods.

full_forecasts = sf.forecast(df=df_weekly_train, h=FORECAST_HORIZON)

# %%
# 4. ROLLING CROSS-VALIDATION SETUP  # ──────────────────────────────────────────
# Compute number of rolling windows based on initial train size, horizon & step.

n_windows = (
    df_weekly_train["ds"].nunique()
    - INITIAL_TRAIN_WEEKS
    - FORECAST_HORIZON
) // CV_STEP_WEEKS + 1

print(f"\nRolling CV: {n_windows} windows, each {INITIAL_TRAIN_WEEKS} weeks long, "
      f"with {CV_STEP_WEEKS} week steps.")

# %%
# 5. EXECUTE CROSS-VALIDATION  # ────────────────────────────────────────────────
# Evaluate each model’s performance on multiple holdout splits to assess stability.

cv_results = sf.cross_validation(
    df         = df_weekly_train,
    h          = FORECAST_HORIZON,
    step_size  = CV_STEP_WEEKS,
    n_windows  = n_windows
)

# %%
# See the cross-validation results
print("\n--- Cross-Validation Results (first 5 rows) ---")
print(cv_results.head())

# %%
# 6. COMPUTE ACCURACY METRICS  # ────────────────────────────────────────────────
# Calculate MAPE and MSE for each model-store combination over all CV folds.

cv_eval = evaluate(
    df       = cv_results.drop(columns=["cutoff"]),
    train_df = df_weekly_train,
    metrics  = [ufl.mape, ufl.mse]
)


# 7. IDENTIFY BEST MODEL PER STORE  # ─────────────────────────────────────────
# For each store, pick the model with lowest MAPE and MSE across CV.

mape_df = cv_eval.query("metric=='mape'").copy()
mse_df  = cv_eval.query("metric=='mse'").copy()

mape_df["Best_MAPE_Model"] = (
    mape_df.drop(columns=["unique_id","metric"])
           .idxmin(axis=1)
)
mape_df["MAPE"] = (
    mape_df.drop(columns=["unique_id","metric"], axis=1)
           .min(axis=1, numeric_only=True)
)

mse_df["Best_MSE_Model"] = (
    mse_df.drop(columns=["unique_id","metric"])
          .idxmin(axis=1)
)
mse_df["MSE"] = (
    mse_df.drop(columns=["unique_id","metric"], axis=1)
          .min(axis=1, numeric_only=True)
)

best_by_store = (
    mape_df[["unique_id","Best_MAPE_Model","MAPE"]]
    .merge(
        mse_df[["unique_id","Best_MSE_Model","MSE"]],
        on="unique_id"
    )
)

print("\nBest models per store:")
print(best_by_store)

# %%
# 8. SUMMARIZE MODEL PERFORMANCE ACROSS STORES  # ──────────────────────────────
# Count wins and average errors to rank baseline effectiveness.

model_names = [m.alias for m in baseline_models]

mape_wins = best_by_store["Best_MAPE_Model"] \
                .value_counts() \
                .reindex(model_names, fill_value=0)
mse_wins  = best_by_store["Best_MSE_Model"] \
                .value_counts() \
                .reindex(model_names, fill_value=0)

summary_by_model = pd.DataFrame({
    "MAPE_wins": mape_wins,
    "MSE_wins":  mse_wins
})

avg_mape = (
    mape_df
    .drop(columns=["unique_id","metric","Best_MAPE_Model"])
    .mean()
    .rename("Avg_MAPE")
)
avg_mse = (
    mse_df
    .drop(columns=["unique_id","metric","Best_MSE_Model"])
    .mean()
    .rename("Avg_MSE")
)
avg_metrics = pd.concat([avg_mape, avg_mse], axis=1)

print("\n— Wins per Model (by store) —")
print(summary_by_model)
print("\n— Average CV Metrics per Model —")
print(avg_metrics)


# 9. VISUALIZE FORECAST COVERAGE  # ─────────────────────────────────────────────
# Plot historical vs. forecasted values for all stores to inspect baseline fit.

StatsForecast.plot(
    df            = df_weekly_train,
    forecasts_df = full_forecasts
)

# %%
# 10. SEGMENT‐LEVEL BREAKDOWN OF BEST BASELINE MODELS  # ──────────────────────────
# Evaluate which statistical model performs best within each store_type × assortment segment.

store_meta = (
    merged
    .loc[:, ['unique_id','store_type','assortment']]
    .drop_duplicates()
)
best_meta = best_by_store.merge(store_meta, on='unique_id')

mape_breakdown = (
    best_meta
    .groupby(['store_type','assortment','Best_MAPE_Model'])
    .size()
    .unstack(fill_value=0)
)
print("\nStores winning on MAPE by segment:")
print(mape_breakdown)

mse_breakdown = (
    best_meta
    .groupby(['store_type','assortment','Best_MSE_Model'])
    .size()
    .unstack(fill_value=0)
)
print("\nStores winning on MSE by segment:")
print(mse_breakdown)

# %%
# 11. BASELINE MODELS – ERROR DISTRIBUTION HISTOGRAMS  # ─────────────────────────
# Visualize the spread of MAPE and MSE for each baseline to assess consistency.

mape_df = (
    cv_eval
    .query("metric == 'mape'")
    .drop(columns="metric")
    .rename(columns={
        "SNaive":    "MAPE_SNaive",
        "AutoETS":   "MAPE_AutoETS",
        "AutoARIMA": "MAPE_AutoARIMA"
    })
)
mse_df = (
    cv_eval
    .query("metric == 'mse'")
    .drop(columns="metric")
    .rename(columns={
        "SNaive":    "MSE_SNaive",
        "AutoETS":   "MSE_AutoETS",
        "AutoARIMA": "MSE_AutoARIMA"
    })
)

base_metrics = (
    mape_df
    .merge(mse_df, on="unique_id")
    .merge(meta[['unique_id','store_type','assortment']],
           on="unique_id", how="left")
)

for model in ["SNaive", "AutoETS", "AutoARIMA"]:
    plt.figure()
    base_metrics[f"MAPE_{model}"].hist(bins=20)
    plt.title(f"{model}: MAPE Distribution")
    plt.xlabel("MAPE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure()
    base_metrics[f"MSE_{model}"].hist(bins=20)
    plt.title(f"{model}: MSE Distribution")
    plt.xlabel("MSE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# 12. TOP/BOTTOM PERFORMERS  # ─────────────────────────────────────────────────
# List the top 10 and bottom 10 stores by error for each baseline, highlighting extremes.

for model in ["SNaive", "AutoETS", "AutoARIMA"]:
    col_mape = f"MAPE_{model}"
    col_mse  = f"MSE_{model}"
    print(f"\n>>> {model} – Top 10 stores by lowest {col_mape}")
    print(base_metrics.nsmallest(10, col_mape)[
        ["unique_id","store_type","assortment",col_mape]
    ])
    print(f"\n>>> {model} – Bottom 10 stores by highest {col_mape}")
    print(base_metrics.nlargest(10, col_mape)[
        ["unique_id","store_type","assortment",col_mape]
    ])
    print(f"\n>>> {model} – Top 10 stores by lowest {col_mse}")
    print(base_metrics.nsmallest(10, col_mse)[
        ["unique_id","store_type","assortment",col_mse]
    ])
    print(f"\n>>> {model} – Bottom 10 stores by highest {col_mse}")
    print(base_metrics.nlargest(10, col_mse)[
        ["unique_id","store_type","assortment",col_mse]
    ])

# 13. ERROR VS. VOLUME SCATTERPLOTS  # ────────────────────────────────────────
# Investigate whether stores with larger sales volumes systematically have higher errors.

avg_sales = merged.groupby("unique_id")["y"].mean().rename("mean_y").reset_index()
df_scatter = base_metrics.merge(avg_sales, on="unique_id")

plt.figure()
plt.scatter(df_scatter["mean_y"], df_scatter["MAPE_AutoARIMA"], alpha=0.6)
plt.title("Avg Weekly Sales vs. MAPE (AutoARIMA)")
plt.xlabel("Average Weekly Sales")
plt.ylabel("MAPE")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(df_scatter["mean_y"], df_scatter["MSE_AutoARIMA"], alpha=0.6)
plt.title("Avg Weekly Sales vs. MSE (AutoARIMA)")
plt.xlabel("Average Weekly Sales")
plt.ylabel("MSE")
plt.tight_layout()
plt.show()

# 14. SEGMENT “GOOD” STORE PROPORTIONS  # ────────────────────────────────────
# Compute proportion of stores per segment achieving MAPE/MSE thresholds under AutoARIMA.

good_mape = (
    base_metrics
    .assign(is_good=lambda d: d.MAPE_AutoARIMA < 0.15)
    .groupby(["store_type","assortment"])["is_good"]
    .mean()
    .unstack(fill_value=0)
)
print("\nProportion with MAPE < 0.15 by segment (AutoARIMA):")
print(good_mape)

median_mse = base_metrics["MSE_AutoARIMA"].median()
good_mse = (
    base_metrics
    .assign(is_good=lambda d: d.MSE_AutoARIMA < median_mse)
    .groupby(["store_type","assortment"])["is_good"]
    .mean()
    .unstack(fill_value=0)
)
print(f"\nProportion with MSE < {median_mse:.0f} by segment (AutoARIMA):")
print(good_mse)

# 15. SUMMARY STATISTICS TABLE  # ─────────────────────────────────────────────
# Provide overall means, medians, and spread of errors across all three baselines.

summary_stats = (
    base_metrics[
        ["MAPE_SNaive", "MAPE_AutoETS", "MAPE_AutoARIMA",
         "MSE_SNaive", "MSE_AutoETS", "MSE_AutoARIMA"]
    ]
    .agg(["mean", "median", "std", "min", "max"])
)
print("\nSummary statistics for baseline models:")
print(summary_stats)

# %%
# ================================================================================
# IX. HIERARCHICAL FORECASTING & RECONCILIATION (WEEKLY)
# ================================================================================

# 1. PREPARE HTS DATAFRAME WITH TOTAL LEVEL  # ─────────────────────────────────
# Add a global 'Total' series alongside bottom-level series for hierarchical modeling.

hts_df = (
    df_weekly_train
    .merge(
        meta[['unique_id', 'assortment', 'store_type']],
        on='unique_id',
        how='left'
    )
    .assign(All='Total')                  # Global aggregate level
    .rename(columns={'unique_id': 'Series'})
)

print("HTS sample:")
print(hts_df.head())


# 2. DEFINE HIERARCHY SPECIFICATION & AGGREGATE  # ─────────────────────────────
# Build aggregation matrix S and tag structure for reconciliation.

hier_spec = [
    ['All'],                                      # Level 0: Total
    ['All', 'assortment'],                        # Level 1: by assortment
    ['All', 'assortment', 'store_type']           # Level 2: by store type
]
train_hts, S_df, hierarchy_tags = aggregate(df=hts_df, spec=hier_spec)

print("\nHierarchy tags:", hierarchy_tags)
print("Number of bottom series:", len(train_hts.drop_duplicates('unique_id')))
print("Series per level:", [len(hierarchy_tags[level]) for level in hierarchy_tags])


# 3. GENERATE RAW BOTTOM-UP FORECASTS  # ───────────────────────────────────────
# Forecast every series independently using AutoARIMA as the base model.

sf_hier = StatsForecast(
    models=[AutoARIMA(season_length=52, alias='ARIMA')],
    freq=WEEKLY_FREQ
)
raw_btup = sf_hier.forecast(df=train_hts, h=FORECAST_HORIZON)


# 4. RECONCILE VIA BOTTOM-UP  # ────────────────────────────────────────────────
# Adjust forecasts to ensure hierarchical coherency (total = sum(children)).

hrec    = HierarchicalReconciliation(reconcilers=[BottomUp()])
fc_btup = hrec.reconcile(
    Y_hat_df=raw_btup,
    Y_df=train_hts,
    S=S_df,
    tags=hierarchy_tags
).rename(columns={'ARIMA/BottomUp': 'BottomUp'})

print("\nReconciled forecasts sample:")
print(fc_btup.head())


# 5. PLOT FORECASTS AT EACH LEVEL  # ──────────────────────────────────────────
# Visualize total, assortment-level, and store-type–level forecasts for sanity checking.

level0 = hierarchy_tags['All']
level1 = hierarchy_tags['All/assortment']
level2 = hierarchy_tags['All/assortment/store_type']

# %%
# Plot Total and Assortment levels
StatsForecast.plot(
    df            = train_hts.rename(columns={'Series':'unique_id'}),
    forecasts_df = fc_btup,
    unique_ids   = list(level0) + list(level1)
)
# %%
# Plot Store-Type level
StatsForecast.plot(
    df            = train_hts.rename(columns={'Series':'unique_id'}),
    forecasts_df = fc_btup,
    unique_ids   = level2
)

# %%
# 6. HTS CROSS-VALIDATION & RECONCILIATION EVALUATION  # ─────────────────────
# Assess reconciliation performance via rolling CV.

# Raw CV forecasts
cv_hts_raw = sf_hier.cross_validation(
    df        = train_hts,
    h         = FORECAST_HORIZON,
    step_size = CV_STEP_WEEKS,
    n_windows = n_windows
)
print("\nHTS raw CV head:")
print(cv_hts_raw.head())
print("HTS raw CV tail:")
print(cv_hts_raw.tail())

# Reconcile each CV fold
hrec_btup   = HierarchicalReconciliation(reconcilers=[BottomUp()])
cv_hts_list = []
for cutoff_i in cv_hts_raw['cutoff'].unique():
    fold = cv_hts_raw[cv_hts_raw['cutoff']==cutoff_i].copy()
    Y_hat = fold.rename(columns={'ARIMA':'ARIMA_raw'})[['unique_id','ds','ARIMA_raw']]
    rec   = (
        hrec_btup
          .reconcile(Y_hat_df=Y_hat, Y_df=train_hts, S=S_df, tags=hierarchy_tags)
          .rename(columns={'ARIMA_raw/BottomUp':'BottomUp'})
    )
    fold = fold.merge(
        rec[['unique_id','ds','BottomUp']],
        on=['unique_id','ds'], how='left'
    )
    cv_hts_list.append(fold)

cv_hts_rec = pd.concat(cv_hts_list, ignore_index=True)
print("\nHTS reconciled CV head:")
print(cv_hts_rec.head())
print("HTS reconciled CV tail:")
print(cv_hts_rec.tail())

# Evaluate MAPE & MSE for reconciled vs. raw
hts_errors = evaluate(
    df         = cv_hts_rec,
    train_df   = train_hts,
    metrics    = [ufl.mape, ufl.mse],
    models     = ['ARIMA','BottomUp'],
    target_col = 'y'
)
print("\nHTS CV Errors:")
print(hts_errors)

# Identify best reconciliation method per series
mape_hts      = hts_errors.query("metric=='mape'").copy()
mse_hts       = hts_errors.query("metric=='mse'").copy()
mape_hts['best_mape'] = mape_hts[['ARIMA','BottomUp']].idxmin(axis=1)
mse_hts ['best_mse']  = mse_hts[['ARIMA','BottomUp']].idxmin(axis=1)

hts_best = (
    mape_hts[['unique_id','best_mape']]
      .merge(mse_hts[['unique_id','best_mse']], on='unique_id')
)
print("\nBest reconciliation per series:")
print(hts_best)

# %%
# ================================================================================
# X. Machine-Learning Forecasts (Weekly)
# ================================================================================

# ──────────────────────────────────────────────────────────────────────────────
# X.A UNIVARIATE RANDOM FOREST FORECASTS 
# ──────────────────────────────────────────────────────────────────────────────

# 1. PREPARE TRAINING DATA  # ─────────────────────────────────────────────────────
# Use historical weekly sales up to the cutoff to train a per‐store RF model.
uni_train = (
    merged
    .loc[merged.ds <= CUTOFF_DATE, ["unique_id","ds","y"]]
    .dropna()
)

# 2. DEFINE RANDOM FOREST & FORECASTING PIPELINE  # ──────────────────────────────
# Configure RF with lag features and date encodings to capture temporal patterns.
rf_uni = RandomForestRegressor(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=10,
    random_state=SEED
)
fcst_uni = MLForecast(
    models            = {"rf_uni": rf_uni},
    freq              = WEEKLY_FREQ,
    lags              = [1, 2, 4, 26, 52],       # capture short‐ and long‐term dependencies
    date_features     = ["week", "month"],        # include seasonal date effects
    target_transforms = [Differences([1])],       # stabilize series via first‐difference
    lag_transforms    = {
        1: [RollingMean(window_size=4)],
        2: [RollingMean(window_size=13)]
    }                                             # smooth noisy lag features
)

# 3. FIT & FORECAST  # ───────────────────────────────────────────────────────────
fcst_uni.fit(df=uni_train)
uni_forecasts = fcst_uni.predict(h=FORECAST_HORIZON)
print("\nUnivariate RF forecasts sample:")
print(uni_forecasts.head())


# %%
# 4. ROLLING CROSS-VALIDATION & EVALUATION  # ───────────────────────────────────
# Assess out‐of‐sample performance via n_windows−7 splits to respect lag setup.
cv_uni = fcst_uni.cross_validation(
    df        = uni_train,
    h         = FORECAST_HORIZON,
    n_windows = n_windows - 7
)
metrics_df_uni = evaluate(
    df      = cv_uni.drop(columns=["cutoff"]),
    metrics = [ufl.mse, ufl.mape]
)
print("\nUnivariate RF Cross-Validation Results (first 5 rows):")
print(metrics_df_uni)

# %%
# 5. PIVOT & ENRICH METRICS WITH META  # ─────────────────────────────────────────
uni_metrics = (
    metrics_df_uni
    .pivot(index="unique_id", columns="metric", values="rf_uni")
    .rename(columns={"mse":"MSE", "mape":"MAPE"})
    .reset_index()
    .merge(meta[["unique_id","store_type","assortment"]],
           on="unique_id", how="left")
)

# 6. VISUALIZE ERROR DISTRIBUTIONS & TOP/BOTTOM PERFORMERS  # ───────────────────
# Provides insight into model consistency and identifies extreme cases.

# a) Histograms
plt.figure(); uni_metrics["MAPE"].hist(bins=20)
plt.title("Univariate RF: MAPE Distribution"); plt.xlabel("MAPE"); plt.ylabel("Count"); plt.show()

plt.figure(); uni_metrics["MSE"].hist(bins=20)
plt.title("Univariate RF: MSE Distribution"); plt.xlabel("MSE"); plt.ylabel("Count"); plt.show()

# b) Top/Bottom 10 by error
print("Top-10 Stores by Lowest MAPE:\n", uni_metrics.nsmallest(10, "MAPE")[["unique_id","store_type","assortment","MAPE"]])
print("\nBottom-10 Stores by Highest MAPE:\n", uni_metrics.nlargest(10, "MAPE")[["unique_id","store_type","assortment","MAPE"]])
print("Top-10 Stores by Lowest MSE:\n", uni_metrics.nsmallest(10, "MSE")[["unique_id","store_type","assortment","MSE"]])
print("\nBottom-10 Stores by Highest MSE:\n", uni_metrics.nlargest(10, "MSE")[["unique_id","store_type","assortment","MSE"]])

# c) Error vs. avg sales scatter
avg_sales = merged.groupby("unique_id")["y"].mean().rename("mean_y").reset_index()
uni_metrics = uni_metrics.merge(avg_sales, on="unique_id")
plt.figure(); plt.scatter(uni_metrics["mean_y"], uni_metrics["MAPE"], alpha=0.6)
plt.title("Avg Weekly Sales vs. MAPE (Univariate RF)"); plt.xlabel("Average Weekly Sales"); plt.ylabel("MAPE"); plt.show()
plt.figure(); plt.scatter(uni_metrics["mean_y"], uni_metrics["MSE"], alpha=0.6)
plt.title("Avg Weekly Sales vs. MSE (Univariate RF)"); plt.xlabel("Average Weekly Sales"); plt.ylabel("MSE"); plt.show()

# d) Segment “good” store proportions
good_mape = (
    uni_metrics
    .assign(is_good=lambda df: df.MAPE < 0.15)
    .groupby(["store_type","assortment"])["is_good"]
    .mean()
    .unstack(fill_value=0)
)
print("Proportion with MAPE < 0.15 by segment:\n", good_mape)

median_mse = uni_metrics["MSE"].median()
good_mse = (
    uni_metrics
    .assign(is_good=lambda df: df.MSE < median_mse)
    .groupby(["store_type","assortment"])["is_good"]
    .mean()
    .unstack(fill_value=0)
)
print(f"\nProportion with MSE < {median_mse:.0f} by segment:\n", good_mse)

# e) Summary table for error metrics
print("\nSummary statistics for Univariate RF metrics:")
print(uni_metrics[["MAPE","MSE"]].describe().loc[["mean","50%","std","min","max"]])

# %%
# ──────────────────────────────────────────────────────────────────────────────
# X.B Multivariate Random Forest with Exogenous
# ──────────────────────────────────────────────────────────────────────────────

# 1. PREPARE HISTORICAL & FUTURE EXOGENOUS DATA  # ────────────────────────────────
# Include key covariates (e.g., promotion, holiday, competition, assortment) 
# in both training and forecasting datasets to enable richer modeling.

train_exog = (
    merged
    .loc[merged.ds <= CUTOFF_DATE, 
         ["unique_id","ds","y",
          "open_frac","promo_frac","school_hol_frac",
          "competition_distance","assortment"]]
    .dropna()
)
future_exog = (
    merged
    .loc[
        (merged.ds > CUTOFF_DATE) &
        (merged.ds <= CUTOFF_DATE + pd.Timedelta(weeks=FORECAST_HORIZON)),
        ["unique_id","ds","y",
         "open_frac","promo_frac","school_hol_frac",
         "competition_distance","assortment"]
    ]
)


# 2. ONE-HOT ENCODE CATEGORICAL FEATURES  # ─────────────────────────────────────
# Convert 'assortment' into dummy variables to allow RF to leverage category effects.

cat_cols = ["assortment"]
train_exog  = pd.get_dummies(train_exog,  columns=cat_cols, drop_first=False)
future_exog = pd.get_dummies(future_exog, columns=cat_cols, drop_first=False)


# 3. DEFINE RF & FORECASTING PIPELINE  # ────────────────────────────────────────
# Mirror the univariate setup but now incorporating exogenous predictors.

rf_exog = RandomForestRegressor(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=10,
    random_state=SEED
)
fcst_exog = MLForecast(
    models            = {"rf_exog": rf_exog},
    freq              = WEEKLY_FREQ,
    lags              = [1,2,4,26,52],
    date_features     = ["week","month"],
    target_transforms = [Differences([1])],
    lag_transforms    = {
        1: [RollingMean(window_size=4), RollingMean(window_size=13)]
    }
)

# %%
# 4. FIT & FORECAST WITH EXOGENOUS VARIABLES  # ─────────────────────────────────
fcst_exog.fit(df=train_exog, static_features=[])
exog_forecasts = fcst_exog.predict(h=FORECAST_HORIZON, X_df=future_exog)

# Visual check to ensure forecasts align with historical patterns
StatsForecast.plot(
    df            = train_exog[['unique_id','ds','y']],
    forecasts_df = exog_forecasts,
    max_insample_length = 120
)

# %%
# 5. FEATURE IMPORTANCE VIA PERMUTATION  # ──────────────────────────────────────
# Assess which inputs most impact RF error, highlighting influential exogenous factors.

all_feats  = fcst_exog.preprocess(train_exog, static_features=[])
rf_model   = fcst_exog.models_["rf_exog"]
perm_results = []
for sid in all_feats.unique_id.unique():
    df_s       = all_feats[all_feats.unique_id == sid].copy()
    eval_slice = df_s.tail(26)
    X_eval     = eval_slice.drop(columns=["unique_id","ds","y"])
    y_eval     = eval_slice["y"]
    perm       = permutation_importance(
                    rf_model, X_eval, y_eval,
                    scoring="neg_mean_squared_error",
                    n_repeats=10,
                    random_state=SEED,
                    n_jobs=-1
                  )
    imp_df     = pd.DataFrame({
                    "feature": X_eval.columns,
                    "importance": perm.importances_mean
                 })
    perm_results.append(imp_df)

avg_imp = (
    pd.concat(perm_results)
      .groupby("feature")["importance"].mean()
      .sort_values(ascending=False)
      .reset_index(name="avg_importance")
)
print("Average permutation importance:\n", avg_imp)

plt.figure(figsize=(8, len(avg_imp)*0.3))
plt.barh(avg_imp["feature"], avg_imp["avg_importance"])
plt.gca().invert_yaxis()
plt.title("Permutation Importance (Multivariate RF)")
plt.xlabel("Avg Increase in MSE when Permuted")
plt.tight_layout()
plt.show()

# %%
# 6. CROSS-VALIDATION & EVALUATION FOR MULTIVARIATE RF  # ────────────────────────
cv_exog = fcst_exog.cross_validation(
    df              = train_exog,
    h               = FORECAST_HORIZON,
    step_size       = CV_STEP_WEEKS,
    n_windows       = n_windows - 7,
    static_features = []
)
metrics_df_multi = evaluate(
    df      = cv_exog.drop(columns=["cutoff"]),
    metrics = [ufl.mse, ufl.mape]
)
print("\nMultivariate RF Cross-Validation Results (first 5 rows):")
print(metrics_df_multi)

# %%
# 7. PIVOT METRICS & MERGE METADATA  # ──────────────────────────────────────────
multi_metrics = (
    metrics_df_multi
      .pivot(index="unique_id", columns="metric", values="rf_exog")
      .rename(columns={"mse":"MSE", "mape":"MAPE"})
      .reset_index()
      .merge(
          meta[["unique_id","store_type","assortment"]],
          on="unique_id", how="left"
      )
)


# 8. VISUALIZE MULTIVARIATE RF PERFORMANCE  # ───────────────────────────────────
# Error distributions
plt.figure(); multi_metrics["MAPE"].hist(bins=20)
plt.title("Multivariate RF: MAPE Distribution"); plt.xlabel("MAPE"); plt.ylabel("Count"); plt.show()
plt.figure(); multi_metrics["MSE"].hist(bins=20)
plt.title("Multivariate RF: MSE Distribution"); plt.xlabel("MSE"); plt.ylabel("Count"); plt.show()

# Top/Bottom performers
print("Top-10 Stores by Lowest MAPE (Multivariate RF):")
print(multi_metrics.nsmallest(10, "MAPE")[["unique_id","store_type","assortment","MAPE"]])
print("\nBottom-10 Stores by Highest MAPE (Multivariate RF):")
print(multi_metrics.nlargest(10, "MAPE")[["unique_id","store_type","assortment","MAPE"]])

print("Top-10 Stores by Lowest MSE (Multivariate RF):")
print(multi_metrics.nsmallest(10, "MSE")[["unique_id","store_type","assortment","MSE"]])
print("\nBottom-10 Stores by Highest MSE (Multivariate RF):")
print(multi_metrics.nlargest(10, "MSE")[["unique_id","store_type","assortment","MSE"]])

# Error vs. volume scatter
avg_sales = merged.groupby("unique_id")["y"].mean().rename("mean_y").reset_index()
multi_metrics = multi_metrics.merge(avg_sales, on="unique_id")
plt.figure(); plt.scatter(multi_metrics["mean_y"], multi_metrics["MAPE"], alpha=0.6)
plt.title("Avg Weekly Sales vs. MAPE (Multivariate RF)"); plt.xlabel("Average Weekly Sales"); plt.ylabel("MAPE"); plt.show()
plt.figure(); plt.scatter(multi_metrics["mean_y"], multi_metrics["MSE"], alpha=0.6)
plt.title("Avg Weekly Sales vs. MSE (Multivariate RF)"); plt.xlabel("Average Weekly Sales"); plt.ylabel("MSE"); plt.show()

# Segment “good” proportions
good_mape_mv = (
    multi_metrics
      .assign(is_good=lambda df: df.MAPE < 0.15)
      .groupby(["store_type","assortment"])["is_good"]
      .mean().unstack(fill_value=0)
)
print("Proportion with MAPE < 0.15 by segment (Multivariate RF):\n", good_mape_mv)

median_mse_mv = multi_metrics["MSE"].median()
good_mse_mv = (
    multi_metrics
      .assign(is_good=lambda df: df.MSE < median_mse_mv)
      .groupby(["store_type","assortment"])["is_good"]
      .mean().unstack(fill_value=0)
)
print(f"\nProportion with MSE < {median_mse_mv:.0f} by segment (Multivariate RF):\n", good_mse_mv)

# Summary statistics
print("\nSummary statistics for Multivariate RF metrics:")
print(multi_metrics[["MAPE","MSE"]].describe().loc[["mean","50%","std","min","max"]])


# %%
# ──────────────────────────────────────────────────────────────────────────────
# X.C Compare univariate vs. multivariate
# ──────────────────────────────────────────────────────────────────────────────

# 1. AVERAGE CV METRICS COMPARISON  # ─────────────────────────────────────────────
# Compare overall mean MAPE/MSE to quantify the benefit of adding exogenous features.
avg_uni   = metrics_df_uni.groupby("metric")["rf_uni"].mean().rename("uni_avg")
avg_multi = metrics_df_multi.groupby("metric")["rf_exog"].mean().rename("multi_avg")
print("\nAverage CV metrics comparison:\n", pd.concat([avg_uni, avg_multi], axis=1))


# 2. PER-STORE PERFORMANCE DELTA ANALYSIS  # ─────────────────────────────────────
# Calculate absolute and relative improvements for each store when using multivariate RF.
uni_wide = (
    metrics_df_uni
    .pivot(index="unique_id", columns="metric", values="rf_uni")
    .rename(columns={"mse":"uni_MSE","mape":"uni_MAPE"})
    .reset_index()
)
multi_wide = (
    metrics_df_multi
    .pivot(index="unique_id", columns="metric", values="rf_exog")
    .rename(columns={"mse":"multi_MSE","mape":"multi_MAPE"})
    .reset_index()
)
comparison = (
    uni_wide
    .merge(multi_wide, on="unique_id")
    .assign(
        ΔMSE     = lambda df: df.uni_MSE   - df.multi_MSE,
        ΔMAPE    = lambda df: df.uni_MAPE  - df.multi_MAPE,
        MSE_red  = lambda df: (df.uni_MSE   - df.multi_MSE)  / df.uni_MSE,
        MAPE_red = lambda df: (df.uni_MAPE  - df.multi_MAPE) / df.uni_MAPE
    )
)
print("\nPer-store uni vs. multi comparison (first 5 rows):\n", comparison.head())


# 3. IDENTIFY BEST AND WORST MULTIVARIATE PERFORMERS  # ─────────────────────────
# Highlight the extreme cases to understand where exogenous features help most/least.
mape_df = metrics_df_multi.query("metric == 'mape'").copy()
mape_df = mape_df.rename(columns={"rf_exog":"MAPE"}).sort_values("MAPE")

best_store  = mape_df.iloc[0]
worst_store = mape_df.iloc[-1]

print(f"\nBest (lowest MAPE) multivariate store: {best_store.unique_id} → MAPE = {best_store.MAPE:.3f}")
print(f"Worst (highest MAPE) multivariate store: {worst_store.unique_id} → MAPE = {worst_store.MAPE:.3f}")

# %%
# ================================================================================
# XI. SINGLE‐STORE DEMONSTRATION (WEEKLY)
# ================================================================================

# 1. SELECT A STORE & EXTRACT ITS WEEKLY SERIES  # ───────────────────────────────
store_id = "store_68"
df_store = (
    merged
    .loc[lambda d: d.unique_id == store_id, ["unique_id","ds","y"]]
    .sort_values("ds")
)

# 2. SPLIT INTO IN‐SAMPLE & ACTUAL SERIES  # ──────────────────────────────────────
df_train = df_store.loc[df_store.ds <= CUTOFF_DATE]
y_actual = df_store.set_index("ds")["y"]


# A. AUTOARIMA DEMONSTRATION  # ==================================================

# 3. DEFINE & FIT ONE‐SERIES AutoARIMA  # ────────────────────────────────────────
sf_demo = StatsForecast(
    models=[AutoARIMA(season_length=52, alias="AutoARIMA")],
    freq=WEEKLY_FREQ
)
sf_demo.fit(df=df_train)

# 4. FORECAST NEXT H WEEKS  # ───────────────────────────────────────────────────
fcst_demo = sf_demo.predict(h=FORECAST_HORIZON)

# 5. ONE‐STEP‐AHEAD IN‐SAMPLE FIT VIA CV  # ─────────────────────────────────────
cv1 = sf_demo.cross_validation(
    df         = df_train,
    h          = 1,
    step_size  = 1,
    n_windows  = len(df_train) - 1
)
fitted_insample = (
    cv1
      .query("unique_id == @store_id")
      .set_index("ds")["AutoARIMA"]
      .rename("InSample_Fit")
)

# 6. BUILD OUT‐OF‐SAMPLE FORECAST SERIES  # ────────────────────────────────────
idx_fcst    = pd.date_range(
                  start=CUTOFF_DATE + pd.Timedelta(weeks=1),
                  periods=FORECAST_HORIZON,
                  freq=WEEKLY_FREQ
              )
forecast_8w = pd.Series(
                  fcst_demo.AutoARIMA.values,
                  index=idx_fcst,
                  name="Forecast_8w"
              )

# 7. COMBINE ACTUAL, FIT & FORECAST  # ──────────────────────────────────────────
df_demo = pd.concat([
    y_actual.rename("Actual"),
    fitted_insample,
    forecast_8w
], axis=1)

# 8. PLOT DEMONSTRATION  # ───────────────────────────────────────────────────────
plt.figure(figsize=(12,6))
plt.plot(df_demo.index, df_demo["Actual"],       label="Actual",       color="black", linewidth=2)
plt.plot(df_demo.index, df_demo["InSample_Fit"], label="In‐Sample Fit", linestyle="-.")
plt.plot(df_demo.index, df_demo["Forecast_8w"],  label="8-Week Forecast", linestyle="--")
plt.axvline(CUTOFF_DATE, color="grey", linestyle=":", label="Train/Test Split")
plt.title(f"{store_id} — AutoARIMA (Weekly)")
plt.xlabel("Week")
plt.ylabel("Weekly Sales")
plt.legend()
plt.tight_layout()
plt.show()


# %%
# B. RF MULTIVARIATE DEMONSTRATION  # ==========================================

# 1. PREPROCESS FEATURES FOR IN‐SAMPLE FIT  # ──────────────────────────────────
feats      = fcst_exog.preprocess(train_exog, static_features=[])
feats["y_diff"] = feats.groupby("unique_id")["y"].diff()
feats_diff = feats.dropna(subset=["y_diff"])

# 2. FIT RF ON FIRST‐DIFFERENCE  # ──────────────────────────────────────────────
X_all      = feats_diff.drop(columns=["unique_id","ds","y","y_diff"])
y_diff_all = feats_diff["y_diff"]
rf_demo    = clone(rf_exog)
rf_demo.fit(X_all, y_diff_all)

# 3. PREDICT IN‐SAMPLE DIFFERENCES & INVERT TO LEVELS  # ────────────────────────
df_store   = (
    feats
    .query("unique_id == @store_id")
    .assign(y_diff=lambda d: d.y.diff())
    .dropna(subset=["y_diff"])
    .set_index("ds")
)
X_insample = df_store.drop(columns=["unique_id","y","y_diff"])
diff_pred  = pd.Series(rf_demo.predict(X_insample), index=X_insample.index)
y_prev     = (
    train_exog
    .query("unique_id == @store_id")
    .set_index("ds")["y"]
    .shift(1)
)
in_sample  = (y_prev + diff_pred).rename("InSample_Fit")

# 4. BUILD ACTUAL SERIES & RAW FORECAST  # ─────────────────────────────────────
act_train   = train_exog.query("unique_id == @store_id").set_index("ds")["y"]
act_future  = future_exog.query("unique_id == @store_id").set_index("ds")["y"]
actual      = pd.concat([act_train, act_future]).sort_index().rename("Actual")
forecast_raw= exog_forecasts.query("unique_id == @store_id").set_index("ds")["rf_exog"].rename("Forecast_8w")

# 5. APPLY SMOOTHING & PLOT FINAL RESULTS  # ───────────────────────────────────
SMOOTH_W   = 3
in_sample_sm = in_sample.rolling(window=SMOOTH_W, min_periods=1).mean()
forecast_sm  = forecast_raw.rolling(window=SMOOTH_W, min_periods=1).mean()

plt.figure(figsize=(12,6))
plt.plot(actual.index, actual.values, color="black", lw=2, label="Actual")
plt.plot(in_sample_sm.index, in_sample_sm.values,
         linestyle="-.", label=f"In-Sample Fit (RM{SMOOTH_W})")
plt.plot(forecast_sm.index, forecast_sm.values,
         linestyle="--", label=f"8-Week Forecast (RM{SMOOTH_W})")
plt.axvline(CUTOFF_DATE, color="grey", linestyle=":", label="Train/Test Split")
plt.title(f"{store_id} — RF Multivariate (Smoothed {SMOOTH_W}-wk RM)")
plt.xlabel("Week")
plt.ylabel("Weekly Sales")
plt.legend()
plt.tight_layout()
plt.show()
# %%
# ================================================================================
# XII. FINAL FORECAST GENERATION & MODEL SELECTION
# ================================================================================

# 1. COLLATE CV‐MAPE FOR ALL CANDIDATES  # ───────────────────────────────────────
# Gather the CV‐derived MAPE for each store across baseline, univariate, and multivariate models.
baseline_mape = (
    best_by_store[["unique_id", "Best_MAPE_Model", "MAPE"]]
      .rename(columns={"Best_MAPE_Model": "model", "MAPE": "mape"})
)
uni_mape = (
    metrics_df_uni
      .query("metric == 'mape'")
      .rename(columns={"rf_uni": "mape"})
      .assign(model="rf_uni")[["unique_id","model","mape"]]
)
multi_mape = (
    metrics_df_multi
      .query("metric == 'mape'")
      .rename(columns={"rf_exog": "mape"})
      .assign(model="rf_exog")[["unique_id","model","mape"]]
)

# 2. SELECT BEST‐MAPE MODEL PER STORE  # ─────────────────────────────────────────
# For each store, pick the model with the lowest cross‐validated MAPE.
all_mape = pd.concat([baseline_mape, uni_mape, multi_mape], ignore_index=True)
final_choice = (
    all_mape
      .sort_values(["unique_id","mape"])
      .groupby("unique_id", as_index=False)
      .first()
      .rename(columns={"model": "Chosen_Model", "mape": "Chosen_MAPE"})
)

print("\nFinal Model Choices (first 5 rows):")
print(final_choice.head())
print("\nChosen Model Distribution:")
print(final_choice["Chosen_Model"].value_counts())


# 3. ASSEMBLE CANDIDATE FORECASTS  # ─────────────────────────────────────────────
# Melt baseline forecasts and append RF forecasts to create a unified table.
baseline_fc = (
    full_forecasts
      .melt(
         id_vars=["unique_id","ds"],
         value_vars=[m.alias for m in baseline_models],
         var_name="model",
         value_name="forecast"
      )
)
uni_fc = (
    uni_forecasts
      .rename(columns={"rf_uni":"forecast"})
      .assign(model="rf_uni")[["unique_id","ds","model","forecast"]]
)
multi_fc = (
    exog_forecasts
      .rename(columns={"rf_exog":"forecast"})
      .assign(model="rf_exog")[["unique_id","ds","model","forecast"]]
)
all_fc = pd.concat([baseline_fc, uni_fc, multi_fc], ignore_index=True)

# 4. FILTER TO CHOSEN MODEL FORECASTS  # ────────────────────────────────────────
# Merge to keep only the forecast from each store’s selected model.
final_fc = (
    final_choice[["unique_id","Chosen_Model"]]
      .merge(all_fc,
             left_on=["unique_id","Chosen_Model"],
             right_on=["unique_id","model"],
             how="left")
      [["unique_id","ds","forecast"]]
      .sort_values(["unique_id","ds"])
)

print("\nFinal Forecast DataFrame (first 5 rows):")
print(final_fc.head())
print("\nFinal Forecast Distribution:")
print(final_fc["forecast"].describe())

# %%
# ================================================================================
# XIII. EXPORT RESULTS TO CSV
# ================================================================================

# 1. EXPORT WEEKLY FORECASTS FOR THE NEXT 8 WEEKS
#    -> One row per store per week
final_fc.to_csv("8wk_weekly_forecasts_per_store.csv", index=False)
print("Written 8wk_weekly_forecasts_per_store.csv")

# 2. EXPORT MODEL SELECTION SUMMARY
#    -> Shows the best model for each store (by MAPE)
final_choice.to_csv("model_selection_summary_per_store.csv", index=False)
print("Written model_selection_summary_per_store.csv")

# 3. EXPORT TOTAL SALES FORECAST OVER 8 WEEKS
#    -> Sum of the 8-week forecast for each store
total_8wk = (
    final_fc
    .groupby("unique_id", as_index=False)["forecast"]
    .sum()
    .rename(columns={"forecast": "total_8wk_forecast"})
)
total_8wk.to_csv("total_8wk_sales_forecast_per_store.csv", index=False)
print("Written total_8wk_sales_forecast_per_store.csv")

# 4. GLOBAL AGGREGATE FOR CFO
#    -> Overall total expected sales across all stores
global_total = total_8wk["total_8wk_forecast"].sum()
with open("global_8wk_total_sales.txt", "w") as f:
    f.write(f"Global 8-week sales forecast: {global_total:.2f}\n")
print("Written global_8wk_total_sales.txt")

# %%
# ================================================================================
# XIV. VISUALIZE TOTAL FORECAST
# ================================================================================

# 1. Top 10 stores by total 8-week forecast
top10 = total_8wk.sort_values("total_8wk_forecast", ascending=False).head(10)

plt.figure(figsize=(10, 5))
plt.bar(top10["unique_id"].astype(str), top10["total_8wk_forecast"])
plt.title("Top 10 Stores by Total 8-Week Sales Forecast")
plt.xlabel("Store ID")
plt.ylabel("Total Forecast Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top10_stores_total_forecast.png")
plt.show()

# 2. Aggregated weekly forecast across all stores
agg_weekly = (
    final_fc
    .groupby("ds", as_index=False)["forecast"]
    .sum()
)

plt.figure(figsize=(10, 5))
plt.plot(agg_weekly["ds"], agg_weekly["forecast"], marker="o")
plt.title("Aggregated Sales Forecast Over 8 Weeks (All Stores)")
plt.xlabel("Week")
plt.ylabel("Total Forecast Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("aggregated_weekly_forecast.png")
plt.show()
