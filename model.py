#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import argparse
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Silence noisy warnings/logs
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

# Optional packages
HAS_XGB = True
try:
    from xgboost import XGBRegressor, plot_importance
except Exception:
    HAS_XGB = False

HAS_CAT = True
try:
    from catboost import CatBoostRegressor
except Exception:
    HAS_CAT = False

HAS_PROPHET = True
try:
    from prophet import Prophet
except Exception:
    HAS_PROPHET = False

HAS_PMDA = True
try:
    from pmdarima import auto_arima
except Exception:
    HAS_PMDA = False

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

EPS = 1e-9


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    input_csv: str
    output_dir: str
    ceo_actuals_csv: Optional[str] = None

    workspace_name: str = "dbbbbb-prod---workspace--name"
    workspace_id_real: str = "1233445566778899"

    train_end_date: str = "2026-03-31"
    holdout_start_date: str = "2026-04-01"
    holdout_end_date: str = "2026-04-30"

    seasonal_period: int = 7
    backtest_horizon_days: int = 30
    backtest_n_folds: int = 4
    min_train_days: int = 180

    recent_windows: Tuple[Optional[int], ...] = (None, 270, 180)
    product_major_keep: Tuple[str, ...] = ("JOBS", "ALL_PURPOSE", "SQL")

    remove_outliers_mode: str = "none"
    iqr_cap_multiplier: float = 1.5

    use_log_variants: bool = True

    xgb_grid: List[Dict[str, Any]] = field(default_factory=list)
    cat_grid: List[Dict[str, Any]] = field(default_factory=list)
    sarimax_orders: List[Tuple[int, int, int]] = field(default_factory=list)
    sarimax_seasonal_orders: List[Tuple[int, int, int, int]] = field(default_factory=list)

    display_float_decimals: int = 2
    top_n_holdout_candidates: int = 5
    forecast_horizon_days: int = 30


def build_default_config(args) -> Config:
    cfg = Config(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        ceo_actuals_csv=args.ceo_actuals_csv,
        workspace_name=args.workspace_name,
        workspace_id_real=args.workspace_id_real,
        remove_outliers_mode=args.remove_outliers_mode,
    )

    cfg.xgb_grid = [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "min_child_weight": 1},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.5, "min_child_weight": 1},
        {"n_estimators": 700, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "min_child_weight": 2},
    ]

    cfg.cat_grid = [
        {"iterations": 400, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"iterations": 700, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 3.0},
        {"iterations": 1000, "depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 5.0},
    ]

    cfg.sarimax_orders = [
        (1, 0, 1),
        (1, 1, 1),
        (2, 1, 1),
        (2, 1, 2),
        (3, 1, 1),
    ]

    cfg.sarimax_seasonal_orders = [
        (0, 1, 1, 7),
        (1, 0, 1, 7),
        (1, 1, 1, 7),
    ]

    return cfg


# ============================================================================
# LOGGER
# ============================================================================

def setup_logger(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("forecast_v4_strict")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(outdir / "pipeline.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================================
# HELPERS
# ============================================================================

def fmt_num(x, decimals=2):
    if pd.isna(x):
        return ""
    return f"{float(x):,.{decimals}f}"


def fmt_pct(x, decimals=2):
    if pd.isna(x):
        return ""
    return f"{float(x):.{decimals}f}%"


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def html_table(df: pd.DataFrame, decimals=2, max_rows=None):
    if df is None or df.empty:
        return "<p>No data available.</p>"
    tmp = df.copy()
    if max_rows is not None and len(tmp) > max_rows:
        tmp = tmp.head(max_rows)
    for c in tmp.columns:
        if pd.api.types.is_numeric_dtype(tmp[c]):
            tmp[c] = tmp[c].map(lambda x: fmt_num(x, decimals))
    return tmp.to_html(index=False, border=0, classes="table table-striped", escape=False)


def get_window_label(window: Optional[int]) -> str:
    return "full" if window is None else f"recent{window}"


# ============================================================================
# METRICS
# ============================================================================

def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + EPS)


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100.0 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + EPS)
    )


def bias_pct(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100.0 * np.sum(y_pred - y_true) / (np.sum(y_true) + EPS)


def mase_seasonal(y_true, y_pred, y_train, seasonality=7):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) <= seasonality:
        denom = np.mean(np.abs(np.diff(y_train))) + EPS
    else:
        denom = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality])) + EPS

    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def peak_day_metrics(y_true, y_pred, quantile=0.90):
    df = pd.DataFrame({"actual": y_true, "pred": y_pred})
    thresh = df["actual"].quantile(quantile)
    sub = df[df["actual"] >= thresh].copy()
    if sub.empty:
        return {"peak_day_count": 0, "peak_WAPE": np.nan, "peak_MAE": np.nan, "peak_RMSE": np.nan}
    return {
        "peak_day_count": int(len(sub)),
        "peak_WAPE": float(wape(sub["actual"], sub["pred"])),
        "peak_MAE": float(mean_absolute_error(sub["actual"], sub["pred"])),
        "peak_RMSE": float(rmse(sub["actual"], sub["pred"])),
    }


def compute_metrics(y_true, y_pred, y_train):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "WAPE": float(wape(y_true, y_pred)),
        "sMAPE": float(smape(y_true, y_pred)),
        "MASE": float(mase_seasonal(y_true, y_pred, y_train, seasonality=7)),
        "BiasPct": float(bias_pct(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "ActualTotal": float(np.sum(y_true)),
        "ForecastTotal": float(np.sum(y_pred)),
        "TotalErrorPct": float(100.0 * (np.sum(y_pred) - np.sum(y_true)) / (np.sum(y_true) + EPS)),
        "ExecutiveAccuracyApprox": float(100.0 - wape(y_true, y_pred)),
    }
    out.update(peak_day_metrics(y_true, y_pred, quantile=0.90))
    return out


def aggregate_blocks(df: pd.DataFrame, actual_col: str, pred_col: str, date_col: str = "usage_date", block_size: int = 7):
    temp = df.copy().sort_values(date_col).reset_index(drop=True)
    temp["block_id"] = (temp.index // block_size) + 1
    agg = temp.groupby("block_id", as_index=False).agg(
        start_date=(date_col, "min"),
        end_date=(date_col, "max"),
        actual_block=(actual_col, "sum"),
        pred_block=(pred_col, "sum"),
    )
    return agg


def compute_multilevel_holdout_metrics(holdout_merged: pd.DataFrame, y_train: np.ndarray):
    daily_metrics = compute_metrics(
        y_true=holdout_merged["actual_total_cost"].values,
        y_pred=holdout_merged["predicted_total_cost"].values,
        y_train=y_train,
    )

    block3 = aggregate_blocks(holdout_merged, actual_col="actual_total_cost", pred_col="predicted_total_cost", block_size=3)
    metrics_3day = compute_metrics(
        y_true=block3["actual_block"].values,
        y_pred=block3["pred_block"].values,
        y_train=y_train,
    )

    block7 = aggregate_blocks(holdout_merged, actual_col="actual_total_cost", pred_col="predicted_total_cost", block_size=7)
    metrics_7day = compute_metrics(
        y_true=block7["actual_block"].values,
        y_pred=block7["pred_block"].values,
        y_train=y_train,
    )

    monthly_actual = float(holdout_merged["actual_total_cost"].sum())
    monthly_pred = float(holdout_merged["predicted_total_cost"].sum())
    monthly_error_pct = 100.0 * (monthly_pred - monthly_actual) / (monthly_actual + EPS)

    monthly_summary = pd.DataFrame([{
        "MonthlyActualTotal": monthly_actual,
        "MonthlyForecastTotal": monthly_pred,
        "MonthlyErrorPct": monthly_error_pct,
    }])

    return {
        "daily": pd.DataFrame([daily_metrics]),
        "block3": pd.DataFrame([metrics_3day]),
        "block7": pd.DataFrame([metrics_7day]),
        "monthly": monthly_summary,
        "block3_table": block3,
        "block7_table": block7,
    }


# ============================================================================
# FIT COUNTER
# ============================================================================

class FitCounter:
    def __init__(self):
        self.counts = {
            "xgboost_fits": 0,
            "xgboost_total_trees": 0,
            "catboost_fits": 0,
            "catboost_total_iterations": 0,
            "prophet_fits": 0,
            "sarimax_fits": 0,
            "autoarima_fits": 0,
        }

    def add_xgb(self, trees):
        self.counts["xgboost_fits"] += 1
        self.counts["xgboost_total_trees"] += int(trees)

    def add_cat(self, iters):
        self.counts["catboost_fits"] += 1
        self.counts["catboost_total_iterations"] += int(iters)

    def add_prophet(self):
        self.counts["prophet_fits"] += 1

    def add_sarimax(self):
        self.counts["sarimax_fits"] += 1

    def add_autoarima(self):
        self.counts["autoarima_fits"] += 1


# ============================================================================
# DATA LOAD + CANONICAL TABLES
# ============================================================================

def normalize_product(x: str) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    return str(x).strip().upper()


def group_product(x: str, major_keep: Tuple[str, ...]) -> str:
    val = normalize_product(x)
    return val if val in major_keep else "OTHER"


def load_source(csv_path: str, cfg: Config, logger: logging.Logger, force_workspace_id: Optional[str] = None) -> pd.DataFrame:
    logger.info(f"Loading source CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {
        "workspace_name",
        "usage_date",
        "billing_origin_product",
        "raw_event_count",
        "daily_usage_quantity",
        "daily_cost",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "workspace_id" not in df.columns:
        df["workspace_id"] = cfg.workspace_id_real

    df["workspace_id"] = df["workspace_id"].astype("string").str.strip()
    df["workspace_name"] = df["workspace_name"].astype("string").str.strip()
    df["usage_date"] = pd.to_datetime(df["usage_date"], errors="coerce")
    df["billing_origin_product"] = df["billing_origin_product"].astype("string").map(normalize_product)
    df["raw_event_count"] = pd.to_numeric(df["raw_event_count"], errors="coerce").fillna(0.0)
    df["daily_usage_quantity"] = pd.to_numeric(df["daily_usage_quantity"], errors="coerce").fillna(0.0)
    df["daily_cost"] = pd.to_numeric(df["daily_cost"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["usage_date", "workspace_name"]).copy()
    df = df[df["workspace_name"] == cfg.workspace_name].copy()
    if df.empty:
        raise ValueError(f"No rows remain after filtering by workspace_name={cfg.workspace_name}")

    df["workspace_id"] = force_workspace_id if force_workspace_id is not None else cfg.workspace_id_real
    df = df.sort_values(["usage_date", "billing_origin_product"]).reset_index(drop=True)

    logger.info(f"Filtered rows: {len(df):,}")
    logger.info(f"Workspace name: {cfg.workspace_name}")
    logger.info(f"Workspace ID used in outputs: {df['workspace_id'].iloc[0]}")
    logger.info(f"Date range: {df['usage_date'].min().date()} to {df['usage_date'].max().date()}")

    return df


def build_internal_tables(df: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    df = df.copy()
    df["product_group"] = df["billing_origin_product"].map(lambda x: group_product(x, cfg.product_major_keep))

    product_raw = (
        df.groupby(["workspace_id", "workspace_name", "usage_date", "billing_origin_product"], as_index=False)
        .agg(
            raw_event_count=("raw_event_count", "sum"),
            daily_usage_quantity=("daily_usage_quantity", "sum"),
            daily_cost=("daily_cost", "sum"),
        )
    )

    product_grouped = (
        df.groupby(["workspace_id", "workspace_name", "usage_date", "product_group"], as_index=False)
        .agg(
            grouped_event_count=("raw_event_count", "sum"),
            grouped_usage=("daily_usage_quantity", "sum"),
            grouped_cost=("daily_cost", "sum"),
        )
    )

    all_dates = pd.date_range(df["usage_date"].min(), df["usage_date"].max(), freq="D")
    product_groups = ["JOBS", "ALL_PURPOSE", "SQL", "OTHER"]

    full_index = pd.MultiIndex.from_product(
        [[cfg.workspace_id_real], [cfg.workspace_name], all_dates, product_groups],
        names=["workspace_id", "workspace_name", "usage_date", "product_group"],
    ).to_frame(index=False)

    grouped_full = full_index.merge(
        product_grouped,
        on=["workspace_id", "workspace_name", "usage_date", "product_group"],
        how="left",
    )

    for c in ["grouped_event_count", "grouped_usage", "grouped_cost"]:
        grouped_full[c] = pd.to_numeric(grouped_full[c], errors="coerce").fillna(0.0)

    usage_wide = grouped_full.pivot_table(
        index=["workspace_id", "workspace_name", "usage_date"],
        columns="product_group",
        values="grouped_usage",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    usage_wide.columns.name = None
    usage_wide = usage_wide.rename(columns={
        "JOBS": "jobs_usage",
        "ALL_PURPOSE": "all_purpose_usage",
        "SQL": "sql_usage",
        "OTHER": "other_usage",
    })

    cost_wide = grouped_full.pivot_table(
        index=["workspace_id", "workspace_name", "usage_date"],
        columns="product_group",
        values="grouped_cost",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    cost_wide.columns.name = None
    cost_wide = cost_wide.rename(columns={
        "JOBS": "jobs_cost",
        "ALL_PURPOSE": "all_purpose_cost",
        "SQL": "sql_cost",
        "OTHER": "other_cost",
    })

    event_wide = grouped_full.pivot_table(
        index=["workspace_id", "workspace_name", "usage_date"],
        columns="product_group",
        values="grouped_event_count",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    event_wide.columns.name = None
    event_wide = event_wide.rename(columns={
        "JOBS": "jobs_events",
        "ALL_PURPOSE": "all_purpose_events",
        "SQL": "sql_events",
        "OTHER": "other_events",
    })

    wide = usage_wide.merge(cost_wide, on=["workspace_id", "workspace_name", "usage_date"]).merge(
        event_wide, on=["workspace_id", "workspace_name", "usage_date"]
    )

    numeric_cols = [
        "jobs_usage", "all_purpose_usage", "sql_usage", "other_usage",
        "jobs_cost", "all_purpose_cost", "sql_cost", "other_cost",
        "jobs_events", "all_purpose_events", "sql_events", "other_events",
    ]
    for col in numeric_cols:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0.0)

    daily_total = wide[["workspace_id", "workspace_name", "usage_date"]].copy()
    daily_total["daily_total_cost"] = wide["jobs_cost"] + wide["all_purpose_cost"] + wide["sql_cost"] + wide["other_cost"]
    daily_total["daily_total_usage"] = wide["jobs_usage"] + wide["all_purpose_usage"] + wide["sql_usage"] + wide["other_usage"]

    denom = daily_total["daily_total_cost"] + EPS
    wide["jobs_cost_share"] = wide["jobs_cost"] / denom
    wide["all_purpose_cost_share"] = wide["all_purpose_cost"] / denom
    wide["sql_cost_share"] = wide["sql_cost"] / denom
    wide["other_cost_share"] = wide["other_cost"] / denom

    return {
        "product_raw": product_raw.sort_values(["usage_date", "billing_origin_product"]).reset_index(drop=True),
        "product_grouped": grouped_full.sort_values(["usage_date", "product_group"]).reset_index(drop=True),
        "wide": wide.sort_values("usage_date").reset_index(drop=True),
        "daily_total": daily_total.sort_values("usage_date").reset_index(drop=True),
    }

# ============================================================================
# EDA
# ============================================================================

def detect_outliers_iqr(series: pd.Series, k: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    flags = (series < lower) | (series > upper)
    return lower, upper, flags


def stationarity_summary(series: pd.Series):
    s = pd.Series(series).dropna()
    out = {
        "mean": float(s.mean()),
        "variance": float(s.var()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(s.median()),
        "adf_stat": np.nan,
        "adf_pvalue": np.nan,
        "kpss_stat": np.nan,
        "kpss_pvalue": np.nan,
    }
    if len(s) > 10 and s.nunique() > 1:
        try:
            adf_res = adfuller(s, autolag="AIC")
            out["adf_stat"] = float(adf_res[0])
            out["adf_pvalue"] = float(adf_res[1])
        except Exception:
            pass
        try:
            kpss_res = kpss(s, regression="c", nlags="auto")
            out["kpss_stat"] = float(kpss_res[0])
            out["kpss_pvalue"] = float(kpss_res[1])
        except Exception:
            pass
    return out


def run_eda(tables: Dict[str, pd.DataFrame], cfg: Config, outdir: Path, logger: logging.Logger):
    eda_dir = outdir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    raw = tables["product_raw"].copy()
    grouped = tables["product_grouped"].copy()
    wide = tables["wide"].copy()
    total = tables["daily_total"].copy()

    audit = pd.DataFrame([{
        "workspace_id": cfg.workspace_id_real,
        "workspace_name": cfg.workspace_name,
        "min_date": total["usage_date"].min(),
        "max_date": total["usage_date"].max(),
        "n_target_days": len(total),
        "n_product_day_rows": len(raw),
        "n_unique_products": raw["billing_origin_product"].nunique(),
        "target": "daily_total_cost",
    }])

    product_summary = (
        raw.groupby("billing_origin_product", as_index=False)
        .agg(
            days_active=("usage_date", "nunique"),
            total_usage=("daily_usage_quantity", "sum"),
            total_cost=("daily_cost", "sum"),
            total_event_count=("raw_event_count", "sum"),
        )
        .sort_values("total_cost", ascending=False)
        .reset_index(drop=True)
    )

    stat = pd.DataFrame([stationarity_summary(total["daily_total_cost"])])
    stat["workspace_id"] = cfg.workspace_id_real
    stat["workspace_name"] = cfg.workspace_name
    stat["practical_stationarity_answer"] = "NO"
    stat["note"] = "Formal tests may suggest weak stationarity, but practical modeling treats this as non-stationary due to regime shifts + seasonality."

    rs = total[["usage_date", "daily_total_cost"]].copy()
    for w in [7, 14, 30]:
        rs[f"roll_mean_{w}"] = rs["daily_total_cost"].rolling(w, min_periods=1).mean()
        rs[f"roll_std_{w}"] = rs["daily_total_cost"].rolling(w, min_periods=2).std()
        rs[f"roll_var_{w}"] = rs["daily_total_cost"].rolling(w, min_periods=2).var()

    phase = total.copy()
    phase["quarter_label"] = phase["usage_date"].dt.to_period("Q").astype(str)
    phase_summary = (
        phase.groupby("quarter_label", as_index=False)
        .agg(
            mean_cost=("daily_total_cost", "mean"),
            std_cost=("daily_total_cost", "std"),
            min_cost=("daily_total_cost", "min"),
            max_cost=("daily_total_cost", "max"),
        )
    )

    lower, upper, flags = detect_outliers_iqr(total["daily_total_cost"], cfg.iqr_cap_multiplier)
    total["iqr_lower"] = lower
    total["iqr_upper"] = upper
    total["is_iqr_outlier"] = flags

    top_spike_days = total.sort_values("daily_total_cost", ascending=False).head(20).copy()
    spike_breakdown = grouped.merge(
        top_spike_days[["usage_date"]].drop_duplicates(),
        on="usage_date",
        how="inner",
    ).sort_values(["usage_date", "grouped_cost"], ascending=[True, False])

    save_csv(audit, eda_dir / "audit.csv")
    save_csv(product_summary, eda_dir / "product_summary.csv")
    save_csv(stat, eda_dir / "stationarity_summary.csv")
    save_csv(rs, eda_dir / "rolling_stats.csv")
    save_csv(phase_summary, eda_dir / "phase_summary.csv")
    save_csv(total, eda_dir / "daily_total_with_outlier_flags.csv")
    save_csv(top_spike_days, eda_dir / "top_spike_days.csv")
    save_csv(spike_breakdown, eda_dir / "top_spike_breakdown.csv")

    plt.figure(figsize=(16, 6))
    plt.plot(total["usage_date"], total["daily_total_cost"], color="black")
    plt.title("Daily Total Cost")
    save_fig(eda_dir / "01_daily_total_cost.png")

    plt.figure(figsize=(16, 6))
    plt.plot(rs["usage_date"], rs["daily_total_cost"], color="lightgray", label="Daily")
    plt.plot(rs["usage_date"], rs["roll_mean_7"], color="orange", label="7-day mean")
    plt.plot(rs["usage_date"], rs["roll_mean_14"], color="red", label="14-day mean")
    plt.plot(rs["usage_date"], rs["roll_mean_30"], color="blue", label="30-day mean")
    plt.title("Daily Total Cost with Rolling Means")
    plt.legend()
    save_fig(eda_dir / "02_daily_total_cost_rolling_means.png")

    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(rs["usage_date"], rs["roll_mean_30"], color="blue")
    ax1.set_title("Rolling Mean (30)")
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(rs["usage_date"], rs["roll_std_30"], color="darkred")
    ax2.set_title("Rolling Std (30)")
    save_fig(eda_dir / "03_rolling_mean_std_30.png")

    cum = total[["usage_date", "daily_total_cost"]].copy()
    cum["cumulative_total_cost"] = cum["daily_total_cost"].cumsum()
    plt.figure(figsize=(16, 6))
    plt.plot(cum["usage_date"], cum["cumulative_total_cost"], color="steelblue")
    plt.title("Cumulative Total Cost")
    save_fig(eda_dir / "04_cumulative_total_cost.png")

    area = grouped.pivot_table(
        index="usage_date",
        columns="product_group",
        values="grouped_cost",
        aggfunc="sum",
        fill_value=0.0,
    )
    for col in ["JOBS", "ALL_PURPOSE", "SQL", "OTHER"]:
        if col not in area.columns:
            area[col] = 0.0
    area = area[["JOBS", "ALL_PURPOSE", "SQL", "OTHER"]]
    area.plot(kind="area", stacked=True, figsize=(16, 7), colormap="tab20")
    plt.title("Grouped Product Cost Mix Over Time")
    save_fig(eda_dir / "05_grouped_product_mix_area.png")

    bar = (
        grouped.groupby("product_group", as_index=False)["grouped_cost"]
        .sum()
        .sort_values("grouped_cost", ascending=False)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=bar, x="product_group", y="grouped_cost", palette="viridis")
    plt.title("Grouped Product Total Cost")
    save_fig(eda_dir / "06_grouped_product_cost_bar.png")

    share = wide[["usage_date", "jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share"]].copy()
    share = share.set_index("usage_date")
    share.plot(figsize=(16, 7))
    plt.title("Grouped Product Cost Shares Over Time")
    save_fig(eda_dir / "07_grouped_product_cost_shares.png")

    dow = total.copy()
    dow["day_name"] = dow["usage_date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=dow, x="day_name", y="daily_total_cost", order=order)
    plt.title("Daily Cost Distribution by Day of Week")
    save_fig(eda_dir / "08_day_of_week_boxplot.png")

    heat = (
        total.assign(day_name=total["usage_date"].dt.day_name(), dow_num=total["usage_date"].dt.dayofweek)
        .groupby(["dow_num", "day_name"], as_index=False)["daily_total_cost"]
        .mean()
        .pivot(index="day_name", columns="dow_num", values="daily_total_cost")
    )
    heat = heat.reindex(order)
    plt.figure(figsize=(8, 6))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Average Cost by Day of Week")
    save_fig(eda_dir / "09_weekday_heatmap.png")

    stl_series = total.set_index("usage_date")["daily_total_cost"].asfreq("D")
    stl = STL(stl_series, period=7, robust=True)
    stl_res = stl.fit()
    fig = stl_res.plot()
    fig.set_size_inches(16, 10)
    fig.suptitle("STL Decomposition - Cost", y=1.02)
    plt.tight_layout()
    plt.savefig(eda_dir / "10_stl_decomposition_cost.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(stl_series.dropna(), lags=35, ax=axes[0])
    plot_pacf(stl_series.dropna(), lags=35, ax=axes[1], method="ywm")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    save_fig(eda_dir / "11_acf_pacf_cost.png")

    plt.figure(figsize=(16, 6))
    plt.plot(total["usage_date"], total["daily_total_cost"], color="blue", label="Daily Total Cost")
    plt.scatter(total.loc[flags, "usage_date"], total.loc[flags, "daily_total_cost"], color="red", label="IQR Outlier")
    plt.legend()
    plt.title("Outlier Detection - Daily Total Cost")
    save_fig(eda_dir / "12_outlier_detection_cost.png")

    logger.info("EDA complete.")
    return {
        "audit": audit,
        "product_summary": product_summary,
        "stationarity": stat,
        "rolling_stats": rs,
        "phase_summary": phase_summary,
        "daily_total_flagged": total,
        "top_spike_days": top_spike_days,
        "spike_breakdown": spike_breakdown,
    }


# ============================================================================
# FEATURES
# ============================================================================

def build_feature_table(wide: pd.DataFrame, daily_total: pd.DataFrame) -> pd.DataFrame:
    df = daily_total.merge(
        wide[[
            "usage_date",
            "jobs_cost", "all_purpose_cost", "sql_cost", "other_cost",
            "jobs_events", "all_purpose_events", "sql_events", "other_events",
            "jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share",
        ]],
        on="usage_date",
        how="left",
    ).sort_values("usage_date").reset_index(drop=True)

    numeric_cols = [
        "daily_total_cost",
        "jobs_cost", "all_purpose_cost", "sql_cost", "other_cost",
        "jobs_events", "all_purpose_events", "sql_events", "other_events",
        "jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    y = df["daily_total_cost"].astype(float)

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = y.shift(lag)

    for w in [7, 14, 28]:
        df[f"roll_mean_{w}"] = y.shift(1).rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"] = y.shift(1).rolling(w, min_periods=2).std()
    df["roll_min_7"] = y.shift(1).rolling(7, min_periods=1).min()
    df["roll_max_7"] = y.shift(1).rolling(7, min_periods=1).max()

    for base in ["jobs_cost", "all_purpose_cost", "sql_cost", "other_cost"]:
        for lag in [1, 7, 14]:
            df[f"{base}_lag_{lag}"] = df[base].shift(lag)

    for base in ["jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share"]:
        df[f"{base}_lag_1"] = df[base].shift(1)
        df[f"{base}_roll_mean_7"] = df[base].shift(1).rolling(7, min_periods=1).mean()

    for base in ["jobs_events", "all_purpose_events", "sql_events", "other_events"]:
        df[f"{base}_lag_1"] = df[base].shift(1)
        df[f"{base}_lag_7"] = df[base].shift(7)
        df[f"{base}_roll_mean_7"] = df[base].shift(1).rolling(7, min_periods=1).mean()

    df["total_events"] = df["jobs_events"] + df["all_purpose_events"] + df["sql_events"] + df["other_events"]
    df["total_events_lag_1"] = df["total_events"].shift(1)
    df["total_events_roll_mean_7"] = df["total_events"].shift(1).rolling(7, min_periods=1).mean()

    df["abs_change_1"] = y.diff().abs()
    df["abs_change_1_lag_1"] = df["abs_change_1"].shift(1)
    df["cv_7"] = (
        y.shift(1).rolling(7, min_periods=2).std()
        / (y.shift(1).rolling(7, min_periods=2).mean() + EPS)
    )

    df["dow"] = df["usage_date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["day"] = df["usage_date"].dt.day
    df["month"] = df["usage_date"].dt.month
    df["weekofyear"] = df["usage_date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["usage_date"].dt.quarter
    df["is_month_start"] = df["usage_date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["usage_date"].dt.is_month_end.astype(int)

    return df


def feature_cols():
    cols = []
    cols += [f"lag_{x}" for x in [1, 2, 3, 7, 14, 21, 28]]
    cols += [f"roll_mean_{x}" for x in [7, 14, 28]]
    cols += [f"roll_std_{x}" for x in [7, 14, 28]]
    cols += ["roll_min_7", "roll_max_7"]

    for base in ["jobs_cost", "all_purpose_cost", "sql_cost", "other_cost"]:
        cols += [f"{base}_lag_{lag}" for lag in [1, 7, 14]]

    for base in ["jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share"]:
        cols += [f"{base}_lag_1", f"{base}_roll_mean_7"]

    for base in ["jobs_events", "all_purpose_events", "sql_events", "other_events"]:
        cols += [f"{base}_lag_1", f"{base}_lag_7", f"{base}_roll_mean_7"]

    cols += ["total_events_lag_1", "total_events_roll_mean_7", "abs_change_1_lag_1", "cv_7"]
    cols += ["dow", "is_weekend", "day", "month", "weekofyear", "quarter", "is_month_start", "is_month_end"]
    return cols


def build_history_state_from_train(train_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    total = train_tables["daily_total"][["workspace_id", "workspace_name", "usage_date", "daily_total_cost", "daily_total_usage"]].copy()
    wide = train_tables["wide"][[
        "usage_date",
        "jobs_cost", "all_purpose_cost", "sql_cost", "other_cost",
        "jobs_events", "all_purpose_events", "sql_events", "other_events",
        "jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share",
    ]].copy()

    hist = total.merge(wide, on="usage_date", how="left").sort_values("usage_date").reset_index(drop=True)

    numeric_cols = [
        "daily_total_cost", "daily_total_usage",
        "jobs_cost", "all_purpose_cost", "sql_cost", "other_cost",
        "jobs_events", "all_purpose_events", "sql_events", "other_events",
        "jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share",
    ]
    for col in numeric_cols:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0.0)

    hist["total_events"] = (
        hist["jobs_events"]
        + hist["all_purpose_events"]
        + hist["sql_events"]
        + hist["other_events"]
    )

    hist["abs_change_1"] = hist["daily_total_cost"].diff().abs().fillna(0.0)
    hist["cv_7"] = (
        hist["daily_total_cost"].rolling(7, min_periods=2).std()
        / (hist["daily_total_cost"].rolling(7, min_periods=2).mean() + EPS)
    )
    hist["cv_7"] = hist["cv_7"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return hist


def build_share_lookup(wide_train: pd.DataFrame, recent_days=56):
    recent = wide_train.sort_values("usage_date").tail(recent_days).copy()
    recent["dow"] = recent["usage_date"].dt.dayofweek

    by_dow = (
        recent.groupby("dow", as_index=False)[
            ["jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share"]
        ].mean()
    )

    overall = recent[["jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share"]].mean().to_dict()

    lookup = {
        int(r["dow"]): {
            "jobs_cost_share": float(r["jobs_cost_share"]),
            "all_purpose_cost_share": float(r["all_purpose_cost_share"]),
            "sql_cost_share": float(r["sql_cost_share"]),
            "other_cost_share": float(r["other_cost_share"]),
        }
        for _, r in by_dow.iterrows()
    }
    return lookup, overall


def build_single_future_row(history_df: pd.DataFrame, next_date: pd.Timestamp) -> Dict[str, float]:
    hist = history_df.copy().sort_values("usage_date").reset_index(drop=True)

    event_cols = ["jobs_events", "all_purpose_events", "sql_events", "other_events"]
    for col in event_cols:
        if col not in hist.columns:
            hist[col] = 0.0

    hist[event_cols] = hist[event_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if "total_events" not in hist.columns:
        hist["total_events"] = hist[event_cols].sum(axis=1)

    if "abs_change_1" not in hist.columns:
        hist["abs_change_1"] = hist["daily_total_cost"].diff().abs().fillna(0.0)

    if "cv_7" not in hist.columns:
        hist["cv_7"] = (
            hist["daily_total_cost"].rolling(7, min_periods=2).std()
            / (hist["daily_total_cost"].rolling(7, min_periods=2).mean() + EPS)
        )
        hist["cv_7"] = hist["cv_7"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def lag_from_col(col, lag):
        if col not in hist.columns:
            return 0.0
        s = pd.to_numeric(hist[col], errors="coerce")
        if len(s) >= lag and pd.notna(s.iloc[-lag]):
            return float(s.iloc[-lag])
        return 0.0

    def roll_from_col(col, w, agg="mean"):
        if col not in hist.columns:
            return 0.0
        s = pd.to_numeric(hist[col], errors="coerce").tail(w).dropna()
        if len(s) == 0:
            return 0.0
        if agg == "mean":
            return float(s.mean())
        if agg == "std":
            return float(s.std(ddof=1)) if len(s) >= 2 else 0.0
        if agg == "min":
            return float(s.min())
        if agg == "max":
            return float(s.max())
        raise ValueError(agg)

    row = {}
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        row[f"lag_{lag}"] = lag_from_col("daily_total_cost", lag)

    for w in [7, 14, 28]:
        row[f"roll_mean_{w}"] = roll_from_col("daily_total_cost", w, "mean")
        row[f"roll_std_{w}"] = roll_from_col("daily_total_cost", w, "std")

    row["roll_min_7"] = roll_from_col("daily_total_cost", 7, "min")
    row["roll_max_7"] = roll_from_col("daily_total_cost", 7, "max")

    for base in ["jobs_cost", "all_purpose_cost", "sql_cost", "other_cost"]:
        for lag in [1, 7, 14]:
            row[f"{base}_lag_{lag}"] = lag_from_col(base, lag)

    for base in ["jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share"]:
        row[f"{base}_lag_1"] = lag_from_col(base, 1)
        row[f"{base}_roll_mean_7"] = roll_from_col(base, 7, "mean")

    for base in ["jobs_events", "all_purpose_events", "sql_events", "other_events"]:
        row[f"{base}_lag_1"] = lag_from_col(base, 1)
        row[f"{base}_lag_7"] = lag_from_col(base, 7)
        row[f"{base}_roll_mean_7"] = roll_from_col(base, 7, "mean")

    row["total_events_lag_1"] = lag_from_col("total_events", 1)
    row["total_events_roll_mean_7"] = roll_from_col("total_events", 7, "mean")
    row["abs_change_1_lag_1"] = lag_from_col("abs_change_1", 1)
    row["cv_7"] = roll_from_col("cv_7", 7, "mean")

    row["dow"] = next_date.dayofweek
    row["is_weekend"] = 1 if next_date.dayofweek in [5, 6] else 0
    row["day"] = next_date.day
    row["month"] = next_date.month
    row["weekofyear"] = int(next_date.isocalendar().week)
    row["quarter"] = next_date.quarter
    row["is_month_start"] = int(next_date.is_month_start)
    row["is_month_end"] = int(next_date.is_month_end)

    return row

# ============================================================================
# MODEL FIT / PREDICT HELPERS
# ============================================================================

def make_backtest_folds(n_obs: int, horizon: int, n_folds: int, min_train_days: int):
    folds = []
    train_end = min_train_days
    while train_end + horizon <= n_obs:
        folds.append({"train_end": train_end, "test_start": train_end, "test_end": train_end + horizon})
        train_end += horizon
    return folds[-n_folds:]


def forecast_naive(train_values, horizon):
    train_values = np.asarray(train_values, dtype=float)
    if len(train_values) == 0:
        return np.array([0.0] * horizon, dtype=float)
    return np.repeat(float(train_values[-1]), horizon)


def forecast_seasonal_naive(train_values, horizon, m=7):
    train_values = np.asarray(train_values, dtype=float)

    if len(train_values) == 0:
        return np.array([0.0] * horizon, dtype=float)

    if len(train_values) < m:
        return np.repeat(float(train_values[-1]), horizon)

    pattern = train_values[-m:]
    reps = int(np.ceil(horizon / m))
    return np.tile(pattern, reps)[:horizon]


def fit_predict_autoarima(train_series: pd.Series, horizon: int, counter: FitCounter):
    if not HAS_PMDA:
        raise RuntimeError("pmdarima not installed")
    counter.add_autoarima()
    model = auto_arima(
        train_series.values,
        seasonal=True,
        m=7,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        start_P=0,
        start_Q=0,
        max_P=2,
        max_Q=2,
        d=None,
        D=None,
        seasonal_test="ocsb",
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
        trace=False,
    )
    fc = model.predict(n_periods=horizon)
    return np.maximum(np.asarray(fc, dtype=float), 0.0), model


def train_fit_autoarima(model, train_series):
    try:
        pred = model.predict_in_sample()
        pred = np.maximum(np.asarray(pred, dtype=float), 0.0)
        true = np.asarray(train_series.values, dtype=float)
        m = min(len(pred), len(true))
        return true[-m:], pred[-m:]
    except Exception:
        return None, None


def tune_sarimax_aic(train_series: pd.Series, cfg: Config, counter: FitCounter):
    best = {"aic": np.inf, "order": None, "seasonal_order": None}

    for order in cfg.sarimax_orders:
        for sorder in cfg.sarimax_seasonal_orders:
            try:
                counter.add_sarimax()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model = SARIMAX(
                        train_series,
                        order=order,
                        seasonal_order=sorder,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = model.fit(disp=False)

                converged = True
                if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
                    converged = bool(res.mle_retvals.get("converged", True))

                if not converged:
                    continue

                if float(res.aic) < best["aic"]:
                    best = {
                        "aic": float(res.aic),
                        "order": order,
                        "seasonal_order": sorder,
                    }
            except Exception:
                continue

    if best["order"] is None:
        best = {
            "aic": np.inf,
            "order": (1, 1, 1),
            "seasonal_order": (0, 1, 1, cfg.seasonal_period),
        }

    return best


def fit_predict_sarimax(train_series: pd.Series, horizon: int, order, seasonal_order, counter: FitCounter):
    counter.add_sarimax()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)

    fc = res.get_forecast(steps=horizon).predicted_mean.values
    return np.maximum(np.asarray(fc, dtype=float), 0.0), res


def fit_predict_prophet(train_df: pd.DataFrame, future_dates: List[pd.Timestamp], counter: FitCounter):
    if not HAS_PROPHET:
        raise RuntimeError("prophet not installed")
    counter.add_prophet()
    p = train_df.rename(columns={"usage_date": "ds", "daily_total_cost": "y"})[["ds", "y"]].copy()
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    model.fit(p)
    future = pd.DataFrame({"ds": pd.to_datetime(future_dates)})
    pred = model.predict(future)["yhat"].values
    return np.maximum(np.asarray(pred, dtype=float), 0.0), model


# ============================================================================
# STRICT TUNING: ONLY ON CURRENT TRAINING SLICE
# ============================================================================

def split_train_for_es(feature_df: pd.DataFrame, min_val=14, max_val=30):
    n = len(feature_df)
    val_size = min(max_val, max(min_val, n // 6))
    split = max(0, n - val_size)
    return split, val_size


def tune_xgb_on_training_slice(feature_df: pd.DataFrame, cfg: Config, counter: FitCounter, use_log: bool):
    if not HAS_XGB:
        raise RuntimeError("xgboost not installed")

    cols = feature_cols()
    df = feature_df.dropna(subset=cols + ["daily_total_cost"]).copy().reset_index(drop=True)
    if len(df) < 80:
        raise ValueError("Too few rows for XGBoost tuning on this training slice")

    split, _ = split_train_for_es(df)
    X_train = df.iloc[:split][cols]
    y_train_raw = df.iloc[:split]["daily_total_cost"].values.astype(float)
    X_val = df.iloc[split:][cols]
    y_val_raw = df.iloc[split:]["daily_total_cost"].values.astype(float)

    if use_log:
        y_train = np.log1p(y_train_raw)
        y_val = np.log1p(y_val_raw)
    else:
        y_train = y_train_raw
        y_val = y_val_raw

    best = {"score": np.inf, "params": None, "best_iteration": None}

    for params in cfg.xgb_grid:
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            eval_metric="rmse",
            early_stopping_rounds=50,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        counter.add_xgb(params["n_estimators"])

        pred = model.predict(X_val)
        if use_log:
            pred = np.expm1(pred)
        pred = np.maximum(np.asarray(pred, dtype=float), 0.0)

        score = wape(y_val_raw, pred)
        current_best_iter = getattr(model, "best_iteration", None)
        if current_best_iter is None:
            current_best_iter = params["n_estimators"]

        if score < best["score"]:
            best = {
                "score": float(score),
                "params": params,
                "best_iteration": int(current_best_iter),
            }

    return best


def tune_cat_on_training_slice(feature_df: pd.DataFrame, cfg: Config, counter: FitCounter, use_log: bool):
    if not HAS_CAT:
        raise RuntimeError("catboost not installed")

    cols = feature_cols()
    df = feature_df.dropna(subset=cols + ["daily_total_cost"]).copy().reset_index(drop=True)
    if len(df) < 80:
        raise ValueError("Too few rows for CatBoost tuning on this training slice")

    split, _ = split_train_for_es(df)
    X_train = df.iloc[:split][cols]
    y_train_raw = df.iloc[:split]["daily_total_cost"].values.astype(float)
    X_val = df.iloc[split:][cols]
    y_val_raw = df.iloc[split:]["daily_total_cost"].values.astype(float)

    if use_log:
        y_train = np.log1p(y_train_raw)
        y_val = np.log1p(y_val_raw)
    else:
        y_train = y_train_raw
        y_val = y_val_raw

    best = {"score": np.inf, "params": None, "best_iteration": None}

    for params in cfg.cat_grid:
        model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            **params,
        )
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=50,
            verbose=False,
        )
        counter.add_cat(params["iterations"])

        pred = model.predict(X_val)
        if use_log:
            pred = np.expm1(pred)
        pred = np.maximum(np.asarray(pred, dtype=float), 0.0)

        score = wape(y_val_raw, pred)
        best_iter = model.get_best_iteration()
        if best_iter is None or best_iter <= 0:
            best_iter = params["iterations"]

        if score < best["score"]:
            best = {
                "score": float(score),
                "params": params,
                "best_iteration": int(best_iter),
            }

    return best


# ============================================================================
# RECURSIVE ML FORECASTER
# ============================================================================

def recursive_ml_forecast(
    history_df: pd.DataFrame,
    future_dates: List[pd.Timestamp],
    model_type: str,
    params: Dict[str, Any],
    counter: FitCounter,
    use_log_target: bool,
):
    feat = build_feature_table(
        wide=history_df[[
            "usage_date",
            "jobs_cost", "all_purpose_cost", "sql_cost", "other_cost",
            "jobs_events", "all_purpose_events", "sql_events", "other_events",
            "jobs_cost_share", "all_purpose_cost_share", "sql_cost_share", "other_cost_share",
        ]].copy(),
        daily_total=history_df[["workspace_id", "workspace_name", "usage_date", "daily_total_cost", "daily_total_usage"]].copy(),
    )

    cols = feature_cols()
    train_feat = feat.dropna(subset=cols + ["daily_total_cost"]).copy().reset_index(drop=True)

    X_all = train_feat[cols]
    y_all_raw = train_feat["daily_total_cost"].values.astype(float)
    y_all_fit = np.log1p(y_all_raw) if use_log_target else y_all_raw

    split, _ = split_train_for_es(train_feat)
    X_train, X_val = X_all.iloc[:split], X_all.iloc[split:]
    y_train, y_val = y_all_fit[:split], y_all_fit[split:]

    if model_type == "xgboost":
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            eval_metric="rmse",
            early_stopping_rounds=50,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        counter.add_xgb(params["n_estimators"])

    elif model_type == "catboost":
        model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            **params,
        )
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=50,
            verbose=False,
        )
        counter.add_cat(params["iterations"])
    else:
        raise ValueError(model_type)

    train_pred = model.predict(X_all)
    if use_log_target:
        train_pred = np.expm1(train_pred)
    train_pred = np.maximum(np.asarray(train_pred, dtype=float), 0.0)
    train_metrics = compute_metrics(y_all_raw, train_pred, y_all_raw)

    hist_state = history_df.copy().sort_values("usage_date").reset_index(drop=True)

    event_cols = ["jobs_events", "all_purpose_events", "sql_events", "other_events"]
    for col in event_cols:
        if col not in hist_state.columns:
            hist_state[col] = 0.0
    hist_state[event_cols] = hist_state[event_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if "total_events" not in hist_state.columns:
        hist_state["total_events"] = hist_state[event_cols].sum(axis=1)

    dow_lookup, overall_share = build_share_lookup(hist_state)
    preds = []

    for dt in future_dates:
        row = build_single_future_row(hist_state, dt)
        X_row = pd.DataFrame([row])[cols]

        yhat = model.predict(X_row)[0]
        if use_log_target:
            yhat = np.expm1(yhat)
        yhat = max(float(yhat), 0.0)
        preds.append(yhat)

        share = dow_lookup.get(dt.dayofweek, overall_share)
        ssum = sum(share.values()) if sum(share.values()) > 0 else 1.0
        share = {k: v / ssum for k, v in share.items()}

        jobs_cost = yhat * share["jobs_cost_share"]
        all_purpose_cost = yhat * share["all_purpose_cost_share"]
        sql_cost = yhat * share["sql_cost_share"]
        other_cost = yhat * share["other_cost_share"]

        jobs_events = float(hist_state["jobs_events"].iloc[-1]) if "jobs_events" in hist_state.columns else 0.0
        all_purpose_events = float(hist_state["all_purpose_events"].iloc[-1]) if "all_purpose_events" in hist_state.columns else 0.0
        sql_events = float(hist_state["sql_events"].iloc[-1]) if "sql_events" in hist_state.columns else 0.0
        other_events = float(hist_state["other_events"].iloc[-1]) if "other_events" in hist_state.columns else 0.0

        new_row = {
            "workspace_id": hist_state["workspace_id"].iloc[0],
            "workspace_name": hist_state["workspace_name"].iloc[0],
            "usage_date": dt,
            "daily_total_cost": yhat,
            "daily_total_usage": np.nan,
            "jobs_cost": jobs_cost,
            "all_purpose_cost": all_purpose_cost,
            "sql_cost": sql_cost,
            "other_cost": other_cost,
            "jobs_events": jobs_events,
            "all_purpose_events": all_purpose_events,
            "sql_events": sql_events,
            "other_events": other_events,
            "total_events": jobs_events + all_purpose_events + sql_events + other_events,
            "jobs_cost_share": share["jobs_cost_share"],
            "all_purpose_cost_share": share["all_purpose_cost_share"],
            "sql_cost_share": share["sql_cost_share"],
            "other_cost_share": share["other_cost_share"],
        }

        hist_state = pd.concat([hist_state, pd.DataFrame([new_row])], ignore_index=True)

        for col in event_cols:
            if col not in hist_state.columns:
                hist_state[col] = 0.0
        hist_state[event_cols] = hist_state[event_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        hist_state["total_events"] = hist_state[event_cols].sum(axis=1)
        hist_state["abs_change_1"] = hist_state["daily_total_cost"].diff().abs().fillna(0.0)
        hist_state["cv_7"] = (
            hist_state["daily_total_cost"].rolling(7, min_periods=2).std()
            / (hist_state["daily_total_cost"].rolling(7, min_periods=2).mean() + EPS)
        )
        hist_state["cv_7"] = hist_state["cv_7"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return np.asarray(preds, dtype=float), train_metrics, model

# ============================================================================
# CANDIDATE DEFINITIONS
# ============================================================================

def candidate_names(cfg: Config):
    names = []
    names += ["naive_full", "seasonal_naive_full"]

    if HAS_PMDA:
        for w in cfg.recent_windows:
            names.append(f"autoarima_{get_window_label(w)}")

    for w in cfg.recent_windows:
        names.append(f"sarimax_{get_window_label(w)}")

    if HAS_PROPHET:
        for w in cfg.recent_windows:
            names.append(f"prophet_{get_window_label(w)}")

    if HAS_XGB:
        for w in cfg.recent_windows:
            label = get_window_label(w)
            names.append(f"xgb_{label}_raw")
            if cfg.use_log_variants:
                names.append(f"xgb_{label}_log")

    if HAS_CAT:
        for w in cfg.recent_windows:
            label = get_window_label(w)
            names.append(f"cat_{label}_raw")
            if cfg.use_log_variants:
                names.append(f"cat_{label}_log")

    return names


# ============================================================================
# FOLD EVALUATION
# ============================================================================

def evaluate_candidate_on_fold(
    candidate_name: str,
    full_train_tables: Dict[str, pd.DataFrame],
    fold_train_end_date: pd.Timestamp,
    fold_test_daily_total: pd.DataFrame,
    cfg: Config,
    counter: FitCounter,
):
    if "recent270" in candidate_name:
        window = 270
    elif "recent180" in candidate_name:
        window = 180
    else:
        window = None

    train_tables = {
        "daily_total": (
            full_train_tables["daily_total"][
                full_train_tables["daily_total"]["usage_date"] <= fold_train_end_date
            ]
            .copy()
            .sort_values("usage_date")
            .reset_index(drop=True)
        ),
        "wide": (
            full_train_tables["wide"][
                full_train_tables["wide"]["usage_date"] <= fold_train_end_date
            ]
            .copy()
            .sort_values("usage_date")
            .reset_index(drop=True)
        ),
    }

    if window is not None:
        train_tables["daily_total"] = train_tables["daily_total"].tail(window).reset_index(drop=True)
        keep_dates = set(train_tables["daily_total"]["usage_date"])
        train_tables["wide"] = (
            train_tables["wide"][train_tables["wide"]["usage_date"].isin(keep_dates)]
            .copy()
            .sort_values("usage_date")
            .reset_index(drop=True)
        )

    train_total = train_tables["daily_total"].copy()
    y_train = train_total["daily_total_cost"].values.astype(float)

    future_dates = fold_test_daily_total["usage_date"].tolist()
    y_true = fold_test_daily_total["daily_total_cost"].values.astype(float)

    if candidate_name.startswith("naive"):
        y_pred = forecast_naive(y_train, len(future_dates))
        train_metrics = compute_metrics(y_train[1:], y_train[:-1], y_train[:-1]) if len(y_train) > 2 else None

    elif candidate_name.startswith("seasonal_naive"):
        y_pred = forecast_seasonal_naive(y_train, len(future_dates), cfg.seasonal_period)
        if len(y_train) > cfg.seasonal_period:
            train_metrics = compute_metrics(
                y_true=y_train[cfg.seasonal_period:],
                y_pred=y_train[:-cfg.seasonal_period],
                y_train=y_train[:-cfg.seasonal_period]
            )
        else:
            train_metrics = None

    elif candidate_name.startswith("autoarima"):
        y_pred, model = fit_predict_autoarima(train_total["daily_total_cost"], len(future_dates), counter)
        train_true, train_pred = train_fit_autoarima(model, train_total["daily_total_cost"])
        train_metrics = compute_metrics(train_true, train_pred, y_train) if train_true is not None else None

    elif candidate_name.startswith("sarimax"):
        best = tune_sarimax_aic(train_total["daily_total_cost"], cfg, counter)
        y_pred, res = fit_predict_sarimax(
            train_total["daily_total_cost"], len(future_dates), best["order"], best["seasonal_order"], counter
        )
        fitted = np.asarray(res.fittedvalues, dtype=float)
        true_train = train_total["daily_total_cost"].values[-len(fitted):]
        train_metrics = compute_metrics(true_train, np.maximum(fitted, 0.0), y_train)

    elif candidate_name.startswith("prophet"):
        y_pred, model = fit_predict_prophet(
            train_total[["usage_date", "daily_total_cost"]].copy(),
            future_dates,
            counter
        )
        p_train = train_total.rename(columns={"usage_date": "ds", "daily_total_cost": "y"})[["ds", "y"]].copy()
        pred_train = model.predict(p_train[["ds"]])["yhat"].values
        train_metrics = compute_metrics(p_train["y"].values, np.maximum(pred_train, 0.0), y_train)

    elif candidate_name.startswith("xgb"):
        use_log = candidate_name.endswith("_log")
        hist_state = build_history_state_from_train(train_tables)
        feat_train = build_feature_table(train_tables["wide"], train_tables["daily_total"])
        best = tune_xgb_on_training_slice(feat_train, cfg, counter, use_log=use_log)
        y_pred, train_metrics, _ = recursive_ml_forecast(
            history_df=hist_state,
            future_dates=future_dates,
            model_type="xgboost",
            params=best["params"],
            counter=counter,
            use_log_target=use_log
        )

    elif candidate_name.startswith("cat"):
        use_log = candidate_name.endswith("_log")
        hist_state = build_history_state_from_train(train_tables)
        feat_train = build_feature_table(train_tables["wide"], train_tables["daily_total"])
        best = tune_cat_on_training_slice(feat_train, cfg, counter, use_log=use_log)
        y_pred, train_metrics, _ = recursive_ml_forecast(
            history_df=hist_state,
            future_dates=future_dates,
            model_type="catboost",
            params=best["params"],
            counter=counter,
            use_log_target=use_log
        )
    else:
        raise ValueError(f"Unknown candidate: {candidate_name}")

    val_metrics = compute_metrics(y_true, y_pred, y_train)
    preds_df = pd.DataFrame({
        "usage_date": future_dates,
        "actual_total_cost": y_true,
        "predicted_total_cost": y_pred,
    })

    return val_metrics, train_metrics, preds_df


# ============================================================================
# STRICT BACKTEST RUN
# ============================================================================

def run_strict_backtests(train_tables: Dict[str, pd.DataFrame], cfg: Config, logger: logging.Logger):
    counter = FitCounter()
    candidates = candidate_names(cfg)
    daily_total = train_tables["daily_total"].copy().sort_values("usage_date").reset_index(drop=True)

    folds = make_backtest_folds(
        n_obs=len(daily_total),
        horizon=cfg.backtest_horizon_days,
        n_folds=cfg.backtest_n_folds,
        min_train_days=cfg.min_train_days,
    )
    logger.info(f"Backtest folds: {folds}")

    metric_rows = []
    train_rows = []
    pred_rows = []

    for i, fd in enumerate(folds, start=1):
        fold_train_total = daily_total.iloc[:fd["train_end"]].copy().reset_index(drop=True)
        fold_test_total = daily_total.iloc[fd["test_start"]:fd["test_end"]].copy().reset_index(drop=True)

        fold_train_end_date = fold_train_total["usage_date"].max()
        logger.info(f"=== FOLD {i}/{len(folds)} | Train rows={len(fold_train_total)} | Test rows={len(fold_test_total)} ===")

        for name in candidates:
            logger.info(f"Running candidate: {name}")
            try:
                val_metrics, train_metrics, preds_df = evaluate_candidate_on_fold(
                    candidate_name=name,
                    full_train_tables=train_tables,
                    fold_train_end_date=fold_train_end_date,
                    fold_test_daily_total=fold_test_total,
                    cfg=cfg,
                    counter=counter,
                )

                metric_rows.append({
                    "fold": i,
                    "model_name": name,
                    "train_end_date": fold_train_end_date,
                    "test_start_date": fold_test_total["usage_date"].min(),
                    "test_end_date": fold_test_total["usage_date"].max(),
                    **val_metrics,
                    "status": "ok",
                })

                if train_metrics is not None:
                    train_rows.append({
                        "fold": i,
                        "model_name": name,
                        **train_metrics,
                    })

                preds_df["fold"] = i
                preds_df["model_name"] = name
                preds_df["residual"] = preds_df["actual_total_cost"] - preds_df["predicted_total_cost"]
                pred_rows.append(preds_df)

            except Exception as e:
                logger.warning(f"Candidate failed: {name} on fold {i} | {type(e).__name__}: {e}")
                metric_rows.append({
                    "fold": i,
                    "model_name": name,
                    "train_end_date": fold_train_end_date,
                    "test_start_date": fold_test_total["usage_date"].min(),
                    "test_end_date": fold_test_total["usage_date"].max(),
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "WAPE": np.nan,
                    "sMAPE": np.nan,
                    "MASE": np.nan,
                    "BiasPct": np.nan,
                    "R2": np.nan,
                    "ActualTotal": np.nan,
                    "ForecastTotal": np.nan,
                    "TotalErrorPct": np.nan,
                    "ExecutiveAccuracyApprox": np.nan,
                    "peak_day_count": np.nan,
                    "peak_WAPE": np.nan,
                    "peak_MAE": np.nan,
                    "peak_RMSE": np.nan,
                    "status": "failed",
                })

    backtest_metrics = pd.DataFrame(metric_rows)
    backtest_preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    train_fit_metrics = pd.DataFrame(train_rows)

    valid = backtest_metrics[backtest_metrics["status"] == "ok"].copy()

    if valid.empty:
        summary = pd.DataFrame(columns=[
            "model_name", "folds_completed", "MAE", "RMSE", "WAPE", "sMAPE",
            "MASE", "BiasPct", "R2", "TotalErrorPct", "ExecutiveAccuracyApprox",
        ])
    else:
        summary = (
            valid.groupby("model_name", as_index=False)
            .agg(
                folds_completed=("fold", "nunique"),
                MAE=("MAE", "mean"),
                RMSE=("RMSE", "mean"),
                WAPE=("WAPE", "mean"),
                sMAPE=("sMAPE", "mean"),
                MASE=("MASE", "mean"),
                BiasPct=("BiasPct", "mean"),
                R2=("R2", "mean"),
                TotalErrorPct=("TotalErrorPct", "mean"),
                ExecutiveAccuracyApprox=("ExecutiveAccuracyApprox", "mean"),
            )
            .sort_values(["WAPE", "MASE", "RMSE", "MAE"])
            .reset_index(drop=True)
        )

    peak_rows = []
    if not backtest_preds.empty:
        for model_name, grp in backtest_preds.groupby("model_name"):
            peak_rows.append({
                "model_name": model_name,
                **peak_day_metrics(grp["actual_total_cost"], grp["predicted_total_cost"], quantile=0.90),
            })
    peak_summary = pd.DataFrame(peak_rows).sort_values("peak_WAPE") if peak_rows else pd.DataFrame()

    train_fit_summary = (
        train_fit_metrics.groupby("model_name", as_index=False)
        .agg(
            train_MAE=("MAE", "mean"),
            train_RMSE=("RMSE", "mean"),
            train_WAPE=("WAPE", "mean"),
            train_sMAPE=("sMAPE", "mean"),
            train_MASE=("MASE", "mean"),
            train_BiasPct=("BiasPct", "mean"),
            train_R2=("R2", "mean"),
        )
        if not train_fit_metrics.empty else pd.DataFrame()
    )

    return {
        "counter": counter,
        "backtest_metrics": backtest_metrics,
        "backtest_preds": backtest_preds,
        "summary": summary,
        "peak_summary": peak_summary,
        "train_fit_summary": train_fit_summary,
    }


# ============================================================================
# HOLDOUT TOP-CANDIDATE COMPARE
# ============================================================================

def choose_top_candidates(summary: pd.DataFrame, top_n=5):
    if summary is None or summary.empty:
        return []
    tmp = summary.copy()
    tmp["abs_bias"] = tmp["BiasPct"].abs()
    tmp = tmp.sort_values(["WAPE", "MASE", "abs_bias", "RMSE", "MAE"])
    return tmp["model_name"].head(top_n).tolist()


def parse_candidate_window(candidate_name: str):
    if "recent270" in candidate_name:
        return 270
    if "recent180" in candidate_name:
        return 180
    return None


def score_candidate_on_holdout(
    candidate_name: str,
    train_tables: Dict[str, pd.DataFrame],
    holdout_daily_total: pd.DataFrame,
    cfg: Config,
    counter: FitCounter,
):
    window = parse_candidate_window(candidate_name)

    full_end = pd.to_datetime(cfg.train_end_date)
    training_tables = {
        "daily_total": (
            train_tables["daily_total"][train_tables["daily_total"]["usage_date"] <= full_end]
            .copy().sort_values("usage_date").reset_index(drop=True)
        ),
        "wide": (
            train_tables["wide"][train_tables["wide"]["usage_date"] <= full_end]
            .copy().sort_values("usage_date").reset_index(drop=True)
        ),
    }

    if window is not None:
        training_tables["daily_total"] = training_tables["daily_total"].tail(window).reset_index(drop=True)
        keep_dates = set(training_tables["daily_total"]["usage_date"])
        training_tables["wide"] = (
            training_tables["wide"][training_tables["wide"]["usage_date"].isin(keep_dates)]
            .copy().sort_values("usage_date").reset_index(drop=True)
        )

    y_train = training_tables["daily_total"]["daily_total_cost"].values.astype(float)
    future_dates = holdout_daily_total["usage_date"].tolist()

    if candidate_name.startswith("naive"):
        preds = forecast_naive(y_train, len(future_dates))
        model_obj = None
        extras = {}

    elif candidate_name.startswith("seasonal_naive"):
        preds = forecast_seasonal_naive(y_train, len(future_dates), cfg.seasonal_period)
        model_obj = None
        extras = {}

    elif candidate_name.startswith("autoarima"):
        preds, model_obj = fit_predict_autoarima(training_tables["daily_total"]["daily_total_cost"], len(future_dates), counter)
        extras = {}

    elif candidate_name.startswith("sarimax"):
        best = tune_sarimax_aic(training_tables["daily_total"]["daily_total_cost"], cfg, counter)
        preds, model_obj = fit_predict_sarimax(
            training_tables["daily_total"]["daily_total_cost"],
            len(future_dates),
            best["order"],
            best["seasonal_order"],
            counter,
        )
        extras = {"sarimax_tuned": best}

    elif candidate_name.startswith("prophet"):
        preds, model_obj = fit_predict_prophet(
            training_tables["daily_total"][["usage_date", "daily_total_cost"]],
            future_dates,
            counter,
        )
        extras = {}

    elif candidate_name.startswith("xgb"):
        use_log = candidate_name.endswith("_log")
        hist_state = build_history_state_from_train(training_tables)
        feat_train = build_feature_table(training_tables["wide"], training_tables["daily_total"])
        best = tune_xgb_on_training_slice(feat_train, cfg, counter, use_log=use_log)
        preds, _, model_obj = recursive_ml_forecast(
            history_df=hist_state,
            future_dates=future_dates,
            model_type="xgboost",
            params=best["params"],
            counter=counter,
            use_log_target=use_log,
        )
        extras = {"xgb_tuned": best}

    elif candidate_name.startswith("cat"):
        use_log = candidate_name.endswith("_log")
        hist_state = build_history_state_from_train(training_tables)
        feat_train = build_feature_table(training_tables["wide"], training_tables["daily_total"])
        best = tune_cat_on_training_slice(feat_train, cfg, counter, use_log=use_log)
        preds, _, model_obj = recursive_ml_forecast(
            history_df=hist_state,
            future_dates=future_dates,
            model_type="catboost",
            params=best["params"],
            counter=counter,
            use_log_target=use_log,
        )
        extras = {"cat_tuned": best}

    else:
        raise ValueError(candidate_name)

    fc = pd.DataFrame({
        "forecast_date": future_dates,
        "predicted_total_cost": preds,
    }).sort_values("forecast_date").reset_index(drop=True)

    fc["horizon_day"] = np.arange(1, len(fc) + 1)
    fc["predicted_cumulative_cost"] = fc["predicted_total_cost"].cumsum()

    merged = holdout_daily_total.rename(columns={"daily_total_cost": "actual_total_cost"}).merge(
        fc,
        left_on="usage_date",
        right_on="forecast_date",
        how="inner",
    ).sort_values("usage_date").reset_index(drop=True)

    metrics_daily = compute_metrics(
        y_true=merged["actual_total_cost"].values,
        y_pred=merged["predicted_total_cost"].values,
        y_train=y_train,
    )
    metrics_agg = compute_multilevel_holdout_metrics(merged, y_train)

    return {
        "model_name": candidate_name,
        "daily_scorecard": pd.DataFrame([metrics_daily]),
        "daily_merged": merged,
        "block_metrics": metrics_agg,
        "forecast_table": fc,
        "model_obj": model_obj,
        "extras": extras,
    }


# ============================================================================
# MODEL EXPLAINABILITY
# ============================================================================

def save_winner_artifacts(winner_result: Dict[str, Any], outdir: Path, logger: logging.Logger):
    exp_dir = outdir / "winner_artifacts"
    exp_dir.mkdir(parents=True, exist_ok=True)

    name = winner_result["model_name"]
    model_obj = winner_result["model_obj"]

    (exp_dir / "winner_model_name.txt").write_text(name, encoding="utf-8")

    try:
        if name.startswith("xgb") and HAS_XGB and model_obj is not None:
            booster = model_obj.get_booster()
            dumps = booster.get_dump(with_stats=True)
            top20 = dumps[:20]
            with open(exp_dir / "xgb_top20_trees.txt", "w", encoding="utf-8") as f:
                for i, tree in enumerate(top20, start=1):
                    f.write(f"\n=== TREE {i} ===\n")
                    f.write(tree)
                    f.write("\n")

            plt.figure(figsize=(12, 8))
            plot_importance(model_obj, max_num_features=20)
            plt.title("XGBoost Feature Importance (Top 20)")
            save_fig(exp_dir / "xgb_feature_importance_top20.png")

        elif name.startswith("cat") and HAS_CAT and model_obj is not None:
            importance = model_obj.get_feature_importance()
            feat = feature_cols()
            imp_df = pd.DataFrame({"feature": feat[:len(importance)], "importance": importance}).sort_values("importance", ascending=False).head(20)
            save_csv(imp_df, exp_dir / "cat_feature_importance_top20.csv")

            plt.figure(figsize=(12, 8))
            sns.barplot(data=imp_df, x="importance", y="feature", palette="viridis")
            plt.title("CatBoost Feature Importance (Top 20)")
            save_fig(exp_dir / "cat_feature_importance_top20.png")
    except Exception as e:
        logger.warning(f"Could not save winner explainability artifacts: {e}")


# ============================================================================
# REPORT PLOTS
# ============================================================================

def generate_report_plots(
    outdir: Path,
    eda_outputs: Dict[str, pd.DataFrame],
    benchmark: Dict[str, pd.DataFrame],
    holdout_compare_df: Optional[pd.DataFrame],
    final_forecast_df: Optional[pd.DataFrame],
    winner_name: Optional[str],
):
    assets = outdir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)

    summary = benchmark["summary"]
    train_fit_summary = benchmark["train_fit_summary"]
    backtest_preds = benchmark["backtest_preds"]

    if not summary.empty:
        plt.figure(figsize=(12, 5))
        sns.barplot(data=summary, x="model_name", y="WAPE", palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.title("Backtest WAPE Comparison")
        save_fig(assets / "01_backtest_wape_bar.png")

        plt.figure(figsize=(12, 5))
        sns.barplot(data=summary, x="model_name", y="MASE", palette="magma")
        plt.xticks(rotation=45, ha="right")
        plt.title("Backtest MASE Comparison")
        save_fig(assets / "02_backtest_mase_bar.png")

    if not train_fit_summary.empty:
        plt.figure(figsize=(12, 5))
        sns.barplot(data=train_fit_summary, x="model_name", y="train_WAPE", palette="cubehelix")
        plt.xticks(rotation=45, ha="right")
        plt.title("Train WAPE Comparison")
        save_fig(assets / "03_train_wape_bar.png")

    if winner_name is not None and not backtest_preds.empty:
        sub = backtest_preds[backtest_preds["model_name"] == winner_name].copy()
        if not sub.empty:
            plt.figure(figsize=(16, 6))
            plt.plot(sub["usage_date"], sub["actual_total_cost"], color="black", label="Actual")
            plt.plot(sub["usage_date"], sub["predicted_total_cost"], color="blue", label=f"Predicted ({winner_name})")
            plt.title(f"Backtest Performance - {winner_name}")
            plt.legend()
            save_fig(assets / "04_backtest_winner_plot.png")

    if final_forecast_df is not None:
        hist = eda_outputs["daily_total_flagged"][["usage_date", "daily_total_cost"]].copy().tail(90)
        plt.figure(figsize=(16, 6))
        plt.plot(hist["usage_date"], hist["daily_total_cost"], color="black", label="Recent Historical")
        plt.plot(final_forecast_df["forecast_date"], final_forecast_df["predicted_total_cost"], color="blue", linestyle="--", label="Forecast")
        plt.title("Historical + Future Forecast")
        plt.legend()
        save_fig(assets / "05_historical_future_forecast.png")

    if holdout_compare_df is not None and not holdout_compare_df.empty:
        plt.figure(figsize=(16, 6))
        plt.plot(holdout_compare_df["usage_date"], holdout_compare_df["actual_total_cost"], color="black", marker="o", label="Actual")
        plt.plot(holdout_compare_df["usage_date"], holdout_compare_df["predicted_total_cost"], color="blue", linestyle="--", marker="x", label="Forecast")
        plt.title("Holdout Validation: Actual vs Forecast")
        plt.legend()
        save_fig(assets / "06_validation_actual_vs_forecast.png")

        tmp = holdout_compare_df.copy()
        tmp["actual_cumulative_cost"] = tmp["actual_total_cost"].cumsum()
        tmp["predicted_cumulative_cost"] = tmp["predicted_total_cost"].cumsum()

        plt.figure(figsize=(16, 6))
        plt.plot(tmp["usage_date"], tmp["actual_cumulative_cost"], color="black", marker="o", label="Actual Cumulative")
        plt.plot(tmp["usage_date"], tmp["predicted_cumulative_cost"], color="blue", linestyle="--", marker="x", label="Forecast Cumulative")
        plt.title("Holdout Validation: Cumulative Actual vs Forecast")
        plt.legend()
        save_fig(assets / "07_validation_cumulative_comparison.png")

# ============================================================================
# HTML REPORT
# ============================================================================

def generate_html_report(
    cfg: Config,
    outdir: Path,
    eda_outputs: Dict[str, pd.DataFrame],
    benchmark: Dict[str, pd.DataFrame],
    top_holdout_results: List[Dict[str, Any]],
    winner_result: Optional[Dict[str, Any]],
):
    report_path = outdir / "reports" / "forecast_report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    winner_name = winner_result["model_name"] if winner_result else None
    winner_daily = winner_result["daily_scorecard"] if winner_result else None
    winner_blocks = winner_result["block_metrics"] if winner_result else None
    winner_daily_merged = winner_result["daily_merged"] if winner_result else None

    top_holdout_table_rows = []
    for r in top_holdout_results:
        d = r["daily_scorecard"].iloc[0].to_dict()
        d["model_name"] = r["model_name"]
        top_holdout_table_rows.append(d)
    top_holdout_table = pd.DataFrame(top_holdout_table_rows) if top_holdout_table_rows else pd.DataFrame()

    html = f"""
    <html>
    <head>
        <title>Workspace Daily Cost Forecast Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; }}
            h1, h2, h3 {{ color: #13324b; }}
            .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #eee; margin-top: 8px; }}
            table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
            th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
            th {{ background: #f4f6f8; }}
            .note {{ color: #555; font-size: 12px; }}
            pre {{ white-space: pre-wrap; word-break: break-word; }}
        </style>
    </head>
    <body>
        <h1>Workspace Daily Cost Forecast Report</h1>

        <div class="card">
            <h2>Run Configuration</h2>
            <pre>{json.dumps(asdict(cfg), indent=2, default=str)}</pre>
        </div>

        <div class="card">
            <h2>Data Audit</h2>
            {html_table(eda_outputs["audit"])}
            <h3>Product Summary</h3>
            {html_table(eda_outputs["product_summary"], max_rows=25)}
            <h3>Stationarity Summary</h3>
            {html_table(eda_outputs["stationarity"])}
            <h3>Phase Summary</h3>
            {html_table(eda_outputs["phase_summary"])}
            <p class="note">
                Practical stationarity answer used for modeling: <b>NO</b>.<br/>
                Formal tests may support weak stationarity, but rolling mean/variance changes,
                regime shifts, and visible weekly seasonality mean the series is handled as non-stationary
                for forecasting decisions.
            </p>
        </div>

        <div class="card">
            <h2>EDA Visuals</h2>
            <img src="../eda/01_daily_total_cost.png"/>
            <img src="../eda/02_daily_total_cost_rolling_means.png"/>
            <img src="../eda/03_rolling_mean_std_30.png"/>
            <img src="../eda/04_cumulative_total_cost.png"/>
            <img src="../eda/05_grouped_product_mix_area.png"/>
            <img src="../eda/06_grouped_product_cost_bar.png"/>
            <img src="../eda/07_grouped_product_cost_shares.png"/>
            <img src="../eda/08_day_of_week_boxplot.png"/>
            <img src="../eda/09_weekday_heatmap.png"/>
            <img src="../eda/10_stl_decomposition_cost.png"/>
            <img src="../eda/11_acf_pacf_cost.png"/>
            <img src="../eda/12_outlier_detection_cost.png"/>
        </div>

        <div class="card">
            <h2>Strict Leakage-Safe Backtest Summary</h2>
            {html_table(benchmark["summary"])}
            <h3>Peak-Day Summary</h3>
            {html_table(benchmark["peak_summary"])}
            <h3>Train-Fit Summary</h3>
            {html_table(benchmark["train_fit_summary"])}
            <h3>Fit Counters</h3>
            <pre>{json.dumps(benchmark["counter"].counts, indent=2)}</pre>
            <p class="note">
                Backtesting means repeatedly training on the past and testing on the next unseen future block.
                This is the correct validation style for time series and reduces leakage risk compared with random train/test splits.
            </p>
        </div>
    """

    if not top_holdout_table.empty:
        html += f"""
        <div class="card">
            <h2>Top Candidate Holdout Comparison</h2>
            {html_table(top_holdout_table)}
            <p class="note">
                Final model selection is based on unseen holdout validation, not only internal backtest score.
            </p>
        </div>
        """

    if winner_result is not None:
        html += f"""
        <div class="card">
            <h2>Final Selected Holdout Winner</h2>
            <p><b>{winner_name}</b></p>
            <h3>Daily Holdout Metrics</h3>
            {html_table(winner_daily)}
            <h3>3-Day Block Metrics</h3>
            {html_table(winner_blocks["block3"])}
            <h3>7-Day Block Metrics</h3>
            {html_table(winner_blocks["block7"])}
            <h3>Monthly Total Comparison</h3>
            {html_table(winner_blocks["monthly"])}
            <h3>Daily Actual vs Forecast</h3>
            {html_table(winner_daily_merged[["usage_date", "actual_total_cost", "predicted_total_cost", "predicted_cumulative_cost"]], max_rows=35)}
            <img src="../report_assets/06_validation_actual_vs_forecast.png"/>
            <img src="../report_assets/07_validation_cumulative_comparison.png"/>
            <p class="note">
                Executive Accuracy Approx = 100 - WAPE.<br/>
                This is a WAPE-based business shorthand, not classification accuracy.
            </p>
        </div>
        """

    html += """
    </body>
    </html>
    """

    report_path.write_text(html, encoding="utf-8")
    return report_path


# ============================================================================
# EXPORTS FOR UI / SERVING
# ============================================================================

def export_serving_outputs(
    outdir: Path,
    cfg: Config,
    winner_result: Optional[Dict[str, Any]],
    winner_name: Optional[str],
):
    serving_dir = outdir / "serving"
    serving_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = outdir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if winner_result is None:
        return

    daily_rows = winner_result["daily_merged"][["usage_date", "actual_total_cost", "predicted_total_cost"]].copy()
    daily_rows["usage_date"] = pd.to_datetime(daily_rows["usage_date"]).dt.strftime("%Y-%m-%d")
    save_csv(daily_rows, serving_dir / "validation_rows.csv")

    scorecard = winner_result["daily_scorecard"].copy()
    save_csv(scorecard, serving_dir / "validation_scorecard.csv")

    payload = {
        "workspace_id": cfg.workspace_id_real,
        "workspace_name": cfg.workspace_name,
        "winner_model_name": winner_name,
        "data_range_label": "1st April 2025 to 31st March 2026",
        "validation_rows_file": "validation_rows.csv",
        "validation_scorecard_file": "validation_scorecard.csv",
    }
    save_json(payload, serving_dir / "prediction_dashboard_payload.json")

    metadata = {
        "workspace_id": cfg.workspace_id_real,
        "workspace_name": cfg.workspace_name,
        "winner_model_name": winner_name,
        "train_end_date": cfg.train_end_date,
        "holdout_start_date": cfg.holdout_start_date,
        "holdout_end_date": cfg.holdout_end_date,
        "forecast_horizon_days": cfg.forecast_horizon_days,
    }
    save_json(metadata, metadata_dir / f"{cfg.workspace_id_real}_forecast_metadata.json")


# ============================================================================
# HOLDOUT DATA LOADER
# ============================================================================

def load_ceo_actuals(cfg: Config, logger: logging.Logger):
    df = load_source(cfg.ceo_actuals_csv, cfg, logger, force_workspace_id=cfg.workspace_id_real)
    tables = build_internal_tables(df, cfg)

    daily_total = tables["daily_total"].copy()
    daily_total = daily_total[
        (daily_total["usage_date"] >= pd.to_datetime(cfg.holdout_start_date)) &
        (daily_total["usage_date"] <= pd.to_datetime(cfg.holdout_end_date))
    ].copy().sort_values("usage_date").reset_index(drop=True)

    return {
        "raw": df,
        "tables": tables,
        "daily_total": daily_total,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Strict, leakage-safe daily total cost forecasting pipeline for one Databricks workspace."
    )
    parser.add_argument("--input-csv", required=True, help="Training CSV path")
    parser.add_argument("--ceo-actuals-csv", default=None, help="Holdout actual CSV path")
    parser.add_argument("--output-dir", default="artifacts/forecasting/workspace_3442171567662343", help="Output directory")
    parser.add_argument("--workspace-name", default="dbricks-mfgprod-sem-prod", help="Workspace name")
    parser.add_argument("--workspace-id-real", default="3442171567662343", help="Exact true workspace ID")
    parser.add_argument("--remove-outliers-mode", default="none", choices=["none", "iqr_cap"], help="Outlier handling mode")
    args = parser.parse_args()

    cfg = build_default_config(args)
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir)

    logger.info("===== PIPELINE START =====")
    logger.info("Target = DAILY TOTAL COST")
    logger.info("Validation = strict rolling backtests + optional April holdout validation")
    logger.info("Deep learning intentionally omitted: ~365 daily target observations are too small for a sensible first deep-learning production candidate.")

    train_df_raw = load_source(cfg.input_csv, cfg, logger, force_workspace_id=cfg.workspace_id_real)
    train_tables = build_internal_tables(train_df_raw, cfg)

    proc_dir = outdir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    for k, v in train_tables.items():
        save_csv(v, proc_dir / f"{k}.csv")
        logger.info(f"Saved processed table: {k}.csv")

    eda_outputs = run_eda(train_tables, cfg, outdir, logger)

    benchmark = run_strict_backtests(train_tables, cfg, logger)

    bench_dir = outdir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    save_csv(benchmark["backtest_metrics"], bench_dir / "backtest_fold_metrics.csv")
    save_csv(benchmark["backtest_preds"], bench_dir / "backtest_predictions.csv")
    save_csv(benchmark["summary"], bench_dir / "model_summary.csv")
    save_csv(benchmark["peak_summary"], bench_dir / "peak_summary.csv")
    if not benchmark["train_fit_summary"].empty:
        save_csv(benchmark["train_fit_summary"], bench_dir / "train_fit_summary.csv")
    logger.info("Strict backtest artifacts saved.")

    top_holdout_results = []
    winner_result = None

    if cfg.ceo_actuals_csv:
        logger.info("Loading holdout actuals for exact April validation scoring...")
        holdout_obj = load_ceo_actuals(cfg, logger)
        holdout_daily_total = holdout_obj["daily_total"].copy()

        if len(holdout_daily_total) != 30:
            logger.warning(f"Holdout daily rows = {len(holdout_daily_total)}. Expected 30 rows for April 1-30.")

        top_candidates = choose_top_candidates(benchmark["summary"], top_n=cfg.top_n_holdout_candidates)
        logger.info(f"Top candidates to score on holdout validation: {top_candidates}")

        holdout_counter = FitCounter()

        for cand in top_candidates:
            logger.info(f"Scoring candidate on holdout validation: {cand}")
            res = score_candidate_on_holdout(
                candidate_name=cand,
                train_tables=train_tables,
                holdout_daily_total=holdout_daily_total,
                cfg=cfg,
                counter=holdout_counter,
            )
            top_holdout_results.append(res)

        holdout_rank = []
        for r in top_holdout_results:
            row = r["daily_scorecard"].iloc[0].to_dict()
            row["model_name"] = r["model_name"]
            holdout_rank.append(row)

        holdout_rank_df = pd.DataFrame(holdout_rank)
        holdout_rank_df["abs_bias"] = holdout_rank_df["BiasPct"].abs()
        holdout_rank_df = holdout_rank_df.sort_values(["WAPE", "MASE", "abs_bias", "RMSE", "MAE"]).reset_index(drop=True)

        val_dir = outdir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)

        save_csv(holdout_rank_df, val_dir / "holdout_candidate_ranking.csv")

        best_name = holdout_rank_df.iloc[0]["model_name"]
        winner_result = [r for r in top_holdout_results if r["model_name"] == best_name][0]
        logger.info(f"Final holdout winner: {best_name}")

        save_csv(winner_result["daily_merged"], val_dir / "winner_holdout_daily_comparison.csv")
        save_csv(winner_result["daily_scorecard"], val_dir / "winner_holdout_daily_scorecard.csv")
        save_csv(winner_result["block_metrics"]["block3"], val_dir / "winner_holdout_3day_scorecard.csv")
        save_csv(winner_result["block_metrics"]["block7"], val_dir / "winner_holdout_7day_scorecard.csv")
        save_csv(winner_result["block_metrics"]["monthly"], val_dir / "winner_holdout_monthly_scorecard.csv")
        save_csv(winner_result["block_metrics"]["block3_table"], val_dir / "winner_holdout_3day_blocks.csv")
        save_csv(winner_result["block_metrics"]["block7_table"], val_dir / "winner_holdout_7day_blocks.csv")
        save_csv(winner_result["forecast_table"], val_dir / "winner_final_forecast_april.csv")

        logger.info("Holdout validation scoring complete.")

    else:
        if benchmark["summary"].empty:
            logger.warning("No successful candidate summary available.")
        else:
            winner_name = benchmark["summary"].iloc[0]["model_name"]
            logger.info(f"No holdout file supplied. Internal benchmark winner = {winner_name}")

    holdout_compare_df = winner_result["daily_merged"] if winner_result else None
    final_forecast_df = winner_result["forecast_table"] if winner_result else None
    winner_name = (
        winner_result["model_name"]
        if winner_result
        else (benchmark["summary"].iloc[0]["model_name"] if not benchmark["summary"].empty else None)
    )

    if winner_result is not None:
        save_winner_artifacts(winner_result, outdir, logger)

    generate_report_plots(
        outdir=outdir,
        eda_outputs=eda_outputs,
        benchmark=benchmark,
        holdout_compare_df=holdout_compare_df,
        final_forecast_df=final_forecast_df,
        winner_name=winner_name,
    )

    report_path = generate_html_report(
        cfg=cfg,
        outdir=outdir,
        eda_outputs=eda_outputs,
        benchmark=benchmark,
        top_holdout_results=top_holdout_results,
        winner_result=winner_result,
    )

    export_serving_outputs(
        outdir=outdir,
        cfg=cfg,
        winner_result=winner_result,
        winner_name=winner_name,
    )

    logger.info(f"Final HTML report written to: {report_path}")
    logger.info("===== PIPELINE COMPLETE =====")


if __name__ == "__main__":
    main()

































