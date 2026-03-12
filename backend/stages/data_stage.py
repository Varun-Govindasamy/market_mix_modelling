import pandas as pd
import numpy as np
from backend.utils.transforms import adstock_transform, saturation_transform

SPEND_COLS = ["tv_spend", "google_ads_spend", "meta_ads_spend", "influencer_spend"]

DEFAULT_DECAYS = {
    "tv_spend": 0.6,
    "google_ads_spend": 0.3,
    "meta_ads_spend": 0.4,
    "influencer_spend": 0.5,
}


def data_stage(state: dict) -> dict:
    logs = []
    logs.append("📊 [Data Stage] Starting data ingestion...")

    # ── load ────────────────────────────────────────────────────
    path = state["dataset_path"]
    df = pd.read_csv(path)
    logs.append(f"📊 [Data Stage] Loaded {path}: {df.shape[0]} rows × {df.shape[1]} cols")

    raw_data = df.to_dict(orient="records")

    # ── clean ───────────────────────────────────────────────────
    missing = int(df.isnull().sum().sum())
    if missing > 0:
        df = df.fillna(df.median(numeric_only=True))
        logs.append(f"📊 [Data Stage] Filled {missing} missing values (median imputation)")
    else:
        logs.append("📊 [Data Stage] No missing values — data is clean")

    dupes = int(df.duplicated().sum())
    if dupes > 0:
        df = df.drop_duplicates()
        logs.append(f"📊 [Data Stage] Dropped {dupes} duplicate rows")

    # ── validate schema ─────────────────────────────────────────
    required = SPEND_COLS + ["discount", "seasonality", "sales", "week"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        logs.append(f"⚠️  [Data Stage] Missing columns: {missing_cols}")
    else:
        logs.append("📊 [Data Stage] Schema validation passed ✓")

    # ── summary stats ───────────────────────────────────────────
    summary = {
        "total_weeks": int(df.shape[0]),
        "channels": len(SPEND_COLS),
        "avg_sales": round(float(df["sales"].mean()), 2),
        "total_sales": round(float(df["sales"].sum()), 2),
        "week_range": f"Week {int(df['week'].min())} – {int(df['week'].max())}",
        "avg_spend_per_channel": {
            col: round(float(df[col].mean()), 2) for col in SPEND_COLS
        },
    }
    logs.append(
        f"📊 [Data Stage] {summary['total_weeks']} weeks, "
        f"{summary['channels']} channels, "
        f"avg sales ₹{summary['avg_sales']:,.0f}"
    )

    # ── feature engineering: adstock + saturation ───────────────
    for col, decay in DEFAULT_DECAYS.items():
        adstocked = adstock_transform(df[col], decay)
        df[f"{col}_adstock"] = adstocked
        df[f"{col}_saturated"] = saturation_transform(adstocked, alpha=0.00005)

    logs.append("📊 [Data Stage] Applied adstock + saturation transformations")
    logs.append("✅ [Data Stage] Data processing complete")

    return {
        "raw_data": raw_data,
        "processed_data": df.to_dict(orient="records"),
        "data_summary": summary,
        "spend_columns": SPEND_COLS,
        "logs": logs,
    }
