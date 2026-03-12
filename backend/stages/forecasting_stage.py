"""Proper time-series forecasting stage.

Decomposes historical sales into trend + seasonality + marketing effect,
then projects forward N weeks with 80% and 95% prediction intervals.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def forecasting_stage(state: dict) -> dict:
    logs = []
    logs.append("📈 [Forecasting Stage] Starting time-series decomposition...")

    df = pd.DataFrame(state["processed_data"])
    spend_cols = state["spend_columns"]
    best_params = state["best_params"]
    coef = np.array(state["model_coef"])
    intercept = state["model_intercept"]
    X_mean = np.array(state["feature_mean"])
    X_std = np.array(state["feature_std"])
    y_mean = state["target_mean"]
    y_std = state["target_std"]
    opt_alloc = state["optimal_allocation"]

    actual_sales = np.array(state["actual_sales"], dtype=float)
    n = len(actual_sales)

    # ── 1. Compute marketing effect for historical period ───────
    logs.append("📈 [Forecasting Stage] Computing historical marketing effects...")

    def _marketing_effect(spend_array, discount, seasonality):
        """Predict sales from a single observation's features."""
        features = []
        for i, col in enumerate(spend_cols):
            alpha = best_params[f"{col}_alpha"]
            features.append(1 - np.exp(-alpha * spend_array[i]))
        features.append(discount)
        features.append(seasonality)
        features = np.array(features)
        scaled = (features - X_mean) / X_std
        return float((intercept + scaled @ coef) * y_std + y_mean)

    mkt_effects = np.zeros(n)
    for idx in range(n):
        row_spend = np.array([float(df[c].iloc[idx]) for c in spend_cols])
        mkt_effects[idx] = _marketing_effect(
            row_spend,
            float(df["discount"].iloc[idx]),
            float(df["seasonality"].iloc[idx]),
        )

    # ── 2. Residuals = actual - marketing model ─────────────────
    residuals = actual_sales - mkt_effects
    logs.append(f"📈 [Forecasting Stage] Residual mean={residuals.mean():,.0f}, std={residuals.std():,.0f}")

    # ── 3. STL decomposition of residuals → trend + seasonal ────
    period = min(13, max(3, n // 4))  # ~quarterly, but adapt to data size
    if period % 2 == 0:
        period += 1  # STL needs odd period

    logs.append(f"📈 [Forecasting Stage] STL decomposition (period={period})...")
    residual_series = pd.Series(residuals, index=pd.RangeIndex(n))
    stl = STL(residual_series, period=period, robust=True)
    stl_result = stl.fit()

    trend = stl_result.trend.values
    seasonal = stl_result.seasonal.values
    stl_resid = stl_result.resid.values

    logs.append(f"📈 [Forecasting Stage] Trend range: {trend.min():,.0f} — {trend.max():,.0f}")
    logs.append(f"📈 [Forecasting Stage] Seasonal amplitude: {seasonal.max() - seasonal.min():,.0f}")

    # ── 4. Extrapolate trend via linear fit on last half ────────
    logs.append("📈 [Forecasting Stage] Extrapolating trend (linear fit)...")
    half = max(n // 2, 10)
    t_idx = np.arange(n)
    trend_slope, trend_intercept = np.polyfit(t_idx[-half:], trend[-half:], 1)
    logs.append(f"📈 [Forecasting Stage] Trend slope: {trend_slope:+.2f} per week")

    # ── 5. Build N-week forecast ────────────────────────────────
    horizon = state.get("forecast_weeks", 12)
    logs.append(f"📈 [Forecasting Stage] Projecting {horizon} weeks ahead...")
    last_week = int(df["week"].max())

    # Optimal spend from simulation stage
    opt_spend = np.array([
        next(a["optimal_spend"] for a in opt_alloc if a["channel"] == c)
        for c in spend_cols
    ])

    # Residual std for prediction intervals
    sigma = float(np.std(stl_resid))
    z80 = 1.282
    z95 = 1.960

    forecast = []
    discount_med = float(df["discount"].median())

    for h in range(horizon):
        wk = last_week + h + 1
        future_t = n + h

        # Trend: linear extrapolation
        trend_h = trend_intercept + trend_slope * future_t

        # Seasonal: cyclic repetition
        seasonal_h = seasonal[future_t % len(seasonal)]

        # Marketing effect with optimal spend (use cyclic seasonality from data)
        season_val = float(df["seasonality"].iloc[future_t % n])
        mkt_h = _marketing_effect(opt_spend, discount_med, season_val)

        # Point forecast = marketing model + trend residual + seasonal residual
        point = mkt_h + trend_h + seasonal_h

        # Prediction intervals widen with horizon
        spread = sigma * np.sqrt(1 + h / horizon)

        forecast.append({
            "week": wk,
            "predicted_sales": round(float(point), 2),
            "lower_80": round(float(point - z80 * spread), 2),
            "upper_80": round(float(point + z80 * spread), 2),
            "lower_95": round(float(point - z95 * spread), 2),
            "upper_95": round(float(point + z95 * spread), 2),
            "trend": round(float(trend_h), 2),
            "seasonal": round(float(seasonal_h), 2),
            "marketing": round(float(mkt_h), 2),
            "seasonality_input": round(season_val, 3),
        })

    fc_df = pd.DataFrame(forecast)
    fc_total = float(fc_df["predicted_sales"].sum())
    fc_avg = float(fc_df["predicted_sales"].mean())

    logs.append(f"📈 [Forecasting Stage] {horizon}-week total: ₹{fc_total:,.0f}")
    logs.append(f"📈 [Forecasting Stage] Weekly average: ₹{fc_avg:,.0f}")
    logs.append(f"📈 [Forecasting Stage] 95% interval width: ±₹{z95 * sigma:,.0f}")
    logs.append("✅ [Forecasting Stage] Forecast complete")

    return {
        "forecast": forecast,
        "forecast_decomposition": {
            "trend_slope": round(float(trend_slope), 4),
            "seasonal_period": period,
            "residual_std": round(sigma, 2),
            "historical_trend": [round(float(t), 2) for t in trend],
            "historical_seasonal": [round(float(s), 2) for s in seasonal],
        },
        "logs": logs,
    }
