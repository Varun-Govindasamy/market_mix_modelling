import numpy as np
import pandas as pd
from scipy.optimize import minimize


def simulation_stage(state: dict) -> dict:
    logs = []
    logs.append("🔮 [Simulation Stage] Starting simulations...")

    df = pd.DataFrame(state["processed_data"])
    spend_cols = state["spend_columns"]
    best_params = state["best_params"]
    coef = np.array(state["model_coef"])
    intercept = state["model_intercept"]
    X_mean = np.array(state["feature_mean"])
    X_std = np.array(state["feature_std"])
    y_mean = state["target_mean"]
    y_std = state["target_std"]
    total_budget = state["total_budget"]

    discount_med = float(df["discount"].median())
    season_med = float(df["seasonality"].median())

    # ── prediction helper ───────────────────────────────────────
    def predict(spend_array, discount=discount_med, seasonality=season_med):
        features = []
        for i, col in enumerate(spend_cols):
            alpha = best_params[f"{col}_alpha"]
            features.append(1 - np.exp(-alpha * spend_array[i]))
        features.append(discount)
        features.append(seasonality)
        features = np.array(features)
        scaled = (features - X_mean) / X_std
        return float((intercept + scaled @ coef) * y_std + y_mean)

    baseline_spend = np.array([df[c].mean() for c in spend_cols])
    baseline_sales = predict(baseline_spend)

    # ── counterfactual scenarios ────────────────────────────────
    scenarios = [
        ("Baseline",          baseline_spend * np.array([1.0, 1.0, 1.0, 1.0])),
        ("Google Ads +30%",   baseline_spend * np.array([1.0, 1.3, 1.0, 1.0])),
        ("TV Ads Stop",       baseline_spend * np.array([0.0, 1.0, 1.0, 1.0])),
        ("Influencer 2×",     baseline_spend * np.array([1.0, 1.0, 1.0, 2.0])),
        ("Meta Ads +50%",     baseline_spend * np.array([1.0, 1.0, 1.5, 1.0])),
        ("All Digital +20%",  baseline_spend * np.array([1.0, 1.2, 1.2, 1.2])),
        ("Cut All 25%",       baseline_spend * 0.75),
    ]

    # add user-defined scenarios
    for idx, us in enumerate(state.get("user_scenarios", []) or []):
        spend = np.array([us.get(c, df[c].mean()) for c in spend_cols])
        scenarios.append((f"Custom {idx + 1}", spend))

    sim_results = []
    for name, spend in scenarios:
        pred = predict(spend)
        delta = pred - baseline_sales
        pct = (delta / baseline_sales) * 100 if baseline_sales else 0
        sim_results.append({
            "scenario": name,
            "spend": {c: round(float(s), 2) for c, s in zip(spend_cols, spend)},
            "predicted_sales": round(pred, 2),
            "delta_sales": round(delta, 2),
            "delta_pct": round(pct, 2),
        })

    logs.append(f"🔮 [Simulation Stage] Evaluated {len(scenarios)} scenarios:")
    for r in sim_results:
        logs.append(f"    {r['scenario']}: ₹{r['predicted_sales']:,.0f} ({r['delta_pct']:+.1f}%)")

    # ── budget optimisation (SciPy SLSQP) ──────────────────────
    logs.append(f"🔮 [Simulation Stage] Optimising budget (₹{total_budget:,.0f})...")

    constraints = {"type": "eq", "fun": lambda x: x.sum() - total_budget}
    bounds = [(0, total_budget)] * len(spend_cols)
    x0 = np.full(len(spend_cols), total_budget / len(spend_cols))
    result = minimize(lambda x: -predict(x), x0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    opt_spend = result.x
    opt_sales = predict(opt_spend)
    curr_sales = predict(baseline_spend)

    optimal_allocation = []
    for i, col in enumerate(spend_cols):
        optimal_allocation.append({
            "channel": col,
            "current_spend": round(float(baseline_spend[i]), 2),
            "optimal_spend": round(float(opt_spend[i]), 2),
            "pct_of_budget": round(float(opt_spend[i] / total_budget * 100), 1),
        })

    opt_summary = {
        "total_budget": round(float(total_budget), 2),
        "current_predicted_sales": round(curr_sales, 2),
        "optimal_predicted_sales": round(opt_sales, 2),
        "uplift": round(float((opt_sales - curr_sales) / curr_sales * 100), 1)
                  if curr_sales else 0,
    }

    logs.append("🔮 [Simulation Stage] Optimal allocation:")
    for oa in optimal_allocation:
        logs.append(f"    {oa['channel']}: ₹{oa['optimal_spend']:,.0f} ({oa['pct_of_budget']}%)")
    logs.append(f"🔮 [Simulation Stage] Predicted uplift: {opt_summary['uplift']:+.1f}%")

    # ── marketing response curves ───────────────────────────────
    logs.append("🔮 [Simulation Stage] Computing response curves...")
    response_curves = {}
    for i, col in enumerate(spend_cols):
        max_s = float(df[col].max()) * 2.5
        spend_range = np.linspace(0, max_s, 60)
        preds = []
        for s in spend_range:
            test = baseline_spend.copy()
            test[i] = s
            preds.append(round(predict(test), 2))
        response_curves[col] = {
            "spend": [round(float(x), 2) for x in spend_range],
            "predicted_sales": preds,
        }

    logs.append("✅ [Simulation Stage] Simulations complete")

    return {
        "simulation_results": sim_results,
        "optimal_allocation": optimal_allocation,
        "optimization_summary": opt_summary,
        "response_curves": response_curves,
        "logs": logs,
    }
