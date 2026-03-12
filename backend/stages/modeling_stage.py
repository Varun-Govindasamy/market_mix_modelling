import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

from backend.utils.transforms import adstock_transform, saturation_transform


def _build_features(df, spend_cols, params):
    """Transform spend columns using adstock + saturation with given hyperparams."""
    feats = pd.DataFrame()
    for col in spend_cols:
        decay = params[f"{col}_decay"]
        alpha = params[f"{col}_alpha"]
        adstocked = adstock_transform(df[col], decay)
        feats[col] = saturation_transform(adstocked, alpha)
    feats["discount"] = df["discount"].values
    feats["seasonality"] = df["seasonality"].values
    return feats


def tuning_stage(state: dict) -> dict:
    logs = []
    logs.append("🔍 [Tuning Stage] Starting hyperparameter search...")

    df = pd.DataFrame(state["processed_data"])
    spend_cols = state["spend_columns"]
    y = df["sales"].values

    logs.append(f"  🔍 [Optuna] Search space: {len(spend_cols)} channels × 2 params (decay, alpha)")
    for col in spend_cols:
        logs.append(f"  🔍 [Optuna]   {col}_decay ∈ [0.0, 0.9] │ {col}_alpha ∈ [1e-6, 5e-4] (log)")
    logs.append("  🔍 [Optuna] Sampler: TPE │ Scoring: 5-fold CV R² │ Estimator: Ridge")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {}
        for col in spend_cols:
            params[f"{col}_decay"] = trial.suggest_float(f"{col}_decay", 0.0, 0.9)
            params[f"{col}_alpha"] = trial.suggest_float(f"{col}_alpha", 1e-6, 5e-4, log=True)
        X = _build_features(df, spend_cols, params).values
        scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring="r2")
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    _best_score_so_far = [float("-inf")]
    def _optuna_callback(study, trial):
        n = trial.number + 1
        is_best = trial.value > _best_score_so_far[0]
        if is_best:
            _best_score_so_far[0] = trial.value
        if n == 1 or n % 10 == 0 or is_best or n == 100:
            tag = " ★ NEW BEST" if is_best else ""
            logs.append(f"  🔍 [Optuna] Trial {n:>3}/100 │ CV R² = {trial.value:.4f}{tag}")

    study.optimize(objective, n_trials=100, callbacks=[_optuna_callback])

    best_params = study.best_params
    logs.append("  🔍 [Optuna] ── Search complete ──")
    logs.append(f"  🔍 [Optuna] Best trial: #{study.best_trial.number + 1} │ CV R² = {study.best_value:.4f}")
    top_trials = sorted(study.trials, key=lambda t: t.value or 0.0, reverse=True)[:3]
    for rank, t in enumerate(top_trials, 1):
        logs.append(f"  🔍 [Optuna] Top-{rank}: Trial #{t.number + 1} │ R² = {t.value:.4f}")
    logs.append("🔍 [Tuning Stage] Best hyperparameters:")
    for k, v in best_params.items():
        logs.append(f"    {k}: {v:.6f}")

    logs.append("✅ [Tuning Stage] Hyperparameter search complete")

    return {
        "best_params": best_params,
        "logs": logs,
    }


def training_stage(state: dict) -> dict:
    import pymc as pm
    import arviz as az

    logs = []
    logs.append("🧠 [Training Stage] Starting Bayesian model training...")

    df = pd.DataFrame(state["processed_data"])
    spend_cols = state["spend_columns"]
    best_params = state["best_params"]
    y = df["sales"].values

    X_best = _build_features(df, spend_cols, best_params).values
    feature_names = spend_cols + ["discount", "seasonality"]
    n_features = len(feature_names)

    X_mean = X_best.mean(axis=0)
    X_std = X_best.std(axis=0)
    X_scaled = (X_best - X_mean) / X_std

    y_mean = float(y.mean())
    y_std = float(y.std())
    y_scaled = (y - y_mean) / y_std

    logs.append(f"  ⛓ [NUTS] Features ({n_features}): {', '.join(feature_names)}")
    logs.append("  ⛓ [NUTS] Config: 2 chains × 1000 draws + 500 tuning steps (cores=1)")
    logs.append("  ⛓ [NUTS] Priors: β ~ Normal(0,1) │ intercept ~ Normal(0,1) │ σ ~ HalfNormal(1)")
    logs.append("  ⛓ [NUTS] Initializing sampler & tuning step size...")

    with pm.Model() as mmm_model:
        # Priors
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_features)
        alpha_intercept = pm.Normal("intercept", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Linear model
        mu = alpha_intercept + pm.math.dot(X_scaled, beta)

        # Likelihood
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)

        # Posterior sampling
        trace = pm.sample(
            draws=1000,
            tune=500,
            cores=1,
            chains=2,
            random_seed=42,
            progressbar=False,
        )

    # Extract posterior means as point estimates
    summary_df = az.summary(trace, var_names=["beta", "intercept", "sigma"])
    coef = az.summary(trace, var_names=["beta"])["mean"].values
    intercept = float(az.summary(trace, var_names=["intercept"])["mean"].values[0])

    logs.append("  ⛓ [NUTS] Sampling complete")
    ess_bulk = summary_df["ess_bulk"].values
    r_hat = summary_df["r_hat"].values
    logs.append(f"  ⛓ [NUTS] Effective sample size: min={ess_bulk.min():.0f}, median={np.median(ess_bulk):.0f}")
    converged = r_hat.max() < 1.05
    logs.append(f"  ⛓ [NUTS] R̂ convergence: max={r_hat.max():.4f} {'✓ chains converged' if converged else '⚠ check convergence'}")
    logs.append("  ⛓ [NUTS] Posterior mean coefficients:")
    for i, name in enumerate(feature_names):
        logs.append(f"    β[{name}] = {coef[i]:.4f}")
    logs.append(f"  ⛓ [NUTS] Intercept = {intercept:.4f}")

    y_pred_scaled = intercept + X_scaled @ coef
    y_pred = y_pred_scaled * y_std + y_mean

    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    logs.append(f"🧠 [Training Stage] R² = {r2:.4f}")
    logs.append(f"🧠 [Training Stage] MAPE = {mape:.2f}%")
    logs.append(f"🧠 [Training Stage] RMSE = ₹{rmse:,.0f}")

    # ── channel contributions ───────────────────────────────────
    abs_contribs = np.abs(X_scaled * coef)
    totals = abs_contribs.sum(axis=0)
    pct = (totals / totals.sum()) * 100

    channel_contributions = []
    for i, name in enumerate(feature_names):
        channel_contributions.append({
            "channel": name,
            "contribution_pct": round(float(pct[i]), 1),
            "coefficient": round(float(coef[i]), 6),
        })

    logs.append("🧠 [Training Stage] Channel contributions:")
    for cc in sorted(channel_contributions, key=lambda x: x["contribution_pct"], reverse=True):
        logs.append(f"    {cc['channel']}: {cc['contribution_pct']}%")

    # ── ROI per channel ─────────────────────────────────────────
    roi_list = []
    for i, col in enumerate(spend_cols):
        total_spend = float(df[col].sum())
        attr = float(np.abs(X_scaled[:, i] * coef[i]).sum() * y_std)
        roi = attr / total_spend if total_spend > 0 else 0
        roi_list.append({
            "channel": col,
            "total_spend": round(total_spend, 2),
            "attributed_sales": round(attr, 2),
            "roi": round(roi, 4),
        })

    logs.append("🧠 [Training Stage] ROI per channel:")
    for r in sorted(roi_list, key=lambda x: x["roi"], reverse=True):
        logs.append(f"    {r['channel']}: {r['roi']}x")

    logs.append("✅ [Training Stage] Model training complete")

    return {
        "model_coef": coef.tolist(),
        "model_intercept": intercept,
        "feature_mean": X_mean.tolist(),
        "feature_std": X_std.tolist(),
        "target_mean": y_mean,
        "target_std": y_std,
        "channel_contributions": channel_contributions,
        "roi_per_channel": roi_list,
        "model_metrics": {
            "r2": round(float(r2), 4),
            "mape": round(float(mape), 2),
            "rmse": round(float(rmse), 2),
        },
        "actual_sales": y.tolist(),
        "predicted_sales": y_pred.tolist(),
        "logs": logs,
    }
