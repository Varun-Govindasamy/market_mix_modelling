from typing import TypedDict, Annotated
import operator


class MMMState(TypedDict):
    # ── input ───────────────────────────────────────────────────
    dataset_path: str
    total_budget: float
    forecast_weeks: int
    user_scenarios: list

    # ── data stage ──────────────────────────────────────────────
    raw_data: list          # list of row dicts
    processed_data: list    # list of row dicts (after transforms)
    data_summary: dict
    spend_columns: list

    # ── causal stage ────────────────────────────────────────────
    causal_summary: dict

    # ── tuning stage ────────────────────────────────────────────
    best_params: dict

    # ── training stage ──────────────────────────────────────────
    model_coef: list
    model_intercept: float
    feature_mean: list
    feature_std: list
    target_mean: float
    target_std: float
    channel_contributions: list
    roi_per_channel: list
    model_metrics: dict
    actual_sales: list
    predicted_sales: list

    # ── simulation stage ────────────────────────────────────────
    simulation_results: list
    optimal_allocation: list
    optimization_summary: dict
    response_curves: dict

    # ── forecasting stage ───────────────────────────────────────
    forecast: list
    forecast_decomposition: dict

    # ── strategy stage ──────────────────────────────────────────
    strategy_text: str

    # ── logs (append-only across nodes) ─────────────────────────
    logs: Annotated[list, operator.add]
