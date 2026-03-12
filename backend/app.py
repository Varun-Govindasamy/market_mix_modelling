import json
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd

from backend.stages.pipeline import build_pipeline
from backend.config import DATASET_PATH


app = FastAPI(title="MMM Pipeline API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache of the last pipeline run (single-user prototype)
_pipeline_cache: dict = {}

# Dir for uploaded datasets
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "mmm_pipeline_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Request / Response schemas ──────────────────────────────────
class PipelineRequest(BaseModel):
    dataset_path: str = DATASET_PATH
    total_budget: float = 35000.0
    user_scenarios: list = []


class SimulateRequest(BaseModel):
    spend: dict  # channel_name → spend amount


# Which stage comes next in the fixed pipeline order?
_NEXT_STAGE = {
    "data_stage": "causal_stage",
    "causal_stage": "tuning_stage",
    "tuning_stage": "training_stage",
    "training_stage": "simulation_stage",
    "simulation_stage": "forecasting_stage",
    "forecasting_stage": "strategy_stage",
}


def _run_pipeline_stream(dataset_path: str, total_budget: float, user_scenarios: list = None):
    """Shared generator: runs the pipeline and yields NDJSON log/result events."""
    pipeline = build_pipeline()

    initial_state = {
        "dataset_path": dataset_path,
        "total_budget": total_budget,
        "user_scenarios": user_scenarios or [],
        "logs": [],
    }

    # Emit the very first phase so the animation starts immediately
    yield json.dumps({"type": "phase", "stage": "data_stage"}) + "\n"

    all_logs: list[str] = []
    final_state: dict = {}

    for event in pipeline.stream(initial_state, stream_mode="updates"):
        for node_name, updates in event.items():
            # Emit new logs produced by this node
            for log_msg in updates.get("logs", []):
                all_logs.append(log_msg)
                print(log_msg, flush=True)
                yield json.dumps({"type": "log", "message": log_msg}) + "\n"
            # Merge updates into final_state
            final_state.update(updates)
            # Signal the *next* stage phase so the animation transitions
            # while that stage is running
            next_stage = _NEXT_STAGE.get(node_name)
            if next_stage:
                yield json.dumps({"type": "phase", "stage": next_stage}) + "\n"

    final_state["logs"] = all_logs
    _pipeline_cache.update(final_state)

    result = {
        "data_summary":          final_state.get("data_summary", {}),
        "raw_data":              final_state.get("raw_data", []),
        "spend_columns":         final_state.get("spend_columns", []),
        "causal_summary":        final_state.get("causal_summary", {}),
        "best_params":           final_state.get("best_params", {}),
        "model_coef":            final_state.get("model_coef", []),
        "model_intercept":       final_state.get("model_intercept", 0.0),
        "feature_mean":          final_state.get("feature_mean", []),
        "feature_std":           final_state.get("feature_std", []),
        "target_mean":           final_state.get("target_mean", 0.0),
        "target_std":            final_state.get("target_std", 0.0),
        "model_metrics":         final_state.get("model_metrics", {}),
        "channel_contributions": final_state.get("channel_contributions", []),
        "roi_per_channel":       final_state.get("roi_per_channel", []),
        "actual_sales":          final_state.get("actual_sales", []),
        "predicted_sales":       final_state.get("predicted_sales", []),
        "simulation_results":    final_state.get("simulation_results", []),
        "optimal_allocation":    final_state.get("optimal_allocation", []),
        "optimization_summary":  final_state.get("optimization_summary", {}),
        "response_curves":       final_state.get("response_curves", {}),
        "forecast":              final_state.get("forecast", []),
        "forecast_decomposition": final_state.get("forecast_decomposition", {}),
        "strategy_text":         final_state.get("strategy_text", ""),
    }

    yield json.dumps({"type": "result", "data": result}) + "\n"


# ── Endpoints ───────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/pipeline/upload")
def upload_and_run(
    file: UploadFile = File(...),
    total_budget: float = Form(35000.0),
):
    """Accept a CSV upload, save it, and stream the pipeline."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(file.file.read())

    return StreamingResponse(
        _run_pipeline_stream(save_path, total_budget),
        media_type="application/x-ndjson",
    )


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRequest):
    """Run the full 7-stage MMM pipeline and stream logs + results as NDJSON."""

    return StreamingResponse(
        _run_pipeline_stream(request.dataset_path, request.total_budget, request.user_scenarios),
        media_type="application/x-ndjson",
    )


@app.post("/api/simulate")
def simulate(request: SimulateRequest):
    """Predict sales for a custom spend mix using the cached model."""
    if not _pipeline_cache:
        raise HTTPException(400, "Run the pipeline first.")

    spend_cols = _pipeline_cache["spend_columns"]
    best_params = _pipeline_cache["best_params"]
    coef = np.array(_pipeline_cache["model_coef"])
    intercept = _pipeline_cache["model_intercept"]
    X_mean = np.array(_pipeline_cache["feature_mean"])
    X_std = np.array(_pipeline_cache["feature_std"])
    y_mean = _pipeline_cache["target_mean"]
    y_std = _pipeline_cache["target_std"]

    df = pd.DataFrame(_pipeline_cache["processed_data"])

    features = []
    for col in spend_cols:
        alpha = best_params[f"{col}_alpha"]
        spend = request.spend.get(col, float(df[col].mean()))
        features.append(1 - np.exp(-alpha * spend))
    features.append(float(df["discount"].median()))
    features.append(float(df["seasonality"].median()))
    features = np.array(features)

    scaled = (features - X_mean) / X_std
    pred = float((intercept + scaled @ coef) * y_std + y_mean)

    return {"predicted_sales": round(pred, 2)}
