# MMM Pipeline

A **Market Mix Modeling** pipeline with a retro-themed Streamlit dashboard. Runs a 7-stage LangGraph pipeline that ingests marketing spend data, discovers causal relationships, trains a Bayesian media-mix model, optimises budget allocation, forecasts future sales, and generates an AI strategy recommendation.

## Architecture

```
Streamlit frontend  ──HTTP/NDJSON──▶  FastAPI backend
                                          │
                                    LangGraph pipeline
                                          │
          ┌───────┬────────┬────────┬─────┴──────┬─────────────┬──────────┐
        DATA   CAUSAL   TUNING  TRAINING  SIMULATION  FORECASTING  STRATEGY
```

### Pipeline stages

| # | Stage | What it does |
|---|-------|-------------|
| 1 | **Data** | Loads CSV, engineers features (ad-stock, seasonality, etc.) |
| 2 | **Causal** | Builds a DAG with DoWhy + NetworkX, estimates causal effects |
| 3 | **Tuning** | Optuna TPE search (100 trials, 5-fold CV) for ad-stock α params |
| 4 | **Training** | PyMC Bayesian regression (NUTS, 2 chains × 1000 draws) |
| 5 | **Simulation** | Counterfactual scenarios, SciPy SLSQP budget optimisation, response curves |
| 6 | **Forecasting** | STL decomposition (trend + seasonality) + marketing effect → 12-week forecast with 80%/95% prediction intervals |
| 7 | **Strategy** | LLM-generated executive recommendation (LangChain + OpenAI) |

## Tech stack

- **Pipeline orchestration** — LangGraph
- **Bayesian modelling** — PyMC, Optuna, ArviZ
- **Causal inference** — DoWhy, NetworkX
- **Forecasting** — statsmodels (STL), NumPy
- **Optimisation** — SciPy (SLSQP)
- **LLM** — LangChain + OpenAI
- **Backend** — FastAPI, NDJSON streaming
- **Frontend** — Streamlit, Plotly

## Quick start

```bash
# 1. Clone & enter
git clone <repo-url> && cd agentic_mmm

# 2. Create venv & install deps (requires uv)
uv sync

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 4. Start the backend
uvicorn backend.app:app --reload

# 5. Start the frontend (separate terminal)
streamlit run frontend/app.py
```

## Project structure

```
├── backend/
│   ├── app.py                 # FastAPI server, NDJSON streaming
│   ├── config.py              # Paths & settings
│   └── stages/
│       ├── pipeline.py        # LangGraph 7-stage graph
│       ├── state.py           # MMMState TypedDict
│       ├── data_stage.py      # Data ingestion & feature engineering
│       ├── causal_stage.py    # Causal discovery (DoWhy)
│       ├── modeling_stage.py  # Tuning (Optuna) + Training (PyMC)
│       ├── simulation_stage.py# Scenarios, optimisation, response curves
│       ├── forecasting_stage.py # STL decomposition + forecast
│       └── strategy_stage.py  # LLM strategy generation
├── frontend/
│   └── app.py                 # Streamlit retro CRT dashboard
├── mmm_dataset.csv            # Sample dataset
├── pyproject.toml             # Dependencies (uv/pip)
└── .env.example               # Environment variable template
```