import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from backend.config import OPENAI_API_KEY, OPENAI_MODEL


SYSTEM_PROMPT = """\
You are a senior marketing strategist analysing Market Mix Modeling results.
Based on the data provided, write a clear, actionable strategy recommendation.

Include these sections (use markdown formatting):
1. **Executive Summary** — 2-3 sentences on key findings
2. **Channel Performance** — which channels drive the most value & ROI
3. **Budget Reallocation** — specific changes with ₹ amounts and percentages
4. **Scenario Insights** — what the simulations reveal
5. **Action Items** — 4-5 concrete next steps
6. **Expected Impact** — projected sales lift and confidence
7. **Risks & Caveats** — things to watch out for

Use the actual numbers from the data. Be direct, no fluff."""


def strategy_stage(state: dict) -> dict:
    logs = []
    logs.append("💡 [Strategy Stage] Generating marketing strategy...")

    contributions = state["channel_contributions"]
    roi = state["roi_per_channel"]
    sim_results = state["simulation_results"]
    opt_alloc = state["optimal_allocation"]
    opt_summary = state["optimization_summary"]
    metrics = state["model_metrics"]

    # ── build LLM context ───────────────────────────────────────
    context = f"""## MMM Analysis Results

### Model Performance
- R²: {metrics['r2']}
- MAPE: {metrics['mape']}%
- RMSE: ₹{metrics['rmse']:,.0f}

### Channel Contributions
{json.dumps(contributions, indent=2)}

### ROI per Channel
{json.dumps(roi, indent=2)}

### Counterfactual Scenarios
{json.dumps(sim_results, indent=2)}

### Optimal Budget Allocation
{json.dumps(opt_alloc, indent=2)}

### Optimisation Summary
- Current predicted weekly sales: ₹{opt_summary['current_predicted_sales']:,.0f}
- Optimal predicted weekly sales: ₹{opt_summary['optimal_predicted_sales']:,.0f}
- Expected uplift: {opt_summary['uplift']}%
"""

    # ── call LLM ────────────────────────────────────────────────
    try:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")

        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.3,
        )
        logs.append(f"💡 [Strategy Stage] Calling {OPENAI_MODEL}...")

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
        strategy_text = response.content
        logs.append("💡 [Strategy Stage] LLM response received ✓")

    except Exception as e:
        logs.append(f"⚠️  [Strategy Stage] LLM unavailable ({e}), using rule-based fallback")
        strategy_text = _fallback_strategy(contributions, roi, opt_alloc, opt_summary, metrics)

    logs.append("✅ [Strategy Stage] Strategy generation complete")

    return {
        "strategy_text": strategy_text,
        "logs": logs,
    }


# ── deterministic fallback (no LLM needed) ─────────────────────
def _fallback_strategy(contributions, roi, opt_alloc, opt_summary, metrics):
    sorted_roi = sorted(roi, key=lambda x: x["roi"], reverse=True)
    sorted_contrib = sorted(contributions, key=lambda x: x["contribution_pct"], reverse=True)
    best_roi = sorted_roi[0]
    worst_roi = sorted_roi[-1]
    top = sorted_contrib[0]

    alloc_lines = "\n".join(
        f"- **{a['channel']}**: ₹{a['optimal_spend']:,.0f} ({a['pct_of_budget']}%)"
        for a in opt_alloc
    )

    return f"""## Marketing Strategy Recommendation

### Executive Summary
The MMM model (R² = {metrics['r2']}) reveals that **{top['channel']}** is the
strongest sales driver at **{top['contribution_pct']}%** contribution.
Reallocating the budget to the optimal mix is projected to deliver a
**{opt_summary['uplift']:+.1f}%** uplift in weekly sales.

### Channel Performance
| Channel | Contribution | ROI |
|---------|-------------|-----|
""" + "\n".join(
        f"| {c['channel']} | {c['contribution_pct']}% | "
        f"{next((r['roi'] for r in roi if r['channel'] == c['channel']), 'N/A')}× |"
        for c in sorted_contrib
    ) + f"""

### Budget Reallocation
{alloc_lines}

### Action Items
1. **Increase {best_roi['channel']}** — highest ROI at {best_roi['roi']}× return
2. **Reduce {worst_roi['channel']}** — lowest ROI at {worst_roi['roi']}× return
3. Shift 10-15% of budget incrementally before full commitment
4. Re-run the model monthly to track changing dynamics
5. A/B test the new allocation for 4 weeks before scaling

### Expected Impact
- Current weekly sales: ₹{opt_summary['current_predicted_sales']:,.0f}
- Projected weekly sales: ₹{opt_summary['optimal_predicted_sales']:,.0f}
- **Uplift: {opt_summary['uplift']:+.1f}%**

### Risks & Caveats
- Model trained on {metrics['r2']*100:.0f}% of variance; remaining variance is unexplained
- Saturation effects may shift if competitor landscape changes
- Macro-economic factors (inflation, demand shocks) are not modelled
"""
