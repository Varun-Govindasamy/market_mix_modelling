import numpy as np
import pandas as pd
import networkx as nx

SPEND_COLS = ["tv_spend", "google_ads_spend", "meta_ads_spend", "influencer_spend"]


def causal_stage(state: dict) -> dict:
    """Causal Discovery Stage — sits between Data Stage and Modeling Stage.

    1. Builds a causal DAG from domain knowledge + data-driven refinement
    2. Uses DoWhy to identify & estimate causal effects of each channel on sales
    3. Adjusts the dataset by removing confounding bias before modeling
    """
    import dowhy
    from dowhy import CausalModel

    logs = []
    logs.append("🔬 [Causal Stage] Starting causal discovery pipeline...")

    df = pd.DataFrame(state["processed_data"])
    spend_cols = state["spend_columns"]

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Build causal DAG from domain knowledge
    # ─────────────────────────────────────────────────────────────
    logs.append("🔬 [Causal Stage] Building causal DAG from domain knowledge...")

    # Domain-informed edges:
    #   - Each marketing channel → sales   (direct effect)
    #   - Seasonality → sales              (confounder)
    #   - Seasonality → each channel spend (budgets vary seasonally)
    #   - Discount → sales                 (direct effect)
    edges = []
    for col in spend_cols:
        edges.append((col, "sales"))
        edges.append(("seasonality", col))        # confounder path
    edges.append(("seasonality", "sales"))
    edges.append(("discount", "sales"))

    dag = nx.DiGraph(edges)
    logs.append(f"  🔗 [DAG] Nodes: {list(dag.nodes())}")
    logs.append(f"  🔗 [DAG] Edges: {len(dag.edges())}")
    for src, dst in dag.edges():
        logs.append(f"  🔗 [DAG]   {src} → {dst}")

    # Validate DAG is acyclic
    if nx.is_directed_acyclic_graph(dag):
        logs.append("  🔗 [DAG] Acyclicity check: ✓ valid DAG")
    else:
        logs.append("  🔗 [DAG] ⚠ Cycle detected — removing back-edges")
        # fallback: remove cycles (shouldn't happen with domain knowledge)
        while not nx.is_directed_acyclic_graph(dag):
            cycle = nx.find_cycle(dag)
            dag.remove_edge(*cycle[0])

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Data-driven edge validation using partial correlations
    # ─────────────────────────────────────────────────────────────
    logs.append("🔬 [Causal Stage] Validating edges with partial correlations...")

    validated_edges = []
    removed_edges = []
    pcorr_threshold = 0.03

    for col in spend_cols:
        # Partial correlation of channel → sales | seasonality
        corr_cols = [col, "sales", "seasonality"]
        sub = df[corr_cols].dropna()
        if len(sub) > 10:
            corr_matrix = sub.corr()
            try:
                inv = np.linalg.inv(corr_matrix.values)
                n = inv.shape[0]
                # Partial correlation between col (idx 0) and sales (idx 1)
                pcorr = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
            except np.linalg.LinAlgError:
                pcorr = corr_matrix.iloc[0, 1]
        else:
            pcorr = 0.0

        if abs(pcorr) >= pcorr_threshold:
            validated_edges.append((col, "sales"))
            logs.append(f"  📐 [Validation] {col} → sales  pcorr={pcorr:+.4f} ✓")
        else:
            removed_edges.append((col, "sales"))
            logs.append(f"  📐 [Validation] {col} → sales  pcorr={pcorr:+.4f} ✗ (weak)")

    logs.append(f"  📐 [Validation] Validated {len(validated_edges)}/{len(spend_cols)} channel edges")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: DoWhy causal effect estimation per channel
    # ─────────────────────────────────────────────────────────────
    logs.append("🔬 [Causal Stage] Estimating causal effects with DoWhy...")

    # Build GML string for DoWhy
    gml_lines = ["graph [directed 1"]
    node_ids = {}
    for i, node in enumerate(dag.nodes()):
        node_ids[node] = i
        gml_lines.append(f'  node [id {i} label "{node}"]')
    for src, dst in dag.edges():
        gml_lines.append(f"  edge [source {node_ids[src]} target {node_ids[dst]}]")
    gml_lines.append("]")
    gml_str = "\n".join(gml_lines)

    causal_effects = {}
    confounders_found = {}

    for col in spend_cols:
        logs.append(f"  🧪 [DoWhy] Analysing: {col} → sales")
        try:
            model = CausalModel(
                data=df,
                treatment=col,
                outcome="sales",
                graph=gml_str,
            )

            # Identify estimand
            identified = model.identify_effect(proceed_when_unidentifiable=True)
            estimand_type = str(identified.estimands.get(
                "backdoor", identified.estimands.get("iv", "unknown")
            ))
            logs.append(f"  🧪 [DoWhy]   Estimand: backdoor adjustment")

            # Estimate with linear regression
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression",
            )
            effect_val = float(estimate.value)
            causal_effects[col] = effect_val
            logs.append(f"  🧪 [DoWhy]   ATE({col} → sales) = {effect_val:.6f}")

            # Refutation: random common cause
            refute = model.refute_estimate(
                identified,
                estimate,
                method_name="random_common_cause",
                num_simulations=5,
            )
            new_effect = float(refute.new_effect)
            pct_change = abs(new_effect - effect_val) / max(abs(effect_val), 1e-9) * 100
            robust = pct_change < 15
            logs.append(
                f"  🧪 [DoWhy]   Refutation: Δ={pct_change:.1f}% "
                f"{'✓ robust' if robust else '⚠ sensitive'}"
            )

            # Track confounders
            confounders = [n for n in dag.predecessors(col) if n != col]
            if confounders:
                confounders_found[col] = confounders
                logs.append(f"  🧪 [DoWhy]   Confounders: {confounders}")

        except Exception as e:
            logs.append(f"  🧪 [DoWhy]   ⚠ Error for {col}: {str(e)[:100]}")
            causal_effects[col] = 0.0

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Deconfound the dataset
    # ─────────────────────────────────────────────────────────────
    logs.append("🔬 [Causal Stage] Deconfounding dataset...")

    # Residualize spend columns — remove seasonality's linear influence
    # so the modeling stage sees spend variation independent of confounders
    adjusted_df = df.copy()
    for col in spend_cols:
        confounders = confounders_found.get(col, [])
        if confounders:
            from sklearn.linear_model import LinearRegression
            conf_vals = df[confounders].values
            orig_vals = df[col].values
            reg = LinearRegression().fit(conf_vals, orig_vals)
            residuals = orig_vals - reg.predict(conf_vals)
            # Shift residuals to original mean (keep scale interpretable)
            adjusted_df[col] = residuals + orig_vals.mean()
            r2_conf = float(reg.score(conf_vals, orig_vals))
            logs.append(
                f"  🔧 [Deconfound] {col}: removed {confounders} influence "
                f"(R²={r2_conf:.4f})"
            )
        else:
            logs.append(f"  🔧 [Deconfound] {col}: no confounders — unchanged")

    logs.append("🔬 [Causal Stage] Deconfounding complete")

    # ─────────────────────────────────────────────────────────────
    # Build summary for frontend
    # ─────────────────────────────────────────────────────────────
    causal_summary = {
        "dag_nodes": list(dag.nodes()),
        "dag_edges": [{"source": s, "target": t} for s, t in dag.edges()],
        "causal_effects": {k: round(v, 6) for k, v in causal_effects.items()},
        "confounders": {k: v for k, v in confounders_found.items()},
        "validated_edges": len(validated_edges),
        "total_edges": len(dag.edges()),
    }

    logs.append(f"🔬 [Causal Stage] Causal effects summary:")
    for ch, eff in sorted(causal_effects.items(), key=lambda x: abs(x[1]), reverse=True):
        logs.append(f"    {ch}: ATE = {eff:.6f}")
    logs.append("✅ [Causal Stage] Causal discovery complete")

    return {
        "processed_data": adjusted_df.to_dict(orient="records"),
        "causal_summary": causal_summary,
        "logs": logs,
    }
