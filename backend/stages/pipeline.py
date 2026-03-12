from langgraph.graph import StateGraph, START, END

from backend.stages.state import MMMState
from backend.stages.data_stage import data_stage
from backend.stages.causal_stage import causal_stage
from backend.stages.modeling_stage import tuning_stage, training_stage
from backend.stages.simulation_stage import simulation_stage
from backend.stages.forecasting_stage import forecasting_stage
from backend.stages.strategy_stage import strategy_stage


def build_pipeline():
    """Build and compile the 7-stage MMM pipeline.

    Flow: START → data → causal → tuning → training → simulation → forecasting → strategy → END
    """
    graph = StateGraph(MMMState)

    graph.add_node("data_stage", data_stage)
    graph.add_node("causal_stage", causal_stage)
    graph.add_node("tuning_stage", tuning_stage)
    graph.add_node("training_stage", training_stage)
    graph.add_node("simulation_stage", simulation_stage)
    graph.add_node("forecasting_stage", forecasting_stage)
    graph.add_node("strategy_stage", strategy_stage)

    graph.add_edge(START, "data_stage")
    graph.add_edge("data_stage", "causal_stage")
    graph.add_edge("causal_stage", "tuning_stage")
    graph.add_edge("tuning_stage", "training_stage")
    graph.add_edge("training_stage", "simulation_stage")
    graph.add_edge("simulation_stage", "forecasting_stage")
    graph.add_edge("forecasting_stage", "strategy_stage")
    graph.add_edge("strategy_stage", END)

    return graph.compile()
