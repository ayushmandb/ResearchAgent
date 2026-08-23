"""
Builds and compiles the StateGraph.

TODO:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from state import ResearchState
    from nodes import planner_node, approval_node, run_worker, writer_node, route_to_workers

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("approval", approval_node)
    graph.add_node("run_worker", run_worker)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "approval")
    graph.add_conditional_edges("approval", route_to_workers, ["run_worker"])
    graph.add_edge("run_worker", "writer")
    graph.add_edge("writer", END)

    compiled = graph.compile(checkpointer=MemorySaver())

Keep `compiled` as the single object app.py imports and streams against.
Test this file headless (plain python script / pytest) before touching Streamlit.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import ResearchState
from nodes import (
    planner_node, approval_node, route_to_workers, run_worker,
    section_planner_node, route_to_sections, section_writer_node, final_polish_node, critic_node, route_after_critic,
    )

graph = StateGraph(ResearchState)

graph.add_node("planner", planner_node)
graph.add_node("approval", approval_node)
graph.add_node("run_worker", run_worker)
graph.add_node("section_planner", section_planner_node)
graph.add_node("section_writer", section_writer_node)
graph.add_node("final_polish", final_polish_node)
graph.add_node("critic", critic_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "approval")
graph.add_conditional_edges("approval", route_to_workers, ["run_worker"])
graph.add_edge("run_worker", "section_planner")
graph.add_conditional_edges("section_planner", route_to_sections, ["section_writer"])
graph.add_edge("section_writer", "final_polish")
graph.add_edge("final_polish", "critic")
graph.add_conditional_edges("critic", route_after_critic, {"end": END, "revise": "final_polish"})
compiled = graph.compile(checkpointer=MemorySaver())