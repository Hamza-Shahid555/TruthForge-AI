from typing import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph


class LoopState(TypedDict):
    count: int


def tick(state: LoopState) -> dict:
    print(f"tick {state['count']}")
    return {"count": state["count"] + 1}


def always_loop(state: LoopState) -> str:
    return "tick"  # never says "stop" -> loops forever


graph = StateGraph(LoopState)
graph.add_node("tick", tick)
graph.add_edge(START, "tick")
graph.add_conditional_edges("tick", always_loop, {"tick": "tick"})
app = graph.compile()

print("--- Running a loop with NO real exit condition ---")
try:
    app.invoke({"count": 0})
except GraphRecursionError as e:
    print(f"\nSTOPPED by LangGraph's built-in safety net: {type(e).__name__}: {e}")
