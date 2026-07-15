import operator
from typing import Annotated, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph


# --- Version 1: a field with NO reducer, written by two parallel nodes ---
class BadState(TypedDict):
    counter: int  # no Annotated -> default "last write wins" behavior


def node_a_bad(state: BadState) -> dict:
    return {"counter": 1}


def node_b_bad(state: BadState) -> dict:
    return {"counter": 2}


bad_graph = StateGraph(BadState)
bad_graph.add_node("a", node_a_bad)
bad_graph.add_node("b", node_b_bad)
bad_graph.add_edge(START, "a")
bad_graph.add_edge(START, "b")  # fan-out: a and b run in the SAME super-step
bad_graph.add_edge("a", END)
bad_graph.add_edge("b", END)
bad_app = bad_graph.compile()

print("--- No reducer, parallel writes to the same key ---")
try:
    bad_app.invoke({"counter": 0})
except InvalidUpdateError as e:
    print(f"CRASHED: {type(e).__name__}: {e}")


# --- Version 2: a field WITH operator.add, written by two parallel nodes ---
class GoodState(TypedDict):
    log: Annotated[list[str], operator.add]


def node_a_good(state: GoodState) -> dict:
    return {"log": ["node_a ran"]}


def node_b_good(state: GoodState) -> dict:
    return {"log": ["node_b ran"]}


good_graph = StateGraph(GoodState)
good_graph.add_node("a", node_a_good)
good_graph.add_node("b", node_b_good)
good_graph.add_edge(START, "a")
good_graph.add_edge(START, "b")
good_graph.add_edge("a", END)
good_graph.add_edge("b", END)
good_app = good_graph.compile()

print("\n--- With operator.add reducer, parallel writes to the same key ---")
result = good_app.invoke({"log": []})
print(result)
