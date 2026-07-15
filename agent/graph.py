import os
import sqlite3

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.nodes.debater import proponent_graph, skeptic_graph
from agent.nodes.judge import judge_node
from agent.state import DebateState


def round_manager(state: DebateState) -> dict:
    next_round = state["round"] + 1
    print(f"[round_manager] starting round {next_round}")
    return {"round": next_round}


def proponent(state: DebateState) -> dict:
    print("[proponent] researching and arguing FOR the claim...")
    result = proponent_graph.invoke(
        {"claim": state["claim"], "stance": "for", "messages": [], "argument": None}
    )
    argument = result["argument"]
    return {"proponent_args": [argument["content"]], "citations": argument["citations"]}


def skeptic(state: DebateState) -> dict:
    print("[skeptic] researching and arguing AGAINST the claim...")
    result = skeptic_graph.invoke(
        {"claim": state["claim"], "stance": "against", "messages": [], "argument": None}
    )
    argument = result["argument"]
    return {"skeptic_args": [argument["content"]], "citations": argument["citations"]}


def finalize(state: DebateState) -> dict:
    proponent_wins = sum(1 for v in state["verdicts"] if v["winner"] == "proponent")
    skeptic_wins = sum(1 for v in state["verdicts"] if v["winner"] == "skeptic")
    if proponent_wins > skeptic_wins:
        overall = "proponent"
    elif skeptic_wins > proponent_wins:
        overall = "skeptic"
    else:
        overall = "tie"
    summary = (
        f"After {state['round']} round(s): Proponent won {proponent_wins}, "
        f"Skeptic won {skeptic_wins}. Overall winner: {overall}."
    )
    print(f"[finalize] {summary}")
    return {"final_verdict": summary}


def route_debate(state: DebateState) -> str:
    if state["round"] >= state["max_rounds"]:
        return "finalize"
    if state["verdicts"] and state["verdicts"][-1]["concluded"]:
        return "finalize"
    return "round_manager"


graph = StateGraph(DebateState)
graph.add_node("round_manager", round_manager)
graph.add_node("proponent", proponent)
graph.add_node("skeptic", skeptic)
graph.add_node("judge", judge_node)
graph.add_node("finalize", finalize)

graph.add_edge(START, "round_manager")
graph.add_edge("round_manager", "proponent")
graph.add_edge("round_manager", "skeptic")
graph.add_edge("proponent", "judge")
graph.add_edge("skeptic", "judge")
graph.add_conditional_edges(
    "judge",
    route_debate,
    {"round_manager": "round_manager", "finalize": "finalize"},
)
graph.add_edge("finalize", END)

# Same checkpointer INTERFACE either way - only the storage backend changes.
# Locally (no DATABASE_URL): a single file, great for learning.
# In Docker Compose (DATABASE_URL set): a real Postgres database, built for production.
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver

    pg_conn = psycopg.connect(DATABASE_URL, autocommit=True)
    checkpointer = PostgresSaver(pg_conn)
    checkpointer.setup()
else:
    from langgraph.checkpoint.sqlite import SqliteSaver

    # check_same_thread=False: needed because LangGraph may touch this connection
    # from more than one internal thread during a single run.
    conn = sqlite3.connect("data/checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

workflow = graph.compile(checkpointer=checkpointer)


def _run_and_print(stream_input, config):
    """Run the graph, printing progress live, and handling a human-tiebreak pause if it happens."""
    for update in workflow.stream(stream_input, config=config, stream_mode="updates"):
        if "__interrupt__" in update:
            payload = update["__interrupt__"][0].value
            print("\n>>> PAUSED — the Judge needs a human tiebreak <<<")
            print(f"Reason: {payload['reason']}")
            print(f"Claim: {payload['claim']}")
            print(f"Proponent argued: {payload['proponent_argument']}")
            print(f"Skeptic argued: {payload['skeptic_argument']}")
            winner = input("Your ruling - proponent, skeptic, or tie? ").strip()
            note = input("Why? ").strip()
            _run_and_print(Command(resume={"winner": winner, "note": note}), config)
        else:
            for node_name, node_output in update.items():
                print(f"  >> [{node_name}] just finished, returned: {node_output}")


if __name__ == "__main__":
    # New slot, so this run starts fresh instead of resuming a finished debate.
    config = {"configurable": {"thread_id": "coffee-debate-3"}}

    initial_state = {
        "claim": "Coffee is good for your health",
        "round": 0,
        "max_rounds": 2,
        "proponent_args": [],
        "skeptic_args": [],
        "verdicts": [],
        "final_verdict": None,
        "citations": [],
        "messages": [],
    }

    print("--- Streaming live progress ---")
    _run_and_print(initial_state, config)

    print("\n--- Proving it was actually saved ---")
    saved = workflow.get_state(config)
    print("Re-read from data/checkpoints.sqlite:", saved.values["final_verdict"])
