import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DebateState(TypedDict):
    claim: str
    round: int
    max_rounds: int
    proponent_args: Annotated[list[str], operator.add]
    skeptic_args: Annotated[list[str], operator.add]
    verdicts: Annotated[list[dict], operator.add]
    final_verdict: str | None
    citations: Annotated[list[str], operator.add]
    messages: Annotated[list[BaseMessage], add_messages]
