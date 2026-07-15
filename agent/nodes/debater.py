from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from agent.tools.evidence_search import evidence_search

load_dotenv()

TOOLS = [evidence_search]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(TOOLS)


class Argument(BaseModel):
    stance: str = Field(description="Either 'for' or 'against' the claim")
    content: str = Field(description="The final argument, 2-4 sentences")
    citations: list[str] = Field(description="Short quotes or facts used as evidence")


class DebaterState(TypedDict):
    claim: str
    stance: str
    messages: Annotated[list[BaseMessage], add_messages]
    argument: dict | None


def make_chat_node(stance: str):
    def chat_node(state: DebaterState) -> dict:
        if not state["messages"]:
            side = "FOR" if stance == "for" else "AGAINST"
            system_prompt = (
                f"You are debating {side} this claim: \"{state['claim']}\". "
                f"Use the evidence_search tool if you need facts before making your argument. "
                f"Keep it concise."
            )
            messages = [SystemMessage(content=system_prompt), HumanMessage(content="Make your case.")]
        else:
            messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return chat_node


def finalize_node(state: DebaterState) -> dict:
    structured_llm = llm.with_structured_output(Argument)
    closing_request = HumanMessage(content="Now summarize your final argument in the required format.")
    argument = structured_llm.invoke(state["messages"] + [closing_request])
    return {"argument": argument.model_dump()}


def build_debater_subgraph(stance: str):
    graph = StateGraph(DebaterState)
    graph.add_node("chat", make_chat_node(stance))
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition, {"tools": "tools", END: "finalize"})
    graph.add_edge("tools", "chat")
    graph.add_edge("finalize", END)

    return graph.compile()


proponent_graph = build_debater_subgraph("for")
skeptic_graph = build_debater_subgraph("against")


if __name__ == "__main__":
    result = proponent_graph.invoke(
        {"claim": "Coffee is good for your health", "stance": "for", "messages": [], "argument": None}
    )
    print("Final argument:")
    print(result["argument"])
