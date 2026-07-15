from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agent.state import DebateState

load_dotenv()


class RoundVerdict(BaseModel):
    winner: Literal["proponent", "skeptic", "tie"] = Field(
        description="Who made the stronger case this round"
    )
    reasoning: str = Field(description="A short explanation of the decision")
    concluded: bool = Field(
        description="True if the debate is settled and no more rounds are needed"
    )


llm = ChatOpenAI(model="gpt-4o-mini")
judge_llm = llm.with_structured_output(RoundVerdict)


def judge_node(state: DebateState) -> dict:
    prompt = f"""Claim: {state['claim']}

Proponent's argument this round: {state['proponent_args'][-1]}

Skeptic's argument this round: {state['skeptic_args'][-1]}

Prior round verdicts so far: {state['verdicts']}

Decide which side made the stronger case THIS round, explain briefly why,
and say whether the debate is settled (concluded) or should continue.
"""
    verdict = judge_llm.invoke(prompt)
    verdict_dict = verdict.model_dump()

    # Genuinely undecided, and we've already given it more than one round -> pause for a human.
    if verdict.winner == "tie" and state["round"] >= 2:
        human_ruling = interrupt(
            {
                "reason": "Judge is still undecided after multiple rounds.",
                "claim": state["claim"],
                "proponent_argument": state["proponent_args"][-1],
                "skeptic_argument": state["skeptic_args"][-1],
                "question": "Who should win: 'proponent', 'skeptic', or 'tie'? And why?",
            }
        )
        verdict_dict["winner"] = human_ruling["winner"]
        verdict_dict["reasoning"] = f"Human tiebreak: {human_ruling['note']}"
        verdict_dict["concluded"] = True

    return {"verdicts": [verdict_dict]}


if __name__ == "__main__":
    fake_state: DebateState = {
        "claim": "Coffee is good for your health",
        "round": 1,
        "max_rounds": 2,
        "proponent_args": ["Coffee is linked to lower risk of type 2 diabetes and Parkinson's disease."],
        "skeptic_args": ["High caffeine intake can increase anxiety and disrupt sleep quality."],
        "verdicts": [],
        "final_verdict": None,
        "citations": [],
        "messages": [],
    }
    result = judge_node(fake_state)
    print("Verdict:")
    print(result)
