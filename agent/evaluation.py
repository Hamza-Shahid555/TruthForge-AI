import uuid

from dotenv import load_dotenv
from langgraph.types import Command
from langsmith import Client
from langsmith.evaluation import evaluate

from agent.graph import workflow

load_dotenv()

client = Client()

DATASET_NAME = "truthforge-ai-claims"

# The "answer key": claims with a clear, known correct side.
EXAMPLES = [
    {"claim": "The Earth is round", "expected_winner": "proponent"},
    {"claim": "The Earth is flat", "expected_winner": "skeptic"},
    {"claim": "Vaccines cause autism", "expected_winner": "skeptic"},
    {"claim": "Smoking is harmful to health", "expected_winner": "proponent"},
    {"claim": "The moon landing was faked", "expected_winner": "skeptic"},
    {"claim": "Regular exercise improves mental health", "expected_winner": "proponent"},
]


def _get_or_create_dataset():
    if client.has_dataset(dataset_name=DATASET_NAME):
        return client.read_dataset(dataset_name=DATASET_NAME)
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"claim": ex["claim"]} for ex in EXAMPLES],
        outputs=[{"expected_winner": ex["expected_winner"]} for ex in EXAMPLES],
    )
    return dataset


def run_debate(inputs: dict) -> dict:
    """The thing being graded: run one claim through the real graph, fresh, and return its verdict."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    workflow.invoke(
        {
            "claim": inputs["claim"],
            "round": 0,
            "max_rounds": 2,
            "proponent_args": [],
            "skeptic_args": [],
            "verdicts": [],
            "final_verdict": None,
            "citations": [],
            "messages": [],
        },
        config=config,
    )

    # If the Judge paused for a human tiebreak, no one's watching this batch run ->
    # auto-resolve with a default answer so the debate can actually finish.
    state = workflow.get_state(config)
    while state.next:
        workflow.invoke(
            Command(resume={"winner": "tie", "note": "auto-resolved during batch evaluation"}),
            config=config,
        )
        state = workflow.get_state(config)

    return {"final_verdict": state.values["final_verdict"]}


def correct_winner(outputs: dict, reference_outputs: dict) -> bool:
    """The grader: does the actual final verdict match the expected winner from the answer key?"""
    expected = reference_outputs["expected_winner"]
    final_verdict = outputs.get("final_verdict")
    if not final_verdict:
        return False
    return f"overall winner: {expected}" in final_verdict.lower()


if __name__ == "__main__":
    _get_or_create_dataset()
    results = evaluate(
        run_debate,
        data=DATASET_NAME,
        evaluators=[correct_winner],
        experiment_prefix="truthforge-ai",
    )
    print(results)
