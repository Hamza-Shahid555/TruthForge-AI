import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from agent.graph import workflow

app = FastAPI(title="TruthForge AI")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Simple in-memory "order ticket" board: thread_id -> the debate's starting inputs.
# (Fine for learning; a real deployment would use something more durable than a dict.)
_pending_debates: dict[str, dict] = {}


class StartDebateRequest(BaseModel):
    claim: str
    max_rounds: int = 2


class StartDebateResponse(BaseModel):
    thread_id: str


class ResumeRequest(BaseModel):
    winner: str
    note: str


def _stream_graph(stream_input, config):
    """Shared by /stream and /resume: run the graph, sending each node's result as an
    SSE event the moment it's ready, and unwrapping a human-tiebreak pause into plain JSON."""

    def event_generator():
        for update in workflow.stream(stream_input, config=config, stream_mode="updates"):
            for node_name, node_output in update.items():
                if node_name == "__interrupt__":
                    node_output = node_output[0].value
                payload = {"node": node_name, "output": node_output}
                yield f"data: {json.dumps(payload)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/")
def index():
    """Serve the live UI from the same origin as the API, so browser requests just work."""
    return FileResponse(WEB_DIR / "index.html")


@app.post("/debates", response_model=StartDebateResponse)
def start_debate(request: StartDebateRequest):
    """Take the order: remember what to debate, hand back a ticket number."""
    thread_id = str(uuid.uuid4())
    _pending_debates[thread_id] = {
        "claim": request.claim,
        "round": 0,
        "max_rounds": request.max_rounds,
        "proponent_args": [],
        "skeptic_args": [],
        "verdicts": [],
        "final_verdict": None,
        "citations": [],
        "messages": [],
    }
    return StartDebateResponse(thread_id=thread_id)


@app.get("/debates/{thread_id}/stream")
def stream_debate(thread_id: str):
    """Actually run the debate now, streaming live progress back as it happens."""
    initial_state = _pending_debates.pop(thread_id, None)
    if initial_state is None:
        raise HTTPException(status_code=404, detail="Unknown or already-started thread_id")

    config = {"configurable": {"thread_id": thread_id}}
    return _stream_graph(initial_state, config)


@app.post("/debates/{thread_id}/resume")
def resume_debate(thread_id: str, request: ResumeRequest):
    """Answer a paused human-tiebreak and stream the rest of the debate to its finish."""
    config = {"configurable": {"thread_id": thread_id}}
    resume_command = Command(resume={"winner": request.winner, "note": request.note})
    return _stream_graph(resume_command, config)


@app.get("/debates/{thread_id}")
def get_debate(thread_id: str):
    """Check the current (or final) status of a debate that's already been started."""
    config = {"configurable": {"thread_id": thread_id}}
    state = workflow.get_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    return {
        "claim": state.values.get("claim"),
        "round": state.values.get("round"),
        "final_verdict": state.values.get("final_verdict"),
        "verdicts": state.values.get("verdicts"),
    }
