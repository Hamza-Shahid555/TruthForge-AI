# TruthForge AI

A multi-agent fact-checking debate system built with [LangGraph](https://github.com/langchain-ai/langgraph).

A **Proponent** and a **Skeptic** agent research and argue opposite sides of a claim, in parallel, over multiple rounds. A **Judge** agent scores every round with structured output and decides whether to reconvene or close the case. When the Judge genuinely can't decide, the whole system pauses and asks a human to break the tie — then resumes exactly where it stopped.

This project was built end-to-end while learning LangGraph, following the step-by-step plan in [`FactCheck_Debate_Agent_Plan.md`](./FactCheck_Debate_Agent_Plan.md).

## How it works

```
START → round_manager → Proponent ─┐
                       → Skeptic  ─┴→ Judge ──concluded?──→ finalize → END
                                       │
                                       └──not yet──→ round_manager (loop)
```

- **Proponent / Skeptic** — each is its own LangGraph subgraph: a tool-calling loop that researches (RAG over a local evidence corpus, plus live web search via an MCP tool server) before producing a structured `Argument`.
- **Judge** — scores each round with a structured `RoundVerdict` (`winner`, `reasoning`, `concluded`). If the Judge is still undecided after multiple rounds, it calls `interrupt()` and the graph pauses for a human ruling.
- **Persistence** — every debate is checkpointed (SQLite locally, PostgreSQL in the Docker setup), scoped to a `thread_id`, so a run can be inspected or resumed at any point.
- **Observability** — full execution tracing via LangSmith, plus an automated evaluation set that scores debate accuracy against a set of claims with known outcomes.

## Features

- Parallel multi-agent debate with a cyclic, conditionally-routed graph
- RAG tool (FAISS) over a local evidence corpus
- Live web search via a custom MCP tool server
- Structured output for both arguments and verdicts
- Human-in-the-loop tiebreaking (`interrupt()` / `Command(resume=...)`)
- Streaming progress (CLI and Server-Sent Events)
- FastAPI backend + a live browser UI
- Dockerized, with a Postgres-backed checkpointer for deployment

## Project structure

```
agent/
  state.py           DebateState — the shared graph state
  graph.py            the main debate graph (nodes, edges, checkpointer)
  nodes/
    debater.py        Proponent / Skeptic subgraph (shared blueprint, two stances)
    judge.py           Judge node, structured verdict + human tiebreak
  tools/
    evidence_search.py  RAG tool over the local corpus
    web_search.py        MCP client for live web search
  evaluation.py       LangSmith dataset + evaluator for debate accuracy
api/
  main.py             FastAPI app: start a debate, stream it, check its status
web/
  index.html          Live browser UI, wired to the real API
mcp_server/
  search_server.py    Standalone MCP server exposing a web_search tool
data/
  coffee_benefits.txt / coffee_risks.txt   seed evidence corpus
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=truthforge-ai
```

(`LANGCHAIN_*` is optional — only needed for tracing/evaluation.)

## Running it

**A single debate from the command line:**
```bash
python -m agent.graph
```

**The full API + live browser UI:**
```bash
python -m uvicorn api.main:app --reload
```
Then open `http://127.0.0.1:8000/` and file a claim.

**With Docker (API + PostgreSQL):**
```bash
docker compose up --build
```

**Run the evaluation set** (scores real debates against claims with known outcomes):
```bash
python -m agent.evaluation
```

## Tech stack

LangGraph · LangChain · OpenAI · FAISS · MCP · FastAPI · SQLite / PostgreSQL · LangSmith · Docker
