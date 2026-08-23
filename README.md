# Research Agent

LangGraph multi-worker research agent with a Streamlit UI. Planner LLM proposes
a research plan, you approve which workers run (human-in-the-loop), approved
workers run in parallel (LLM knowledge / Tavily / Wikipedia — extensible via a
worker registry), results merge automatically, and a writer LLM produces a
cited markdown report you can download.

## Local setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your GROQ_API_KEY and TAVILY_API_KEY
```

## Run

```bash
streamlit run app.py
```

## Project layout

- `state.py`    — shared graph state schema (topic, plan, documents, report)
- `workers.py`  — worker functions + WORKER_REGISTRY (add new workers here)
- `llm.py`      — shared Groq client
- `nodes.py`    — planner / approval (HITL) / run_worker / writer nodes
- `graph.py`    — builds and compiles the StateGraph
- `app.py`      — Streamlit UI, drives the compiled graph
- `outputs/`    — generated reports land here

## Build status

- [ ] state.py + workers.py (Document/Plan schemas, worker registry)
- [ ] planner node
- [ ] HITL approval node
- [ ] fan-out + implicit reducer
- [ ] writer node
- [ ] full graph compiled + tested headless
- [ ] Streamlit UI
- [ ] Dockerize
- [ ] Deploy to Render
