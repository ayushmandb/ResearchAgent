"""
The graph nodes: planner, approval (HITL interrupt), run_worker, writer.

TODO:
- Plan(BaseModel): assignments: dict[Literal["llm_knowledge","tavily","wikipedia"], str]
    (update this Literal whenever workers.py gains a new worker)

- planner_node(state) -> calls llm.call_llm() with structured_output_model=Plan,
    returns {"plan": plan.dict()}

- approval_node(state) -> decision = interrupt({"plan": state["plan"]})
    returns {"approved_workers": decision["approved"]}

- route_to_workers(state) -> conditional edge function, NOT a node.
    Returns a list of Send("run_worker", {...}) for each approved worker.
    This is the fan-out step.

- run_worker(payload) -> looks up WORKER_REGISTRY[payload["worker_name"]],
    calls it with payload["query"], returns {"documents": result}
    (the merge/reduce happens automatically via the Annotated field in state.py —
    no separate reducer node needed)

- writer_node(state) -> calls llm.call_llm() with a prompt built from
    state["documents"] + state["topic"], returns {"report": markdown_str}
"""
from typing import Literal
from pydantic import BaseModel
from langgraph.types import Send,interrupt
from datetime import date
from state import ResearchState
from workers import WORKER_REGISTRY 
from llm import call_llm, call_llm_structured
from workers import WORKER_REGISTRY, has_rag_index
WorkerName = Literal["llm_knowledge", "tavily", "wikipedia", "arxiv", "semantic_scholar", "hackernews","supporting_document"]

class Plan(BaseModel):
    reasoning: str
    assignments: dict[WorkerName, str]  # worker_name -> query


class CriticVerdict(BaseModel):
    approved: bool
    issues: list[str] = []


def planner_node(state: ResearchState) -> dict:
    worker_lines = [
        "- llm_knowledge: your own general knowledge, good for definitions/background",
        "- tavily: live web search, good for recent or current information",
        "- wikipedia: encyclopedic background, good for established facts/history",
        "- arxiv: scientific/technical paper abstracts, good for cutting-edge research topics",
        "- semantic_scholar: broader academic paper search across all fields",
        "- hackernews: tech community discussion, good for practitioner sentiment",
    ]
    if has_rag_index():
        worker_lines.append("- supporting_document: search documents the user personally uploaded — use this whenever the topic relates to material they provided")

    prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Available workers:\n" + "\n".join(worker_lines) + "\n\n"
        "Decide which workers are useful for this topic and write one specific "
        "search query for each chosen worker. Not every worker needs to be used."
    )
    plan = call_llm_structured(prompt, Plan)
    return {"plan": plan.model_dump()}

def approval_node(state: ResearchState) -> dict:
    decision = interrupt({"plan": state["plan"]})
    updated_plan = dict(state["plan"])
    updated_plan["assignments"] = decision["queries"]
    return {"plan": updated_plan, "approved_workers": decision["approved"]}


def route_to_workers(state: ResearchState):
    return [
        Send("run_worker", {"worker_name": w, "query": q})
        for w, q in state["plan"]["assignments"].items()
        if w in state["approved_workers"]
    ]

def run_worker(payload: dict) -> dict:
    fn = WORKER_REGISTRY[payload["worker_name"]]
    try:
        docs = fn(payload["query"])
    except Exception as e:
        print(f"[worker failed] {payload['worker_name']}: {e}")
        docs = []
    return {"documents": docs}

from datetime import date as date_cls   # avoid shadowing the `date` field name

class Section(BaseModel):
    title: str
    document_indices: list[int]  # 1-based, matching the numbered source list


class SectionPlan(BaseModel):
    sections: list[Section]


def section_planner_node(state: ResearchState) -> dict:
    source_summaries = "\n".join(
        f"[{i+1}] {d['source']}: {d['content'][:150]}"
        for i, d in enumerate(state["documents"])
    )
    prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Numbered sources gathered:\n{source_summaries}\n\n"
        "Plan a report structure with 3 to 4 sections (not more). Each section needs "
        "a clear, distinct title and a list of which source numbers belong to it. "
        "Sections must not overlap in subject matter. Every source doesn't need to "
        "be used, and a source can belong to more than one section only if genuinely relevant to both."
    )
    plan = call_llm_structured(prompt, SectionPlan)
    return {"section_plan": plan.model_dump()}

def route_to_sections(state: ResearchState):
    all_titles = [s["title"] for s in state["section_plan"]["sections"]]
    sends = []
    for section in state["section_plan"]["sections"]:
        assigned_docs = [
            {"index": i, **state["documents"][i - 1]}
            for i in section["document_indices"]
            if 0 < i <= len(state["documents"])
        ]
        sends.append(Send("section_writer", {
            "title": section["title"],
            "all_titles": all_titles,
            "documents": assigned_docs,
        }))
    return sends

def section_writer_node(payload: dict) -> dict:
    docs_text = "\n\n".join(
        f"[{d['index']}] {d['source']}" + (f", dated {d['date']}" if d.get("date") else "") + f": {d['content'][:400]}"
        for d in payload["documents"]
    )
    prompt = (
        f"You are writing ONE section of a larger multi-section research report.\n"
        f"All sections in this report: {', '.join(payload['all_titles'])}\n"
        f"Your assigned section: \"{payload['title']}\"\n\n"
        "Write ONLY this section's body content — no report title, no overall "
        "introduction or conclusion, since this is one chapter in a larger document. "
        "Don't repeat information that belongs to the other listed sections. Cite "
        "sources using their [n] numbers as given below. If you lack relevant source "
        "material for a claim, say so rather than filling in from general knowledge.\n\n"
        f"Source material:\n{docs_text}"
    )
    body = call_llm(prompt, system="You write one focused section of a research report.")
    return {"section_texts": [f"## {payload['title']}\n\n{body}"]}


def final_polish_node(state: ResearchState) -> dict:
    sources_list = "\n".join(
        f"[{i+1}] {d['source']}" + (f" — {d['url']}" if d["url"] else " — no URL available")
        for i, d in enumerate(state["documents"])
    )

    if state.get("critic_verdict") and not state["critic_verdict"]["approved"]:
        issues = "\n".join(f"- {i}" for i in state["critic_verdict"]["issues"])
        prompt = (
            f"Revise this report to fix ONLY these specific issues found by a fact-checker:\n{issues}\n\n"
            "Do not introduce new facts or rewrite unrelated parts. Fix or remove just the "
            "flagged content, keep everything else unchanged.\n\n"
            f"Report:\n{state['report']}\n\n"
            f"Numbered sources:\n{sources_list}"
        )
    else:
        sections_text = "\n\n".join(state["section_texts"])
        prompt = (
            f"Today's date is {date_cls.today().isoformat()}.\n"
            f"Topic: {state['topic']}\n\n"
            "Below are independently written sections of a research report. Do the following:\n"
            "1. Write a short 'Bottom Line' paragraph at the top directly answering the topic.\n"
            "2. Lightly edit for smooth flow between sections and cut duplicated points — "
            "do not rewrite content wholesale or invent new facts.\n"
            "3. Append a 'Sources' section at the end using the numbered list provided.\n\n"
            f"Sections:\n{sections_text}\n\n"
            f"Numbered sources:\n{sources_list}"
        )

    report = call_llm(prompt, system="You assemble and lightly edit report sections. Never invent facts or URLs.")
    return {"report": report}


def critic_node(state: ResearchState) -> dict:
    sources_list = "\n".join(
        f"[{i+1}] {d['source']}" + (f" — {d['url']}" if d["url"] else " — no URL available")
        for i, d in enumerate(state["documents"])
    )
    prompt = (
        "Review this research report for factual discipline. Check specifically for:\n"
        "1. Specific numbers, statistics, specs, or claims NOT backed by a citation to the "
        "numbered sources below.\n"
        "2. Any URL, date, or figure that looks fabricated rather than pulled from a real source.\n"
        "3. Citation numbers that don't correspond to anything in the source list.\n\n"
        f"Numbered sources:\n{sources_list}\n\n"
        f"Report:\n{state['report']}\n\n"
        "Respond with approved=true if clean. If not, approved=false with a short list of "
        "specific issues — quote the exact problematic text for each one."
    )
    verdict = call_llm_structured(prompt, CriticVerdict, system="You are a strict fact-checking editor.")
    return {
        "critic_verdict": verdict.model_dump(),
        "revision_count": state.get("revision_count", 0) + 1,
    }


def route_after_critic(state: ResearchState) -> str:
    if state["critic_verdict"]["approved"] or state["revision_count"] >= 2:
        return "end"
    return "revise"