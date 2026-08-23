"""
ResearchState: the single object every graph node reads from and writes to.

Fill this in first — every other file (workers.py, nodes.py, graph.py)
depends on the shape defined here.

Expected fields:
    topic            : str            -> user's research topic from Streamlit
    plan             : dict           -> planner node's output (worker -> query)
    approved_workers : list[str]      -> set after the HITL interrupt() step
    documents        : list[Document] -> Annotated with operator.add so LangGraph
                                          auto-merges results from parallel workers
    report           : str            -> final markdown produced by the writer node

TODO:
- define Document as a TypedDict: content, source, url
- define ResearchState as a TypedDict using Annotated[list[Document], operator.add]
  for the `documents` field
"""

from typing import TypedDict, Annotated , Optional
import operator

class Document(TypedDict):
    content: str
    source: Optional[str] # which worker produced this document
    url: Optional[str]   # Link to the source document, if available
    date: Optional[str]  # publish date if the source provides one, else None

class ResearchState(TypedDict):
    topic: str
    plan: dict
    approved_workers: list[str]
    documents: Annotated[list[Document], operator.add]
    section_plan: dict
    section_texts: Annotated[list[str], operator.add]
    report: str
    critic_verdict: dict
    revision_count: int
