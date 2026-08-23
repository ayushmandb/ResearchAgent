"""
Worker functions + the WORKER_REGISTRY dict.

Every worker follows the same contract:
    def worker_fn(query: str) -> list[Document]

TODO:
- llm_knowledge_worker(query)  -> calls llm.call_llm(), wraps answer as one Document
- tavily_worker(query)         -> calls Tavily's free search API, one Document per result
- wikipedia_worker(query)      -> calls Wikipedia REST/summary API, one Document per page

WORKER_REGISTRY = {
    "llm_knowledge": llm_knowledge_worker,
    "tavily": tavily_worker,
    "wikipedia": wikipedia_worker,
}

Adding a new worker later:
    1. Write the function here (or split into workers/ once this file gets crowded)
    2. Add it to WORKER_REGISTRY
    3. Update the Plan schema's Literal[...] in nodes.py so the planner LLM
       knows the new worker exists (easy step to forget)
"""

import os 
import wikipedia
from tavily import TavilyClient
from llm import call_llm
from state import Document
import xml.etree.ElementTree as ET

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def llm_knowledge_worker(query: str) -> list[Document]:
    answer = call_llm(query, system="Answer concisely from your own knowledge.")
    return [Document(content=answer, source="llm_knowledge", url=None,date=None)]

def tavily_worker(query: str) -> list[Document]:
    results = tavily_client.search(query, max_results=3)
    return [
        Document(content=r["content"], source="tavily", url=r["url"],date=r.get("published_date") or None)
        for r in results["results"]
    ]


import requests

WIKI_HEADERS = {"User-Agent": "ResearchAgent/1.0 (student project; contact: your_email@example.com)"}


import re

def wikipedia_worker(query: str) -> list[Document]:
    resp = requests.get(
        "https://api.wikimedia.org/core/v1/wikipedia/en/search/page",
        params={"q": query, "limit": 1},
        headers=WIKI_HEADERS,
        timeout=10,
    )
    if resp.status_code != 200:
        return []
    pages = resp.json().get("pages", [])
    if not pages:
        return []
    page = pages[0]
    excerpt = re.sub("<[^<]+?>", "", page.get("excerpt", ""))  # strip <span> highlight tags
    url = f"https://en.wikipedia.org/wiki/{page['key']}"
    return [Document(content=excerpt, source="wikipedia", url=url, date=None)]



def arxiv_worker(query: str) -> list[Document]:
    resp = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": 3},
        timeout=20,
    )
    if resp.status_code != 200:
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    docs = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()
        link = entry.find("atom:id", ns).text.strip()
        published = entry.find("atom:published", ns)
        pub_date = published.text[:10] if published is not None else None
        docs.append(Document(content=f"{title}\n{summary}", source="arxiv", url=link, date=pub_date))
    return docs


def semantic_scholar_worker(query: str) -> list[Document]:
    resp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": 3, "fields": "title,abstract,url"},
        timeout=10,
    )
    if resp.status_code != 200:
        return []
    docs = []
    for p in resp.json().get("data", []):
        if not p.get("abstract"):
            continue
        docs.append(Document(content=f"{p['title']}\n{p['abstract']}", source="semantic_scholar", url=p.get("url"), date=str(p["year"]) if p.get("year") else None))
    return docs


def hackernews_worker(query: str) -> list[Document]:
    resp = requests.get(
        "https://hn.algolia.com/api/v1/search",
        params={"query": query, "tags": "story", "hitsPerPage": 3},
        timeout=10,
    )
    if resp.status_code != 200:
        return []
    docs = []
    for h in resp.json().get("hits", []):
        title = h.get("title")
        if not title:
            continue
        content = title
        if h.get("story_text"):
            content += "\n" + h["story_text"]
        content += f" (points: {h.get('points', 0)}, comments: {h.get('num_comments', 0)})"
        url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        date = h.get("created_at", "")[:10] or None
        docs.append(Document(content=content, source="hackernews", url=url,date=date))
    return docs

# ################### RAG ################
import os
import requests
import numpy as np

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
EMBED_MODEL = "models/gemini-embedding-001"

_rag_chunks: list[dict] = []
_rag_embeddings = None  # numpy array, one row per chunk

def _chunk_text(text: str, filename: str, chunk_size: int = 250) -> list[dict]:
    words = text.split()
    return [
        {"text": " ".join(words[i:i + chunk_size]), "filename": filename}
        for i in range(0, len(words), chunk_size)
    ]

def _embed_texts(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/{EMBED_MODEL}:batchEmbedContents",
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json={"requests": [{"model": EMBED_MODEL, "content": {"parts": [{"text": t}]}} for t in texts]},
        timeout=30,
    )
    resp.raise_for_status()
    return [e["values"] for e in resp.json()["embeddings"]]


def build_rag_index(documents: dict[str, str]) -> int:
    global _rag_chunks, _rag_embeddings
    _rag_chunks = []
    for filename, text in documents.items():
        _rag_chunks.extend(_chunk_text(text, filename, chunk_size=250))

    if not _rag_chunks:
        _rag_embeddings = None
        return 0

    vectors = _embed_texts([c["text"] for c in _rag_chunks])
    _rag_embeddings = np.array(vectors)
    return len(_rag_chunks)


def has_rag_index() -> bool:
    return _rag_embeddings is not None


def supporting_document_worker(query: str) -> list[Document]:
    if _rag_embeddings is None:
        return []
    query_vec = np.array(_embed_texts([query])[0])
    norms = np.linalg.norm(_rag_embeddings, axis=1) * np.linalg.norm(query_vec)
    scores = (_rag_embeddings @ query_vec) / (norms + 1e-8)

    ranked = scores.argsort()[::-1]
    top_indices = [i for i in ranked if scores[i] > 0.5][:5]
    return [
        Document(content=_rag_chunks[i]["text"], source="supporting_document", url=f"Uploaded: {_rag_chunks[i]['filename']}", date=None)
        for i in top_indices
    ]

# ################
WORKER_REGISTRY = {
    "llm_knowledge": llm_knowledge_worker,
    "tavily": tavily_worker,
    "wikipedia": wikipedia_worker,
    "arxiv": arxiv_worker,
    "semantic_scholar": semantic_scholar_worker,
    "hackernews": hackernews_worker,
    "supporting_document": supporting_document_worker
}