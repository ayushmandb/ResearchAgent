"""
Streamlit UI. No business logic here — only:
    - topic input + "level of research" option
    - streaming the compiled graph, catching __interrupt__ to show approval checkboxes
    - resuming with Command(resume=...) once the user confirms
    - showing the final report + download button

TODO (rough shape):

import streamlit as st
from langgraph.types import Command
from graph import compiled

st.title("Research Agent")
topic = st.text_input("Research topic")
config = {"configurable": {"thread_id": st.session_state.get("thread_id", "default")}}

if st.button("Start research") and topic:
    for event in compiled.stream({"topic": topic}, config, stream_mode="updates"):
        if "__interrupt__" in event:
            st.session_state.pending_plan = event["__interrupt__"][0].value["plan"]

if st.session_state.get("pending_plan"):
    st.write(st.session_state.pending_plan)
    approved = [w for w in st.session_state.pending_plan["assignments"] if st.checkbox(w, value=True)]
    if st.button("Confirm workers"):
        for event in compiled.stream(Command(resume={"approved": approved}), config, stream_mode="updates"):
            pass
        final_state = compiled.get_state(config).values
        st.markdown(final_state["report"])
        st.download_button("Download report", final_state["report"], file_name="report.md")
"""
from dotenv import load_dotenv
load_dotenv()
import re
import uuid
import datetime
import pathlib
import streamlit as st
from langgraph.types import Command
import pypdf
from workers import build_rag_index
from graph import compiled


st.set_page_config(page_title="Research Agent", page_icon="📚")
st.title("📚 AI Research Agent")


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "stage" not in st.session_state:
    st.session_state.stage = "input"

config = {"configurable": {"thread_id": st.session_state.thread_id}}


def save_report(topic: str, report: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic.lower()).strip("_")[:50]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = pathlib.Path("outputs") / f"{slug}_{timestamp}.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return str(path)

def extract_text(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


with st.sidebar:
    st.subheader("📎 Supporting Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if st.button("Learn documents") and uploaded_files:
        docs_dict = {f.name: extract_text(f) for f in uploaded_files}
        num_chunks = build_rag_index(docs_dict)
        st.success(f"Indexed {num_chunks} chunks from {len(docs_dict)} document(s).")

if st.session_state.stage == "input":
    topic = st.text_input("What do you want to research?")
    if st.button("Generate plan") and topic:
        st.session_state.topic = topic
        with st.spinner("Planning research..."):
            for event in compiled.stream({"topic": topic}, config, stream_mode="updates"):
                if "__interrupt__" in event:
                    st.session_state.pending_plan = event["__interrupt__"][0].value["plan"]
        st.session_state.stage = "approval"
        st.rerun()

elif st.session_state.stage == "approval":
    plan = st.session_state.pending_plan
    st.subheader("Proposed research plan")
    st.write(plan["reasoning"])

    approved = []
    edited_queries = {}
    for worker, query in plan["assignments"].items():
        col1, col2 = st.columns([1, 4])
        with col1:
            use_worker = st.checkbox(worker, value=True, key=f"use_{worker}")
        with col2:
            edited_query = st.text_input(
                f"Query for {worker}", value=query, key=f"query_{worker}",
                label_visibility="collapsed",
            )
        if use_worker:
            approved.append(worker)
            edited_queries[worker] = edited_query

################# v1 
    
    # if st.button("Run research"):
    #     with st.spinner("Researching..."):
    #         for event in compiled.stream(
    #             Command(resume={"approved": approved, "queries": edited_queries}),
    #             config,
    #             stream_mode="updates",
    #         ):
    #             pass
    #     final_state = compiled.get_state(config).values
    #     st.session_state.report = final_state["report"]
    #     st.session_state.saved_path = save_report(st.session_state.topic, final_state["report"])
    #     st.session_state.stage = "done"
    #     st.rerun()

    ################# v2 
    if st.button("Run research"):
        with st.status("Starting research...", expanded=True) as status:
            worker_log = st.empty()
            section_log = st.empty()
            completed_workers = []
            section_state = {}

            for event in compiled.stream(
                Command(resume={"approved": approved, "queries": edited_queries}),
                config,
                stream_mode="updates",
            ):
                if "run_worker" in event:
                    docs = event["run_worker"]["documents"]
                    if docs:
                        completed_workers.append(f"✅ {docs[0]['source']} — {len(docs)} doc(s)")
                    else:
                        completed_workers.append("⚠️ a worker returned no results")
                    status.update(label=f"Researching... ({len(completed_workers)}/{len(approved)} workers done)")
                    worker_log.markdown("\n".join(completed_workers))

                elif "section_planner" in event:
                    titles = [s["title"] for s in event["section_planner"]["section_plan"]["sections"]]
                    section_state = {t: "⏳" for t in titles}
                    status.update(label=f"Planning report structure... {len(titles)} sections")
                    section_log.markdown("\n".join(f"{icon} {t}" for t, icon in section_state.items()))

                elif "section_writer" in event:
                    text = event["section_writer"]["section_texts"][0]
                    title = text.split("\n")[0].replace("## ", "").strip()
                    if title in section_state:
                        section_state[title] = "✅"
                    status.update(label="Writing report sections...")
                    section_log.markdown("\n".join(f"{icon} {t}" for t, icon in section_state.items()))

                elif "final_polish" in event:
                    status.update(label="Polishing final report...")

                elif "critic" in event:
                   verdict = event["critic"]["critic_verdict"]
                   if verdict["approved"]:
                       status.update(label="Fact-check passed ✅")
                   else:
                       status.update(label=f"Fact-check found {len(verdict['issues'])} issue(s), revising...")

            status.update(label="Research complete!", state="complete", expanded=False)

        final_state = compiled.get_state(config).values
        st.session_state.report = final_state["report"]
        st.session_state.saved_path = save_report(st.session_state.topic, final_state["report"])
        st.session_state.stage = "done"
        st.rerun()

elif st.session_state.stage == "done":
    st.subheader("Report")
    st.caption(f"Saved to {st.session_state.saved_path}")
    st.markdown(st.session_state.report)
    st.download_button("Download report (.md)", st.session_state.report, file_name="research_report.md")
    if st.button("Start new research"):
        st.session_state.stage = "input"
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()