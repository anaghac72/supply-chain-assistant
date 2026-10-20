from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# Run this file from the repository root so router.py and its data folders
# are importable using the same Python environment as the existing backend.
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual theme ----------
st.markdown(
    """
    <style>
    :root { --navy:#071421; --panel:#0d2030; --panel2:#10283a;
            --teal:#26d7c4; --muted:#9bb0bf; --line:#1c394b; }
    .stApp { background:linear-gradient(145deg,#071421 0%,#081927 55%,#0a1e2d 100%); }
    [data-testid="stHeader"] { background:rgba(7,20,33,.88); }
    [data-testid="stSidebar"] { background:#081724; border-right:1px solid #193446; }
    [data-testid="stSidebar"] * { color:#e7f0f5; }
    h1,h2,h3 { color:#f1f7fa !important; letter-spacing:-.02em; }
    p,li,label { color:#d5e2e9; }
    .eyebrow { color:#26d7c4; text-transform:uppercase; letter-spacing:.16em;
               font-size:.72rem; font-weight:800; }
    .hero { background:linear-gradient(120deg,rgba(19,55,72,.92),rgba(10,31,47,.92));
            border:1px solid #1c4858; border-radius:20px; padding:28px 30px;
            margin:4px 0 22px 0; }
    .hero h1 { font-size:2.1rem; margin:.3rem 0 .5rem 0; }
    .hero p { color:#b8cbd5; margin:0; max-width:780px; }
    .metric-card { background:rgba(13,32,48,.92); border:1px solid #1c394b;
                   border-radius:15px; padding:17px 18px; min-height:105px; }
    .metric-label { color:#9bb0bf; font-size:.8rem; }
    .metric-value { color:#f2f8fa; font-size:1.35rem; font-weight:750; margin-top:7px; }
    .metric-note { color:#26d7c4; font-size:.76rem; margin-top:4px; }
    .section-note { color:#9bb0bf; font-size:.88rem; }
    .source-card { background:#0b1d2b; border:1px solid #1b3a4b; border-radius:12px;
                   padding:14px 16px; margin:8px 0; }
    .source-title { color:#eaf4f8; font-weight:700; }
    .source-meta { color:#26d7c4; font-size:.8rem; margin-top:4px; }
    div[data-testid="stChatMessage"] { border:1px solid #193547; border-radius:14px;
                                       background:rgba(11,29,43,.65); }
    .stButton > button { border-radius:10px; border:1px solid #1d5260;
                         background:#103849; color:#e9ffff; }
    .stButton > button:hover { border-color:#26d7c4; color:#26d7c4; }
    .stChatInput textarea { background:#0d2030 !important; color:#f1f7fa !important; }
    div[data-testid="stExpander"] { border:1px solid #1b3a4b; border-radius:12px; }
    .footer { color:#7893a3; font-size:.75rem; padding-top:20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Backend connection ----------
@st.cache_resource(show_spinner=False)
def load_backend():
    """Import the existing assistant without reimplementing its routing logic."""
    from router import answer_question
    return answer_question


def safe_text(value: Any, fallback: str = "Not provided") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def source_name(source: dict[str, Any]) -> str:
    return safe_text(source.get("document") or source.get("source") or source.get("file"), "SOP document")


def show_sources(sources: Any) -> None:
    if not sources:
        st.info("The backend did not return SOP source passages for this response.")
        return
    if not isinstance(sources, list):
        st.write(sources)
        return
    st.markdown("#### Evidence from SOP documents")
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            with st.expander(f"Source {index}"):
                st.write(source)
            continue
        name = source_name(source)
        page = safe_text(source.get("page"), "?")
        chunk = safe_text(source.get("chunk_number"), "?")
        with st.expander(f"📄 {name}  ·  Page {page}"):
            st.caption(f"Chunk: {chunk}")
            text_value = source.get("text") or source.get("page_content") or source.get("content")
            if text_value:
                st.write(text_value)
            else:
                st.json(source)


def show_database_results(result: dict[str, Any]) -> None:
    rows = result.get("rows")
    sql = result.get("sql")
    if sql:
        with st.expander("View generated read-only SQL"):
            st.code(str(sql), language="sql")
    if rows is not None:
        with st.expander(f"View database results ({len(rows) if isinstance(rows, list) else 'available'})"):
            if isinstance(rows, list) and rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            elif isinstance(rows, list):
                st.info("No matching database rows were returned.")
            else:
                st.write(rows)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📦 SC / INTELLIGENCE")
    st.caption("SUPPLY CHAIN DECISION SUPPORT")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Assistant", "About the system"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**CONNECTED WORKFLOWS**")
    st.markdown("🟢 SOP knowledge retrieval")
    st.markdown("🟢 Order data lookup")
    st.markdown("🟢 Combined policy + order questions")
    st.caption("Availability depends on local configuration and backend data.")
    st.markdown("---")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Powered by your existing Python router, Groq, FAISS and PostgreSQL.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# ---------- Assistant page ----------
if page == "Assistant":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">AI-powered operations workspace</div>
          <h1>Supply Chain Intelligence</h1>
          <p>Ask questions about SOPs, inventory procedures and order records.
          Answers are generated by your existing assistant, with supporting evidence
          shown when the backend returns it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    cards = [
        ("KNOWLEDGE BASE", "SOP / Policy RAG", "FAISS retrieval + document citations"),
        ("LIVE OPERATIONS", "Order lookup", "Read-only PostgreSQL query workflow"),
        ("ANSWER QUALITY", "Evidence checked", "Existing answer verification retained"),
    ]
    for col, (label, value, note) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Ask your assistant")
    st.markdown(
        '<div class="section-note">Try a suggested question, or type your own below.</div>',
        unsafe_allow_html=True,
    )
    suggestions = [
        "What is the inventory ordering procedure?",
        "What information is needed when placing an order?",
        "Show the status of order 123 for customer 45",
        "What does the SOP say about ordering, and what is the status of order 123?",
    ]
    s_cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with s_cols[i % 2]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_question = suggestion

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("route"):
                    st.caption(f"Workflow: {message['route']}")
                if message.get("sources"):
                    show_sources(message["sources"])
                if message.get("result"):
                    show_database_results(message["result"])

    pending = st.session_state.pop("pending_question", None)
    prompt = st.chat_input("Ask about SOPs, policies, or order records…")
    question = prompt or pending

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                answer_question = load_backend()
                with st.spinner("Routing your question and checking the available evidence…"):
                    result = answer_question(question)
                if not isinstance(result, dict):
                    raise TypeError("router.answer_question() did not return a dictionary.")
                answer = safe_text(result.get("answer"), "The assistant returned no answer.")
                st.markdown(answer)
                route = result.get("route")
                if route:
                    st.caption(f"Workflow: {route}")
                sources = result.get("sources", [])
                if sources:
                    show_sources(sources)
                show_database_results(result)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "route": route,
                        "sources": sources,
                        "result": result if ("rows" in result or "sql" in result) else None,
                    }
                )
                st.session_state.question_count += 1
            except Exception as exc:
                message = (
                    "I couldn't run the existing assistant. Check that the project dependencies, "
                    "Groq API key, PostgreSQL connection, and FAISS/SOP files are configured. "
                    "The technical detail is shown below."
                )
                st.error(message)
                with st.expander("Technical error details"):
                    st.code(f"{type(exc).__name__}: {exc}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": message}
                )

    st.markdown(
        '<div class="footer">Use the returned answer and source evidence to support decisions; '
        'review important operational decisions before acting.</div>',
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">System overview</div>
          <h1>How the assistant works</h1>
          <p>The interface calls the repository's existing backend. It does not replace
          or bypass routing, retrieval, SQL validation, or answer verification.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Request flow")
    st.code(
        "Question from Streamlit\n"
        "        ↓\n"
        "router.answer_question(question)\n"
        "        ↓\n"
        "Groq classifies: policy / orders / both\n"
        "        ↓\n"
        "SOP path: embeddings → FAISS → answer + citations\n"
        "Order path: SQL generation → validation → read-only PostgreSQL\n"
        "Combined path: order rows + SOP evidence\n"
        "        ↓\n"
        "Existing answer verification\n"
        "        ↓\n"
        "Answer + available sources/results shown in Streamlit",
        language="text",
    )
    st.markdown("### Required configuration")
    st.markdown(
        """
        - `GROQ_API_KEY` in the project's local `.env` file.
        - `DATABASE_URL` pointing to the PostgreSQL database for order questions.
        - Existing `vector_db/` FAISS index and metadata, plus the backend modules.
        - The dependencies required by the repository's runtime modules.
        """
    )
    st.warning(
        "The UI does not create sample order records or pretend the database is connected. "
        "Order questions work only when PostgreSQL is configured."
    )
    st.markdown("### Backend modules used")
    st.markdown(
        "- `router.py` — routes each question and returns the answer payload.\n"
        "- `sop_rag.py` — retrieves SOP passages and produces policy answers.\n"
        "- `sql_lookup.py` — generates, validates and executes read-only SQL.\n"
        "- `llm_client.py` — shared Groq client and evidence-based verification."
    )
