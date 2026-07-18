"""
Streamlit chat UI for the Wine Sommelier RAG assistant.
"""
import sys
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from db_init import init_db
from db_save import save_conversation, save_feedback
from judge import evaluate_relevance
from rag import WineRAG

init_db()

st.set_page_config(page_title="Wine Sommelier", page_icon="🍷", layout="centered")
st.title("🍷 Wine Sommelier Assistant")
st.caption("Ask me anything about wine — pairings, regions, grape varieties, and recommendations.")

with st.sidebar:
    st.header("Settings")
    search_method = st.selectbox(
        "Search method",
        ["vector"],
        index=0,
        help="pgvector cosine similarity search",
    )
    prompt_style = st.selectbox(
        "Answer style",
        ["detailed", "concise"],
        index=0,
    )
    use_rewrite = st.checkbox("Query rewriting", value=True, help="Rewrite query for better retrieval")
    use_rerank = st.checkbox("Result reranking", value=True, help="Local ONNX cross-encoder reranking")

@st.cache_resource
def get_rag(search_method, prompt_style, use_rewrite, use_rerank):
    return WineRAG(
        search_method=search_method,
        prompt_style=prompt_style,
        use_rewrite=use_rewrite,
        use_rerank=use_rerank,
    )

rag = get_rag(search_method, prompt_style, use_rewrite, use_rerank)

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

user_input = st.text_input(
    "Your question:",
    placeholder="e.g. What's a good Pinot Noir under $30 from Oregon?",
)

if st.button("Ask", type="primary") and user_input.strip():
    with st.spinner("Consulting the cellar..."):
        answer = rag.rag(user_input)
        st.session_state.last_answer = answer
        record = rag.last_call

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Metrics"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Response time", f"{record.response_time:.2f}s")
        col2.metric("Prompt tokens", record.prompt_tokens)
        col3.metric("Completion tokens", record.completion_tokens)
        col4.metric("Cost", f"${record.cost:.5f}")

    conversation_id = save_conversation(record)
    st.session_state.conversation_id = conversation_id

    relevance, explanation = evaluate_relevance(user_input, answer)
    save_feedback(conversation_id, "judge", relevance=relevance, explanation=explanation)

    badge_color = {"RELEVANT": "green", "PARTLY_RELEVANT": "orange", "NON_RELEVANT": "red"}.get(relevance, "gray")
    st.markdown(f"**Relevance:** :{badge_color}[{relevance}]")
    with st.expander("Judge explanation"):
        st.write(explanation)

st.divider()
st.markdown("**Was this answer helpful?**")
col1, col2 = st.columns(2)
with col1:
    if st.button("👍 Yes") and st.session_state.conversation_id:
        save_feedback(st.session_state.conversation_id, "user", score=1)
        st.success("Thanks!")
with col2:
    if st.button("👎 No") and st.session_state.conversation_id:
        save_feedback(st.session_state.conversation_id, "user", score=-1)
        st.info("Thanks for the feedback!")
