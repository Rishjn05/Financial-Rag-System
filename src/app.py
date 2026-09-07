# src/app.py — Streamlit demo for the 10-K Financial RAG system.

import streamlit as st
from rag import ask

st.set_page_config(page_title="10-K Financial RAG", page_icon="📊", layout="centered")

# --- Header ---
st.title("📊 10-K Financial RAG")
st.caption(
    "Ask questions about Apple's 2025 10-K filing. "
    "Answers are grounded **only** in the retrieved filing text — "
    "the system refuses when the answer isn't in the document."
)

# --- Example questions (helps a viewer try it instantly) ---
st.markdown("**Try:**")
examples = [
    "What was Apple's net income in 2025?",
    "Why did iPhone net sales increase in 2025?",
    "What is the total number of employees in Google?",  # shows refusal
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["question"] = ex

# --- Question input ---
question = st.text_input(
    "Your question:",
    key="question",
    placeholder="e.g. What were Apple's total net sales in 2025?",
)

# --- Run ---
if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving from the filing and generating a grounded answer..."):
        try:
            answer, sources = ask(question, n_results=8)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # --- Answer ---
    st.markdown("### Answer")
    st.write(answer)

    # --- Sources (the grounding — this is what makes it RAG, not a chatbot) ---
    st.markdown("### Retrieved source passages")
    st.caption("The answer is grounded in these chunks from the 10-K:")
    for i, chunk in enumerate(sources, 1):
        with st.expander(f"Source {i}"):
            st.text(chunk[:800])

# --- Footer: honest framing ---
st.divider()
st.caption(
    "Pipeline: PyMuPDF → chunking → sentence-transformers embeddings → "
    "ChromaDB + BM25 hybrid retrieval (RRF) → Gemini (grounded prompt). "
    "Known limitation: large tables split across chunks can cause retrieval misses."
)