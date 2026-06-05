"""
HR Policy Documents tab — browse the knowledge base.
"""

import streamlit as st
from src.hr_docs import HR_DOCUMENTS


def render_docs_tab() -> None:
    st.divider()
    st.subheader("Knowledge Base — Acme Corp HR Policies")
    st.caption(
        f"{len(HR_DOCUMENTS)} documents · chunked into ~500-char segments · "
        "indexed in FAISS using BAAI/bge-small-en-v1.5 embeddings · "
        "top-3 chunks retrieved per query"
    )
    st.divider()

    for doc in HR_DOCUMENTS:
        with st.expander(f"📄  {doc['title']}"):
            for line in doc["content"].split("\n"):
                stripped = line.strip()
                if stripped and stripped == stripped.upper() and len(stripped) > 3:
                    st.markdown(f"**{stripped}**")
                elif stripped:
                    st.write(stripped)
                else:
                    st.write("")
