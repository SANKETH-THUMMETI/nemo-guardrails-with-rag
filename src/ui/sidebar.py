"""
Sidebar UI — BYOK key input + model selectors.
Returns (groq_key, guard_model, chat_model).
"""

import sys
import streamlit as st
from src.config import GROQ_MODELS, GUARD_MODEL_DEFAULT, CHAT_MODEL_DEFAULT


def render_sidebar() -> tuple:
    with st.sidebar:
        st.title("🛡️ HR Policy Assistant")
        st.caption("NeMo Guardrails + FAISS RAG — Acme Corp")
        st.divider()

        st.subheader("🔑 Bring Your Own Key")
        st.caption("Keys are used only for this session and never stored.")

        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com",
        )

        if groq_key:
            st.success("Key loaded ✓", icon="🔒")
        else:
            st.info("Paste your Groq key above to start.", icon="ℹ️")

        st.divider()

        st.subheader("🤖 Two LLMs Per Request")
        st.caption(
            "Every message makes **2 separate LLM calls** via Groq — "
            "one for guardrail classification, one for the final answer. "
            "You can pick different models for each."
        )

        guard_model = st.selectbox(
            "① Guard model — NeMo intent classification",
            options=list(GROQ_MODELS.keys()),
            index=list(GROQ_MODELS.keys()).index(GUARD_MODEL_DEFAULT),
            format_func=lambda m: GROQ_MODELS[m],
            help=(
                "NeMo uses this model to decide whether your message is off-topic, "
                "a jailbreak, a request for confidential data, etc. "
                "A stronger model catches more subtle attacks."
            ),
        )
        chat_model = st.selectbox(
            "② Chat model — RAG answer generation",
            options=list(GROQ_MODELS.keys()),
            index=list(GROQ_MODELS.keys()).index(CHAT_MODEL_DEFAULT),
            format_func=lambda m: GROQ_MODELS[m],
            help=(
                "Used to generate the final answer from the top-3 HR policy chunks "
                "retrieved by FAISS. A faster/cheaper model is fine here."
            ),
        )

        if guard_model == "llama-3.1-8b-instant":
            st.warning("8B models may miss subtle jailbreaks. 70B+ is recommended for the guard model.")

        st.divider()

        st.subheader("⚙️ Pipeline")
        st.markdown("""
**Per message:**
1. PII check (Python regex, systematic)
2. Intent classification → LLM ① (guard)
3. FAISS retrieves top-3 chunks
4. Answer generation → LLM ② (chat)
5. Output sanitizer (Python regex)
        """)
        st.divider()
        st.caption(f"Python {sys.version.split()[0]}")

    return groq_key, guard_model, chat_model
