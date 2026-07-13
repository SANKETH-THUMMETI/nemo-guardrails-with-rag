"""
Sidebar UI — BYOK key input + model selectors for NVIDIA NIM.
Returns (nvidia_api_key, guard_model, chat_model).
"""

import sys
import streamlit as st


def render_sidebar() -> tuple:
    with st.sidebar:
        st.title("🛡️ HR Policy Assistant")
        st.caption("NeMo Guardrails + FAISS RAG — Acme Corp")
        st.divider()

        st.subheader("🔑 Bring Your Own Key")
        st.caption("Keys are used only for this session and never stored.")

        # CHANGED: Replaced Groq with NVIDIA NIM API configuration
        nvidia_api_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            placeholder="nvapi-...",
            help="Get your key at build.nvidia.com or integrate.api.nvidia.com",
        )

        if nvidia_api_key:
            st.success("Key loaded ✓", icon="🔒")
        else:
            st.info("Paste your NVIDIA key above to start.", icon="ℹ️")

        st.divider()

        st.subheader("🤖 Two LLMs Per Request")
        st.caption(
            "Every message makes **2 separate LLM calls** via NVIDIA NIM — "
            "one for guardrail classification, one for the final answer. "
            "You can pick different models for each."
        )

        # CHANGED: Hardcoded choices to use your requested NVIDIA NIM models explicitly
        guard_options = [
            "z-ai/glm-5.2",
            "deepseek-ai/deepseek-v4-pro",
            "meta/llama-3.3-70b-instruct"
        ]
        
        guard_model = st.selectbox(
            "① Guard model — NeMo intent classification",
            options=guard_options,
            index=0,
            help=(
                "NeMo uses this model to decide whether your message is off-topic, "
                "a jailbreak, a request for confidential data, etc. "
                "A stronger model catches more subtle attacks."
            ),
        )

        chat_options = [
            "deepseek-ai/deepseek-v4-flash",
            "meta/llama-3.3-70b-instruct"
        ]

        chat_model = st.selectbox(
            "② Chat model — RAG answer generation",
            options=chat_options,
            index=0,
            help=(
                "Used to generate the final answer from the top-3 HR policy chunks "
                "retrieved by FAISS. A faster/cheaper model is fine here."
            ),
        )

        # Updated model rule warning check for smaller engines if applicable
        if "flash" in chat_model or "5.2" in guard_model:
            st.caption("⚡ Running high-efficiency inference options.")

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
        st.caption(f"Python {sys.version.split()[0]} · NVIDIA NIM Integration")

    return nvidia_api_key, guard_model, chat_model
