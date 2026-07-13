"""
Sidebar UI — BYOK key input + model selectors for NVIDIA NIM.
Returns (nvidia_api_key, use_guardrails, guard_model, chat_model).
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

        st.subheader("⚙️ Execution Strategy")
        
        # Toggle choice: 1 model or 2 models execution
        use_guardrails = st.checkbox(
            "Enable Guardrail Shield (Uses 2 Models)", 
            value=True,
            help="Uncheck this to skip intent classification and only use 1 model for faster responses."
        )

        st.divider()

        st.subheader("🤖 Model Selection")

        guard_options = [
            "z-ai/glm-5.2",
            "deepseek-ai/deepseek-v4-pro",
            "meta/llama-3.3-70b-instruct"
        ]
        
        # This dropdown turns gray/disabled if "Enable Guardrail Shield" is unchecked
        guard_model = st.selectbox(
            "① Guard model — NeMo intent classification",
            options=guard_options,
            index=0,
            disabled=not use_guardrails,
            help="NeMo uses this model to decide whether your message is off-topic, unsafe, or a jailbreak.",
        )

        chat_options = [
            "deepseek-ai/deepseek-v4-flash",
            "meta/llama-3.3-70b-instruct"
        ]

        chat_model = st.selectbox(
            "② Chat model — RAG answer generation",
            options=chat_options,
            index=0,
            help="Used to generate the final answer from the top-3 HR policy chunks retrieved by FAISS.",
        )

        st.divider()

        st.subheader("⚙️ Pipeline Status")
        st.markdown(f"**Mode:** {'🛡️ Dual-Model Guarded' if use_guardrails else '⚡ Single-Model RAG Only'}")
        st.divider()
        st.caption(f"Python {sys.version.split()[0]} · NVIDIA NIM")

    return nvidia_api_key, use_guardrails, guard_model, chat_model
