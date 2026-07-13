"""
Chat interface tab layout logic.
Manages session message states and passes parameters to the execution pipeline.
"""

import streamlit as st
import traceback as tb
from src.pipeline import run_pipeline

# Local helper functions assumed from your codebase
def _render_examples():
    # Placeholder for rendering example prompts if defined in your file
    pass

def _render_trace(trace_data):
    # Placeholder for rendering the execution trace sidebar layout if defined
    if trace_data:
        st.subheader("🔍 Execution Trace")
        st.json(trace_data)


def render_chat_tab(vectorstore, nvidia_api_key: str, use_guardrails: bool, guard_model: str, chat_model: str) -> None:
    st.divider()
    col_chat, col_trace = st.columns([6, 4], gap="large")

    with col_chat:
        _render_examples()

        if "chat" not in st.session_state:
            st.session_state["chat"] = []

        for msg in st.session_state["chat"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        injected   = st.session_state.pop("inject", None)
        user_input = injected or st.chat_input("Ask an HR policy question…")

        if user_input:
            st.session_state["chat"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Processing…"):
                    try:
                        # UPDATED: Corrected pipeline execution parameter mapping 
                        reply, trace = run_pipeline(
                            message=user_input,
                            nvidia_api_key=nvidia_api_key,
                            use_guardrails=use_guardrails,
                            guard_model=guard_model,
                            chat_model=chat_model,
                            vectorstore=vectorstore
                        )
                        st.session_state["last_trace"] = trace
                        st.write(reply)
                        st.caption(f"⏱ {trace.get('total_ms', '?')} ms total")
                        st.session_state["chat"].append({"role": "assistant", "content": reply})
                    except Exception as e:
                        err_trace = tb.format_exc()
                        st.error(f"**{type(e).__name__}:** {e}")
                        with st.expander("Full traceback"):
                            st.code(err_trace, language="python")
                        st.session_state["last_trace"] = {"error": err_trace}

        if st.session_state.get("chat"):
            if st.button("🗑 Clear chat"):
                st.session_state["chat"] = []
                st.session_state.pop("last_trace", None)
                st.rerun()

    with col_trace:
        _render_trace(st.session_state.get("last_trace"))
