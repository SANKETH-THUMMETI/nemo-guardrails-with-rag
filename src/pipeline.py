"""
Core pipeline logic — no Streamlit imports.

run_pipeline() wires together:
  NeMo input rails (LLM ①) → FAISS retrieval → answer generation (LLM ②) → output sanitizer
"""

import asyncio
import re
import time
import traceback as tb
from concurrent.futures import ThreadPoolExecutor

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

from src.guards import build_rails, parse_nemo_response
from src.rag import retrieve
from src.config import HR_SYSTEM_PROMPT, SENSITIVE_OUTPUT_PATTERNS

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nemo")


def check_output(text: str) -> list:
    return [label for label, pat in SENSITIVE_OUTPUT_PATTERNS.items()
            if re.search(pat, text)]


def run_pipeline(message: str, nvidia_api_key: str, use_guardrails: bool, guard_model: str, chat_model: str, vectorstore) -> tuple:
    """
    Returns (reply: str, trace: dict).
    Dynamically routes work through either 1 or 2 NVIDIA NIM models based on user selection.
    """
    trace   = {}
    t_total = time.time()

    # ── LLM ① — NeMo input rails ─────────────────────────────────────────────
    if use_guardrails:
        def _nemo_worker():
            llm = ChatNVIDIA(
                nvidia_api_key=nvidia_api_key, 
                model=guard_model, 
                temperature=0.0
            )
            rails = build_rails(llm)
            async def _run():
                return await rails.generate_async(
                    messages=[{"role": "user", "content": message}]
                )
            return asyncio.run(_run())

        t0 = time.time()
        nemo_error = None
        try:
            raw = _executor.submit(_nemo_worker).result(timeout=60)
        except Exception:
            raw        = f"Guard error"
            nemo_error = tb.format_exc()
        rail_ms = round((time.time() - t0) * 1000)

        text, is_blocked, block_reason, is_dialog, needs_rag = parse_nemo_response(raw)

        trace["rail"] = {
            "active": True,
            "blocked": is_blocked,
            "reason":  block_reason,
            "dialog":  is_dialog,
            "ms":      rail_ms,
            "model":   guard_model,
            "error":   nemo_error,
        }

        if is_blocked or is_dialog:
            trace["total_ms"] = round((time.time() - t_total) * 1000)
            return text, trace
            
    else:
        # Code path taken when single model mode is active
        trace["rail"] = {
            "active": False,
            "blocked": False,
            "reason": "Bypassed by choice",
            "ms": 0,
            "model": "None"
        }

    # ── FAISS retrieval ───────────────────────────────────────────────────────
    t1 = time.time()
    rag_error = None
    try:
        chunks = retrieve(message, vectorstore, k=3)
    except Exception:
        chunks    = []
        rag_error = tb.format_exc()
    rag_ms = round((time.time() - t1) * 1000)
    trace["retrieval"] = {"chunks": chunks, "ms": rag_ms, "error": rag_error}

    # ── LLM ② — answer generation ────────────────────────────────────────────
    context_text = "\n\n---\n\n".join(
        f"[{c['source']}]\n{c['content']}" for c in chunks
    ) if chunks else "No relevant policy excerpts were retrieved."

    t2 = time.time()
    gen_error = None
    answer    = ""
    try:
        llm = ChatNVIDIA(
            nvidia_api_key=nvidia_api_key, 
            model=chat_model, 
            temperature=1.0,
            top_p=1.0,
            max_tokens=16384,
            seed=42
        )
        resp = llm.invoke([
            SystemMessage(content=HR_SYSTEM_PROMPT.format(context=context_text)),
            HumanMessage(content=message),
        ])
        answer = resp.content
    except Exception:
        gen_error = tb.format_exc()
        answer    = "LLM call failed — see trace for details."
    gen_ms = round((time.time() - t2) * 1000)
    trace["generation"] = {"ms": gen_ms, "model": chat_model, "error": gen_error}

    # ── Output sanitizer ──────────────────────────────────────────────────────
    t3     = time.time()
    issues = check_output(answer)
    out_ms = round((time.time() - t3) * 1000)

    if issues:
        answer = "My response contained potentially sensitive information and has been withheld. Please contact hr@acmecorp.com directly."
        trace["output_rail"] = {"blocked": True, "issues": issues, "ms": out_ms}
    else:
        trace["output_rail"] = {"blocked": False, "ms": out_ms}

    trace["total_ms"] = round((time.time() - t_total) * 1000)
    return answer, trace
