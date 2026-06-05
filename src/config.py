"""
App-wide constants — models, prompts, and output-scan patterns.
"""

GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 · 70B  ★ recommended for guardrails",
    "llama-3.1-8b-instant":    "Llama 3.1 · 8B  ★ fast & cheap",
    "openai/gpt-oss-120b":     "OpenAI OSS · 120B  — strong reasoning",
    "openai/gpt-oss-20b":      "OpenAI OSS · 20B  — fast",
}

GUARD_MODEL_DEFAULT = "llama-3.3-70b-versatile"
CHAT_MODEL_DEFAULT  = "llama-3.1-8b-instant"

HR_SYSTEM_PROMPT = (
    "You are the HR Policy Assistant for Acme Corp. "
    "Answer the employee's question using ONLY the policy excerpts provided below. "
    "Be concise, cite the relevant policy section, and include specific numbers or rules where applicable. "
    "If the answer is not covered in the excerpts, say so clearly and direct the employee to hr@acmecorp.com.\n\n"
    "Policy Excerpts:\n{context}"
)

SENSITIVE_OUTPUT_PATTERNS = {
    "credential_leak":  r"(?i)(password|passwd|secret|api[_\-]?key|token)\s*[:=]\s*['\"]?\w{6,}",
    "ssn_in_output":    r"\b\d{3}-\d{2}-\d{4}\b",
    "hardcoded_salary": r"(?i)\b(earns?|is paid|salary of)\s+\$[\d,]+\b",
}
