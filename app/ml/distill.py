from typing import Dict


def distill_chunk(chunk: str) -> Dict[str, str]:
    """
    Minimal placeholder for knowledge distillation.
    Replace this with an LLM call in next iterations.
    """
    summary = chunk[:260].strip()
    if len(chunk) > 260:
        summary += "..."
    return {"fact_summary": summary}
