import html
import re
from typing import Dict


def _clean_text(text: str) -> str:
    """Strip HTML tags/entities and normalize whitespace."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def distill_chunk(chunk: str) -> Dict[str, str]:
    """
    Minimal knowledge distillation: clean raw text and extract a compact summary.
    Replace this with an LLM call in next iterations.
    """
    cleaned = _clean_text(chunk)
    if len(cleaned) <= 300:
        return {"fact_summary": cleaned}
    # Try to trim at a sentence boundary within the first 400 chars
    window = cleaned[:400]
    last_stop = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if last_stop > 100:
        return {"fact_summary": cleaned[: last_stop + 1]}
    return {"fact_summary": cleaned[:300] + "\u2026"}
