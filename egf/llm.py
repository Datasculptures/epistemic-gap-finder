"""Ollama HTTP client for candidate generation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from egf.domain import DomainTemplate

logger = logging.getLogger(__name__)

DEFAULT_LLM_HOST = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3"
LLM_TIMEOUT = 30.0
LLM_RETRIES = 1


@dataclass(frozen=True)
class LLMCandidate:
    name: str
    function_summary: str
    positioning_summary: str


def health_check(host: str = DEFAULT_LLM_HOST) -> bool:
    """
    Return True if the ollama server at host is reachable, False otherwise.
    Never raises.
    """
    try:
        response = httpx.get(f"{host}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def generate_candidate(
    gap_id: int,
    bounding_items: list[str],
    vocabulary_terms: list[str],
    domain: DomainTemplate,
    host: str = DEFAULT_LLM_HOST,
    model: str = DEFAULT_LLM_MODEL,
) -> LLMCandidate | None:
    """
    Ask ollama to describe a candidate concept for the given gap.

    Returns LLMCandidate on success, None on any failure.
    Never raises.
    """
    clean_names = [n.rsplit(".", 1)[0] for n in bounding_items]
    terms_str = ", ".join(vocabulary_terms) if vocabulary_terms else "(none)"
    names_str = ", ".join(clean_names)

    system_prompt = (
        f"{domain.system_prompt_fragment}\n\n"
        "You identify gaps in conceptual space and describe what would occupy them. "
        "Respond ONLY with valid JSON. No preamble, no markdown fences, no explanation."
    )

    user_prompt = (
        f"The following {domain.label_plural} form a conceptual cluster:\n"
        f"  {names_str}\n\n"
        f"Key vocabulary terms near the gap between them:\n"
        f"  {terms_str}\n\n"
        f"Describe a {domain.label_noun} that would occupy the conceptual gap "
        f"between these {domain.label_plural}.\n\n"
        f"Respond with ONLY this JSON object (no other text):\n"
        f'{{"name": "...", "function_summary": "...", "positioning_summary": "..."}}'
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
    }

    for attempt in range(LLM_RETRIES + 1):
        try:
            response = httpx.post(
                f"{host}/api/chat",
                json=payload,
                timeout=LLM_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"]

            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            parsed = json.loads(content)

            name = str(parsed.get("name", "")).strip()
            function_summary = str(parsed.get("function_summary", "")).strip()
            positioning_summary = str(
                parsed.get("positioning_summary", "")
            ).strip()

            if not name or not function_summary or not positioning_summary:
                logger.warning(
                    "Gap %d: LLM response missing required fields", gap_id
                )
                return None

            return LLMCandidate(
                name=name,
                function_summary=function_summary,
                positioning_summary=positioning_summary,
            )

        except httpx.TimeoutException:
            if attempt < LLM_RETRIES:
                logger.warning("Gap %d: LLM timeout, retrying...", gap_id)
                continue
            logger.warning(
                "Gap %d: LLM timed out after %d attempt(s)",
                gap_id,
                LLM_RETRIES + 1,
            )
            return None

        except Exception as e:
            logger.warning("Gap %d: LLM error — %s", gap_id, e)
            return None

    return None
