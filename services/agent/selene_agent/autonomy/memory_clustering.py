"""HDBSCAN clustering + LLM cluster summarization for memory_review.

`cluster_vectors` is a thin, testable wrapper. `summarize_cluster` performs
one LLM call per cluster and normalizes the structured output.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import hdbscan
import numpy as np


CLUSTER_SUMMARIZER_SYSTEM = (
    "You consolidate related memories into a single durable summary. "
    "You will receive N short memory texts that have been clustered by "
    "semantic similarity. Produce ONE consolidated summary capturing the "
    "stable pattern across them, plus up to 3 tags. If the texts do not "
    "share a coherent pattern, return null as the summary. "
    "Respond with ONE JSON object and nothing else: "
    '{"summary": string|null, "tags": array of <=3 strings, "rationale": string}. '
    "No prose, no code fences, no preamble."
)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def cluster_vectors(
    vectors: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> List[int]:
    """Run HDBSCAN with cosine metric. Returns a label per input row.

    Label -1 indicates noise (not clustered). Returns all-noise if there are
    fewer input rows than ``min_cluster_size``.
    """
    if len(vectors) < min_cluster_size:
        return [-1] * len(vectors)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # HDBSCAN doesn't support cosine natively
    )
    # Normalize vectors so euclidean ≈ cosine.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = np.where(norms > 0, vectors / norms, vectors)
    labels = clusterer.fit_predict(normed)
    return [int(l) for l in labels]


async def summarize_cluster(
    *,
    client,
    model_name: str,
    member_texts: List[str],
    max_tokens: int = 1500,
    temperature: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """Call the LLM once to summarize a cluster. Returns normalized dict or None.

    None is returned when the LLM says the pattern is not coherent, or when
    the LLM output cannot be parsed as the expected JSON shape.
    """
    user_prompt = "Memory texts to consolidate:\n\n" + "\n".join(
        f"- {t}" for t in member_texts
    ) + "\n\nOutput the JSON object only."
    resp = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": CLUSTER_SUMMARIZER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    parsed = _extract_json(content)
    if parsed is None:
        return None
    summary = parsed.get("summary")
    if summary is None or not isinstance(summary, str) or not summary.strip():
        return None
    tags = parsed.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t)[:64] for t in tags[:3]]
    rationale = str(parsed.get("rationale") or "")[:280]
    return {"summary": summary.strip(), "tags": tags, "rationale": rationale}
