"""Tests for clustering wrapper and cluster summarization."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from selene_agent.autonomy import memory_clustering


def _planted_vectors(n_per_cluster=6, n_clusters=3, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)) * 5
    vecs, truth = [], []
    for c in range(n_clusters):
        for _ in range(n_per_cluster):
            vecs.append(centers[c] + rng.normal(size=dim) * 0.3)
            truth.append(c)
    return np.array(vecs), truth


def test_hdbscan_finds_planted_clusters():
    vecs, truth = _planted_vectors()
    labels = memory_clustering.cluster_vectors(
        vecs, min_cluster_size=3, min_samples=2,
    )
    # At least 2 distinct non-noise labels should map to our 3 planted clusters.
    non_noise = [l for l in labels if l != -1]
    assert len(set(non_noise)) >= 2


def test_hdbscan_returns_all_noise_when_too_few_points():
    vecs = np.random.default_rng(0).normal(size=(2, 16))
    labels = memory_clustering.cluster_vectors(
        vecs, min_cluster_size=5, min_samples=3,
    )
    assert all(l == -1 for l in labels)


@pytest.mark.asyncio
async def test_summarize_cluster_happy_path():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "summary": "Matt prefers oat milk in coffee.",
            "tags": ["coffee", "preferences", "core_fact"],
            "rationale": "Four independent mentions of oat milk preference.",
        })))
    ]
    out = await memory_clustering.summarize_cluster(
        client=client,
        model_name="gpt-3.5-turbo",
        member_texts=["oat milk pls", "no dairy in coffee", "oat milk again", "oat milk"],
    )
    assert out is not None
    assert "oat milk" in out["summary"].lower()
    assert "coffee" in out["tags"]


@pytest.mark.asyncio
async def test_summarize_cluster_null_pattern_returns_none():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "summary": None, "tags": [], "rationale": "No coherent pattern."
        })))
    ]
    out = await memory_clustering.summarize_cluster(
        client=client,
        model_name="gpt-3.5-turbo",
        member_texts=["a", "b", "c"],
    )
    assert out is None


@pytest.mark.asyncio
async def test_summarize_cluster_malformed_json_returns_none():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="sorry I can't help"))
    ]
    out = await memory_clustering.summarize_cluster(
        client=client,
        model_name="gpt-3.5-turbo",
        member_texts=["a", "b"],
    )
    assert out is None
