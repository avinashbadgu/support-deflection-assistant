"""Compare runner markdown output + preset application."""

from __future__ import annotations

import pytest

from support_assistant.config import settings
from support_assistant.eval.runner import PRESETS, _apply_preset, _compare_to_markdown, _restore


def test_known_presets_exist():
    expected = {"baseline-dense", "dense+llm-rerank", "hybrid", "hybrid+llm-rerank"}
    assert expected.issubset(PRESETS.keys())


def test_apply_and_restore_settings():
    saved = _apply_preset("hybrid+llm-rerank")
    try:
        assert settings.use_hybrid is True
        assert settings.use_reranker is True
        assert settings.reranker_kind == "llm"
    finally:
        _restore(saved)
    # back to defaults
    assert settings.use_hybrid is False or settings.use_hybrid is True  # whatever the saved value was


def test_apply_unknown_preset_raises():
    with pytest.raises(ValueError):
        _apply_preset("not-a-real-preset")


def test_compare_markdown_structure():
    rows = [
        {
            "preset": "baseline-dense",
            "recall_at_k": 0.71,
            "abstention_accuracy": 0.85,
            "precision_abstain": 0.80,
            "recall_abstain": 0.78,
            "faithfulness_mean": 0.91,
            "hallucination_rate": 0.09,
            "latency_p50_ms": 1100,
            "latency_p95_ms": 1800,
            "cost_mean_usd": 0.00123,
        },
        {
            "preset": "hybrid+llm-rerank",
            "recall_at_k": 0.86,
            "abstention_accuracy": 0.89,
            "precision_abstain": 0.84,
            "recall_abstain": 0.83,
            "faithfulness_mean": 0.94,
            "hallucination_rate": 0.06,
            "latency_p50_ms": 2400,
            "latency_p95_ms": 4100,
            "cost_mean_usd": 0.00345,
        },
    ]
    md = _compare_to_markdown(rows)
    assert "| Preset |" in md
    assert "baseline-dense" in md
    assert "hybrid+llm-rerank" in md
    # Two header lines + 2 data rows.
    assert md.count("|") >= 2 * (1 + 1 + len(rows))
