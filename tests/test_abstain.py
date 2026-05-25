"""Abstention threshold logic. The LLM judge is stubbed — we test gate behavior."""

from __future__ import annotations

from support_assistant import abstain
from support_assistant.store import Retrieved


def _r(score: float) -> Retrieved:
    return Retrieved(chunk_id="x:0", url="https://x", title="t", text="...", score=score)


class _FakeJudge:
    def __init__(self, can_answer: bool, confidence: float = 0.9):
        self.can_answer = can_answer
        self.confidence = confidence
        self.calls = 0

    def judge(self, question, context):
        self.calls += 1
        return self.can_answer, self.confidence, "stubbed"


def test_abstain_when_no_context():
    d = abstain.decide("q", [], use_judge=False)
    assert d.abstain
    assert d.reason == "no_context_retrieved"


def test_abstain_when_top_score_below_floor():
    d = abstain.decide("q", [_r(0.10)], min_score=0.50, use_judge=False)
    assert d.abstain
    assert d.reason.startswith("top_score_below_floor")


def test_no_abstain_when_floor_passes_and_judge_off():
    d = abstain.decide("q", [_r(0.80)], min_score=0.50, use_judge=False)
    assert not d.abstain


def test_judge_can_force_abstain_even_when_floor_passes():
    j = _FakeJudge(can_answer=False)
    d = abstain.decide("q", [_r(0.90)], min_score=0.30, judge=j, use_judge=True)
    assert d.abstain
    assert j.calls == 1
    assert d.judge_confidence == 0.9


def test_judge_can_allow_when_floor_passes():
    j = _FakeJudge(can_answer=True)
    d = abstain.decide("q", [_r(0.90)], min_score=0.30, judge=j, use_judge=True)
    assert not d.abstain
    assert j.calls == 1


def test_floor_short_circuits_judge():
    """If the score floor fires, the judge should not be called (saves cost)."""
    j = _FakeJudge(can_answer=True)
    d = abstain.decide("q", [_r(0.05)], min_score=0.50, judge=j, use_judge=True)
    assert d.abstain
    assert j.calls == 0


def test_top_score_override_used_when_provided():
    """Hybrid retrieval supplies the original dense cosine via override — the
    candidates' own scores (RRF-fused, on a different scale) should NOT be
    used for the floor decision."""
    # context score is high (0.90) but the dense signal is low (0.10) → should abstain.
    d = abstain.decide(
        "q",
        [_r(0.90)],
        min_score=0.30,
        use_judge=False,
        top_score_override=0.10,
    )
    assert d.abstain
    assert "0.100" in d.reason


def test_top_score_override_lets_through_when_dense_is_strong():
    # Conversely: candidate scores look low (RRF ~0.03) but dense was strong.
    d = abstain.decide(
        "q",
        [_r(0.03)],
        min_score=0.30,
        use_judge=False,
        top_score_override=0.80,
    )
    assert not d.abstain
