"""Eval runner: drive the full pipeline against the eval set, compute metrics, write a report.

Three entry points:
- run_eval(config_tag)        : one pass with current settings.
- sweep_abstention()          : grid over abstain_min_score → precision/recall curve.
- run_compare(presets)        : run N named config presets, emit a markdown
                                 comparison table — the M4/M8/M9 benchmark.

All math lives in metrics.py so it can be unit-tested without API calls.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from tqdm import tqdm

from .. import pipeline
from ..config import settings
from .dataset import EvalCase, load_eval_set, report_dir
from .judge import FaithfulnessJudge, FaithfulnessResult
from .metrics import (
    AbstentionCounts,
    CategoryBucket,
    percentile,
    recall_at_k,
    update_abstention,
)


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    expected_abstain: bool
    did_abstain: bool
    abstain_reason: str | None
    answer: str
    retrieved_urls: list[str]
    recall_at_k: float
    faithfulness: float | None
    unsupported_claims: list[str]
    latency_ms: float
    cost_usd: float


@dataclass
class Report:
    config_tag: str
    n_cases: int
    recall_at_k_mean: float
    abstention: dict[str, float]
    faithfulness_mean: float | None
    hallucination_rate: float | None
    latency_p50_ms: float
    latency_p95_ms: float
    cost_total_usd: float
    cost_mean_usd: float
    settings_snapshot: dict[str, Any]
    by_category: list[dict] = field(default_factory=list)
    cases: list[dict] = field(default_factory=list)


def _run_one(case: EvalCase, judge: FaithfulnessJudge | None) -> CaseResult:
    t0 = time.perf_counter()
    out = pipeline.ask(case.question, use_cache=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    retrieved_urls = [c["url"] for c in out["citations"]]
    rk = recall_at_k(case.expected_source_urls, retrieved_urls)

    faith_score: float | None = None
    unsupported: list[str] = []
    if not out["abstained"] and judge is not None:
        f: FaithfulnessResult = judge.judge(
            case.question,
            out["answer"],
            [{"title": c.get("title", ""), "text": ""} for c in out["citations"]],
        )
        faith_score = f.score
        unsupported = f.unsupported_claims

    return CaseResult(
        case_id=case.id,
        category=case.category,
        question=case.question,
        expected_abstain=case.expected_abstain,
        did_abstain=bool(out["abstained"]),
        abstain_reason=out.get("abstain_reason"),
        answer=out["answer"],
        retrieved_urls=retrieved_urls,
        recall_at_k=rk,
        faithfulness=faith_score,
        unsupported_claims=unsupported,
        latency_ms=latency_ms,
        cost_usd=float(out.get("cost_usd", 0.0)),
    )


def run_eval(
    config_tag: str = "default",
    with_judge: bool = True,
    on_each: Callable[[CaseResult], None] | None = None,
    limit: int | None = None,
) -> Report:
    cases = load_eval_set()
    if limit:
        cases = cases[:limit]
    judge = FaithfulnessJudge() if with_judge else None

    results: list[CaseResult] = []
    counts = AbstentionCounts()
    buckets: dict[str, CategoryBucket] = {}

    for c in tqdm(cases, desc=f"eval[{config_tag}]"):
        r = _run_one(c, judge)
        results.append(r)
        update_abstention(counts, expected_abstain=c.expected_abstain, did_abstain=r.did_abstain)

        b = buckets.setdefault(c.category, CategoryBucket(category=c.category))
        b.n += 1
        if not c.expected_abstain:
            b.recalls.append(r.recall_at_k)
        update_abstention(b.abstention, c.expected_abstain, r.did_abstain)
        if r.faithfulness is not None:
            b.faithfulnesses.append(r.faithfulness)
        b.latencies_ms.append(r.latency_ms)
        if on_each:
            on_each(r)

    answered_recalls = [r.recall_at_k for r in results if not r.expected_abstain]
    answered_faiths = [r.faithfulness for r in results if r.faithfulness is not None]
    latencies = [r.latency_ms for r in results]
    costs = [r.cost_usd for r in results]

    report = Report(
        config_tag=config_tag,
        n_cases=len(results),
        recall_at_k_mean=(sum(answered_recalls) / len(answered_recalls)) if answered_recalls else 0.0,
        abstention={
            "accuracy": counts.accuracy,
            "precision_abstain": counts.precision_abstain,
            "recall_abstain": counts.recall_abstain,
            "tp": counts.tp,
            "tn": counts.tn,
            "fp": counts.fp,
            "fn": counts.fn,
        },
        faithfulness_mean=(sum(answered_faiths) / len(answered_faiths)) if answered_faiths else None,
        hallucination_rate=(
            sum(1 for f in answered_faiths if f < 1.0) / len(answered_faiths)
            if answered_faiths else None
        ),
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        cost_total_usd=sum(costs),
        cost_mean_usd=(sum(costs) / len(costs)) if costs else 0.0,
        settings_snapshot=_settings_snapshot(),
        by_category=[buckets[k].summary() for k in sorted(buckets)],
        cases=[asdict(r) for r in results],
    )

    out_path = report_dir() / f"{int(time.time())}_{config_tag}.json"
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    _print_summary(report, out_path)
    return report


def _settings_snapshot() -> dict[str, Any]:
    return {
        "embed_model": settings.embed_model,
        "gen_model": settings.gen_model,
        "judge_model": settings.judge_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "rerank_top_n": settings.rerank_top_n,
        "use_reranker": settings.use_reranker,
        "reranker_kind": settings.reranker_kind,
        "use_cache": settings.use_cache,
        "use_hybrid": settings.use_hybrid,
        "bm25_top_k": settings.bm25_top_k,
        "rrf_k": settings.rrf_k,
        "abstain_min_score": settings.abstain_min_score,
        "abstain_use_judge": settings.abstain_use_judge,
    }


def _print_summary(r: Report, out_path) -> None:
    print()
    print(f"=== eval report [{r.config_tag}]  ({r.n_cases} cases) ===")
    print(f"  recall@k (answerable):     {r.recall_at_k_mean:.3f}")
    print(f"  abstention accuracy:       {r.abstention['accuracy']:.3f}")
    print(f"    tp={r.abstention['tp']}  tn={r.abstention['tn']}  fp={r.abstention['fp']}  fn={r.abstention['fn']}")
    print(f"    precision (abstain):     {r.abstention['precision_abstain']:.3f}")
    print(f"    recall    (abstain):     {r.abstention['recall_abstain']:.3f}")
    if r.faithfulness_mean is not None:
        print(f"  faithfulness mean:         {r.faithfulness_mean:.3f}")
        print(f"  hallucination rate:        {r.hallucination_rate:.3f}")
    print(f"  latency p50/p95:           {r.latency_p50_ms:.0f}ms / {r.latency_p95_ms:.0f}ms")
    print(f"  cost total / mean:         ${r.cost_total_usd:.4f} / ${r.cost_mean_usd:.4f}")
    if r.by_category:
        print()
        print("  by category:")
        for c in r.by_category:
            rk = "—" if c["recall_at_k_mean"] is None else f"{c['recall_at_k_mean']:.3f}"
            ab = "—" if c["abstention_accuracy"] is None else f"{c['abstention_accuracy']:.3f}"
            fa = "—" if c["faithfulness_mean"] is None else f"{c['faithfulness_mean']:.3f}"
            print(f"    {c['category']:<22} n={c['n']:>3}  recall={rk}  abstain_acc={ab}  faith={fa}")
    print(f"  report:                    {out_path}")
    print()


def sweep_abstention(
    config_tag: str = "sweep",
    thresholds: list[float] | None = None,
) -> list[dict]:
    thresholds = thresholds or [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    original = settings.abstain_min_score
    rows: list[dict] = []
    try:
        for t in thresholds:
            settings.abstain_min_score = t
            r = run_eval(config_tag=f"{config_tag}-floor{t:.2f}", with_judge=False)
            rows.append(
                {
                    "threshold": t,
                    "abstention_accuracy": r.abstention["accuracy"],
                    "precision_abstain": r.abstention["precision_abstain"],
                    "recall_abstain": r.abstention["recall_abstain"],
                    "fn": r.abstention["fn"],
                    "fp": r.abstention["fp"],
                }
            )
    finally:
        settings.abstain_min_score = original

    out_path = report_dir() / f"{int(time.time())}_sweep.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print()
    print(f"=== abstain threshold sweep ({len(rows)} points) ===")
    print(f"{'thr':>5}  {'acc':>5}  {'precA':>6}  {'recA':>6}  {'fp':>3}  {'fn':>3}")
    for r in rows:
        print(f"{r['threshold']:>5.2f}  {r['abstention_accuracy']:>5.3f}  {r['precision_abstain']:>6.3f}  {r['recall_abstain']:>6.3f}  {r['fp']:>3}  {r['fn']:>3}")
    print(f"  sweep report: {out_path}\n")
    return rows


# ---------- Comparison runner (the benchmark table) ----------

PRESETS: dict[str, dict[str, Any]] = {
    "baseline-dense": {
        "use_hybrid": False, "use_reranker": False, "reranker_kind": "none",
    },
    "dense+llm-rerank": {
        "use_hybrid": False, "use_reranker": True, "reranker_kind": "llm",
    },
    "dense+crossencoder": {
        "use_hybrid": False, "use_reranker": True, "reranker_kind": "crossencoder",
    },
    "hybrid": {
        "use_hybrid": True, "use_reranker": False, "reranker_kind": "none",
    },
    "hybrid+llm-rerank": {
        "use_hybrid": True, "use_reranker": True, "reranker_kind": "llm",
    },
    "hybrid+crossencoder": {
        "use_hybrid": True, "use_reranker": True, "reranker_kind": "crossencoder",
    },
}


def _apply_preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}. Known: {list(PRESETS)}")
    saved = {k: getattr(settings, k) for k in PRESETS[name]}
    for k, v in PRESETS[name].items():
        setattr(settings, k, v)
    return saved


def _restore(saved: dict[str, Any]) -> None:
    for k, v in saved.items():
        setattr(settings, k, v)


def run_compare(
    preset_names: list[str] | None = None,
    with_judge: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the eval set under each named preset and produce a markdown table.

    Writes JSON + markdown under data/eval_reports/. NO numbers are fabricated;
    rows are filled only by actually running.
    """
    preset_names = preset_names or list(PRESETS.keys())
    rows: list[dict[str, Any]] = []
    for name in preset_names:
        print(f"\n>>> running preset: {name}")
        saved = _apply_preset(name)
        try:
            r = run_eval(config_tag=f"compare-{name}", with_judge=with_judge, limit=limit)
            rows.append(
                {
                    "preset": name,
                    "settings": PRESETS[name],
                    "recall_at_k": r.recall_at_k_mean,
                    "abstention_accuracy": r.abstention["accuracy"],
                    "precision_abstain": r.abstention["precision_abstain"],
                    "recall_abstain": r.abstention["recall_abstain"],
                    "faithfulness_mean": r.faithfulness_mean,
                    "hallucination_rate": r.hallucination_rate,
                    "latency_p50_ms": r.latency_p50_ms,
                    "latency_p95_ms": r.latency_p95_ms,
                    "cost_mean_usd": r.cost_mean_usd,
                    "by_category": r.by_category,
                }
            )
        finally:
            _restore(saved)

    ts = int(time.time())
    json_path = report_dir() / f"{ts}_compare.json"
    md_path   = report_dir() / f"{ts}_compare.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(_compare_to_markdown(rows), encoding="utf-8")

    print()
    print(_compare_to_markdown(rows))
    print(f"\nReports written:\n  {json_path}\n  {md_path}\n")
    return {"rows": rows, "json": str(json_path), "md": str(md_path)}


def _fmt(v, d=3):
    if v is None:
        return "—"
    return f"{v:.{d}f}"


def _compare_to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    head = (
        "| Preset | Recall@k | Abstain Acc | Abstain Prec | Abstain Rec | Faith. mean | Halluc. rate | p50 ms | p95 ms | $/q (mean) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    body = "\n".join(
        "| {preset} | {rk} | {aacc} | {ap} | {ar} | {fm} | {hr} | {p50:.0f} | {p95:.0f} | ${cm} |".format(
            preset=r["preset"],
            rk=_fmt(r["recall_at_k"]),
            aacc=_fmt(r["abstention_accuracy"]),
            ap=_fmt(r["precision_abstain"]),
            ar=_fmt(r["recall_abstain"]),
            fm=_fmt(r["faithfulness_mean"]),
            hr=_fmt(r["hallucination_rate"]),
            p50=r["latency_p50_ms"],
            p95=r["latency_p95_ms"],
            cm=_fmt(r["cost_mean_usd"], 5),
        )
        for r in rows
    )
    return f"## Benchmark comparison\n\n{head}\n{body}\n"
