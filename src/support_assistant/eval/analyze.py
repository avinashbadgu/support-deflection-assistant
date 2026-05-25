"""Failure analysis on an eval report.

Reads a JSON report produced by `support-assistant eval` and prints (and saves
as markdown) a per-category failure breakdown:

  - false abstentions     should have answered, abstained instead
  - missed abstentions    should have abstained, answered instead (hallucination risk)
  - low recall            recall@k = 0
  - low faithfulness      faith < 1.0 with the unsupported claims listed

This is what the user reads after a benchmark run to find the *interesting*
failure cases — the ones to write better eval questions about, fix in the
prompt, or address with a config change.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dataset import report_dir


def analyze(report_path: str | Path) -> dict[str, Any]:
    p = Path(report_path)
    if not p.is_absolute():
        p = report_dir() / p
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return _analyze_sweep(data, p)
    return _analyze_report(data, p)


def _analyze_report(data: dict, src: Path) -> dict[str, Any]:
    cases = data.get("cases", [])
    config_tag = data.get("config_tag", "(untagged)")

    false_abstain: list[dict] = []   # should answer, abstained
    missed_abstain: list[dict] = []  # should abstain, answered
    low_recall: list[dict] = []
    low_faith: list[dict] = []

    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {
        "n": 0, "false_abstain": 0, "missed_abstain": 0,
        "low_recall": 0, "low_faith": 0,
    })

    for c in cases:
        cat = c.get("category", "?")
        by_cat[cat]["n"] += 1
        if c.get("expected_abstain") is False and c.get("did_abstain") is True:
            false_abstain.append(c); by_cat[cat]["false_abstain"] += 1
        if c.get("expected_abstain") is True and c.get("did_abstain") is False:
            missed_abstain.append(c); by_cat[cat]["missed_abstain"] += 1
        if not c.get("expected_abstain") and c.get("recall_at_k") == 0:
            low_recall.append(c); by_cat[cat]["low_recall"] += 1
        f = c.get("faithfulness")
        if f is not None and f < 1.0:
            low_faith.append(c); by_cat[cat]["low_faith"] += 1

    md = _to_markdown(config_tag, by_cat, false_abstain, missed_abstain, low_recall, low_faith)
    out_md = src.with_name(src.stem + "_failures.md")
    out_md.write_text(md, encoding="utf-8")

    print(md)
    print(f"\nWritten: {out_md}")
    return {
        "by_category": by_cat,
        "false_abstain_count": len(false_abstain),
        "missed_abstain_count": len(missed_abstain),
        "low_recall_count": len(low_recall),
        "low_faith_count": len(low_faith),
        "md_path": str(out_md),
    }


def _analyze_sweep(rows: list[dict], src: Path) -> dict[str, Any]:
    # Find the threshold that maximises abstention accuracy.
    best = max(rows, key=lambda r: r.get("abstention_accuracy", 0))
    md = f"## Sweep summary — {src.name}\n\n"
    md += "| Threshold | Accuracy | Prec | Rec | FP | FN |\n|---:|---:|---:|---:|---:|---:|\n"
    md += "\n".join(
        f"| {r['threshold']:.2f} | {r['abstention_accuracy']:.3f} | "
        f"{r['precision_abstain']:.3f} | {r['recall_abstain']:.3f} | "
        f"{r['fp']} | {r['fn']} |"
        for r in rows
    )
    md += f"\n\n**Best threshold:** {best['threshold']:.2f} (accuracy {best['abstention_accuracy']:.3f})\n"
    out_md = src.with_name(src.stem + "_summary.md")
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWritten: {out_md}")
    return {"best_threshold": best, "md_path": str(out_md)}


def _to_markdown(
    tag: str,
    by_cat: dict[str, dict[str, int]],
    false_abstain: list[dict],
    missed_abstain: list[dict],
    low_recall: list[dict],
    low_faith: list[dict],
) -> str:
    parts: list[str] = []
    parts.append(f"# Failure analysis — {tag}\n")
    parts.append(_cat_table(by_cat))

    parts.append(_section("Missed abstentions (answered when it should have refused)", missed_abstain, why_field="abstain_reason"))
    parts.append(_section("False abstentions (refused when it should have answered)", false_abstain, why_field="abstain_reason"))
    parts.append(_section("Low recall (none of the expected sources retrieved)", low_recall, why_field=None))
    parts.append(_section("Low faithfulness (some claim unsupported by citations)", low_faith, why_field="unsupported_claims"))

    return "\n".join(parts)


def _cat_table(by_cat: dict[str, dict[str, int]]) -> str:
    head = "## Per-category failure counts\n\n| Category | N | False abstain | Missed abstain | Low recall | Low faith |\n|---|---:|---:|---:|---:|---:|"
    body = "\n".join(
        f"| {cat} | {v['n']} | {v['false_abstain']} | {v['missed_abstain']} | {v['low_recall']} | {v['low_faith']} |"
        for cat, v in sorted(by_cat.items())
    )
    return f"{head}\n{body}\n"


def _section(title: str, cases: list[dict], why_field: str | None) -> str:
    if not cases:
        return f"## {title}\n\n_None._\n"
    parts = [f"## {title}  ({len(cases)})\n"]
    for c in cases[:25]:  # cap detail; full list is in JSON
        why = ""
        if why_field:
            v = c.get(why_field)
            if isinstance(v, list):
                why = " — " + "; ".join(v[:3])
            elif v:
                why = " — " + str(v)
        parts.append(
            f"- **{c.get('case_id')}** ({c.get('category')}) — _{(c.get('question') or '').replace(chr(10),' ')}_{why}"
        )
    if len(cases) > 25:
        parts.append(f"- _(... {len(cases) - 25} more, see JSON)_")
    return "\n".join(parts) + "\n"
