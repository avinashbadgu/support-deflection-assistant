"""CLI: ingest, ask, eval, compare, analyze, serve."""

from __future__ import annotations

import typer

from . import pipeline

app = typer.Typer(add_completion=False, help="Support-deflection assistant.")


@app.command()
def ingest(
    no_cache: bool = typer.Option(False, "--no-cache", help="Refetch even if cached on disk."),
    full_rebuild: bool = typer.Option(False, "--full", help="Skip delta — re-embed everything."),
) -> None:
    """Fetch the curated Stripe doc seed list, chunk, embed (delta), upsert, rebuild BM25."""
    counts = pipeline.build_index(use_cache=not no_cache, use_delta=not full_rebuild)
    typer.echo(f"index: {counts}")


@app.command()
def ask(question: str, no_cache: bool = typer.Option(False, "--no-cache")) -> None:
    """Ask a question. Prints answer, abstention status, citations."""
    out = pipeline.ask(question, use_cache=not no_cache)
    typer.echo("\n--- ANSWER ---\n")
    typer.echo(out["answer"])
    typer.echo(f"\nabstained: {out['abstained']}  reason: {out['abstain_reason']}")
    typer.echo(f"cache_hit: {out['cache_hit']}  cost: ${out['cost_usd']:.5f}")
    typer.echo("\n--- SOURCES ---")
    for i, c in enumerate(out["citations"], start=1):
        typer.echo(f"[{i}] {c['title']}  ({c['url']})  score={c['score']:.4f}")


@app.command()
def eval(
    config: str = typer.Option("default", help="Variant tag for the report (e.g. baseline, with-rerank)."),
    sweep_abstain: bool = typer.Option(False, "--sweep-abstain", help="Sweep abstention thresholds."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip the faithfulness LLM judge (faster, cheaper)."),
    limit: int = typer.Option(0, help="Run only the first N cases (0 = all)."),
) -> None:
    """Run the eval set and print metrics. Writes a JSON report under data/eval_reports/."""
    from .eval.runner import run_eval, sweep_abstention

    if sweep_abstain:
        sweep_abstention(config_tag=config)
    else:
        run_eval(config_tag=config, with_judge=not no_judge, limit=limit or None)


@app.command()
def compare(
    presets: str = typer.Option(
        "baseline-dense,dense+llm-rerank,hybrid,hybrid+llm-rerank",
        help="Comma-separated preset names. Run `support-assistant compare --list` to see all.",
    ),
    list_presets: bool = typer.Option(False, "--list", help="Print available presets and exit."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip faithfulness judge (faster, cheaper)."),
    limit: int = typer.Option(0, help="Run only the first N cases per preset (0 = all)."),
) -> None:
    """Benchmark N config presets and emit a markdown comparison table.

    The benchmark is what fills DECISIONS.md and the README. No row is filled
    unless a real run produced it.
    """
    from .eval.runner import PRESETS, run_compare

    if list_presets:
        for name, cfg in PRESETS.items():
            typer.echo(f"  {name}: {cfg}")
        return

    names = [s.strip() for s in presets.split(",") if s.strip()]
    run_compare(preset_names=names, with_judge=not no_judge, limit=limit or None)


@app.command()
def analyze(report: str = typer.Argument(..., help="Path or filename inside data/eval_reports/.")) -> None:
    """Failure analysis on an eval report. Writes a sibling _failures.md."""
    from .eval.analyze import analyze as _analyze

    _analyze(report)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI server (chat + dashboard + sources + eval + feedback + settings)."""
    import uvicorn

    uvicorn.run("support_assistant.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
