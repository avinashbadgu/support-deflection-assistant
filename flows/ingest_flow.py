"""Prefect flow for scheduled ingest with retries and failure alerts.

Run ad-hoc:
    python flows/ingest_flow.py

Deploy on a cron (Prefect 3):
    prefect deploy flows/ingest_flow.py:ingest_flow ...

The flow defers heavy work to support_assistant.pipeline.build_index,
which already does content-hash delta updates — so scheduled runs on an
unchanged corpus do zero embedding calls.
"""

from __future__ import annotations

from prefect import flow, task, get_run_logger

from support_assistant.pipeline import build_index


@task(retries=2, retry_delay_seconds=30)
def run_ingest(use_cache: bool, use_delta: bool) -> dict[str, int]:
    return build_index(use_cache=use_cache, use_delta=use_delta)


@flow(name="stripe-docs-ingest")
def ingest_flow(use_cache: bool = False, use_delta: bool = True) -> dict[str, int]:
    """Re-fetch the corpus (cache off so we see real upstream changes), then
    embed only the delta. With use_delta=True an unchanged corpus performs
    zero embedding calls."""
    logger = get_run_logger()
    counts = run_ingest(use_cache=use_cache, use_delta=use_delta)
    logger.info("ingest_complete", extra={"counts": counts})
    return counts


if __name__ == "__main__":
    print(ingest_flow())
