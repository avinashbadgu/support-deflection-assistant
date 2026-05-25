"""HTTP routes.

Pages (HTML):
  GET /            chat UI
  GET /dashboard   metrics dashboard
  GET /sources     corpus browser
  GET /eval        eval reports viewer
  GET /feedback    feedback inbox
  GET /settings    runtime tunables

JSON:
  POST /api/ask                  { question } -> answer payload
  POST /api/feedback             { event_id, rating, note, ... }
  GET  /api/metrics              aggregate stats
  GET  /api/events               recent events
  GET  /api/sources              corpus inventory
  GET  /api/eval/reports         list of eval report files
  GET  /api/eval/reports/{name}  one report
  GET  /api/feedback             list flagged answers
  GET  /api/settings             current tunables (read)
  POST /api/settings             update tunables (subset, runtime-safe only)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .. import cache as cache_mod
from .. import feedback as feedback_mod
from .. import pipeline
from ..config import settings
from ..eval.metrics import percentile

router = APIRouter()

# Settings that are safe to change at runtime (no data reshape required).
RUNTIME_TUNABLE = {
    "top_k": int,
    "rerank_top_n": int,
    "use_reranker": bool,
    "use_cache": bool,
    "abstain_min_score": float,
    "abstain_use_judge": bool,
    "cache_similarity": float,
    "cache_ttl_seconds": int,
}


class AskRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    event_id: str
    rating: str
    note: str = ""
    question: str
    answer: str
    citations: list[dict] = []


class SettingsPatch(BaseModel):
    top_k: int | None = None
    rerank_top_n: int | None = None
    use_reranker: bool | None = None
    use_cache: bool | None = None
    abstain_min_score: float | None = None
    abstain_use_judge: bool | None = None
    cache_similarity: float | None = None
    cache_ttl_seconds: int | None = None


# ---------- HTML pages ----------

def _render(request: Request, name: str) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(request, name)


@router.get("/", response_class=HTMLResponse)
def chat(request: Request) -> HTMLResponse:
    return _render(request, "chat.html")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return _render(request, "dashboard.html")


@router.get("/sources", response_class=HTMLResponse)
def sources(request: Request) -> HTMLResponse:
    return _render(request, "sources.html")


@router.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request) -> HTMLResponse:
    return _render(request, "eval.html")


@router.get("/feedback", response_class=HTMLResponse)
def feedback_page(request: Request) -> HTMLResponse:
    return _render(request, "feedback.html")


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    return _render(request, "settings.html")


# ---------- Core JSON ----------

@router.post("/api/ask")
def api_ask(body: AskRequest) -> JSONResponse:
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    out = pipeline.ask(body.question)
    return JSONResponse(out)


@router.post("/api/feedback")
def api_feedback(body: FeedbackRequest) -> JSONResponse:
    try:
        feedback_mod.submit(
            event_id=body.event_id,
            rating=body.rating,
            note=body.note,
            question=body.question,
            answer=body.answer,
            citations=body.citations,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"ok": True})


def _read_events_tail(limit: int = 200) -> list[dict[str, Any]]:
    p: Path = settings.events_path
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


@router.get("/api/metrics")
def api_metrics() -> JSONResponse:
    events = _read_events_tail(limit=1000)
    n = len(events)
    if n == 0:
        return JSONResponse(
            {
                "n": 0,
                "abstention_rate": 0.0,
                "cache_hit_rate": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "cost_total_usd": 0.0,
                "cost_mean_usd": 0.0,
                "cache_stats": cache_mod.cache.stats(),
                "stages_p95_ms": {"embed": 0, "retrieve": 0, "rerank": 0, "abstain": 0, "generate": 0},
                "series": [],
            }
        )

    abstained = sum(1 for e in events if e.get("abstained"))
    cache_hits = sum(1 for e in events if e.get("cache_hit"))

    def stage(e: dict, k: str) -> float:
        return float(e.get("timings", {}).get(k, 0))

    embeds = [stage(e, "embed_ms") for e in events]
    retrieves = [stage(e, "retrieve_ms") for e in events]
    reranks = [stage(e, "rerank_ms") for e in events]
    abstains = [stage(e, "abstain_ms") for e in events]
    generates = [stage(e, "generate_ms") for e in events]
    totals = [a + b + c + d + g for a, b, c, d, g in zip(embeds, retrieves, reranks, abstains, generates)]
    costs = [float(e.get("cost_usd", 0.0)) for e in events]

    series = [
        {
            "ts": e.get("ts"),
            "latency_ms": tot,
            "abstained": bool(e.get("abstained")),
            "cache_hit": bool(e.get("cache_hit")),
            "cost_usd": float(e.get("cost_usd", 0.0)),
        }
        for e, tot in zip(events, totals)
    ]

    return JSONResponse(
        {
            "n": n,
            "abstention_rate": abstained / n,
            "cache_hit_rate": cache_hits / n,
            "p50_ms": percentile(totals, 0.50),
            "p95_ms": percentile(totals, 0.95),
            "cost_total_usd": sum(costs),
            "cost_mean_usd": sum(costs) / n,
            "cache_stats": cache_mod.cache.stats(),
            "stages_p95_ms": {
                "embed": percentile(embeds, 0.95),
                "retrieve": percentile(retrieves, 0.95),
                "rerank": percentile(reranks, 0.95),
                "abstain": percentile(abstains, 0.95),
                "generate": percentile(generates, 0.95),
            },
            "series": series,
        }
    )


@router.get("/api/events")
def api_events(limit: int = 50) -> JSONResponse:
    return JSONResponse({"events": _read_events_tail(limit=limit)})


# ---------- Sources (corpus browser) ----------

@router.get("/api/sources")
def api_sources() -> JSONResponse:
    """Return a per-URL summary of what's in the vector store."""
    from ..store import VectorStore

    store = VectorStore()
    coll = store._collection  # noqa: SLF001 — internal use is fine here
    try:
        data = coll.get(include=["metadatas", "documents"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chroma read failed: {e}")

    by_url: dict[str, dict[str, Any]] = defaultdict(lambda: {"chunks": 0, "chars": 0, "title": "", "url": ""})
    for meta, doc in zip(data.get("metadatas") or [], data.get("documents") or []):
        url = (meta or {}).get("url", "")
        title = (meta or {}).get("title", "")
        rec = by_url[url]
        rec["url"] = url
        rec["title"] = title or rec["title"]
        rec["chunks"] += 1
        rec["chars"] += len(doc or "")

    sources_list = sorted(by_url.values(), key=lambda r: r["chunks"], reverse=True)
    return JSONResponse(
        {
            "total_docs": len(sources_list),
            "total_chunks": sum(r["chunks"] for r in sources_list),
            "total_chars": sum(r["chars"] for r in sources_list),
            "sources": sources_list,
        }
    )


# ---------- Eval reports ----------

def _eval_dir() -> Path:
    p = settings.data_dir / "eval_reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


@router.get("/api/eval/reports")
def api_eval_reports() -> JSONResponse:
    out: list[dict[str, Any]] = []
    for f in sorted(_eval_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            out.append({"name": f.name, "kind": "sweep", "n": len(data), "ts": f.stat().st_mtime})
            continue
        out.append(
            {
                "name": f.name,
                "kind": "report",
                "config_tag": data.get("config_tag"),
                "n_cases": data.get("n_cases"),
                "recall_at_k_mean": data.get("recall_at_k_mean"),
                "abstention_accuracy": (data.get("abstention") or {}).get("accuracy"),
                "faithfulness_mean": data.get("faithfulness_mean"),
                "hallucination_rate": data.get("hallucination_rate"),
                "latency_p95_ms": data.get("latency_p95_ms"),
                "cost_total_usd": data.get("cost_total_usd"),
                "ts": f.stat().st_mtime,
            }
        )
    return JSONResponse({"reports": out})


@router.get("/api/eval/reports/{name}")
def api_eval_report_one(name: str) -> JSONResponse:
    # Defend against path traversal.
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid name")
    p = _eval_dir() / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


# ---------- Feedback inbox ----------

@router.get("/api/feedback")
def api_feedback_list() -> JSONResponse:
    items = [
        {
            "event_id": fb.event_id,
            "ts": fb.ts,
            "rating": fb.rating,
            "note": fb.note,
            "question": fb.question,
            "answer": fb.answer,
            "citations": fb.citations,
        }
        for fb in feedback_mod.load_all()
    ]
    items.reverse()
    return JSONResponse({"feedback": items, "count": len(items)})


# ---------- Settings (runtime tunables) ----------

def _current_settings() -> dict[str, Any]:
    return {
        "embed_model": settings.embed_model,
        "gen_model": settings.gen_model,
        "judge_model": settings.judge_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "rerank_top_n": settings.rerank_top_n,
        "use_reranker": settings.use_reranker,
        "use_cache": settings.use_cache,
        "abstain_min_score": settings.abstain_min_score,
        "abstain_use_judge": settings.abstain_use_judge,
        "cache_similarity": settings.cache_similarity,
        "cache_ttl_seconds": settings.cache_ttl_seconds,
    }


@router.get("/api/settings")
def api_settings_get() -> JSONResponse:
    return JSONResponse(
        {
            "settings": _current_settings(),
            "tunable_keys": list(RUNTIME_TUNABLE.keys()),
            "locked_reason": {
                "chunk_size": "changing this requires re-chunk + re-embed",
                "chunk_overlap": "changing this requires re-chunk + re-embed",
                "embed_model": "changing this requires re-embed",
                "gen_model": "set via .env (no runtime change to avoid mid-conversation drift)",
                "judge_model": "set via .env",
            },
        }
    )


@router.post("/api/settings")
def api_settings_set(patch: SettingsPatch) -> JSONResponse:
    applied: dict[str, Any] = {}
    data = patch.model_dump(exclude_none=True)
    for key, value in data.items():
        if key not in RUNTIME_TUNABLE:
            continue
        setattr(settings, key, value)
        applied[key] = value
    return JSONResponse({"applied": applied, "settings": _current_settings()})


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
