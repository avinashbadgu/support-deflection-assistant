# Decisions log

One line per non-trivial choice: **what / alternatives rejected / why / metric (if any)**.
This is the interview-defense doc. Numbers marked `<run eval>` are deliberately
unfilled — they must come from a real run.

---

## Cross-cutting

### Stack: plain Python + provider SDKs, NO LangChain / LlamaIndex
- **Rejected:** LangChain, LlamaIndex, Haystack.
- **Why:** their abstractions hide the exact decisions an interviewer probes
  ("what does your retriever do? why that chunker?"). Plain SDK calls keep the
  code legible and the credit ours.

### Dependencies (each justified)
- `openai`, `chromadb` — provider + store.
- `httpx` — modern sync HTTP, retry-friendly.
- `beautifulsoup4` + `lxml` — HTML cleaning.
- `pydantic` + `pydantic-settings` — typed env config.
- `tenacity` — retries on flaky network/API calls.
- `structlog` — JSON logs for the dashboard pipeline.
- `typer` — CLI; less boilerplate than argparse.
- `tqdm` — progress for ingest/eval.
- `fastapi` + `uvicorn` + `jinja2` — API + server-rendered UI (no JS build step).
- `prefect` — orchestration in M5 (named by spec).

---

## M1 — thin end-to-end slice

### Corpus: Stripe public docs, curated 15-page seed list
- **Rejected:** PostgreSQL docs (hard to source realistic Q/A pairs); HF Transformers (good Q/A but harder business framing); a full live crawl (overkill, irreproducible).
- **Why:** Stripe has a clean "who would pay for this" story and clean structured HTML. The 15 URLs cover diverse topics (payments / billing / connect / radar / webhooks / API ref / testing) so the eval set can discriminate.
- **Metric:** n/a (corpus-choice decision).

### Vector store: Chroma local persistent client
- **Rejected:** pgvector (best DE story but extra infra for a portfolio project); Qdrant (slightly more setup).
- **Why:** zero infra, behind a thin wrapper in `store.py`. Migration to pgvector in M5 would be one file change.
- **Metric:** n/a.

### Providers: OpenAI text-embedding-3-small + gpt-4o-mini
- **Rejected:** Claude for generation + OpenAI for embeddings (two keys, more moving parts); fully local sentence-transformers + Ollama (slower, weaker, harder to defend "why local" in interview).
- **Why:** cheapest credible setup; both behind Protocols for swap.
- **Metric:** n/a.

---

## M2 — eval set + metric harness

### Metric set: recall@k, abstention accuracy + precision/recall, faithfulness (LLM judge), hallucination rate, latency p50/p95, cost-per-query
- **Rejected:** BLEU/ROUGE (don't measure groundedness, the actual failure mode); end-to-end accuracy as a single number (loses signal).
- **Why:** these five dimensions map 1:1 to the spec's "what interviewers score." Faithfulness is the one that catches hallucinations directly.
- **Metric:** `<run support-assistant eval --config baseline>` to fill in:

  | metric | baseline |
  |---|---|
  | recall@k (k=top_k → rerank → top_n) | `<run>` |
  | abstention accuracy | `<run>` |
  | abstention precision / recall | `<run>` / `<run>` |
  | faithfulness mean | `<run>` |
  | hallucination rate | `<run>` |
  | latency p50 / p95 (ms) | `<run>` / `<run>` |
  | cost mean / total (USD) | `<run>` / `<run>` |

### Eval set: 20 hand-written cases (15 answerable + 5 must-abstain)
- **Rejected:** auto-generating Q/A from the corpus with an LLM (circular: same model writes and grades).
- **Why:** spec wants real questions a customer would ask; abstention cases must be deliberately out-of-corpus to test the signature feature.
- **Weakness:** 20 is below the 50–150 spec target. Stated openly in README. The closed feedback loop (M6) is the path to growing it.

### Metric math is pure and unit-tested
- **Why:** CLAUDE.md: "tests for the evaluation harness itself — the metrics must be trustworthy."
- **Where:** [tests/test_metrics.py](tests/test_metrics.py).

---

## M3 — abstention

### Two-gate design: cosine score floor + LLM judge
- **Rejected:** floor only (cheap but lexically-similar off-topic chunks slip through with high scores); judge only (expensive, every query pays the judge cost).
- **Why:** the floor short-circuits the cheap obvious cases (truly off-topic, no retrieval hit); the judge catches the hard "lexically similar but not actually answering" failure. Tests verify the floor short-circuits the judge so we don't pay twice ([tests/test_abstain.py::test_floor_short_circuits_judge](tests/test_abstain.py)).
- **Metric:** abstention accuracy. **Threshold is NOT picked by default** — run `eval --sweep-abstain`:

  | threshold | accuracy | precision (abstain) | recall (abstain) | fp | fn |
  |---|---|---|---|---|---|
  | 0.10 | `<run>` | `<run>` | `<run>` | `<run>` | `<run>` |
  | 0.20 | `<run>` | `<run>` | `<run>` | `<run>` | `<run>` |
  | 0.30 | `<run>` | `<run>` | `<run>` | `<run>` | `<run>` |
  | 0.35 (current .env default) | `<run>` | `<run>` | `<run>` | `<run>` | `<run>` |
  | 0.40 | `<run>` | `<run>` | `<run>` | `<run>` | `<run>` |
  | 0.50 | `<run>` | `<run>` | `<run>` | `<run>` | `<run>` |

  **Pick a threshold from this curve, not from .env.**

### Abstentions are NOT cached
- **Why:** retrying an abstention every time gives the user a chance to phrase the question differently. Caching them would entrench wrong calls.

---

## M4 — one deliberate improvement: LLM reranker

### LLM reranker over cross-encoder (e.g. ms-marco-MiniLM)
- **Rejected:** sentence-transformers cross-encoder.
- **Why:** keeps the dependency surface tiny (no torch / transformers); we're already paying for an LLM, the marginal cost is bounded (~1 small call per query). At scale a cross-encoder is the right move — that's a great interview answer in itself.
- **Metric:** before/after, **rerun both with reranker off vs on**:

  | metric | rerank OFF (baseline) | rerank ON |
  |---|---|---|
  | recall@k | `<run>` | `<run>` |
  | faithfulness mean | `<run>` | `<run>` |
  | hallucination rate | `<run>` | `<run>` |
  | latency p95 (ms) | `<run>` | `<run>` |
  | cost mean (USD) | `<run>` | `<run>` |

  How to run:
  ```powershell
  $env:SA_USE_RERANKER="false"; support-assistant eval --config baseline-no-rerank
  $env:SA_USE_RERANKER="true";  support-assistant eval --config with-rerank
  ```

### Alternative improvement candidates we did NOT pick (and could in a follow-up)
- Switching chunk strategy from 1000/200 fixed to sentence/paragraph aware.
- Hybrid retrieval (BM25 + vector) with reciprocal rank fusion.
- Larger embedding model (`text-embedding-3-large`).
- Different abstention floor.

---

## M5 — pipeline hardening

### Incremental updates via content hashing
- **Rejected:** wholesale re-embed every run; mtime-based (unreliable across re-fetches).
- **Why:** chunk text content is what actually matters. SHA-256 of chunk text → JSON map of `chunk_id -> hash`. Re-running on unchanged corpus performs **zero** embedding API calls. Tested in [tests/test_delta.py](tests/test_delta.py).
- **Metric:** n/a directly; observable as `embedded=0 unchanged=N` in the second `ingest` run.

### Drop log (`data/drops.jsonl`)
- **Why:** spec calls out "log what you dropped and why — this log is interview gold." Records: fetch errors, too-short bodies, duplicate text.

### Retries via tenacity (3 attempts, exponential backoff)
- **Rejected:** ad-hoc try/except (less consistent); no retries (flaky tests/runs).
- **Why:** wraps both `httpx.get` and OpenAI calls with the same policy.

### Orchestration: Prefect 3 flow
- **Rejected:** Dagster (heavier setup); APScheduler (no UI / observability).
- **Why:** spec names Prefect/Dagster; Prefect 3 is the lower-friction modern choice. Flow has retries and uses `build_index(use_delta=True)` so scheduled runs cost nothing on unchanged input.

---

## M6 — production layer

### API: FastAPI
- **Why:** ergonomic typed schemas via Pydantic, async-ready, the standard. Routes are thin — all work delegates to `pipeline.ask`.

### Frontend: server-rendered Jinja + vanilla JS, Chart.js via CDN
- **Rejected:** React/Next.js (build step, separate process — overkill for a portfolio); Streamlit (looks like Streamlit).
- **Why:** zero build, single Python process, easy to read aloud in interview. The chat UI uses fetch + a `<template>`; the dashboard uses Chart.js.

### The one deliberate optimization: in-process semantic cache
- **Rejected:** Redis-backed cache (more infra); exact-string cache (zero hit rate on natural-language queries).
- **Why:** semantic dedupe on question embeddings catches paraphrases. Tunable similarity threshold (`SA_CACHE_SIMILARITY`) and TTL. Spec wants "one optimization with no quality loss" — measure with cache off vs on:

  | metric | cache OFF | cache ON |
  |---|---|---|
  | latency p95 | `<run>` | `<run>` |
  | cost / query (mean) | `<run>` | `<run>` |
  | faithfulness mean | `<run>` | `<run>` |

  Quality loss check: confirm faithfulness mean does not drop.

### Closed feedback loop
- **Why:** thumbs-down on chat UI → `data/feedback.jsonl` → `eval.dataset.load_eval_set(include_feedback=True)` picks them up. Production failures become regression cases — the seniority signal in the spec.

### Observability: per-query event row in `data/events.jsonl`
- **Why:** dashboard reads from this. Per-stage latency (embed / retrieve / rerank / abstain / generate), abstention flag, cache hit, cost, citations. Read aloud in interview.

### Known weakness: cost-per-event is approximated
- **Why:** `cost.jsonl` has no per-call event-id linkage; we take the trailing rows since `t0`. Fine for one user; concurrent calls would interleave. Real fix: add `event_id` to `cost.record`. Recorded openly in README weaknesses.

---

## M7 — Expanded categorized eval set (114 cases)

### What
Replaced the 20-case seed set with a 114-case set, each labelled with a category:
`direct_retrieval` · `exact_api` · `multi_hop` · `ambiguous` · `adversarial` · `stale_conflicting` · `out_of_domain`.

### Why these categories
- **direct_retrieval** (30) — the broad baseline.
- **exact_api** (22) — questions where retrieval has to find the *specific* endpoint / parameter / test card / error code. BM25 typically wins here vs pure dense.
- **multi_hop** (15) — answer requires fusing 2+ source chunks. Tests whether top_k is large enough.
- **ambiguous** (12) — vague phrasing with multiple plausible interpretations. Tests prompt grounding.
- **adversarial** (12) — lexically similar to corpus but actually off-topic (Atlas, Climate, Issuing, Terminal — all "Stripe X" terms not in our 15-page slice). The hardest abstention cases.
- **stale_conflicting** (5) — pricing / API-version / SLA questions. Tests refusal to invent numbers that change.
- **out_of_domain** (18) — completely off-topic. Tests the floor.

### Rejected
- LLM-generated synthetic Q/A from the corpus — circular (same family of models writing and grading).
- A single flat "test set" — loses signal; can't see *where* the system fails.

### Metric
Reports now break out per-category recall@k, abstention accuracy, faithfulness, p95 — see `by_category` in every JSON report and the dashboard table on `/eval`.

### Test
`tests/test_eval_categories.py` asserts: ≥100 cases, every case has a known category, OOD all expect abstain, direct cases all have expected URLs, **no expected URL references a page outside `SEED_URLS`** (catches typos and unrunnable cases).

### Honest weakness still standing
The 114 cases I hand-wrote are mine. Real grade-A eval sets pull from Stack Overflow `[stripe-payments]` and the Stripe community forum — that's the next move and a single page of mined production questions will move every number more than any model swap.

---

## M8 — Hybrid retrieval: BM25 + dense, fused with RRF

### What
Sparse retrieval via `rank_bm25` (pure Python, ~few-hundred-chunks corpus) built at ingest time alongside Chroma. At query time, dense and sparse result lists are fused via **Reciprocal Rank Fusion** (RRF). Tunable via `SA_USE_HYBRID`, `SA_BM25_TOP_K`, `SA_RRF_K`.

### Why RRF, not score-based fusion
- Dense (cosine) and BM25 scores are on different scales and distributions. Any linear-combination weighting is a hyperparameter you'd have to tune *per query type*.
- RRF uses ranks, not raw scores: `score(d) = sum_over_lists(1 / (k + rank))`. Robust to score-scale differences. The original paper picked `k=60` which we keep as default.

### Why BM25 at all (this is the M8 story)
Stripe docs are full of *exact tokens*: `4242 4242 4242 4242`, `payment_intents`, `requires_action`, `Stripe-Signature`. Dense embeddings often blur these into nearby semantic neighbours. BM25 hits them on the nose. The expected story (which the eval will confirm or deny) is that hybrid lifts `exact_api`-category recall noticeably and leaves `direct_retrieval` roughly even, at the cost of more retrieve-stage latency.

### Rejected
- BM25-only — loses semantic matching ("issue a partial refund" → chunks about "refunding"); the eval would catch this immediately.
- ColBERT / late-interaction models — heavy deps; not justified at this scale.
- Lexical pre-filter then dense rerank — couples the stages too tightly; hard to A/B independently.

### Metric — to be filled by `support-assistant compare`
| Preset | Recall@k | Abstain Acc | p95 ms | $/q |
|---|---:|---:|---:|---:|
| baseline-dense | `<run>` | `<run>` | `<run>` | `<run>` |
| hybrid | `<run>` | `<run>` | `<run>` | `<run>` |
| hybrid+llm-rerank | `<run>` | `<run>` | `<run>` | `<run>` |

### Test
`tests/test_bm25.py` and `tests/test_hybrid.py` — tokenizer behavior, persistence/reload, RRF math (rank-summed reciprocal, dedup, metadata preservation, empty inputs).

---

## M9 — Cross-encoder reranker as an alternative to the LLM reranker

### What
Added `CrossEncoderReranker` using `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2`. Selectable via `SA_RERANKER_KIND` ∈ {`none`, `llm`, `crossencoder`}. Cross-encoder dep is **optional** — installed via `pip install -e ".[crossencoder]"`.

### Why both, kept side-by-side
- **LLM rerank** — high ceiling on hard cases, free deps, slow (1 LLM call per query), $$.
- **Cross-encoder** — heavy deps (~1GB torch+transformers stack), but local + free per call + ~10ms inference. The production-shape choice at any scale.

You only know which is *right for your corpus* by measuring. M9's deliverable is the benchmark row showing the latency/quality tradeoff — written into the comparison table when `support-assistant compare` runs.

### Rejected
- Reranker-as-default (no toggle) — locks in the wrong answer if the eval disagrees.
- Hard-coded cross-encoder dep in the core install — too much weight for users who don't need it.

### Metric — to be filled by `support-assistant compare`
Three rows on the comparison table (no rerank / LLM rerank / cross-encoder rerank); pick the winner on the recall × latency × cost frontier.

### Test
`tests/test_compare.py` — preset application and restoration, markdown table generation. No live cross-encoder load (would require the optional dep).

---

## M10 — Compare runner + failure-analysis tool

### `support-assistant compare`
Runs N named config presets, produces both a JSON record and a markdown table under `data/eval_reports/`. Presets (see `eval/runner.py::PRESETS`):
- `baseline-dense`
- `dense+llm-rerank`
- `dense+crossencoder`
- `hybrid`
- `hybrid+llm-rerank`
- `hybrid+crossencoder`

### `support-assistant analyze <report>`
Reads a report JSON, writes a sibling `_failures.md` with:
- Per-category failure count table.
- Missed abstentions (the dangerous failure — answered when it shouldn't have).
- False abstentions (lost-coverage failure — abstained when it shouldn't have).
- Low-recall cases (none of the expected URLs were retrieved).
- Low-faithfulness cases with the unsupported claims listed.

### Why this matters
A benchmark number without a failure analysis is a half-finished story. "Recall@k went from 0.71 to 0.84" is fine; "and here are the 3 exact-api cases that still fail because BM25 is matching on a misleading identifier" is interview-grade. The `_failures.md` companion file is what turns a recruiter scroll into an actual conversation.
