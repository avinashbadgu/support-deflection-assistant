"""Fetch a curated slice of Stripe docs, strip boilerplate, cache on disk.

M1 used a fixed seed list. M5 hardens this: retries via tenacity, a JSONL drop
log (which pages we dropped and why — that log is interview material), and
content hashing so re-ingest only re-embeds the delta.

We still keep a curated seed list rather than a full crawler because (a) docs.stripe.com
is large and a polite crawl would take a long time, (b) for a portfolio project
the corpus needs to be reproducible from the repo.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from .config import settings
from .observability import log

SEED_URLS: list[str] = [
    "https://docs.stripe.com/payments",
    "https://docs.stripe.com/payments/accept-a-payment",
    "https://docs.stripe.com/payments/payment-intents",
    "https://docs.stripe.com/refunds",
    "https://docs.stripe.com/disputes",
    "https://docs.stripe.com/billing/subscriptions/overview",
    "https://docs.stripe.com/billing/invoices/overview",
    "https://docs.stripe.com/connect/overview",
    "https://docs.stripe.com/payouts",
    "https://docs.stripe.com/radar",
    "https://docs.stripe.com/identity",
    "https://docs.stripe.com/webhooks",
    "https://docs.stripe.com/api/charges",
    "https://docs.stripe.com/api/customers",
    "https://docs.stripe.com/testing",
]

USER_AGENT = "support-assistant-portfolio/0.1 (educational; respects robots)"
REQUEST_DELAY_S = 1.0
MIN_BODY_CHARS = 200


@dataclass(frozen=True)
class Document:
    url: str
    title: str
    text: str

    @property
    def doc_id(self) -> str:
        return hashlib.sha1(self.url.encode()).hexdigest()[:12]


def _cache_path(url: str) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()[:12]
    return settings.raw_dir / f"{h}.txt"


def _log_drop(url: str, reason: str, **fields) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with settings.drops_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"url": url, "reason": reason, **fields}) + "\n")


def _clean_html(html: str) -> tuple[str, str, int]:
    """Return (title, body_text, removed_blocks)."""
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""
    removed = 0
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
        removed += 1
    main = soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return title, "\n".join(lines), removed


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPError,)),
    reraise=True,
)
def _fetch(client: httpx.Client, url: str) -> str:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.text


def fetch_documents(urls: list[str] | None = None, use_cache: bool = True) -> list[Document]:
    urls = urls or SEED_URLS
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    docs: list[Document] = []
    seen_text_hashes: set[str] = set()

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True) as client:
        for url in tqdm(urls, desc="fetch"):
            cache = _cache_path(url)
            if use_cache and cache.exists():
                cached = cache.read_text(encoding="utf-8")
                title, _, body = cached.partition("\n\n")
                doc = Document(url=url, title=title, text=body)
            else:
                try:
                    html = _fetch(client, url)
                except httpx.HTTPError as e:
                    log.warning("fetch_failed", url=url, error=str(e))
                    _log_drop(url, "fetch_failed", error=str(e))
                    continue

                title, body, removed = _clean_html(html)
                if len(body) < MIN_BODY_CHARS:
                    log.info("dropped_too_short", url=url, chars=len(body))
                    _log_drop(url, "too_short", chars=len(body), removed_blocks=removed)
                    continue

                cache.write_text(f"{title}\n\n{body}", encoding="utf-8")
                doc = Document(url=url, title=title, text=body)
                time.sleep(REQUEST_DELAY_S)

            # Cross-document dedupe by full-text hash.
            h = hashlib.sha1(doc.text.encode()).hexdigest()
            if h in seen_text_hashes:
                _log_drop(url, "duplicate_text", text_hash=h)
                continue
            seen_text_hashes.add(h)
            docs.append(doc)

    log.info("fetch_complete", n_docs=len(docs))
    return docs
