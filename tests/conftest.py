"""Shared fixtures. Redirects data_dir to a tmp path so tests never touch real data."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path: Path, monkeypatch):
    from support_assistant.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    yield
