"""Offline work stops when local miner liveness degrades."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark_fleurs_dev


class HealthResponse(io.BytesIO):
    status = 200


def test_miner_health_requires_fast_unchanged_revision(monkeypatch):
    monkeypatch.setattr(
        benchmark_fleurs_dev.urllib.request,
        "urlopen",
        lambda _url, timeout: HealthResponse(b'{"ok":true,"model_revision":"baseline"}'),
    )
    ticks = iter([0.0, 0.002])
    monkeypatch.setattr(benchmark_fleurs_dev.time, "perf_counter", lambda: next(ticks))
    assert benchmark_fleurs_dev.miner_revision("http://127.0.0.1/healthz", "baseline") == "baseline"


def test_miner_health_rejects_slow_response(monkeypatch):
    monkeypatch.setattr(
        benchmark_fleurs_dev.urllib.request,
        "urlopen",
        lambda _url, timeout: HealthResponse(b'{"ok":true,"model_revision":"baseline"}'),
    )
    ticks = iter([0.0, 1.001])
    monkeypatch.setattr(benchmark_fleurs_dev.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="too slowly"):
        benchmark_fleurs_dev.miner_revision("http://127.0.0.1/healthz", "baseline")
