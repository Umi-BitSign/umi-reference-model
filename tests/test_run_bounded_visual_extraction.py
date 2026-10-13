"""The extraction supervisor advances bounded batches and stops on errors."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_bounded_visual_extraction as extraction


def test_worker_commands_keep_source_limits_and_health_checks():
    for source, spec in extraction.SOURCES.items():
        command = extraction.worker_command(source, "http://127.0.0.1:8787/healthz", spec["batch"])
        assert command[:3] == ["nice", "-n", "20"]
        assert command[4] == spec["worker"]
        assert command[-2:] == ["--limit", str(spec["batch"])]
        assert "--miner-health-url" in command
        with pytest.raises(ValueError, match="safe batch"):
            extraction.worker_command(source, "http://127.0.0.1:8787/healthz", spec["batch"] + 1)


def test_supervisor_advances_only_complete_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "ROOT", tmp_path)
    monkeypatch.setattr(extraction, "MIN_FREE_GIB", 0)
    monkeypatch.setattr(extraction, "POLL_SECONDS", 1)
    monkeypatch.setattr(extraction, "miner_revision", lambda url, expected: expected)
    monkeypatch.setitem(
        extraction.SOURCES,
        "toy",
        {"index": "runs/toy.jsonl", "target": 3, "batch": 2, "max_rss_gib": 1},
    )
    index = tmp_path / "runs/toy.jsonl"
    assert extraction.index_count(index) == 0

    def command(source, health_url, limit):
        code = (
            "from pathlib import Path; "
            f"path=Path({str(index)!r}); path.parent.mkdir(parents=True,exist_ok=True); "
            f"stream=path.open('a'); stream.write('{{}}\\n'*{limit}); stream.close()"
        )
        return [sys.executable, "-c", code]

    monkeypatch.setattr(extraction, "worker_command", command)
    extraction.run_source("toy", "http://127.0.0.1:8787/healthz", "revision")
    assert extraction.index_count(index) == 3
    assert "exited 0: 3/3" in (
        tmp_path / "runs/visual-extraction-supervisor-v1/toy.log"
    ).read_text()


def test_supervisor_does_not_retry_failed_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "ROOT", tmp_path)
    monkeypatch.setattr(extraction, "MIN_FREE_GIB", 0)
    monkeypatch.setattr(extraction, "POLL_SECONDS", 1)
    monkeypatch.setattr(extraction, "miner_revision", lambda url, expected: expected)
    monkeypatch.setitem(
        extraction.SOURCES,
        "toy_failure",
        {"index": "runs/toy.jsonl", "target": 1, "batch": 1, "max_rss_gib": 1},
    )
    monkeypatch.setattr(
        extraction,
        "worker_command",
        lambda source, health_url, limit: [sys.executable, "-c", "raise RuntimeError('failed')"],
    )
    with pytest.raises(RuntimeError, match="batch failed"):
        extraction.run_source("toy_failure", "http://127.0.0.1:8787/healthz", "revision")
    assert extraction.index_count(tmp_path / "runs/toy.jsonl") == 0


def test_supervisor_refuses_a_second_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "ROOT", tmp_path)
    monkeypatch.setattr(extraction, "MIN_FREE_GIB", 0)
    monkeypatch.setattr(extraction, "miner_revision", lambda url, expected: expected)
    monkeypatch.setattr(extraction, "active_worker_pids", lambda spec: [123])
    monkeypatch.setitem(
        extraction.SOURCES,
        "toy_duplicate",
        {"index": "runs/toy.jsonl", "target": 1, "batch": 1, "max_rss_gib": 1},
    )
    with pytest.raises(RuntimeError, match="already has a writer"):
        extraction.run_source("toy_duplicate", "http://127.0.0.1:8787/healthz", "revision")


def test_supervisor_hands_off_only_after_exact_existing_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "ROOT", tmp_path)
    monkeypatch.setattr(extraction, "MIN_FREE_GIB", 0)
    monkeypatch.setattr(extraction, "POLL_SECONDS", 1)
    monkeypatch.setattr(extraction, "miner_revision", lambda url, expected: expected)
    monkeypatch.setattr(extraction, "rss_kib", lambda pid: 0)
    monkeypatch.setitem(
        extraction.SOURCES,
        "toy_handoff",
        {"index": "runs/toy.jsonl", "target": 3, "batch": 1, "max_rss_gib": 1},
    )
    index = tmp_path / "runs/toy.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text("{}\n")
    live = iter([[123], [], []])
    def active_writer(spec):
        result = next(live, [])
        if not result and extraction.index_count(index) == 1:
            index.write_text("{}\n{}\n")
        return result

    monkeypatch.setattr(extraction, "active_worker_pids", active_writer)
    monkeypatch.setattr(extraction.time, "sleep", lambda seconds: None)

    def command(source, health_url, limit):
        code = f"from pathlib import Path; Path({str(index)!r}).open('a').write('{{}}\\n')"
        return [sys.executable, "-c", code]

    monkeypatch.setattr(extraction, "worker_command", command)
    extraction.run_source(
        "toy_handoff", "http://127.0.0.1:8787/healthz", "revision",
        handoff_pid=123, handoff_count=2,
    )
    assert extraction.index_count(index) == 3
    assert "accepted handoff 123: 2" in (
        tmp_path / "runs/visual-extraction-supervisor-v1/toy_handoff.log"
    ).read_text()


def test_supervisor_rejects_short_existing_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "ROOT", tmp_path)
    monkeypatch.setattr(extraction, "active_worker_pids", lambda spec: [])
    monkeypatch.setitem(
        extraction.SOURCES,
        "toy_short",
        {"index": "runs/toy.jsonl", "target": 3, "batch": 1, "max_rss_gib": 1},
    )
    index = tmp_path / "runs/toy.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text("{}\n")
    with pytest.raises(RuntimeError, match="stopped short"):
        extraction.await_existing_worker(
            "toy_short", 123, 2, "http://127.0.0.1:8787/healthz", "revision"
        )


def test_supervisor_interrupts_existing_writer_on_rss_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "ROOT", tmp_path)
    monkeypatch.setattr(extraction, "MIN_FREE_GIB", 0)
    monkeypatch.setattr(extraction, "miner_revision", lambda url, expected: expected)
    monkeypatch.setattr(extraction, "active_worker_pids", lambda spec: [123])
    monkeypatch.setattr(extraction, "rss_kib", lambda pid: 2 * 1024**2)
    monkeypatch.setitem(
        extraction.SOURCES,
        "toy_rss",
        {"index": "runs/toy.jsonl", "target": 3, "batch": 1, "max_rss_gib": 1},
    )
    signals = []
    monkeypatch.setattr(extraction.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(RuntimeError, match="RSS guard"):
        extraction.await_existing_worker(
            "toy_rss", 123, 2, "http://127.0.0.1:8787/healthz", "revision"
        )
    assert signals == [(123, extraction.signal.SIGINT)]
