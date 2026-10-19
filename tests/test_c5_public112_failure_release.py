"""Synthetic stopped-evaluation admission; no model or external request."""

import json
import os
import random
from types import SimpleNamespace

import pytest

from tests.test_c5_public112_word_diagnostic import diagnostic, put
from tests.test_c5_public112_word_diagnostic import source as source


@pytest.fixture
def http502(source, monkeypatch):  # noqa: F811 (shared pytest fixture)
    root = source.root
    decline = root / "checkpoints/c5-http502-word-evaluation-v1/scoring/decision.json"
    monkeypatch.setattr(diagnostic, "DECLINE", decline)
    put(decline, {"calibration_passed": True, "stop_for_semantic_decline": True})
    word = decline.parent.parent
    put(word / "complete.json", {"complete": True, "decision": diagnostic.artifact(decline)})
    put(word / "worker.json", {"pid": 41101, "uid": os.getuid()})
    put(word / "exit.json", {"pid": 41101, "exit_code": 0})
    sentence = root / "checkpoints/c5-http502-sentence-evaluation-v1"
    scoring = sentence / "scoring"
    semantic = scoring / "semantic"
    for i, (directory, code) in enumerate(
        (
            (source.paths["FINAL"].parent, 1),
            (sentence, 1),
            (scoring, 1),
            (sentence / "inference", 0),
        )
    ):
        put(directory / "worker.json", {"pid": 41200 + i, "uid": os.getuid()})
        put(
            directory / "exit.json",
            {"pid": 41200 + i, "exit_code": code, "observed_at": "2000-01-01T00:00:00+00:00"},
        )
    put(
        source.paths["FINAL"].with_name("failure.json"),
        {
            "automatic_retries": 0,
            "error": "final sentence evaluation stopped; no retry or next pipeline",
        },
    )
    inference = sentence / "inference/receipt.json"
    put(
        inference,
        {
            "complete": True,
            "clips": 34,
            "all34_frozen_C5_controls_reproduced": True,
            "all_fixed_weights_unchanged": True,
        },
    )
    items = [
        {"reference": f"Synthetic reference {i}", "candidate": f"Synthetic output {i}"}
        for i in range(60)
    ]
    comparison = {"items": items}
    put(semantic / "blinded-inputs.json", comparison)
    score_path = root / "runs/c5-http502-sentence-evaluation-v1-scores/plan.json"
    put(score_path, {"decoded": diagnostic.artifact(inference), "comparison": comparison})
    orders = []
    for repeat in range(2):
        order = list(range(60))
        random.Random(diagnostic.original.SCORE_RECIPE["seed"] + repeat).shuffle(order)
        orders.append([order[i : i + 16] for i in range(0, 60, 16)])
    for repeat, count in ((0, 4), (1, 2)):
        joined = [None] * 60
        for batch in range(count):
            path = semantic / f"pass-{repeat + 1}-batch-{batch:03d}"
            chosen = orders[repeat][batch]
            put(path / "request.json", diagnostic.payload([items[i] for i in chosen]))
            put(
                path / "started.json",
                {
                    "automatic_retries": 0,
                    "pairs": len(chosen),
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "medium",
                    "request_sha256": diagnostic.sha256(path / "request.json"),
                },
            )
            if repeat == 1 and batch == 1:
                put(
                    path / "failure.json",
                    {
                        "http_status": 502,
                        "ambiguous_completion": False,
                        "automatic_retry_allowed": False,
                    },
                )
            else:
                response = {
                    "model": "gpt-5.6-terra",
                    "reasoning": {"effort": "medium"},
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {
                                            "scores": [
                                                {"id": f"pair_{i:02d}", "score": 4}
                                                for i in range(len(chosen))
                                            ]
                                        }
                                    ),
                                }
                            ],
                        }
                    ],
                }
                put(path / "response.json", response)
                put(
                    path / "scores.json",
                    {
                        "model": "gpt-5.6-terra",
                        "reasoning_effort": "medium",
                        "scores": [4] * len(chosen),
                        "response_sha256": diagnostic.sha256(path / "response.json"),
                    },
                )
                for i in chosen:
                    joined[i] = 4
        if repeat == 0:
            put(semantic / "pass-1-scores.json", joined)
    monkeypatch.setattr(diagnostic, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    return SimpleNamespace(
        source=source,
        sentence=sentence,
        scoring=scoring,
        semantic=semantic,
        failed=semantic / "pass-2-batch-001",
        inference=inference,
    )


def test_explicit_terminal_http502_releases_only_new_diagnostic_without_completing_scores(http502):
    release = diagnostic.checked_release()
    assert release["kind"] == "terminal_HTTP502_after_complete_sentence_inference"
    assert release["sentence_quality_available"] is False
    assert release["failed_request_replayed"] is False
    assert release["partial_sentence_grades_interpreted"] is False
    assert "complete" not in release


@pytest.mark.parametrize(
    "change",
    (
        "ambiguous",
        "wrong_status",
        "live_worker",
        "response_after_failure",
        "incomplete_inference",
        "changed_prefix",
        "extra_batch",
    ),
)
def test_failure_release_rejects_ambiguous_live_changed_or_replayed_predecessor(
    http502, monkeypatch, change
):
    h = http502
    if change in ("ambiguous", "wrong_status"):
        p = h.failed / "failure.json"
        value = json.loads(p.read_text())
        key = "ambiguous_completion" if change == "ambiguous" else "http_status"
        value[key] = True if change == "ambiguous" else 429
        put(p, value)
    elif change == "live_worker":
        monkeypatch.setattr(
            diagnostic.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="41200"),
        )
    elif change == "response_after_failure":
        put(h.failed / "response.json", {"unexpected": True})
    elif change == "incomplete_inference":
        value = json.loads(h.inference.read_text())
        value["complete"] = False
        put(h.inference, value)
    elif change == "extra_batch":
        (h.semantic / "pass-2-batch-002").mkdir()
    else:
        put(h.semantic / "pass-1-batch-000/request.json", {"changed": True})
    with pytest.raises(ValueError):
        diagnostic.checked_release()


def test_failure_release_uses_terminal_time_for_conservative_api_spacing(http502, monkeypatch):
    seen = []
    monkeypatch.setattr(diagnostic, "wait_for_predecessor", lambda p: seen.append(p) or "ready")
    spacing = diagnostic.wait_after_predecessor()
    assert seen == [{"API_start_not_before": "2000-01-01T00:01:00+00:00"}]
    assert spacing["ready_at"] == "ready"
