from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_terra_word_native
import probe_c5_polarity_semantic as proof
import probe_c5_word_semantic as original
import terra_polarity_judge as judge
from test_probe_c5_word_semantic import candidate_fixture  # noqa: F401


@pytest.fixture
def clarified_candidate(candidate_fixture):  # noqa: F811
    plan_path, path, plan, tokenizer, result, response, directory = candidate_fixture
    paired = result["comparison"]
    indices = original.shuffled_indices(paired)
    rows = [paired["items"][i] for i in indices]
    (directory / "request.json").write_text(json.dumps(judge.payload(rows)))
    scores = json.loads(response["output"][0]["content"][0]["text"])["scores"]
    (directory / "started.json").write_text(
        json.dumps(
            {
                "request_sha256": original.sha256(directory / "request.json"),
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "pairs": len(rows),
                "automatic_retries": 0,
            }
        )
    )
    (directory / "scores.json").write_text(
        json.dumps(
            {
                "response_sha256": original.sha256(directory / "response.json"),
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "scores": [row["score"] for row in scores],
            }
        )
    )
    return plan_path, path, plan, tokenizer, result, directory


def test_qualification_reuses_the_entire_original_numerical_loop_with_two_private_bindings():
    bound = proof.bound_qualification()
    assert bound.__code__ is original.qualify.__code__
    assert {
        key
        for key in bound.__globals__
        if bound.__globals__[key] is not original.qualify.__globals__[key]
    } == {"checked_candidates", "checked"}
    for name in (
        "load_student",
        "load_clip",
        "require_model",
        "compare",
        "visual_gradients",
        "verified_comparison",
        "POLICY",
        "LIMITS",
    ):
        assert bound.__globals__[name] is original.qualify.__globals__[name]
    assert (
        original.POLICY["initialization"]
        == "original public C5 only; no locally trained initializer"
    )


def test_new_candidate_checker_accepts_only_the_new_protocol_and_original_rejects_it(
    clarified_candidate,
):
    plan_path, path, plan, tokenizer, result, directory = clarified_candidate
    assert proof.checked_candidates(path, plan_path, plan, tokenizer) == result
    with pytest.raises(ValueError, match="prompt or response bindings"):
        original.checked_candidates(path, plan_path, plan, tokenizer)
    rows = [
        result["comparison"]["items"][i] for i in original.shuffled_indices(result["comparison"])
    ]
    (directory / "request.json").write_text(json.dumps(judge.original_payload(rows)))
    with pytest.raises(ValueError, match="prompt or response bindings"):
        proof.checked_candidates(path, plan_path, plan, tokenizer)


def test_candidate_checker_rejects_retry_metadata_even_with_identical_scores(clarified_candidate):
    plan_path, path, plan, tokenizer, _, directory = clarified_candidate
    started = json.loads((directory / "started.json").read_text())
    started["automatic_retries"] = 1
    (directory / "started.json").write_text(json.dumps(started))
    with pytest.raises(ValueError, match="request/response bindings"):
        proof.checked_candidates(path, plan_path, plan, tokenizer)


def test_pipeline_uses_its_own_entrypoint_and_serial_original_modes(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(original, "ROOT", tmp_path)
    monkeypatch.setattr(proof, "checked", lambda *a: {})
    monkeypatch.setattr(proof, "checked_qualification", lambda *a: {})
    monkeypatch.setattr(prepare_terra_word_native, "guard", lambda *a: None)
    monkeypatch.setattr(original, "artifact", lambda path: {"path": str(path)})

    def child(command, check):
        assert check is False
        commands.append(command)
        directory = Path(command[command.index("--output") + 1])
        directory.mkdir()
        (directory / "qualification.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(original.subprocess, "run", child)
    monkeypatch.setattr(original, "identity", lambda *a: {"pid": 71})
    proof.pipeline(tmp_path / "plan.json", tmp_path / "checkpoints/proof")
    assert [command[command.index("--mode") + 1] for command in commands] == [
        "candidates",
        "qualify",
    ]
    assert all(command[1] == str(Path(proof.__file__).resolve()) for command in commands)
    assert "--candidates" not in commands[0] and "--candidates" in commands[1]
