from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_word_training as original
import terra_polarity_judge as judge
from probe_jeff_terra_quality import terra_payload
from semantic_quality_judge import calibration_controls
from terra_quality_training import score_with_terra
from test_c5_word_training import FakePacer, Tokenizer, cases, pools_for


def test_append_only_request_keeps_original_schema_inputs_levels_and_controls():
    rows = calibration_controls()
    baseline = terra_payload(rows)
    actual = judge.payload(rows)
    assert actual.pop("instructions") == baseline.pop("instructions") + judge.CLARIFICATION
    assert actual == baseline
    assert terra_payload(rows) != judge.payload(rows)
    assert judge.protocol()["original_controls"] == rows
    assert [row["range"] for row in rows] == [[10, 10], [10, 10], [0, 0], [0, 3]]
    assert all(set(row) == {"id", "reference", "candidate"} for row in json.loads(actual["input"]))


def test_original_client_parser_and_training_loops_have_only_the_declared_private_bindings():
    for actual, before, replacements in (
        (judge.score_with_polarity, score_with_terra, {"terra_payload"}),
        (judge.verified_training_rewards, original.verified_rewards, {"terra_payload"}),
        (judge.score_training_batch, original.score_batch, {"verified_rewards"}),
    ):
        assert actual.__code__ is before.__code__
        assert {
            key
            for key in actual.__globals__
            if actual.__globals__[key] is not before.__globals__[key]
        } == replacements
    assert original.verified_rewards.__globals__["terra_payload"] is terra_payload
    assert score_with_terra.__globals__["terra_payload"] is terra_payload


@pytest.mark.parametrize("negation", [3, 4])
def test_original_control_boundary_is_not_clamped_or_relaxed(tmp_path, monkeypatch, negation):
    records, tokenizer = cases(4), Tokenizer()
    donors, pools = pools_for(records, tokenizer)
    paired = original.paired_inputs(records, donors, pools, tokenizer)
    directory = tmp_path / "batch"
    directory.mkdir()
    original.write(directory / "blinded-inputs.json", paired)
    ordered = original.shuffled_indices(paired, 72)
    pacer = FakePacer()
    monkeypatch.setattr(sys.modules[FakePacer.__module__], "terra_payload", judge.payload)
    for offset in range(0, len(ordered), 16):
        chosen = ordered[offset : offset + 16]
        target = directory / f"request-{offset // 16:03d}"
        pacer.score([paired["items"][i] for i in chosen], target)
        response = json.loads((target / "response.json").read_text())
        content = json.loads(response["output"][0]["content"][0]["text"])
        index = next(
            (j for j, i in enumerate(chosen) if i == paired["controls"][-1]["index"]), None
        )
        if index is not None:
            content["scores"][index]["score"] = negation
            response["output"][0]["content"][0]["text"] = json.dumps(content)
            (target / "response.json").write_text(json.dumps(response))
            saved = json.loads((target / "scores.json").read_text())
            saved["scores"][index] = negation
            saved["response_sha256"] = original.sha256(target / "response.json")
            (target / "scores.json").write_text(json.dumps(saved))
    if negation == 4:
        with pytest.raises(ValueError, match="original calibration"):
            judge.verified_training_rewards(paired, directory, 72)
    else:
        actual = judge.verified_training_rewards(paired, directory, 72)
        assert actual["scores"][paired["controls"][-1]["index"]] == 3
    with pytest.raises(ValueError, match="authentic model, rubric"):
        original.verified_rewards(paired, directory, 72)


def test_pacer_preserves_one_request_spacing_and_never_retries_failure(monkeypatch):
    clock, waits, starts = [0.0], [], []

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    def score(rows, output):
        starts.append(clock[0])
        if len(starts) == 2:
            raise RuntimeError("transport failure")
        return {"scores": [10]}

    monkeypatch.setattr(judge, "score_with_polarity", score)
    pacer = judge.Pacer(clock=lambda: clock[0], sleep=sleep)
    pacer.score([{}], None)
    with pytest.raises(RuntimeError, match="transport failure"):
        pacer.score([{}], None)
    assert starts == [0, 60] and waits == [30, 30]
    with pytest.raises(ValueError, match="1-16"):
        pacer.score([{}] * 17, None)
    assert len(starts) == 2
