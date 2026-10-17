from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_c5_polarity_final as handoff
import evaluate_c5_polarity_sentences as sentences
import evaluate_c5_polarity_words as words
import evaluate_c5_word_final as old_handoff
import evaluate_c5_word_sentences as old_sentences
import evaluate_c5_words as old_words
import terra_polarity_judge as judge
import test_c5_word_training as fixtures
from test_c5_mixed_risk_evidence import predecessor  # noqa: F401
from test_evaluate_c5_word_final import completed  # noqa: F401


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_full_word_and_sentence_inference_keep_original_scientific_bytecode():
    for actual, before in (
        (words.bound_baseline(), old_words.baseline),
        (words.bound_inference(), old_words.infer),
        (sentences.bound_inference(), old_sentences.bound_inference()),
    ):
        assert actual.__code__ is before.__code__
        if hasattr(before, "__wrapped__"):
            assert actual.__wrapped__.__code__ is before.__wrapped__.__code__
            actual, before = actual.__wrapped__, before.__wrapped__
        assert {
            key
            for key in actual.__globals__
            if actual.__globals__[key] is not before.__globals__[key]
        } == {"checked"}


@pytest.mark.parametrize("mode", ["inference", "scores"])
def test_sentence_child_reaches_plan_validation_without_double_interop_setup(tmp_path, mode):
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            sentences.__file__,
            "--plan",
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
            "--mode",
            mode,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "FileNotFoundError" in result.stderr
    assert "cannot set number of interop threads" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("kind", ["word", "sentence"])
@pytest.mark.parametrize("decline", [False, True])
def test_complete_new_prompt_passes_preserve_calibrated_decline_and_no_WER_veto(
    predecessor,  # noqa: F811
    monkeypatch,
    kind,
    decline,
):
    monkeypatch.setattr(fixtures, "terra_payload", judge.payload)
    env = predecessor(kind, decline=decline)
    module = words if kind == "word" else sentences
    monkeypatch.setattr(module, "checked", lambda path: env.inference)
    monkeypatch.setattr(module, "checked_scores", lambda path: env.scores)
    result = handoff.previous_evaluation(env.parent, kind, env.initialization, evaluation_only=True)
    decision = result["decision_values"]
    assert decision["calibration_passed"] and decision["WER_veto_used"] is False
    assert decision["stop_for_semantic_decline"] is decline
    assert result["stage_advance_or_promotion_authorized"] is False
    if decline:
        with pytest.raises(ValueError, match="semantic stop condition"):
            handoff.previous_evaluation(env.parent, kind, env.initialization)


def test_failed_calibration_remains_a_stop_even_in_evaluation_only_mode(predecessor, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fixtures, "terra_payload", judge.payload)
    env = predecessor("word", failed_control=True)
    monkeypatch.setattr(words, "checked", lambda path: env.inference)
    monkeypatch.setattr(words, "checked_scores", lambda path: env.scores)
    with pytest.raises(ValueError, match="semantic stop condition"):
        handoff.previous_evaluation(env.parent, "word", env.initialization, evaluation_only=True)


@pytest.fixture
def fresh_audit(completed, tmp_path, monkeypatch):  # noqa: F811
    env = completed
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    monkeypatch.setattr(handoff, "word_initialization", lambda *a: env.binding)
    source = tmp_path / "tools/audit_c5_polarity_word.py"
    source.parent.mkdir()
    source.write_bytes(env.source.read_bytes())
    digest = handoff.sha256(source)
    protocol = judge.protocol()
    put(
        env.plan,
        {
            "terminal_audit_source": old_handoff.artifact(env.source),
            "source_sha256": {"tools/audit_c5_polarity_word.py": digest},
            "judge_protocol": protocol,
        },
    )
    env.binding["word_training_plan"] = old_handoff.artifact(env.plan)
    env.audit.update(
        plan=env.binding["word_training_plan"],
        original_public_initialization=True,
        judge_protocol=protocol,
    )
    put(env.audit_path, env.audit)
    return env


@pytest.mark.parametrize("change", [None, "protocol", "initializer", "source", "partial"])
def test_handoff_requires_complete856_audit_and_its_exact_new_protocol_and_source(
    fresh_audit, change
):
    env = fresh_audit
    if change == "protocol":
        env.audit["judge_protocol"] = {"changed": True}
    elif change == "initializer":
        env.audit["original_public_initialization"] = False
    elif change == "source":
        env.audit["audit_source"]["sha256"] = "wrong"
    elif change == "partial":
        env.audit["training_examples"] = 642
    put(env.audit_path, env.audit)
    if change is None:
        assert handoff.prerequisites(env.plan, env.result, env.audit_path) == env.binding
    else:
        with pytest.raises(ValueError):
            handoff.prerequisites(env.plan, env.result, env.audit_path)


def test_unfinished_final_stops_before_any_child_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(old_handoff, "ROOT", tmp_path)

    def unfinished(*args):
        raise ValueError("word curriculum incomplete")

    monkeypatch.setattr(handoff, "prerequisites", unfinished)
    monkeypatch.setattr(old_handoff.subprocess, "run", lambda *a, **k: pytest.fail("child started"))
    target = tmp_path / "checkpoints/final"
    with pytest.raises(ValueError, match="curriculum incomplete"):
        handoff.pipeline(tmp_path / "plan", tmp_path / "result", tmp_path / "audit", target)
    assert not target.exists()


def test_new_locations_keep_all96_words_then_all34_sentences_separate_from_legacy():
    assert list(handoff.locations()) == ["word", "sentence"]
    assert handoff.locations()["word"][2] == "evaluate_c5_polarity_words.py"
    assert handoff.locations()["sentence"][2] == "evaluate_c5_polarity_sentences.py"
    assert all(
        value[0] != old_handoff.locations()[kind][0] for kind, value in handoff.locations().items()
    )
