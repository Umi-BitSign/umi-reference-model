from __future__ import annotations

import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_word_evaluation_io as word_io
import evaluate_c5_words as words
import test_c5_word_training as fixtures
from terra_training_recovery import artifact


@pytest.fixture
def word_outputs(tmp_path, monkeypatch):
    """Authenticated synthetic file fixture, not real C5 numerical/quality evidence."""
    descriptor = partial(artifact, root=tmp_path)
    for module in (words, word_io):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "artifact", descriptor)
    monkeypatch.setattr(word_io, "storage", lambda *a, **k: {})
    monkeypatch.setattr(words, "storage", lambda *a, **k: {})
    from transformers import ByT5Tokenizer

    tokenizer = fixtures.Tokenizer()
    monkeypatch.setattr(ByT5Tokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    records = []
    for index in range(96):
        pose = np.full((3, 14), -9999, dtype=np.float32)
        indices = np.asarray([0, 2, 4], dtype=np.int64)
        records.append(
            {
                "record": {
                    "original": {
                        "row": {
                            "example_id": str(index),
                            "reference": f"Word{index % 60}",
                            "label": f"word{index % 60}",
                            "participant": sorted(word_io.RESERVED)[index % 8],
                            "source_group": f"asl_stem:reserved:{index // 4}",
                            "role": "development",
                            "training_allowed": False,
                        }
                    },
                    "ROI_lineage_training_allowed": False,
                    "normalized_probe": {"decoded_frames": 5, "height": 480, "width": 640},
                },
                "receipt": {"path": f"checkpoints/inputs/{index}.json", "sha256": str(index)},
                "verified_input": {
                    "arrays": {
                        "pose_features": {"sha256": hashlib.sha256(pose.tobytes()).hexdigest()},
                        "frame_indices": {"sha256": hashlib.sha256(indices.tobytes()).hexdigest()},
                    }
                },
            }
        )
    plan = {
        "records": records,
        "reference": {"bundle_root": "artifacts/original"},
        "model_plan": {"pretrained_state_sha256": "original"},
        "source_sha256": {},
        "training_result": {"path": "checkpoints/train/result.json", "sha256": "f"},
    }
    path = tmp_path / "runs/word-evaluation/plan.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(words, "checked", lambda path: plan)
    output = tmp_path / "checkpoints/word-evaluation"
    baseline, inference = output / "baseline", output / "inference"
    baseline.mkdir(parents=True)
    inference.mkdir()
    for directory in (baseline, inference):
        words.write(directory / "worker.json", {"pid": 123})
        words.write(directory / "exit.json", {"pid": 123, "exit_code": 0})
    base_predictions, predictions, baseline_files = [], [], []
    for index, case in enumerate(records):
        row = word_io.evaluation_row(case)
        streams = {
            **{name: np.full((3, 384), index, dtype=np.float32) for name in words.STREAMS},
            "pose_features": np.full((3, 14), -9999, dtype=np.float32),
            "frame_indices": np.asarray([0, 2, 4], dtype=np.int64),
        }
        cached = word_io.save_streams(baseline / f"streams-{index:04d}.npz", streams, case)
        text = row["reference"]
        tokens = tokenizer(text, truncation=False).input_ids
        base = words.prediction(row, text)
        filename = baseline / f"baseline-{index:04d}.json"
        words.write(
            filename,
            {
                **base,
                "tokens": tokens,
                "native_crop_parity": True,
                "native_input_side": 224,
                "input_receipt": case["receipt"],
                "original_streams": cached,
            },
        )
        base_predictions.append(base)
        baseline_files.append(descriptor(filename))
        words.write(
            inference / f"control-{index:04d}.json",
            {
                **base,
                "tokens": tokens,
                "matches_frozen_control": True,
                "input_receipt": case["receipt"],
                "original_streams": cached,
            },
        )
        candidate = words.prediction(row, "Other")
        predictions.append(candidate)
        words.write(
            inference / f"candidate-{index:04d}.json",
            {
                **candidate,
                "tokens": tokenizer("Other", truncation=False).input_ids,
                "native_crop_parity": True,
                "input_receipt": case["receipt"],
                "predicted_streams_sha256": "a" * 64,
                "semantic_memory_sha256": "b" * 64,
            },
        )
    words.write(
        baseline / "receipt.json",
        {
            "complete": True,
            "plan_sha256": words.sha256(path),
            "clips": 96,
            "all_original_C5_values_unchanged": True,
            "original_visual_state_sha256": "original",
            "predictions": base_predictions,
            "baseline_files": baseline_files,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "promotion_authorized": False,
            "resources": {
                "GPU_use": False,
                "CPU_threads": 2,
                "process_peak_bytes": 1,
                "memory_policy": "framework_default_no_artificial_six_GiB_cap",
            },
        },
    )
    comparison = words.comparison_items(
        [word_io.evaluation_row(case) for case in records], base_predictions, predictions
    )
    resources = {
        "current_allowed_gpu_duty_fraction": 0.5,
        "current_cpu_threads": 2,
        "mps_active_seconds": 1.0,
        "mps_idle_seconds": 1.0,
        "maximum_allocated_bytes": 1,
        "maximum_driver_allocated_bytes": 1,
        "memory_policy": "framework_default_no_artificial_six_GiB_cap",
    }
    receipt = inference / "receipt.json"
    words.write(
        receipt,
        {
            "complete": True,
            "plan_sha256": words.sha256(path),
            "clips": 96,
            "all96_original_C5_control_tokens_reproduced": True,
            "all_fixed_values_unchanged": True,
            "baseline": descriptor(baseline / "receipt.json"),
            "predictions": predictions,
            "comparison": comparison,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "promotion_authorized": False,
            "resources": resources,
        },
    )
    return SimpleNamespace(
        path=path,
        plan=plan,
        output=output,
        receipt=receipt,
        baseline=baseline,
        inference=inference,
        tokenizer=tokenizer,
    )


def test_word_scoring_rejoins_all96_actual_controls_candidates_and_streams(word_outputs):
    env = word_outputs
    score = words.score_specification(env.path, env.receipt)
    assert (
        len(score["predictions"])
        == len(score["baseline"])
        == len(score["comparison"]["cases"])
        == 96
    )
    assert score["planned_Terra_requests"] == 16
    assert len(score["assets"]) == 385
    assert score["policy"]["WER_is_diagnostic_only"] is True


@pytest.mark.parametrize(
    "filename,change",
    [
        ("baseline/baseline-0000.json", lambda value: value.update(native_input_side=112)),
        ("baseline/baseline-0000.json", lambda value: value["tokens"].append(0)),
        ("baseline/receipt.json", lambda value: value.update(clips=95)),
        ("baseline/receipt.json", lambda value: value["resources"].update(CPU_threads=8)),
        ("inference/control-0000.json", lambda value: value["tokens"].append(0)),
        ("inference/control-0000.json", lambda value: value.update(matches_frozen_control=False)),
        ("inference/candidate-0000.json", lambda value: value.update(input_receipt={})),
        ("inference/candidate-0000.json", lambda value: value.update(native_crop_parity=False)),
        (
            "inference/candidate-0000.json",
            lambda value: value.update(semantic_memory_sha256="wrong"),
        ),
        ("inference/receipt.json", lambda value: value.update(clips=95)),
        ("inference/receipt.json", lambda value: value.update(all_fixed_values_unchanged=False)),
        ("inference/exit.json", lambda value: value.update(exit_code=1)),
    ],
)
def test_word_scoring_rejects_actual_control_candidate_or_terminal_tampering(
    word_outputs, filename, change
):
    env = word_outputs
    fixtures.overwrite(env.output / filename, change)
    with pytest.raises(ValueError):
        words.score_specification(env.path, env.receipt)


def test_word_scoring_rejects_changed_stream_bytes_before_array_load(word_outputs):
    env = word_outputs
    (env.baseline / "streams-0000.npz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        words.score_specification(env.path, env.receipt)


class ScorePacer(fixtures.FakePacer):
    def score(self, rows, directory):
        super().score(rows, directory)
        return json.loads((directory / "scores.json").read_text())


@pytest.fixture
def actual_score_files(word_outputs):
    env = word_outputs
    plan = words.score_specification(env.path, env.receipt)
    output = env.output / "scoring"
    output.mkdir()
    words.score_comparison(plan["comparison"], output / "semantic", ScorePacer())
    return env, plan, output


def test_word_semantics_authenticate_both_shuffled_raw_response_passes(actual_score_files):
    _, plan, output = actual_score_files
    passes = words.authenticated_passes(plan["comparison"], output / "semantic")
    result = words.semantic_decision(
        plan["comparison"], passes, plan["baseline"], plan["predictions"]
    )
    assert result["complete_scoring"] and result["calibration_passed"]
    assert result["C5_mean_quality"] == 10 and result["candidate_mean_quality"] == 3
    assert not result["passes_semantic_research_gate"] and result["WER_veto_used"] is False


@pytest.mark.parametrize(
    "filename,change",
    [
        ("request.json", lambda value: value.update(model="fallback")),
        ("started.json", lambda value: value.update(automatic_retries=1)),
        ("response.json", lambda value: value.update(reasoning={"effort": "low"})),
        ("scores.json", lambda value: value.update(response_sha256="changed")),
        ("scores.json", lambda value: value["scores"].__setitem__(0, 1)),
    ],
)
def test_word_semantics_reject_changed_actual_request_response_score_files(
    actual_score_files, filename, change
):
    _, plan, output = actual_score_files
    fixtures.overwrite(output / "semantic/pass-1-batch-000" / filename, change)
    with pytest.raises(ValueError):
        words.authenticated_passes(plan["comparison"], output / "semantic")


def test_word_final_freeze_rejects_incomplete_training_before_any_output(tmp_path, monkeypatch):
    monkeypatch.setattr(words, "ROOT", tmp_path)
    monkeypatch.setattr(
        words, "checked_final", lambda *a: (_ for _ in ()).throw(ValueError("unfinished"))
    )
    target = tmp_path / "runs/evaluation/plan.json"
    with pytest.raises(ValueError, match="unfinished"):
        words.freeze(target, Path("training"), Path("result"))
    assert not target.parent.exists()


@pytest.mark.parametrize("failure", ["baseline", "inference", "scores", None])
def test_word_pipeline_keeps_original_baseline_inference_and_API_children_serial(
    tmp_path, monkeypatch, failure
):
    import probe_roi_seen_translation

    monkeypatch.setattr(words, "ROOT", tmp_path)
    monkeypatch.setattr(words, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(words, "storage", lambda *a, **k: {})
    path = tmp_path / "runs/evaluation/plan.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    output = tmp_path / "checkpoints/evaluation"
    monkeypatch.setattr(words, "checked", lambda path: {"training_result": {"fixed": True}})
    calls = []

    def child(command, check):
        mode = command[command.index("--mode") + 1]
        calls.append(mode)
        directory = Path(command[command.index("--output") + 1])
        directory.mkdir()
        if mode == failure:
            return SimpleNamespace(returncode=1)
        if mode == "inference":
            assert (
                Path(command[command.index("--baseline") + 1]) == output / "baseline/receipt.json"
            )
        words.write(directory / "worker.json", {"pid": 123})
        words.write(directory / "exit.json", {"pid": 123, "exit_code": 0})
        filename = "decision.json" if mode == "scores" else "receipt.json"
        words.write(
            directory / filename, {"complete": True, "clips": 96, "plan_sha256": words.sha256(path)}
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(words, "subprocess", SimpleNamespace(run=child))

    def specification(plan, decoded):
        assert calls == ["baseline", "inference"]
        assert decoded.exists()
        return {"actual_tokens": True}

    monkeypatch.setattr(words, "score_specification", specification)

    def freeze(plan, specification):
        assert specification == {"actual_tokens": True}
        plan.parent.mkdir()
        plan.write_text("{}")

    monkeypatch.setattr(probe_roi_seen_translation, "freeze", freeze)
    if failure:
        with pytest.raises(RuntimeError):
            words.pipeline(path, output)
        assert not (output / "complete.json").exists()
    else:
        words.pipeline(path, output)
        result = json.loads((output / "complete.json").read_text())
        assert result["optimizer_updates"] == 0 and result["promotion_authorized"] is False
    expected = ["baseline", "inference", "scores"]
    assert calls == (expected[: expected.index(failure) + 1] if failure else expected)


def test_word_pipeline_stops_when_child_zero_exit_lacks_authoritative_receipt(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(words, "ROOT", tmp_path)
    monkeypatch.setattr(words, "checked", lambda path: {})
    monkeypatch.setattr(words, "storage", lambda *a, **k: {})
    path = tmp_path / "runs/evaluation/plan.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    output = tmp_path / "checkpoints/evaluation"
    calls = []
    monkeypatch.setattr(
        words,
        "subprocess",
        SimpleNamespace(run=lambda *a, **k: calls.append(a[0]) or SimpleNamespace(returncode=0)),
    )
    with pytest.raises(FileNotFoundError):
        words.pipeline(path, output)
    assert len(calls) == 1


def test_original_word_vision_uses_native224_two_frame_chunks_without_pose_changes():
    calls = []

    def appearance(pixels):
        calls.append(tuple(pixels.shape))
        return {name: torch.ones((len(pixels), 384), dtype=torch.float32) for name in words.STREAMS}

    pose = np.full((3, 14), -9999, dtype=np.float32)
    values = {
        "pixels": np.zeros((3, 3, 224, 224, 3), dtype=np.uint8),
        "pose_features": pose,
        "frame_indices": np.asarray([0, 2, 4], dtype=np.int64),
    }
    streams = words.original_streams(appearance, values)
    assert calls == [(2, 3, 3, 224, 224), (1, 3, 3, 224, 224)]
    assert streams["face_features"].shape == (3, 384)
    np.testing.assert_array_equal(streams["pose_features"], pose)
    np.testing.assert_array_equal(streams["frame_indices"], [0, 2, 4])


@pytest.mark.parametrize("fail_control", [None, 17])
def test_word_inference_reads_no_candidate_pixels_until_all96_controls_reproduce(
    word_outputs, monkeypatch, fail_control
):
    import c5_certification_budget

    env = word_outputs
    output = env.output / "candidate-inference"
    output.mkdir()
    env.plan["initial_reference_hashes"] = {"encoder": "e", "language": "l"}
    fixed = {"value": torch.ones(1)}
    env.plan["candidate_visual_state_sha256"] = words.state_digest(fixed)
    monkeypatch.setattr(c5_certification_budget, "CertificationComputeWindow", fixtures.FakeBudget)
    monkeypatch.setattr(c5_certification_budget, "WinnerDutyCycle", fixtures.FakeDuty)
    env.plan["resource_policy"] = "runs/resource.json"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(words, "models", lambda plan: (object(), object(), env.tokenizer))
    monkeypatch.setattr(words, "backbone_hashes", lambda *a: env.plan["initial_reference_hashes"])
    monkeypatch.setattr(
        words,
        "memory_for",
        lambda encoder, streams: torch.tensor(float(streams["face_features"][0, 0])),
    )
    monkeypatch.setattr(
        words,
        "legacy_models",
        lambda plan: (SimpleNamespace(model=object()), SimpleNamespace(model=object())),
    )
    events, crops = [], []

    class Candidate:
        def requires_grad_(self, value):
            assert value is False
            return self

        def eval(self):
            return self

        def to(self, device):
            assert device == "mps"
            return self

        def state_dict(self):
            return fixed

    def appearance(*args):
        assert len(list(output.glob("control-*.json"))) == 96
        return Candidate()

    monkeypatch.setattr(words, "candidate_appearance", appearance)

    def decode(language, tokenizer, generation, memory):
        index = int(memory)
        events.append("control" if index < 1000 else "candidate")
        text = f"Word{index % 60}" if index < 1000 else "Other"
        if index == fail_control:
            text = "Mismatch"
        return text, tokenizer(text, truncation=False).input_ids

    monkeypatch.setattr(words, "decode", decode)
    env.plan["reference"]["generation"] = {}

    def load(case, purpose):
        assert purpose == "development"
        assert len(list(output.glob("control-*.json"))) == 96
        index = int(case["record"]["original"]["row"]["example_id"])
        crops.append(index)
        pixels = np.zeros((3, 3, 112, 112, 3), dtype=np.uint8)
        pixels[0, 0, 0, 0, 0] = index
        return {
            "pixels": pixels,
            "pose_features": np.full((3, 14), -9999, dtype=np.float32),
            "frame_indices": np.asarray([0, 2, 4], dtype=np.int64),
        }

    monkeypatch.setattr(words, "load_verified_input", load)

    def streams(appearance, pixels, original, duty):
        index = int(pixels[0, 0, 0, 0, 0])
        return {
            **{
                name: torch.full((3, 384), 1000 + index, dtype=torch.float32)
                for name in words.STREAMS
            },
            "pose_features": torch.from_numpy(original["pose_features"].copy()),
        }

    monkeypatch.setattr(words, "predicted_streams", streams)
    baseline = env.baseline / "receipt.json"
    if fail_control is not None:
        with pytest.raises(ValueError, match="reproduce"):
            words.infer(env.path, output, env.plan, baseline)
        assert not crops and set(events) == {"control"}
        assert not (output / "receipt.json").exists()
    else:
        words.infer(env.path, output, env.plan, baseline)
        assert events == ["control"] * 96 + ["candidate"] * 96
        assert crops == list(range(96))
        assert len(list(output.glob("candidate-*.json"))) == 96


def test_word_score_decision_keeps_word_scope_and_calibration_gate(word_outputs, monkeypatch):
    env = word_outputs
    plan = words.score_specification(env.path, env.receipt)
    path = env.path.with_name("scores.json")
    path.write_text(json.dumps(plan))
    output = env.output / "scoring"
    output.mkdir()
    monkeypatch.setattr(words, "RecoveryPacer", ScorePacer)
    monkeypatch.setattr(words, "checked_scores", lambda path: plan)
    words.score(path, output, plan)
    result = json.loads((output / "decision.json").read_text())
    assert result["clips"] == 96 and result["calibration_passed"]
    assert result["word_recognition_curriculum_only"] is True
    assert result["continuous_translation_win_claimed"] is False
    assert result["promotion_authorized"] is False and result["WER_veto_used"] is False
