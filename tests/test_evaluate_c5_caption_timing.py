"""Full130 synthetic routing plus actual raw-score parsing; no real C5/API work."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_timing_evaluation_protocol as protocol
import c5_timing_feature_data as data
import evaluate_c5_caption_timing as evaluation
import test_c5_caption_timing_candidate as candidate_tests
from test_probe_c5_timing_semantic import Tokenizer

adapter = candidate_tests.adapter


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def inference(adapter, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import c5_caption_timing_candidate as candidate
    import c5_caption_timing_final as final
    import c5_small_input_vision as visual
    import c5_timing_semantic_model as semantic
    import evaluate_c5_words as words
    import evaluate_roi_short_translation as original
    import probe_c5_small_input as legacy
    from c5_timing_native_input import input_receipt
    from probe_tiny_rgb_capacity import state_digest

    model, binding, _, _ = adapter
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "guard", lambda *args: None)
    model_plan = {
        "reference": {
            "bundle_root": "synthetic-model",
            "generation": {
                "max_length": 2048,
                "num_beams": 5,
                "length_penalty": 0.6,
                "no_repeat_ngram_size": 16,
            },
        }
    }
    appearance = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        appearance.weight.fill_(0.7)
    model_plan["pretrained_state_sha256"] = state_digest(appearance.state_dict())
    binding["model_plan"] = model_plan
    audit = tmp_path / "runs/audit.json"
    audit.parent.mkdir()
    save(audit, {"synthetic": True})
    monkeypatch.setattr(final, "checked_final", lambda path: copy.deepcopy(binding))
    monkeypatch.setattr(semantic, "load_student", lambda plan: (model, Tokenizer()))
    monkeypatch.setattr(semantic, "require_model", lambda model, plan: model.require_fixed())
    monkeypatch.setattr(visual, "SmallInputDino", lambda *args, **kwargs: appearance)
    monkeypatch.setattr(
        legacy,
        "legacy_models",
        lambda plan: (SimpleNamespace(model=None), SimpleNamespace(model=None)),
    )
    frame_indices = np.array([0, 2, 4], dtype=np.int64)
    header = {
        "source_fps": 25,
        "source_frames": 5,
        "sampled_frames": 3,
        "source_shape": [480, 640],
        "frame_indices_sha256": data.array_metadata(frame_indices)["sha256"],
    }
    pose = np.ones((3, 14), dtype=np.float32)
    pose[0, 0] = -0.0
    pixels = {
        "header": header,
        "pixels": np.zeros((3, 3, 224, 224, 3), dtype=np.uint8),
        "pose_features": pose,
        "frame_indices": frame_indices,
        "valid_intervals": np.ones(2, dtype=np.bool_),
        "source_hand_tip_visible": np.ones(5, dtype=np.bool_),
    }
    receipt = input_receipt(pixels)
    teacher = tmp_path / "checkpoints/teacher.npz"
    np.savez(teacher, pose_features=pose)
    cases, controls = [], []
    for index in range(130):
        kind = "word" if index < 96 else "sentence"
        row = {
            "source": "synthetic_" + kind,
            "example_id": str(index),
            "source_group": f"group-{index // 3}",
            "signer": "P6",
            "reference": "a",
            "role": "development",
            "training_allowed": False,
        }
        case = {"kind": kind, "row": row}
        if kind == "word":
            case["word"] = {
                "verified_input": {"arrays": {"pose_features": receipt["arrays"]["pose_features"]}}
            }
        else:
            case["sentence"] = {
                "development": {
                    "original_pair": {
                        "teacher": {
                            "cache": data.artifact(teacher)["path"],
                            "cache_sha256": data.sha(teacher),
                        }
                    }
                }
            }
        cases.append(case)
        controls.append(
            {
                "row": row,
                "hypothesis": "b",
                "tokens": [0, 3, 1] if kind == "word" else None,
                "token_identity_required": kind == "word",
            }
        )
    monkeypatch.setattr(evaluation, "development_row", lambda case: case["row"])
    monkeypatch.setattr(evaluation, "expected_header", lambda case: header)
    monkeypatch.setattr(
        evaluation,
        "source_visibility",
        lambda case: (
            header,
            frame_indices,
            pixels["source_hand_tip_visible"],
            pixels["valid_intervals"],
        ),
    )
    monkeypatch.setattr(evaluation, "read_development224", lambda case: copy.deepcopy(pixels))
    streams = {
        name: np.broadcast_to(
            np.array([0.1, 0.8, 1.1], dtype=np.float32)[:, None], (3, width)
        ).copy()
        for name, width in data.WIDTHS.items()
    }
    streams["pose_features"] = pose
    monkeypatch.setattr(
        words,
        "original_streams",
        lambda *args: {**copy.deepcopy(streams), "frame_indices": frame_indices.copy()},
    )
    events = []

    def decode(language, tokenizer, generation, memory):
        assert generation == model_plan["reference"]["generation"]
        trained = torch.count_nonzero(model.timing.rate_weights).item() > 0
        events.append("candidate" if trained else "control")
        return ("a", [0, 2, 1]) if trained else ("b", [0, 3, 1])

    monkeypatch.setattr(original, "decode", decode)
    install = candidate.install_timing

    def install_after_all_controls(model, binding):
        assert events == ["control"] * 130
        install(model, binding)

    monkeypatch.setattr(candidate, "install_timing", install_after_all_controls)
    plan = {
        "final_audit": data.artifact(audit),
        "adapter": binding["adapter"],
        "model_plan": model_plan,
        "inputs": {"cases": cases, "controls": controls},
    }
    path = tmp_path / "runs/evaluation-plan.json"
    save(path, plan)
    monkeypatch.setattr(evaluation, "checked", lambda path: copy.deepcopy(plan))
    output = tmp_path / "checkpoints/evaluation"
    output.mkdir()
    return path, output, plan, model, events


def test_all130_controls_precede_candidate_loading_and_decoded_evidence_replays(
    inference, monkeypatch
):
    from transformers import ByT5Tokenizer

    path, output, plan, model, events = inference
    evaluation.infer(path, output, plan)
    assert events == ["control"] * 130 + ["candidate"] * 130
    monkeypatch.setattr(ByT5Tokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    baseline, predictions, assets = evaluation.decoded(plan, output, path)
    assert len(baseline) == len(predictions) == 130 and len(assets) == 391
    assert all(row["hypothesis"] == "b" for row in baseline)
    assert all(row["hypothesis"] == "a" for row in predictions)
    receipt_path = output / "inference.json"
    original_receipt = receipt_path.read_bytes()
    candidate_path = output / "candidate-0000.json"
    original_candidate = candidate_path.read_bytes()
    for change in ("timing", "adapter", "row", "native", "memory", "plan", "partial"):
        receipt, value = json.loads(original_receipt), json.loads(original_candidate)
        if change == "timing":
            value["timing"]["positions"]["sha256"] = "0" * 64
        elif change == "adapter":
            value["adapter"] = {"different": True}
        elif change == "row":
            value["row"]["example_id"] = "different"
        elif change == "native":
            value["native_crop_parity"] = False
        elif change == "memory":
            value["semantic_memory"]["shape"][0] = 2
        elif change == "plan":
            receipt["plan"] = {"different": True}
        else:
            receipt["candidates"].pop()
        save(candidate_path, value)
        receipt["candidates"][0] = data.artifact(candidate_path)
        save(receipt_path, receipt)
        with pytest.raises(ValueError):
            evaluation.decoded(plan, output, path)
        candidate_path.write_bytes(original_candidate)
        receipt_path.write_bytes(original_receipt)
    assert model.timing.rate_weights.detach().tolist()


def test_failed_original_token_control_prevents_candidate_installation(inference):
    path, output, plan, model, events = inference
    plan["inputs"]["controls"][0]["tokens"] = [0, 4, 1]
    with pytest.raises(ValueError, match="original complete C5 control"):
        evaluation.infer(path, output, plan)
    assert events == ["control"]
    assert torch.count_nonzero(model.timing.rate_weights).item() == 0
    assert not list(output.glob("candidate-*.json"))


class RawFakePacer:
    exact_match_score = 10

    def score(self, rows, output):
        from semantic_quality_judge import calibration_controls
        from terra_polarity_judge import payload

        known = {(c["reference"], c["candidate"]): c["range"][0] for c in calibration_controls()}
        scores = [
            known.get(
                (row["reference"], row["candidate"]),
                self.exact_match_score if row["reference"] == row["candidate"] else 0,
            )
            for row in rows
        ]
        output.mkdir(parents=True)
        save(output / "request.json", payload(rows))
        save(
            output / "started.json",
            {
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "automatic_retries": 0,
                "pairs": len(rows),
                "request_sha256": data.sha(output / "request.json"),
            },
        )
        save(
            output / "response.json",
            {
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
                                            {"id": f"pair_{i:02d}", "score": value}
                                            for i, value in enumerate(scores)
                                        ]
                                    }
                                ),
                            }
                        ],
                    }
                ],
            },
        )
        result = {
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "scores": scores,
            "response_sha256": data.sha(output / "response.json"),
        }
        save(output / "scores.json", result)
        return result


def test_full_word_and_sentence_scoring_uses_real_parser_and_never_claims_confirmation(
    inference, monkeypatch
):
    import terra_polarity_judge as judge
    from transformers import ByT5Tokenizer

    path, output, plan, _, _ = inference
    evaluation.infer(path, output, plan)
    monkeypatch.setattr(ByT5Tokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(judge, "Pacer", RawFakePacer)
    evaluation.score(path, output, plan)
    result = data.read(output / "decision.json")
    assert result["word"]["clips"] == 96 and result["sentence"]["clips"] == 34
    assert result["passes_both_semantic_research_gates"] is True
    assert result["word"]["calibration_passed"] and result["sentence"]["calibration_passed"]
    assert result["new_Terra_requests"] == 4
    assert result["independent_confirmation_complete"] is False
    assert result["promotion_authorized"] is False


def test_failed_original_calibration_preserves_raw_scores_without_quality_decision(
    inference, monkeypatch
):
    import terra_polarity_judge as judge
    from semantic_quality_judge import calibration_controls
    from transformers import ByT5Tokenizer

    class BadCalibration(RawFakePacer):
        def score(self, rows, output):
            result = super().score(rows, output)
            target = calibration_controls()[-1]
            position = next(
                (
                    i
                    for i, row in enumerate(rows)
                    if row["reference"] == target["reference"]
                    and row["candidate"] == target["candidate"]
                ),
                None,
            )
            if position is not None:
                response = data.read(output / "response.json")
                content = json.loads(response["output"][0]["content"][0]["text"])
                content["scores"][position]["score"] = 4
                response["output"][0]["content"][0]["text"] = json.dumps(content)
                save(output / "response.json", response)
                result["scores"][position] = 4
                result["response_sha256"] = data.sha(output / "response.json")
                save(output / "scores.json", result)
            return result

    path, output, plan, _, _ = inference
    evaluation.infer(path, output, plan)
    monkeypatch.setattr(ByT5Tokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(judge, "Pacer", BadCalibration)
    with pytest.raises(ValueError, match="failed original semantic calibration"):
        evaluation.score(path, output, plan)
    assert (output / "word/calibration-failure.json").exists()
    assert not (output / "word/decision.json").exists()
    assert not (output / "sentence").exists() and not (output / "decision.json").exists()
    assert len(list((output / "word/semantic").rglob("scores.json"))) == 2


@pytest.mark.parametrize("improved", [True, False])
def test_controller_releases_neural_lease_and_authenticates_positive_or_negative_decision(
    inference, monkeypatch, improved
):
    from contextlib import contextmanager

    import prepare_c5_timing_features as features
    import terra_polarity_judge as judge
    import train_terra_c5_tail as owner
    from transformers import ByT5Tokenizer

    path, output, _, _, events = inference
    output.rmdir()
    lease = [False]

    @contextmanager
    def fake_lease():
        lease[0] = True
        try:
            yield
        finally:
            lease[0] = False

    class OutsideLease(RawFakePacer):
        exact_match_score = 10 if improved else 0

        def score(self, rows, output):
            assert not lease[0]
            return super().score(rows, output)

    monkeypatch.setattr(features, "require_environment", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(owner, "gpu_lease", fake_lease)
    monkeypatch.setattr(ByT5Tokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(judge, "Pacer", OutsideLease)
    evaluation.run(path, output)
    assert len(events) == 260 and not lease[0]
    assert data.read(output / "exit.json")["exit_code"] == 0
    assert data.read(output / "decision.json")["passes_both_semantic_research_gates"] is improved

    # Join synthetic terminal evidence to the program that a real CLI run records.
    worker_path = output / "worker.json"
    worker = data.read(worker_path)
    worker["argv"] = [
        sys.executable,
        str(evaluation.ROOT / "tools/evaluate_c5_caption_timing.py"),
        "--plan",
        str(path),
        "--output",
        str(output),
    ]
    save(worker_path, worker)
    claim_path = path.with_name("execution-claim.json")
    claim = data.read(claim_path)
    claim["worker"] = data.artifact(worker_path)
    save(claim_path, claim)
    poll_path = path.with_name("terminal-tool-poll.json")
    poll = {
        "tool": "write_stdin",
        "requested_session_id": 123,
        "actual_owned_pid": worker["pid"],
        "tool_result": {"exit_code": 0},
    }
    save(poll_path, poll)
    decision_path = output / "decision.json"
    checked_plan, result, evidence = evaluation.checked_decision(path, decision_path, poll_path)
    assert checked_plan == data.read(path)
    assert result == data.read(decision_path) and data.artifact(claim_path) in evidence

    for changed_path, value in (
        (decision_path, {**result, "independent_confirmation_complete": True}),
        (poll_path, {**poll, "tool_result": {"session_id": 123}}),
        (poll_path, {**poll, "actual_owned_pid": worker["pid"] + 1}),
        (claim_path, {**claim, "worker": {"different": True}}),
        (output / "word/semantic/pass-1-batch-000/response.json", {"status": "failed"}),
    ):
        original_bytes = changed_path.read_bytes()
        save(changed_path, value)
        with pytest.raises((ValueError, KeyError)):
            evaluation.checked_decision(path, decision_path, poll_path)
        changed_path.write_bytes(original_bytes)
    extra = output / "unbound.json"
    save(extra, {})
    with pytest.raises(ValueError, match="extra or aliased"):
        evaluation.checked_decision(path, decision_path, poll_path)
    extra.unlink()
    evaluation.checked_decision(path, decision_path, poll_path)


def test_full_families_cannot_be_replaced_by_partial_or_unknown_scope():
    for kind, count in (("word", 95), ("sentence", 33), ("unknown", 130)):
        with pytest.raises(ValueError):
            protocol.compare(kind, [{}] * count, [{}] * count, [{}] * count)


def test_overall_gate_requires_both_complete_calibrated_families():
    word = {
        "clips": 96,
        "passes_semantic_research_gate": True,
        "complete_scoring": True,
        "calibration_passed": True,
    }
    sentence = {**word, "clips": 34}
    assert protocol.both_families(word, sentence)["passes_both_semantic_research_gates"]
    sentence["calibration_passed"] = False
    assert not protocol.both_families(word, sentence)["passes_both_semantic_research_gates"]
    with pytest.raises(ValueError, match="both complete"):
        protocol.both_families({**word, "clips": 95}, sentence)


def test_early_freeze_refuses_missing_final_before_inputs_or_models(tmp_path, monkeypatch):
    import c5_caption_timing_final as final

    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(
        evaluation, "inventory", lambda *args: pytest.fail("input admission before final")
    )

    def refuse(path):
        raise ValueError("complete actual final unavailable")

    monkeypatch.setattr(final, "checked_final", refuse)
    path = tmp_path / "runs/new/plan.json"
    with pytest.raises(ValueError, match="complete actual final unavailable"):
        evaluation.freeze(
            path, tmp_path / "missing", tmp_path / "a", tmp_path / "b", tmp_path / "c"
        )
    assert not path.parent.exists()
