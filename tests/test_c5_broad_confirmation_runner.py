"""Synthetic end-to-end controller tests; no real clip, model or API execution."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_broad_semantic_scoring as semantic
import run_c5_broad_confirmation as runner


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    media = tmp_path / "runs/media/plan.json"
    claim = media.with_name("consumption.json")
    monkeypatch.setattr(runner, "MEDIA", media)
    monkeypatch.setattr(runner, "CLAIM", claim)

    def artifact(path):
        return {
            "path": str(path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(runner, "artifact", artifact)
    monkeypatch.setattr(runner, "guard", lambda *args: None)
    monkeypatch.setattr(runner.inputs, "owned", lambda p, folder: None)
    monkeypatch.setattr(runner, "require_environment", lambda: None)
    monkeypatch.setattr(
        runner, "identity", lambda pid, command: {"pid": pid, "uid": os.getuid(), "argv": command}
    )
    save(media, {"qualifications": ["Synthetic cohort; no real predictions"]})
    examples = [
        {
            "clip_id": f"synthetic-{i:02d}",
            "family": "how2sign" if i < 24 else "openasl",
            "source_group": ("how2sign:" if i < 24 else "openasl:") + f"synthetic-{i // 2}",
            "reference": "synthetic reference",
            "training_allowed": False,
            "video": {"path": f"data/synthetic-{i:02d}", "sha256": "synthetic"},
        }
        for i in range(48)
    ]
    protocol = {
        "schema": "umi-studio-Terra-polarity-clarification/1",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "original_controls": semantic.calibration_controls(),
        "clarification_append_only": " Synthetic clarification.",
    }
    protocol["full_instructions_sha256"] = hashlib.sha256(
        (
            semantic.original_payload([])["instructions"] + protocol["clarification_append_only"]
        ).encode()
    ).hexdigest()
    preflight = tmp_path / "checkpoints/preflight/result.json"
    descriptors = []
    records = [{"example": row} for row in examples]
    for i, record in enumerate(records):
        path = preflight.with_name(f"input-{i:04d}.json")
        save(
            path,
            {
                "record": record,
                "native_inputs": {
                    str(side): {"side": side, "header": {"sampled_frames": 2}}
                    for side in (112, 224)
                },
            },
        )
        descriptors.append(artifact(path))
    save(preflight, {"receipts": descriptors})
    export = tmp_path / "checkpoints/export/EXPORT.json"
    save(export, {"synthetic": True})
    plan_path = tmp_path / "runs/confirmation/plan.json"
    plan = {
        "records": records,
        "model_plan": {
            "reference": {"bundle_root": "synthetic-model", "generation": {"num_beams": 5}},
            "pretrained_state_sha256": "synthetic-weight",
        },
        "evidence": {"export": artifact(export), "preflight": artifact(preflight)},
        "judge_protocol": protocol,
    }
    save(plan_path, plan)
    monkeypatch.setattr(runner, "checked", lambda path: copy.deepcopy(plan))
    monkeypatch.setattr(
        runner.inputs,
        "read_input",
        lambda record, side: {
            "pixels": np.zeros((2, 3, side, side, 3), dtype=np.uint8),
            "side": side,
        },
    )
    monkeypatch.setattr(
        runner.inputs,
        "array_receipt",
        lambda values, side: {"side": side, "header": {"sampled_frames": 2}},
    )
    streams = {
        name: np.zeros((2, width), dtype=np.float32)
        for name, width in (
            ("face_features", 384),
            ("left_hand_features", 384),
            ("right_hand_features", 384),
            ("pose_features", 14),
        )
    }
    events = []

    class Appearance:
        def __init__(self, *args, **kwargs):
            pass

        def requires_grad_(self, flag):
            return self

        def eval(self):
            return self

        def state_dict(self):
            return {}

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def decode(self, tokens, **kwargs):
            return "synthetic baseline" if tokens == [1] else "synthetic candidate"

    def baseline(appearance, values):
        events.append("baseline-features")
        return {**{k: v.copy() for k, v in streams.items()}, "frame_indices": np.arange(2)}

    def candidate(embedders, values):
        events.append("candidate-features")
        return {k: v.copy() for k, v in streams.items()}

    def decode(*args):
        assert claim.exists(), "cohort must be consumed before every decode"
        arm = "baseline" if len([e for e in events if e.endswith("decode")]) < 48 else "candidate"
        events.append(arm + "-decode")
        return "synthetic " + arm, [1 if arm == "baseline" else 2]

    def load_pair(*args, **kwargs):
        assert events.count("baseline-decode") == 48 and "candidate-decode" not in events
        events.append("candidate-loaded")
        return {}

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(ByT5Tokenizer=Tokenizer))
    monkeypatch.setitem(
        sys.modules, "c5_small_input_vision", SimpleNamespace(SmallInputDino=Appearance)
    )
    monkeypatch.setitem(
        sys.modules, "c5_native112_visual_loader", SimpleNamespace(load_visual_pair=load_pair)
    )
    monkeypatch.setitem(
        sys.modules,
        "compare_c5_native112_export",
        SimpleNamespace(native_translator=lambda plan: object(), decode_native=decode),
    )
    monkeypatch.setitem(
        sys.modules, "evaluate_c5_words", SimpleNamespace(original_streams=baseline)
    )
    monkeypatch.setitem(
        sys.modules,
        "probe_c5_small_input",
        SimpleNamespace(
            legacy_models=lambda plan: (
                SimpleNamespace(model=object()),
                SimpleNamespace(model=object()),
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "probe_tiny_rgb_capacity",
        SimpleNamespace(state_digest=lambda state: "synthetic-weight"),
    )
    import c5_native112_parity

    monkeypatch.setattr(c5_native112_parity, "native_streams", candidate)
    import torch

    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    return SimpleNamespace(
        root=tmp_path,
        plan_path=plan_path,
        plan=plan,
        events=events,
        claim=claim,
        output=tmp_path / "checkpoints/inference",
        streams=streams,
    )


def test_complete_serial_inference_consumes_before_baseline_and_authenticates_all_outputs(
    synthetic,
):
    s = synthetic
    runner.infer(s.plan_path, s.output)
    result = runner.read(s.output / "result.json")
    assert result["complete"] is True and len(result["files"]) == 96
    assert s.events.count("baseline-decode") == s.events.count("candidate-decode") == 48
    assert s.events.index("candidate-loaded") > s.events.index("baseline-decode")
    comparison, authority = runner.checked_inference(s.output / "result.json", s.plan_path, s.plan)
    assert len(comparison["cases"]) == 48 and len(comparison["items"]) == 6
    assert runner.read(s.output / "exit.json")["exit_code"] == 0
    with pytest.raises(ValueError, match="already consumed"):
        runner.infer(s.plan_path, s.root / "checkpoints/second-inference")


def test_failed_first_decode_keeps_cohort_consumed_and_prevents_fresh_run(synthetic, monkeypatch):
    s = synthetic

    def fail(*args):
        assert s.claim.exists()
        raise RuntimeError("synthetic decoder failure")

    monkeypatch.setattr(sys.modules["compare_c5_native112_export"], "decode_native", fail)
    with pytest.raises(RuntimeError, match="decoder failure"):
        runner.infer(s.plan_path, s.output)
    assert s.claim.exists() and runner.read(s.output / "exit.json")["exit_code"] == 1
    assert runner.read(s.output / "failure.json")["cohort_released"] is False
    with pytest.raises(ValueError, match="already consumed"):
        runner.infer(s.plan_path, s.root / "checkpoints/replacement")


def test_preflight_mismatch_is_rejected_before_decode_but_consumption_remains(
    synthetic, monkeypatch
):
    s = synthetic
    monkeypatch.setattr(runner.inputs, "array_receipt", lambda *args: {"changed": True})
    with pytest.raises(ValueError, match="after preflight"):
        runner.infer(s.plan_path, s.output)
    assert s.claim.exists() and not any(e.endswith("decode") for e in s.events)


def test_atomic_consumption_refuses_existing_claim_without_truncation(synthetic):
    s = synthetic
    runner.start(s.output)
    runner.consume(s.plan_path, s.output)
    original = s.claim.read_bytes()
    with pytest.raises(FileExistsError):
        runner.consume(s.plan_path, s.output)
    assert s.claim.read_bytes() == original


@pytest.mark.parametrize("change", ("token", "hypothesis", "features", "file_order", "identity"))
def test_authentication_rejects_changed_actual_inference(synthetic, change):
    s = synthetic
    runner.infer(s.plan_path, s.output)
    result_path = s.output / "result.json"
    result = runner.read(result_path)
    target = s.output / "candidate-0000.json"
    saved = runner.read(target)
    if change == "file_order":
        result["files"][0], result["files"][1] = result["files"][1], result["files"][0]
    elif change == "features":
        feature_path = runner.bound(saved["features"])
        feature_path.write_bytes(b"changed")
    else:
        if change == "token":
            saved["tokens"] = [1]
        elif change == "hypothesis":
            saved["prediction"]["hypothesis"] = "changed"
        else:
            saved["prediction"]["source_group"] = "openasl:substituted"
        save(target, saved)
        result["files"][48] = runner.artifact(target)
    save(result_path, result)
    with pytest.raises(ValueError):
        runner.checked_inference(result_path, s.plan_path, s.plan)


def test_two_complete_raw_grading_passes_and_single_scoring_claim(synthetic, monkeypatch):
    s = synthetic
    runner.infer(s.plan_path, s.output)
    # Avoid wall-clock waits in this synthetic scorer. Real execution uses actual exit time.
    exited_path = s.output / "exit.json"
    exited = runner.read(exited_path)
    exited["observed_at"] = "2000-01-01T00:00:00+00:00"
    save(exited_path, exited)
    calls = []

    class Pacer:
        def score(self, rows, output):
            calls.append(rows)
            output.mkdir()
            save(output / "request.json", semantic.polarity_payload(rows, s.plan["judge_protocol"]))
            save(
                output / "started.json",
                {
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "medium",
                    "automatic_retries": 0,
                    "pairs": len(rows),
                    "request_sha256": hashlib.sha256(
                        (output / "request.json").read_bytes()
                    ).hexdigest(),
                },
            )
            controls = {
                (r["reference"], r["candidate"]): r["range"][0]
                for r in semantic.calibration_controls()
            }
            scores = [
                controls.get(
                    (r["reference"], r["candidate"]),
                    6 if r["candidate"] == "synthetic candidate" else 4,
                )
                for r in rows
            ]
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
                                                {"id": f"pair_{i:02d}", "score": v}
                                                for i, v in enumerate(scores)
                                            ]
                                        }
                                    ),
                                }
                            ],
                        }
                    ],
                },
            )
            save(
                output / "scores.json",
                {
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "medium",
                    "scores": scores,
                    "response_sha256": hashlib.sha256(
                        (output / "response.json").read_bytes()
                    ).hexdigest(),
                },
            )
            return {"scores": scores}

    monkeypatch.setitem(sys.modules, "terra_polarity_judge", SimpleNamespace(Pacer=Pacer))
    output = s.root / "checkpoints/scores"
    runner.score(s.plan_path, s.output / "result.json", output)
    result = runner.read(output / "result.json")
    assert len(calls) == 2 and result["passes_semantic_statistics"] is True
    assert result["raw_response_authentication_performed"] is True
    assert result["candidate_or_runtime_admission_verified"] is True
    assert result["promotion_authorized"] is False
    with pytest.raises(FileExistsError):
        runner.score(s.plan_path, s.output / "result.json", s.root / "checkpoints/rescore")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "change",
    ("incomplete", "wrong_export", "missing_request", "missing_reference", "wrong_repeats"),
)
def test_runtime_admission_rejects_incomplete_or_different_candidate(change):
    export, parity = {"path": "synthetic-export"}, {"path": "synthetic-parity"}
    plan = {
        "schema": "umi-studio-native112-full-video-compatibility/1",
        "export": export,
        "preprocessed_parity": parity,
        "case_indices": [0, 1],
        "repeats": 2,
    }
    result = {
        "schema": plan["schema"],
        "complete": True,
        "full_video_crop_pose_feature_token_parity": True,
        "fresh_linux_detection_per_request": True,
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "requests": [{}] * 8,
        "trained_checkpoint_references_independently_reproduced": [{}, {}],
    }
    runner.check_runtime(result, plan, export, parity)
    if change == "incomplete":
        result["complete"] = False
    elif change == "wrong_export":
        plan["export"] = {"path": "different"}
    elif change == "missing_request":
        result["requests"].pop()
    elif change == "missing_reference":
        result["trained_checkpoint_references_independently_reproduced"].pop()
    else:
        plan["repeats"] = 1
    with pytest.raises(ValueError, match="native video compatibility"):
        runner.check_runtime(result, plan, export, parity)


@pytest.mark.parametrize("change", (None, "preflight_order", "request_order", "input_cohort"))
def test_admission_joins_runtime_features_and_all_reserved_inputs_before_consumption(
    synthetic, monkeypatch, change
):
    s = synthetic
    root = s.root
    export_path = runner.bound(s.plan["evidence"]["export"])
    preflight_path = runner.bound(s.plan["evidence"]["preflight"])
    parity_path = root / "checkpoints/parity/result.json"
    save(parity_path, {"synthetic": True})
    audit_path = root / "runs/landmark-audit/result.json"
    landmark_plan = root / "runs/landmark-plan/plan.json"
    media = runner.read(runner.MEDIA)
    media["examples"] = [r["example"] for r in s.plan["records"]]
    save(runner.MEDIA, media)
    save(landmark_plan, {"media": runner.artifact(runner.MEDIA)})
    save(
        audit_path,
        {
            "plan": runner.artifact(landmark_plan),
            "source_sha256": {"synthetic-source": "synthetic"},
        },
    )
    preflight = runner.read(preflight_path)
    preflight.update(
        {
            "schema": "umi-studio-broad-confirmation-native-inputs/1",
            "complete": True,
            "clips": 48,
            "training_allowed": False,
            "model_or_scorer_execution": False,
            "landmark_audit": runner.artifact(audit_path),
            "source_sha256": {"synthetic-source": "synthetic"},
        }
    )
    if change == "preflight_order":
        preflight["receipts"][0], preflight["receipts"][1] = (
            preflight["receipts"][1],
            preflight["receipts"][0],
        )
    save(preflight_path, preflight)
    monkeypatch.setattr(
        runner.inputs, "bind_records", lambda path: copy.deepcopy(s.plan["records"])
    )
    if change == "input_cohort":
        changed = copy.deepcopy(s.plan["records"])
        changed[0]["example"]["source_group"] = "how2sign:substituted"
        monkeypatch.setattr(runner.inputs, "bind_records", lambda path: changed)
    audit_receipt = root / "runs/training-audit.json"
    save(audit_receipt, {"synthetic": True})
    export = {"evidence": {"terminal_audit": runner.artifact(audit_receipt)}}
    monkeypatch.setitem(
        sys.modules,
        "run_c5_http502_audit_corrected",
        SimpleNamespace(install=lambda: None, verify_audit=lambda path: runner.read(path)),
    )
    monkeypatch.setitem(
        sys.modules,
        "probe_c5_native112_full_runtime",
        SimpleNamespace(
            admit=lambda a, b: (export, s.plan["model_plan"], [], [], {"synthetic": "parity"})
        ),
    )
    runtime_path = root / "checkpoints/runtime/result.json"
    runtime_path.parent.mkdir(parents=True)
    for directory in (runtime_path.parent, preflight_path.parent):
        save(directory / "worker.json", {"uid": os.getuid(), "pid": 123})
        save(directory / "exit.json", {"pid": 123, "exit_code": 0})
    references = []
    for i in range(2):
        target = runtime_path.with_name(f"reference-{i}.npz")
        np.savez(target, **s.streams)
        receipt_path = runtime_path.with_name(f"reference-{i}.json")
        save(
            receipt_path, {"features": runner.artifact(target), "tokens": [1], "text": "synthetic"}
        )
        references.append(runner.artifact(receipt_path))
    source = root / "tools/synthetic.py"
    source.parent.mkdir()
    source.write_text("# Synthetic source\n")
    snapshot = runtime_path.parent / "source-snapshot/tools/synthetic.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    runtime_plan_path = runtime_path.with_name("plan.json")
    runtime_plan = {
        "schema": "umi-studio-native112-full-video-compatibility/1",
        "export": runner.artifact(export_path),
        "preprocessed_parity": runner.artifact(parity_path),
        "case_indices": [0, 1],
        "repeats": 2,
        "cases": [{"synthetic": 0}, {"synthetic": 1}],
        "source_sha256": {"tools/synthetic.py": hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    save(runtime_plan_path, runtime_plan)
    from c5_native112_parity import compare_streams

    requests = []
    for device in ("CPU", "Metal"):
        for repeat in range(2):
            for case in range(2):
                path = runtime_path.with_name(f"request-{len(requests)}.json")
                save(
                    path,
                    {
                        "device": device,
                        "repeat": repeat,
                        "case_index": case,
                        "case": runtime_plan["cases"][case],
                        "exact_token_match": True,
                        "tokens": [1],
                        "text": "synthetic",
                        "feature_arrays": runner.read(runner.bound(references[case]))["features"],
                        "feature_comparison": compare_streams(s.streams, s.streams, 2),
                    },
                )
                requests.append(runner.artifact(path))
    if change == "request_order":
        requests[0], requests[1] = requests[1], requests[0]
    save(
        runtime_path,
        {
            "schema": runtime_plan["schema"],
            "complete": True,
            "plan": runner.artifact(runtime_plan_path),
            "full_video_crop_pose_feature_token_parity": True,
            "fresh_linux_detection_per_request": True,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "requests": requests,
            "trained_checkpoint_references_independently_reproduced": references,
        },
    )
    if change:
        with pytest.raises(ValueError):
            runner.admit(export_path, parity_path, runtime_path, preflight_path)
    else:
        evidence, model_plan, records = runner.admit(
            export_path, parity_path, runtime_path, preflight_path
        )
        assert evidence["runtime"] == runner.artifact(runtime_path) and records == s.plan["records"]
    assert not s.claim.exists() and not s.events
