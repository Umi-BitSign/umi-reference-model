"""Complete48 synthetic native224 confirmation routing and original raw parser."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_temporal_memory_candidate as candidate
import c5_timing_feature_data as data
import evaluate_c5_temporal_memory as evaluation
import run_c5_temporal_memory_confirmation as confirmation
import test_evaluate_c5_temporal_memory as evaluation_tests
from test_evaluate_c5_temporal_memory import RawFakePacer
from test_probe_c5_timing_semantic import Tokenizer

INSTALL = candidate.install_adapter
adapter = evaluation_tests.adapter
inference = evaluation_tests.inference


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def cohort(inference, monkeypatch):
    from contextlib import contextmanager

    import evaluate_roi_short_translation as original
    import train_terra_c5_tail as owner
    from c5_timing_native_input import input_receipt
    from transformers import ByT5Tokenizer

    path, output, development, model, events = inference
    output.rmdir()
    root = output.parents[1]
    broad = confirmation.broad
    monkeypatch.setattr(confirmation, "ROOT", root)
    monkeypatch.setattr(broad, "ROOT", root)
    monkeypatch.setattr(confirmation.native, "ROOT", root)
    monkeypatch.setattr(broad, "artifact", data.artifact)
    monkeypatch.setattr(broad, "bound", data.bound)
    monkeypatch.setattr(broad, "guard", lambda *args: None)
    monkeypatch.setattr(broad, "require_environment", lambda: None)
    monkeypatch.setattr(broad, "MEDIA", root / "runs/media/plan.json")
    monkeypatch.setattr(broad, "CLAIM", broad.MEDIA.with_name("consumption.json"))
    save(broad.MEDIA, {"qualifications": ["synthetic cohort"]})
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    monkeypatch.setattr(ByT5Tokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    values = evaluation.read_development224(development["inputs"]["cases"][0])
    monkeypatch.setattr(
        confirmation.inputs, "read_confirmation224", lambda row: copy.deepcopy(values)
    )
    monkeypatch.setattr(
        confirmation.inputs,
        "source_visibility",
        lambda row: (
            values["header"],
            values["frame_indices"],
            values["source_hand_tip_visible"],
            values["valid_intervals"],
        ),
    )
    receipts, records = [], []
    original_input = {
        k: v for k, v in values.items() if k not in ("source_hand_tip_visible", "valid_intervals")
    }
    for index in range(48):
        family = "how2sign" if index < 24 else "openasl"
        group = index // 2 if index < 24 else index
        identity = f"synthetic-{index:04d}"
        row = {
            "clip_id": identity,
            "family": family,
            "source_group": f"{family}:{group}",
            "reference": "a",
            "video": {"path": "synthetic.mp4"},
            "training_allowed": False,
        }
        record = {
            "example": row,
            "original": {
                "row": {"example_id": identity, "role": "confirmation", "training_allowed": False}
            },
            "ROI_lineage_training_allowed": False,
        }
        target = root / f"checkpoints/preflight/input-{index:04d}.json"
        save(
            target,
            {
                "record": record,
                "native_inputs": {"224": confirmation.native.array_receipt(original_input, 224)},
            },
        )
        records.append(record)
        receipts.append(data.artifact(target))
    preflight = root / "checkpoints/preflight/result.json"
    save(preflight, {"receipts": receipts})
    plan = {
        "adapter": development["adapter"],
        "adapter_configuration": development["adapter_configuration"],
        "final_audit": development["final_audit"],
        "model_plan": development["model_plan"],
        "records": records,
        "evidence": {"preflight": data.artifact(preflight)},
    }
    from terra_polarity_judge import protocol

    plan["judge_protocol"] = protocol()
    save(path, plan)
    monkeypatch.setattr(confirmation, "checked", lambda path: copy.deepcopy(plan))
    lease = [False]

    @contextmanager
    def fake_lease():
        lease[0] = True
        try:
            yield
        finally:
            lease[0] = False

    monkeypatch.setattr(owner, "gpu_lease", fake_lease)
    decode = original.decode

    def after_consumption(*args):
        assert broad.CLAIM.exists() and lease[0]
        return decode(*args)

    monkeypatch.setattr(original, "decode", after_consumption)

    def after_all_originals(model, binding):
        assert events == ["control"] * 48
        INSTALL(model, binding)

    monkeypatch.setattr(candidate, "install_adapter", after_all_originals)
    assert input_receipt(values)["native_side"] == 224
    return path, output, plan, model, events, lease


def test_all48_originals_before_trained_memory_and_single_common_cohort_claim(cohort):
    path, output, plan, _, events, lease = cohort
    confirmation.infer(path, output)
    assert events == ["control"] * 48 + ["candidate"] * 48 and not lease[0]
    comparison, authority = confirmation.checked_inference(output / "result.json", path, plan)
    assert len(comparison["cases"]) == 48
    assert len(authority["memory_inference_assets"]) == 195
    assert data.read(confirmation.broad.CLAIM)["replacement_or_tuning_allowed"] is False
    with pytest.raises(ValueError, match="already consumed"):
        confirmation.infer(path, output.with_name("another"))
    assert not output.with_name("another").exists()
    target = output / "candidate-0000.json"
    original_bytes = target.read_bytes()
    original_result = (output / "result.json").read_bytes()
    baseline_path = output / "baseline-0000.json"
    baseline_bytes = baseline_path.read_bytes()
    memory_path = output / "memory-0000.npy"
    memory_bytes = memory_path.read_bytes()
    for change in ("timing", "adapter", "tokens", "source", "partial", "memory", "base_memory"):
        saved, result = json.loads(original_bytes), json.loads(original_result)
        if change == "timing":
            saved["adapter_details"]["base_memory"]["sha256"] = "0" * 64
        elif change == "adapter":
            saved["adapter"] = {"different": True}
        elif change == "tokens":
            saved["tokens"] = [999]
        elif change == "source":
            saved["prediction"]["source_group"] = "how2sign:other"
        elif change == "memory":
            saved["semantic_memory"]["sha256"] = "0" * 64
        elif change == "base_memory":
            import numpy as np

            control = json.loads(baseline_bytes)
            memory = np.load(memory_path, allow_pickle=False)
            memory[0, 0] += 1
            np.save(memory_path, memory, allow_pickle=False)
            control["base_memory"] = data.artifact(memory_path)
            control["base_memory_metadata"] = data.array_metadata(memory)
            save(baseline_path, control)
            result["controls"][0] = data.artifact(baseline_path)
            saved["control"] = result["controls"][0]
            saved["adapter_details"]["base_memory"] = control["base_memory_metadata"]
        else:
            result["candidates"].pop()
        save(target, saved)
        result["candidates"][0] = data.artifact(target)
        save(output / "result.json", result)
        with pytest.raises(ValueError):
            confirmation.checked_inference(output / "result.json", path, plan)
        target.write_bytes(original_bytes)
        (output / "result.json").write_bytes(original_result)
        baseline_path.write_bytes(baseline_bytes)
        memory_path.write_bytes(memory_bytes)


@pytest.mark.parametrize("improved", [True, False])
def test_original_two_pass_broad_scoring_and_final_positive_or_negative_audit(
    cohort, monkeypatch, improved
):
    import terra_polarity_judge as judge

    path, output, plan, _, _, lease = cohort
    confirmation.infer(path, output)
    exited = data.read(output / "exit.json")
    exited["observed_at"] = "2020-01-01T00:00:00+00:00"
    save(output / "exit.json", exited)
    worker = data.read(output / "worker.json")
    worker["argv"] = [
        sys.executable,
        str(confirmation.ROOT / "tools/run_c5_temporal_memory_confirmation.py"),
        "--plan",
        str(path),
        "--output",
        str(output),
    ]
    save(output / "worker.json", worker)
    claim_path = confirmation.broad.CLAIM
    claim = data.read(claim_path)
    claim["worker"] = data.artifact(output / "worker.json")
    save(claim_path, claim)
    inference_path = output / "result.json"
    decoded = data.read(inference_path)
    decoded["claim"] = data.artifact(claim_path)
    save(inference_path, decoded)
    poll = {
        "tool": "write_stdin",
        "requested_session_id": 123,
        "actual_owned_pid": worker["pid"],
        "tool_result": {"exit_code": 0},
    }
    inference_poll = path.with_name("inference-poll.json")
    save(inference_poll, poll)

    class OutsideLease(RawFakePacer):
        exact_match_score = 10 if improved else 0

        def score(self, rows, output):
            assert not lease[0]
            return super().score(rows, output)

    monkeypatch.setattr(judge, "Pacer", OutsideLease)
    scores = output.with_name("scores")
    confirmation.score(path, output / "result.json", scores)
    result = data.read(scores / "result.json")
    assert result["complete"] and result["clips"] == 48
    assert result["passes_semantic_statistics"] is improved
    assert result["no_family_mean_regression"]
    assert result["raw_response_authentication_performed"] is True
    assert result["candidate_or_runtime_admission_verified"] is True
    assert result["promotion_authorized"] is False
    assert data.read(scores / "exit.json")["exit_code"] == 0
    worker = data.read(scores / "worker.json")
    worker["argv"] = [
        sys.executable,
        str(confirmation.ROOT / "tools/run_c5_temporal_memory_confirmation.py"),
        "--plan",
        str(path),
        "--output",
        str(scores),
        "--inference",
        str(inference_path),
    ]
    save(scores / "worker.json", worker)
    scoring_claim_path = claim_path.with_name("scoring-consumption.json")
    claim = data.read(scoring_claim_path)
    claim["worker"] = data.artifact(scores / "worker.json")
    save(scoring_claim_path, claim)
    scoring_poll = path.with_name("scoring-poll.json")
    save(scoring_poll, {**poll, "requested_session_id": 124})

    def audit():
        return confirmation.checked_confirmation(
            path, inference_path, inference_poll, scores / "result.json", scoring_poll
        )

    validated = audit()
    assert validated["substantial_model_improvement_confirmed"] is improved
    assert validated["independent_confirmation_complete"] is True
    for changed_path, changed in (
        (scoring_poll, {**poll, "tool_result": {"session_id": 124}}),
        (scores / "result.json", {**result, "promotion_authorized": True}),
        (scores / "pass-1-batch-000/response.json", {"status": "failed"}),
        (scoring_claim_path, {**claim, "replacement_or_tuning_allowed": True}),
    ):
        original_bytes = changed_path.read_bytes()
        save(changed_path, changed)
        with pytest.raises((ValueError, KeyError)):
            audit()
        changed_path.write_bytes(original_bytes)
    extra = scores / "unbound.json"
    save(extra, {})
    with pytest.raises(ValueError, match="extra or aliased"):
        audit()
    extra.unlink()
    audit()


def test_failed_local_gate_refuses_before_broad_input_access(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "checked_decision",
        lambda *args: ({}, {"passes_both_semantic_research_gates": False}, []),
    )
    monkeypatch.setattr(confirmation, "read", lambda *args: pytest.fail("broad inputs read early"))
    with pytest.raises(ValueError, match="both complete calibrated development gains"):
        confirmation.specification(
            *(tmp_path / name for name in ("plan", "result", "poll", "input"))
        )


def test_admission_binds_both_gates_native_preflight_and_all_reserved_source_groups(
    cohort, monkeypatch
):
    path, _, plan, _, _, _ = cohort
    root = confirmation.ROOT
    preflight_path = data.bound(plan["evidence"]["preflight"])
    records = plan["records"]
    media = data.read(confirmation.broad.MEDIA)
    reserved = sorted({r["example"]["source_group"] for r in records})
    reserved += [f"how2sign:fallback-{i}" for i in range(45)]
    media.update(examples=[r["example"] for r in records], reserved_source_groups=reserved)
    save(confirmation.broad.MEDIA, media)
    landmark_plan_path = root / "runs/landmarks/plan.json"
    save(landmark_plan_path, {"media": data.artifact(confirmation.broad.MEDIA)})
    audit_path = landmark_plan_path.with_name("audit.json")
    source = root / "tools/synthetic.py"
    source.parent.mkdir()
    source.write_text("# synthetic source")
    for name in ("run_c5_temporal_memory_confirmation.py", "c5_timing_broad_input.py"):
        (source.parent / name).write_text("# synthetic source")
    sources = {"tools/synthetic.py": data.sha(source)}
    save(audit_path, {"plan": data.artifact(landmark_plan_path), "source_sha256": sources})
    preflight = data.read(preflight_path)
    preflight.update(
        schema="umi-studio-broad-confirmation-native-inputs/1",
        complete=True,
        clips=48,
        training_allowed=False,
        model_or_scorer_execution=False,
        candidate_bound_by_preflight=False,
        confirmation_consumed_by_preflight=False,
        source_sha256=sources,
        landmark_audit=data.artifact(audit_path),
    )
    for descriptor in preflight["receipts"]:
        target = data.bound(descriptor)
        saved = data.read(target)
        saved["native_inputs"]["112"] = {"synthetic": True}
        save(target, saved)
        descriptor.update(data.artifact(target))
    save(preflight_path, preflight)
    save(preflight_path.with_name("worker.json"), {"pid": os.getpid(), "uid": os.getuid()})
    save(preflight_path.with_name("exit.json"), {"pid": os.getpid(), "exit_code": 0})
    training_path = root / "runs/training-plan.json"
    training = {
        "records": [
            {
                "row": {
                    "source_group": "asl_stem:training",
                    "role": "training",
                    "training_allowed": True,
                }
            }
            for _ in range(532)
        ]
    }
    save(training_path, training)
    final_path = data.bound(plan["final_audit"])
    save(final_path, {"plan": data.artifact(training_path)})
    development = {**plan, "final_audit": data.artifact(final_path), "source_sha256": {}}
    monkeypatch.setattr(
        evaluation,
        "checked_decision",
        lambda *args: (development, {"passes_both_semantic_research_gates": True}, []),
    )
    monkeypatch.setattr(confirmation.native, "bind_records", lambda path: copy.deepcopy(records))
    monkeypatch.setattr(confirmation.broad, "sources", lambda: {})

    def specification():
        return confirmation.specification(path, path, path, preflight_path)

    result = specification()
    assert result["adapter"] == plan["adapter"]
    assert len(result["records"]) == 48 and len(result["preflight_assets"]) == 52
    assert result["promotion_authorized"] is False
    # A fallback source is excluded just as a primary source is.
    training["records"][0]["row"]["source_group"] = reserved[-1]
    save(training_path, training)
    save(final_path, {"plan": data.artifact(training_path)})
    development["final_audit"] = data.artifact(final_path)
    with pytest.raises(ValueError, match="primary or fallback"):
        specification()
