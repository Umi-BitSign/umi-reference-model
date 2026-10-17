from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_word_semantic_model as model_module
import prepare_terra_word_native
import probe_c5_word_semantic as proof
from probe_jeff_terra_quality import terra_payload
from terra_training_recovery import artifact
from terra_word_semantic_protocol import training_negative


class Tokenizer:
    def __call__(self, text, truncation):
        assert truncation is False
        return SimpleNamespace(input_ids=[ord(character) + 3 for character in text] + [1])

    def decode(self, tokens, **kwargs):
        return "".join(chr(value - 3) for value in tokens if value > 2)


def overwrite_fixture(path, value):
    """Intentional fixture tampering bypasses the production write-once helper."""
    path.write_text(json.dumps(value))


def training_cases():
    return [
        {
            "record": {
                "original": {
                    "row": {
                        "example_id": str(index),
                        "reference": text,
                        "label": text.lower(),
                        "participant": "P2",
                        "role": "training",
                        "training_allowed": True,
                    }
                },
                "ROI_lineage_training_allowed": True,
                "normalized_probe": {"decoded_frames": 5, "width": 640, "height": 480},
            },
            "receipt": {"path": f"checkpoints/inputs/{index}.json", "sha256": str(index)},
        }
        for index, text in enumerate(("Environment", "Electromechanics", "Religion"))
    ]


@pytest.fixture
def candidate_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(proof, "ROOT", tmp_path)
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(proof, "artifact", descriptor)
    monkeypatch.setattr(prepare_terra_word_native, "guard", lambda output=None: None)
    plan_path = tmp_path / "runs/plan.json"
    plan_path.parent.mkdir()
    plan_path.write_text("{}")
    cases, tokenizer = training_cases(), Tokenizer()
    plan = {
        "records": cases,
        "pretrained_state_sha256": "original",
        "initial_reference_hashes": {"encoder": "e", "language": "l"},
    }
    pools = []
    for index, case in enumerate(cases):
        text = case["record"]["original"]["row"]["reference"]
        pools.append(
            [
                {
                    "tokens": tokenizer(text, truncation=False).input_ids,
                    "text": text,
                    "kind": "human_anchor",
                    "truncated": False,
                    "termination": "eos",
                },
                training_negative(case, cases[(index + 1) % len(cases)], tokenizer),
            ]
        )
    paired = proof.paired_inputs(cases, pools, tokenizer)
    scores = [0] * len(paired["items"])
    for mapping in paired["cases"]:
        scores[mapping["indices"][0]] = 10
    for control in paired["controls"]:
        scores[control["index"]] = control["range"][0]
    output = tmp_path / "checkpoints/proof/candidates"
    semantic = output / "semantic"
    semantic.mkdir(parents=True)
    indices = proof.shuffled_indices(paired)
    directory = semantic / "batch-000"
    directory.mkdir()
    request = terra_payload([paired["items"][index] for index in indices])
    payload = {
        "scores": [
            {"id": f"pair_{position:02d}", "score": scores[index]}
            for position, index in enumerate(indices)
        ]
    }
    response = {
        "model": "gpt-5.6-terra",
        "reasoning": {"effort": "medium"},
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]}
        ],
    }
    proof.write(directory / "request.json", request)
    proof.write(directory / "response.json", response)
    proof.write(
        directory / "scores.json", {"response_sha256": proof.sha256(directory / "response.json")}
    )
    proof.write(semantic / "blinded-inputs.json", paired)
    proof.write(output / "worker.json", {"pid": 71})
    proof.write(output / "exit.json", {"pid": 71, "exit_code": 0})
    proof.write(output / "scores.json", scores)
    for index, (case, pool) in enumerate(zip(cases, pools, strict=True)):
        proof.write(output / f"pool-{index:02d}.json", {"case": case, "candidates": pool})
    result = {
        "complete": True,
        "plan_sha256": proof.sha256(plan_path),
        "pools": pools,
        "comparison": paired,
        "qualities": proof.qualities(paired, scores),
        "scores": descriptor(output / "scores.json"),
        "vision_initialization_sha256": "original",
        "initial_reference_hashes": plan["initial_reference_hashes"],
        "optimizer_updates": 0,
        "new_Terra_requests": 1,
        "calibration_passed": True,
        "training_only": True,
        "promotion_authorized": False,
    }
    target = output / "candidates.json"
    proof.write(target, result)
    return plan_path, target, plan, tokenizer, result, response, directory


@pytest.mark.parametrize(
    "mutation",
    [None, "request", "model", "reasoning", "score", "calibration", "tokens", "terminal"],
)
def test_actual_response_payload_model_effort_tokens_calibration_and_grade_mapping_are_required(
    candidate_fixture, mutation
):
    plan_path, path, plan, tokenizer, result, response, directory = candidate_fixture
    if mutation == "request":
        payload = json.loads((directory / "request.json").read_text())
        payload["instructions"] += " changed"
        overwrite_fixture(directory / "request.json", payload)
    elif mutation in ("model", "reasoning", "score", "calibration"):
        if mutation == "model":
            response["model"] = "another-model"
        elif mutation == "reasoning":
            response["reasoning"]["effort"] = "low"
        else:
            emitted = json.loads(response["output"][0]["content"][0]["text"])
            position = 0
            if mutation == "calibration":
                generic = next(c for c in result["comparison"]["controls"] if c["range"] == [0, 0])
                position = proof.shuffled_indices(result["comparison"]).index(generic["index"])
            emitted["scores"][position]["score"] = 1
            response["output"][0]["content"][0]["text"] = json.dumps(emitted)
            if mutation == "calibration":
                scores = json.loads(path.with_name("scores.json").read_text())
                scores[generic["index"]] = 1
                overwrite_fixture(path.with_name("scores.json"), scores)
                result["scores"] = proof.artifact(path.with_name("scores.json"))
                overwrite_fixture(path, result)
        overwrite_fixture(directory / "response.json", response)
        overwrite_fixture(
            directory / "scores.json",
            {"response_sha256": proof.sha256(directory / "response.json")},
        )
    elif mutation == "tokens":
        result["pools"][0][0]["tokens"][0] += 1
        overwrite_fixture(path, result)
    elif mutation == "terminal":
        overwrite_fixture(path.with_name("exit.json"), {"pid": 71, "exit_code": 1})
    if mutation is None:
        assert (
            proof.checked_candidates(path, plan_path, plan, tokenizer)["qualities"] == [[10, 0]] * 3
        )
    else:
        with pytest.raises(ValueError):
            proof.checked_candidates(path, plan_path, plan, tokenizer)


def test_failed_original_controls_do_not_publish_completed_candidates(
    candidate_fixture, monkeypatch
):
    plan_path, _, plan, tokenizer, result, _, _ = candidate_fixture
    output = plan_path.parent.parent / "checkpoints/failed-candidates"
    output.mkdir()
    monkeypatch.setattr(proof, "load_student", lambda plan: (object(), tokenizer))
    monkeypatch.setattr(proof, "prepare_pools", lambda *args: result["pools"])
    indices = proof.shuffled_indices(result["comparison"])
    scores = json.loads((plan_path.parent.parent / result["scores"]["path"]).read_text())
    generic = next(c for c in result["comparison"]["controls"] if c["range"] == [0, 0])
    scores[generic["index"]] = 1
    calls = []

    class Pacer:
        def score(self, rows, directory):
            calls.append(rows)
            return {"scores": [scores[index] for index in indices]}

    monkeypatch.setattr(proof, "RecoveryPacer", Pacer)
    with pytest.raises(ValueError, match="rubric controls"):
        proof.prepare(plan_path, output, plan)
    assert len(calls) == 1
    assert not (output / "candidates.json").exists()


@pytest.mark.parametrize("failure", [None, "candidates", "qualify"])
def test_pipeline_waits_for_actual_child_exit_and_never_retries_or_qualifies_failed_rewards(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(proof, "ROOT", tmp_path)
    monkeypatch.setattr(proof, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(proof, "checked", lambda path: {"records": []})
    monkeypatch.setattr(prepare_terra_word_native, "guard", lambda output=None: None)
    monkeypatch.setattr(proof, "identity", lambda pid, command: {"pid": pid})
    path = tmp_path / "runs/plan.json"
    path.parent.mkdir()
    path.write_text("{}")
    output = tmp_path / "checkpoints/proof"
    events = []

    def child(command, check):
        mode = command[command.index("--mode") + 1]
        events.append(mode)
        assert check is False
        if mode == "qualify":
            assert events == ["candidates", "qualify"] and "--candidates" in command
            folder = Path(command[command.index("--output") + 1])
            folder.mkdir()
            proof.write(folder / "qualification.json", {"complete": True})
        return SimpleNamespace(returncode=1 if failure == mode else 0)

    def verify(*args):
        events.append("verify actual proof")

    monkeypatch.setattr(proof.subprocess, "run", child)
    monkeypatch.setattr(proof, "checked_qualification", verify)
    if failure is None:
        proof.pipeline(path, output)
        assert events == ["candidates", "qualify", "verify actual proof"]
        assert json.loads((output / "complete.json").read_text())["optimizer_updates"] == 0
    else:
        with pytest.raises(RuntimeError, match="no retry"):
            proof.pipeline(path, output)
        assert events == (["candidates"] if failure == "candidates" else ["candidates", "qualify"])
        assert not (output / "complete.json").exists()
    worker = json.loads((output / "worker.json").read_text())
    terminal = json.loads((output / "exit.json").read_text())
    assert worker["pid"] == terminal["pid"]
    assert terminal["exit_code"] == (1 if failure else 0)


@pytest.mark.parametrize(
    "mutation", [None, "duty", "cpu", "allocator", "idle", "nan", "zero_active", "no_memory"]
)
def test_every_resource_receipt_requires_the_original_policy(mutation):
    resource = {
        "current_allowed_gpu_duty_fraction": 0.5,
        "current_cpu_threads": 2,
        "memory_policy": "framework_default_no_artificial_six_GiB_cap",
        "mps_active_seconds": 1.0,
        "mps_idle_seconds": 1.0,
        "maximum_allocated_bytes": 1024,
        "maximum_driver_allocated_bytes": 2048,
    }
    if mutation:
        key, value = {
            "duty": ("current_allowed_gpu_duty_fraction", 1),
            "cpu": ("current_cpu_threads", 8),
            "allocator": ("memory_policy", "six_GiB_cap"),
            "idle": ("mps_idle_seconds", 0),
            "nan": ("mps_active_seconds", float("nan")),
            "zero_active": ("mps_active_seconds", 0),
            "no_memory": ("maximum_allocated_bytes", 0),
        }[mutation]
        resource[key] = value
    if mutation is None:
        proof.require_resources(resource)
    else:
        with pytest.raises(ValueError, match="allocator"):
            proof.require_resources(resource)


def test_incomplete_component_result_stops_before_word_inputs_or_any_model_load(
    tmp_path, monkeypatch
):
    component = tmp_path / "component.json"
    component.write_text("{}")
    result = tmp_path / "qualification.json"
    result.write_text(json.dumps({"complete": False, "corrected_first_case_passes": True}))
    monkeypatch.setattr(proof, "checked_component", lambda path: {})
    monkeypatch.setattr(proof, "terminal", lambda path: None)

    def forbidden():
        raise AssertionError("incomplete component reached word input/model preparation")

    monkeypatch.setattr(proof, "input_evidence", forbidden)
    with pytest.raises(ValueError, match="qualification"):
        proof.specification(component, result)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "count",
        "frames",
        "qualities",
        "gradient",
        "norm",
        "name",
        "resource",
        "fixed_values",
        "actual_file",
    ],
)
def test_full_proof_acceptance_rechecks_all_actual_cases_gradients_values_and_resources(
    candidate_fixture, monkeypatch, mutation
):
    import transformers

    plan_path, candidates, plan, tokenizer, judged, _, _ = candidate_fixture
    monkeypatch.setattr(
        transformers.ByT5Tokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    plan["reference"] = {"bundle_root": "artifacts/fixture"}
    plan["expected_visual_gradient_names"] = [f"vision.fixture{i}" for i in range(350)]
    resources = {
        "current_allowed_gpu_duty_fraction": 0.5,
        "current_cpu_threads": 2,
        "memory_policy": "framework_default_no_artificial_six_GiB_cap",
        "mps_active_seconds": 1.0,
        "mps_idle_seconds": 1.0,
        "maximum_allocated_bytes": 1024,
        "maximum_driver_allocated_bytes": 2048,
    }
    folder = candidates.parent.parent / "qualify"
    folder.mkdir()
    proof.write(folder / "worker.json", {"pid": 72})
    proof.write(folder / "exit.json", {"pid": 72, "exit_code": 0})
    reports = []
    for index, case in enumerate(plan["records"]):
        report = {
            "input_receipt": case["receipt"],
            "frames": 3,
            "actual_Terra_qualities": judged["qualities"][index],
            "CPU_risk": 0.5,
            "Metal_risk": 0.5,
            "resources": dict(resources),
            "comparison": {
                "loss_absolute_difference": 0,
                "logit_maximum_absolute_difference": 0,
                "total_gradient_norm_relative_error": 0,
                "gradient_differences": [
                    {
                        "parameter": name,
                        "relative_l2_error": 0,
                        "finite": True,
                        "cpu_norm": 1,
                        "actual_norm": 1,
                    }
                    for name in plan["expected_visual_gradient_names"]
                ],
            },
        }
        if index == 1:
            if mutation == "frames":
                report["frames"] = 2
            elif mutation == "qualities":
                report["actual_Terra_qualities"] = [10, 10]
            elif mutation == "gradient":
                report["comparison"]["gradient_differences"][-1]["relative_l2_error"] = 0.006
            elif mutation == "norm":
                report["comparison"]["gradient_differences"][-1]["actual_norm"] = 0
            elif mutation == "name":
                report["comparison"]["gradient_differences"][-1]["parameter"] = "vision.changed"
            elif mutation == "resource":
                report["resources"]["current_allowed_gpu_duty_fraction"] = 1
        target = folder / f"comparison-{index:02d}.json"
        proof.write(target, report)
        reports.append(proof.artifact(target))
    result = {
        "complete": True,
        "full_model_numerics_passed": True,
        "plan_sha256": proof.sha256(plan_path),
        "candidates": proof.artifact(candidates),
        "comparisons": reports,
        "vision_initialization_sha256": "original",
        "initial_reference_hashes": dict(plan["initial_reference_hashes"]),
        "trainable_tensors": 350,
        "trainable_parameters": 43260672,
        "pure_Terra_objective": True,
        "all_fixed_values_unchanged": True,
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "resources": resources,
        "model_win_claimed": False,
        "held_out_quality_evaluated": False,
        "promotion_authorized": False,
    }
    if mutation == "count":
        result["comparisons"].pop()
    if mutation == "fixed_values":
        result["initial_reference_hashes"]["language"] = "trained"
    path = folder / "qualification.json"
    proof.write(path, result)
    if mutation == "actual_file":
        (folder / "comparison-02.json").write_text("{}")
    if mutation is None:
        assert proof.checked_qualification(path, plan_path, plan)["complete"] is True
    else:
        with pytest.raises(ValueError):
            proof.checked_qualification(path, plan_path, plan)


@pytest.mark.parametrize("mutation", [None, "vision", "encoder", "language_scope", "frozen_visual"])
def test_model_value_and_scope_guard_with_small_independent_fixture(monkeypatch, mutation):
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.Linear(2, 2)
            self.semantic_encoder = nn.Linear(2, 2).requires_grad_(False)
            self.language = nn.Linear(2, 2).requires_grad_(False)

        def trainable_parameters(self):
            return {name: value for name, value in self.named_parameters() if value.requires_grad}

    student = Student()
    # This fixture tests value/scope rejection; actual C5 retains350 tensors/43M values.
    monkeypatch.setattr(model_module, "TENSORS", 2)
    monkeypatch.setattr(model_module, "PARAMETERS", 6)
    plan = {
        "initial_reference_hashes": model_module.fixed_hashes(student),
        "pretrained_state_sha256": model_module.state_digest(student.vision.state_dict()),
    }
    with torch.no_grad():
        if mutation == "vision":
            student.vision.weight.add_(1)
        elif mutation == "encoder":
            student.semantic_encoder.weight.add_(1)
    if mutation == "language_scope":
        student.language.bias.requires_grad_(True)
    if mutation == "frozen_visual":
        student.vision.bias.requires_grad_(False)
    if mutation is None:
        model_module.require_model(student, plan)
    else:
        with pytest.raises(ValueError, match="C5 values"):
            model_module.require_model(student, plan)


def test_unauthenticated_rewards_stop_before_any_Metal_work(tmp_path, monkeypatch):
    import c5_certification_budget

    class Budget:
        def __init__(self, path):
            pass

        def duty_fraction(self):
            return 0.5

        def cpu_threads(self):
            return 2

    events = []
    monkeypatch.setattr(c5_certification_budget, "CertificationComputeWindow", Budget)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        proof, "load_student", lambda plan: (events.append("CPU load") or object(), Tokenizer())
    )

    def rejected(*args):
        events.append("rejected reward authentication")
        raise ValueError("unauthenticated actual rewards")

    monkeypatch.setattr(proof, "checked_candidates", rejected)
    monkeypatch.setattr(
        proof, "load_clip", lambda case: pytest.fail("bad rewards reached input/Metal work")
    )
    with pytest.raises(ValueError, match="unauthenticated"):
        proof.qualify(
            tmp_path, tmp_path, {"resource_policy": "unused"}, tmp_path / "candidates.json"
        )
    assert events == ["CPU load", "rejected reward authentication"]
