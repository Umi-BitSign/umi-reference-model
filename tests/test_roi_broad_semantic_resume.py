from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import roi_broad_semantic_resume as recovery
import roi_semantic_resume as shared_recovery
from benchmark_fleurs_dev import sha256
from probe_tiny_rgb_capacity import state_digest
from terra_training_recovery import artifact as bound_artifact
from train_tiny_rgb_feature_winner_full import optimizer_payload


def prefix():
    batches = [
        {"stage": "short", "epoch": 0, "training_record_indices": [0, 1, 2, 3]},
        {"stage": "short", "epoch": 0, "training_record_indices": [4, 5, 6]},
        {"stage": "short", "epoch": 1, "training_record_indices": [2, 1, 0, 3]},
    ]
    steps = []
    for i, b in enumerate(batches[:2]):
        steps.append(
            {
                **copy.deepcopy(b),
                "batch": i,
                "Terra_qualities": [[10, 0]] * len(b["training_record_indices"]),
                "contrasting_clips": len(b["training_record_indices"]),
                "gradient_norm_before_clip": 0.4,
                "training_examples": [4, 7][i],
                "optimizer_updates": i + 1,
            }
        )
    return {"batches": batches}, steps


def test_prefix_counts_partial_batches_instead_of_assuming_updates_times_four():
    plan, steps = prefix()
    assert recovery.prefix_state(plan, steps) == (7, 2)
    steps[1]["Terra_qualities"] = [[10, 10]] * 3
    steps[1].update(contrasting_clips=0, optimizer_updates=1, gradient_norm_before_clip=None)
    assert recovery.prefix_state(plan, steps) == (7, 1)


@pytest.mark.parametrize("mutation", ["empty", "full", "order", "counter", "quality", "gradient"])
def test_prefix_requires_exact_unfinished_calibrated_completed_work(mutation):
    plan, steps = prefix()
    if mutation == "empty":
        steps = []
    elif mutation == "full":
        plan["batches"].pop()
    elif mutation == "order":
        steps[0]["training_record_indices"] = [1, 0, 2, 3]
    elif mutation == "counter":
        steps[1]["training_examples"] = 8
    elif mutation == "quality":
        steps[1]["Terra_qualities"][0][0] = 9
    else:
        steps[0]["gradient_norm_before_clip"] = 0
    with pytest.raises(ValueError):
        recovery.prefix_state(plan, steps)


def test_compatibility_allows_only_two_recovery_control_sources():
    before = {
        "recipe": {"risk_alpha": 0.005},
        "batches": ["same"],
        "model_plan": {"fixed": True},
        "source_sha256": {
            "tools/train_roi_broad_semantic.py": "old",
            "tools/roi_semantic_student.py": "same",
        },
    }
    after = copy.deepcopy(before)
    after["source_sha256"].update(
        {"tools/train_roi_broad_semantic.py": "new", "tools/roi_broad_semantic_resume.py": "new"}
    )
    after["resume"] = {"training_examples": 111}
    recovery.compatible(before, after)
    for field in ("recipe", "batches", "model_plan"):
        changed = copy.deepcopy(after)
        changed[field] = None
        with pytest.raises(ValueError):
            recovery.compatible(before, changed)
    after["source_sha256"]["tools/roi_semantic_student.py"] = "changed"
    with pytest.raises(ValueError, match="scientific"):
        recovery.compatible(before, after)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    plan, steps = prefix()
    plan.update(
        {
            "source_sha256": {"tools/train_roi_broad_semantic.py": "old"},
            "recipe": {"learning_rate": 3e-5},
            "vision_initialization_sha256": "origin",
            "model_plan": {
                "initial_reference_hashes": {"encoder": "encoder", "language": "language"},
                "reference": {"translator_checkpoint_sha256": "C5"},
                "policy": {"trainable_tensors": 2},
            },
        }
    )
    source = tmp_path / "runs/source/plan.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(plan))
    output = tmp_path / "checkpoints/source"
    output.mkdir(parents=True)
    model = torch.nn.Module()
    model.vision = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, foreach=False)
    x = torch.ones(2, 3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model.vision(x).square().sum().backward()
        optimizer.step()
    visual = output / "visual-recovery-after-000007.safetensors"
    state = visual.with_name(visual.stem + "-optimizer.safetensors")
    provenance = {"plan_sha256": sha256(source), "update": "7"}
    save_file(
        {n: p.detach().clone() for n, p in model.named_parameters()},
        visual,
        metadata={**provenance, "optimizer_updates": "2", "fixed_winner_checkpoint": "C5"},
    )
    values, metadata = optimizer_payload(
        optimizer,
        dict(model.named_parameters()),
        {
            "cpu": torch.get_rng_state(),
            "mps": torch.tensor([1, 2, 3], dtype=torch.uint8),
        },
    )
    save_file(values, state, metadata={**provenance, **metadata})
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "artifact", lambda path: bound_artifact(path, root=tmp_path))
    monkeypatch.setattr(recovery, "verify_snapshot", lambda *args: None)
    commit = {
        "complete": True,
        "training_examples": 7,
        "optimizer_updates": 2,
        "visual": recovery.artifact(visual),
        "optimizer": recovery.artifact(state),
    }
    (output / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in steps) + "\n")
    visual.with_suffix(".json").write_text(json.dumps(commit))
    (output / "worker.json").write_text(json.dumps({"pid": 1}))
    (output / "exit.json").write_text(json.dumps({"pid": 1, "exit_code": 1}))
    (output / "stopped.json").write_text(
        json.dumps(
            {
                "error": "ROI training judge failed original calibration controls",
                "automatic_retries": 0,
                "training_examples": 7,
                "optimizer_updates": 2,
                "recovery": recovery.artifact(visual),
            }
        )
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "plan_sha256": sha256(source),
                "initial_visual_sha256": "origin",
                "fixed_backbones": plan["model_plan"]["initial_reference_hashes"],
                "recipe": plan["recipe"],
            }
        )
    )
    current = copy.deepcopy(plan)
    current["source_sha256"].update(
        {"tools/train_roi_broad_semantic.py": "new", "tools/roi_broad_semantic_resume.py": "new"}
    )
    return source, visual, current, model


def test_recovery_descriptor_binds_full_visual_optimizer_rng_prefix_and_actual_start(saved):
    source, visual, current, model = saved
    descriptor = recovery.make_resume(source, visual, current)
    assert descriptor["completed_batches"] == 2
    assert descriptor["training_examples"] == 7
    assert descriptor["optimizer_updates"] == 2
    assert descriptor["starting_visual_state_sha256"] == state_digest(model.vision.state_dict())
    assert descriptor["failed_batch_grades_discarded"] is True
    assert descriptor["manual_continuation_limit"] == 1
    assert recovery.restore is shared_recovery.restore


@pytest.mark.parametrize(
    "file,field,value",
    [
        ("exit.json", "exit_code", 0),
        ("exit.json", "pid", 2),
        ("stopped.json", "error", "another failure"),
        ("stopped.json", "automatic_retries", 1),
        ("stopped.json", "training_examples", 8),
        ("run.json", "initial_visual_sha256", "other"),
    ],
)
def test_descriptor_rejects_live_wrong_stop_wrong_counter_or_wrong_initialization(
    saved, file, field, value
):
    source, visual, current, _ = saved
    target = visual.parent / file
    record = json.loads(target.read_text())
    record[field] = value
    target.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        recovery.make_resume(source, visual, current)


def test_a_second_manual_continuation_is_rejected_before_any_restore(saved):
    source, visual, current, _ = saved
    previous = json.loads(source.read_text())
    previous["resume"] = {"training_examples": 4}
    source.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="retry chain"):
        recovery.make_resume(source, visual, current)


def test_numerical_training_loop_and_qualification_are_identical_to_original_archived_arm():
    import train_roi_broad_semantic as training

    tree = ast.parse(Path(training.__file__).read_text())
    # Normalize only empty Python3.12+ generic-parameter fields. Reference hashes
    # come from the immutable source archive parsed by the pinned Python3.10 runtime.
    for node in ast.walk(tree):
        if "type_params" in node._fields:
            assert not node.type_params
            node._fields = tuple(field for field in node._fields if field != "type_params")
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    expected = {
        "batches_for": "33a488dad116c001f47b60edd8f86f895fd678c48604cd0922235a52fc5749a2",
        "verify_coverage": "c06c1b272483ebca92444c558d0acd34a4391565a831d061180718c17a6ffd13",
        "qualified": "5fea2e8831bbc8ee2da0d00447883e2befb9eab27de8a54e340fd4ce0a83c05b",
        "run": "5f4208c1fd0f4020692d58e727b3e050248ae39c07e61be30c5a014c09cb8528",
    }
    for name, digest in expected.items():
        assert (
            hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest()
            == digest
        )
    loop = next(
        n
        for n in ast.walk(functions["fit"])
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Tuple)
        and [p.id for p in n.target.elts] == ["batch_number", "batch"]
    )
    assert ast.unparse(loop.body[0]) == "if batch_number < start_batch:\n    continue"
    assert (
        hashlib.sha256(
            ast.dump(
                ast.Module(body=loop.body[1:], type_ignores=[]), include_attributes=False
            ).encode()
        ).hexdigest()
        == "58254e02f9c96fd5538231a948bd209a6d513b21ceb2576c4c1b60d94b925872"
    )
