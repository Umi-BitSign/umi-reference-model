from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from terra_training_recovery import compatible_plans, restore_checkpoint, restore_state
from train_tiny_rgb_feature_winner_full import optimizer_payload


def test_graceful_checkpoint_signal_only_sets_a_boundary_flag():
    import signal

    from terra_training_recovery import CheckpointRequest

    handlers = {}
    request = CheckpointRequest(install=lambda name, handler: handlers.update({name: handler}))
    assert not request.requested and set(handlers) == {signal.SIGUSR1}
    handlers[signal.SIGUSR1](signal.SIGUSR1, None)
    assert request.requested


def fixture_state():
    torch.manual_seed(317)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False)
    data = torch.randn(4, 3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(data).square().sum().backward()
        optimizer.step()
    parameters = dict(model.named_parameters())
    values, metadata = optimizer_payload(
        optimizer,
        parameters,
        {"cpu": torch.get_rng_state(), "mps": torch.tensor([1, 2, 3], dtype=torch.uint8)},
    )
    # The real saver serializes immediately. This fixture must also own its copies.
    return model, optimizer, data, {k: v.clone() for k, v in values.items()}, metadata


def test_optimizer_restore_has_identical_next_update_and_cpu_step_tensor():
    model, optimizer, data, values, metadata = fixture_state()
    recovered = copy.deepcopy(model)
    resumed = torch.optim.AdamW(recovered.parameters(), lr=3e-4, weight_decay=0.01, foreach=False)
    rng = []
    restore_state(
        dict(recovered.named_parameters()),
        resumed,
        values,
        metadata,
        updates=3,
        restore_rng=lambda cpu, mps: rng.append((cpu.clone(), mps.clone())),
    )
    assert len(rng) == 1 and torch.equal(rng[0][0], values["rng.cpu"])
    for instance, state in ((model, optimizer), (recovered, resumed)):
        state.zero_grad(set_to_none=True)
        instance(data).square().sum().backward()
        state.step()
    for expected, actual in zip(model.parameters(), recovered.parameters(), strict=True):
        assert torch.equal(expected, actual)
        assert resumed.state[actual]["step"].device.type == "cpu"
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(optimizer.state[expected][field], resumed.state[actual][field])


@pytest.mark.parametrize(
    "damage", ["step", "shape", "nan", "negative_variance", "order", "lr_recipe", "rng", "extra"]
)
def test_invalid_optimizer_state_is_rejected_before_optimizer_or_rng_mutation(damage):
    model, _, _, values, metadata = fixture_state()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False)
    rng = []
    if damage == "step":
        values["weight.step"] = torch.tensor(4.0)
    elif damage == "shape":
        values["weight.exp_avg"] = torch.zeros(1)
    elif damage == "nan":
        values["weight.exp_avg"].fill_(float("nan"))
    elif damage == "negative_variance":
        values["weight.exp_avg_sq"].fill_(-1)
    elif damage == "order":
        metadata["parameter_order"] = json.dumps(["bias", "weight"])
    elif damage == "lr_recipe":
        groups = json.loads(metadata["groups_without_parameters"])
        groups[0]["weight_decay"] = 0.9
        metadata["groups_without_parameters"] = json.dumps(groups)
    elif damage == "rng":
        values["rng.cpu"] = torch.zeros(2, dtype=torch.float32)
    else:
        values["unexpected"] = torch.zeros(1)
    with pytest.raises(ValueError):
        restore_state(
            dict(model.named_parameters()),
            optimizer,
            values,
            metadata,
            updates=3,
            restore_rng=lambda *args: rng.append(args),
        )
    assert not optimizer.state and not rng


def test_proof_reuse_allows_only_recovery_control_changes():
    before = {
        "recipe": {"alpha": 0.005},
        "evaluation": [1, 2],
        "source_sha256": {
            "tools/train_terra_visual.py": "old",
            "tools/terra_visual_learning.py": "qualified-learning",
            "tools/terra_quality_training.py": "qualified-judge-and-math",
        },
    }
    after = copy.deepcopy(before)
    after["source_sha256"]["tools/train_terra_visual.py"] = "new"
    after["source_sha256"]["tools/terra_training_recovery.py"] = "new-helper"
    after["resume"] = {"training_examples": 128}
    compatible_plans(before, after)
    for field in ("recipe", "evaluation", "source_sha256"):
        invalid = copy.deepcopy(after)
        if field == "source_sha256":
            invalid[field]["tools/terra_visual_learning.py"] = "different-math"
        else:
            invalid[field] = None
        with pytest.raises(ValueError):
            compatible_plans(before, invalid)


def test_complete_visual_optimizer_restore_checks_provenance_and_exact_next_step(tmp_path):
    from safetensors.torch import save_file

    model, optimizer, data, values, metadata = fixture_state()
    visual, opt, run = (
        tmp_path / name for name in ("visual.safetensors", "opt.safetensors", "run.json")
    )
    common = {"plan_sha256": "source-plan", "update": "24"}
    save_file(
        {name: p.detach().clone() for name, p in model.named_parameters()},
        str(visual),
        metadata={
            **common,
            "optimizer_updates": "3",
            "fixed_winner_checkpoint": "frozen-C5",
        },
    )
    save_file(values, str(opt), metadata={**common, **metadata})
    run.write_text(
        json.dumps({"initial_visual_sha256": "fresh", "fixed_backbones": {"language": "fixed"}})
    )
    plan = {
        "reference": {"translator_checkpoint_sha256": "frozen-C5"},
        "resume": {
            "source_plan": {"sha256": "source-plan"},
            "training_examples": 24,
            "optimizer_updates": 3,
        },
    }
    paths = {"visual": visual, "optimizer": opt, "source_run": run}
    recovered = torch.nn.Linear(3, 2)
    resumed = torch.optim.AdamW(recovered.parameters(), lr=3e-4, weight_decay=0.01, foreach=False)
    kwargs = {
        "initial_visual": "fresh",
        "fixed": {"language": "fixed"},
        "restore_rng": lambda *args: None,
    }
    assert restore_checkpoint(
        plan, paths, dict(recovered.named_parameters()), resumed, **kwargs
    ) == (24, 3)
    for instance, state in ((model, optimizer), (recovered, resumed)):
        state.zero_grad(set_to_none=True)
        instance(data).square().sum().backward()
        state.step()
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), recovered.parameters(), strict=True)
    )
    plan["resume"]["source_plan"]["sha256"] = "wrong-plan"
    with pytest.raises(ValueError, match="provenance"):
        restore_checkpoint(plan, paths, dict(recovered.named_parameters()), resumed, **kwargs)


def test_resume_corpus_offset_does_not_replay_previous_epochs():
    from terra_training_recovery import skip_epoch

    def remaining(length, epochs, start):
        result, skip = [], start
        for epoch in range(1, epochs + 1):
            offset, skip = skip_epoch(skip, length)
            result.extend((epoch, index) for index in range(offset, length))
        return result

    full = remaining(11, 4, 0)
    for start in (0, 8, 11, 19, 22, 43, 44):
        assert remaining(11, 4, start) == full[start:]


def test_streamed_rollouts_release_each_clip_and_preserve_all_ordered_seeds():
    from terra_training_recovery import streamed_pools

    class Clip:
        live = 0
        maximum_live = 0

        def __init__(self):
            Clip.live += 1
            Clip.maximum_live = max(Clip.live, Clip.maximum_live)

        def __del__(self):
            Clip.live -= 1

    pairs = [{"raw": {"source": "training", "example_id": str(i)}} for i in range(8)]
    events = []

    def prepare(pair):
        assert Clip.live == 0
        return Clip(), None

    result = streamed_pools(
        pairs,
        prepare=prepare,
        generate=lambda pair, video, mask, seed: (pair["raw"]["example_id"], seed),
        seed=317,
        emit=events.append,
    )
    assert result == [(str(i), 317 + i) for i in range(8)]
    assert Clip.maximum_live == 1 and Clip.live == 0
    assert [event["stage"] for event in events] == ["rollout_started", "rollout_complete"] * 8


def test_rollout_guard_failure_preserves_exact_clip_and_unchanged_limits():
    from evaluate_tiny_rgb_feature_winner import CPU_LIMITS
    from terra_training_recovery import streamed_pools

    pairs = [{"raw": {"source": "training", "example_id": str(i)}} for i in range(3)]
    events = []

    def generate(pair, video, mask, seed):
        if pair["raw"]["example_id"] == "1":
            raise RuntimeError("full-backbone CPU scoring exceeded its resource policy")
        return seed

    before = dict(CPU_LIMITS)
    with pytest.raises(RuntimeError, match="resource policy") as caught:
        streamed_pools(
            pairs,
            prepare=lambda pair: (None, None),
            generate=generate,
            seed=317,
            emit=events.append,
        )
    context = caught.value.terra_rollout_context
    assert context["example_id"] == "1" and context["seed"] == 318
    assert context["enforced_cpu_limits"] == before == CPU_LIMITS
    assert events[-1]["stage"] == "rollout_failed" and len(events) == 4


@pytest.mark.parametrize("stage", ["rollout", "terra_scoring"])
def test_preupdate_failure_checkpoints_completed_updates_and_never_retries(tmp_path, stage):
    from terra_training_recovery import record_preupdate_failure

    saves = []

    def save():
        # Failure evidence must exist even if checkpoint serialization itself fails.
        before = json.loads((tmp_path / "preupdate-stop.json").read_text())
        assert before["optimizer_updates"] == 100
        saves.append(1)
        return "checkpoints/recovery-at-800.safetensors"

    result = record_preupdate_failure(
        tmp_path,
        stage=stage,
        error=RuntimeError("injected"),
        examples=800,
        updates=100,
        save=save,
    )
    assert saves == [1] and result["failed_batch_optimizer_applied"] is False
    assert result["failed_batch"] == 101 and result["automatic_retries"] == 0
    assert (tmp_path / "scoring-stop.json").exists() == (stage == "terra_scoring")


def test_checkpoint_pair_has_complete_hash_receipt_and_never_overwrites(monkeypatch, tmp_path):
    from benchmark_fleurs_dev import sha256
    from terra_training_recovery import save_checkpoint

    model, optimizer, _, _, _ = fixture_state()
    model.trainable_parameters = lambda: dict(model.named_parameters())
    output = tmp_path / "checkpoints" / "terra-test"
    output.mkdir(parents=True)
    plan_path = tmp_path / "runs" / "terra-test" / "plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("{}\n")
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: torch.tensor([1, 2], dtype=torch.uint8))
    monkeypatch.setattr("terra_training_storage.storage", lambda *args, **kwargs: None)
    checkpoint = save_checkpoint(
        model,
        optimizer,
        output,
        plan_path,
        "fixed-C5",
        24,
        3,
        recovery=True,
        root=tmp_path,
    )
    result = json.loads(checkpoint.with_suffix(".json").read_text())
    assert result["complete"] is True and result["training_examples"] == 24
    assert result["visual"]["sha256"] == sha256(checkpoint)
    assert result["optimizer"]["sha256"] == sha256(tmp_path / result["optimizer"]["path"])
    with pytest.raises(ValueError, match="overwrites"):
        save_checkpoint(
            model,
            optimizer,
            output,
            plan_path,
            "fixed-C5",
            24,
            3,
            recovery=True,
            root=tmp_path,
        )
