from __future__ import annotations

import copy
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_word_training as word
import train_c5_word_semantic as trainer
from probe_jeff_terra_quality import terra_payload
from terra_training_recovery import artifact
from terra_word_semantic_protocol import training_negative


class Tokenizer:
    def __call__(self, text, truncation):
        assert truncation is False
        return SimpleNamespace(input_ids=[ord(character) + 3 for character in text] + [1])

    def decode(self, tokens, **kwargs):
        return "".join(chr(value - 3) for value in tokens if value > 2)


def cases(count=214):
    return [
        {
            "record": {
                "original": {
                    "row": {
                        "example_id": str(index),
                        "reference": f"Word{index % 60}",
                        "label": f"word{index % 60}",
                        "participant": ("P1", "P2", "P3", "P5")[(index // 60) % 4],
                        "role": "training",
                        "training_allowed": True,
                    }
                },
                "ROI_lineage_training_allowed": True,
                "normalized_probe": {
                    "decoded_frames": (57 if index < 163 else 56) * 2 - 1,
                    "width": 640,
                    "height": 480,
                },
            },
            "receipt": {"path": f"checkpoints/inputs/{index}.json", "sha256": str(index)},
        }
        for index in range(count)
    ]


def pools_for(records, tokenizer):
    donors = [records[index] for index in word.donor_indices(records)]
    pools = []
    for case, donor in zip(records, donors, strict=True):
        text = word.training_case(case)["reference"]
        pools.append(
            [
                {
                    "tokens": tokenizer(text, truncation=False).input_ids,
                    "text": text,
                    "kind": "human_anchor",
                    "truncated": False,
                    "termination": "eos",
                },
                training_negative(case, donor, tokenizer),
            ]
        )
    return donors, pools


class FakePacer:
    """Local response fixture; never sends requests or qualifies a real model."""

    def __init__(self, *, fail_call=None, tie_calls=()):
        self.calls = 0
        self.fail_call, self.tie_calls = fail_call, tie_calls

    def score(self, rows, directory):
        self.calls += 1
        directory.mkdir()
        request = terra_payload(rows)
        word.write(directory / "request.json", request)
        values = []
        controls = {
            (row["reference"], row["candidate"]): row["range"][0]
            for row in word.calibration_controls()
        }
        for row in rows:
            key = row["reference"], row["candidate"]
            if key in controls:
                score = controls[key]
                if self.calls == self.fail_call and controls[key] == 0:
                    score = 9
            else:
                score = 10 if row["reference"] == row["candidate"] else 3
                if self.calls in self.tie_calls:
                    score = 10
            values.append(score)
        payload = {
            "scores": [{"id": f"pair_{i:02d}", "score": value} for i, value in enumerate(values)]
        }
        response = {
            "model": "gpt-5.6-terra",
            "reasoning": {"effort": "medium"},
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(payload)}],
                }
            ],
        }
        word.write(
            directory / "started.json",
            {
                "request_sha256": word.sha256(directory / "request.json"),
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "pairs": len(rows),
                "automatic_retries": 0,
            },
        )
        word.write(directory / "response.json", response)
        word.write(
            directory / "scores.json",
            {
                "response_sha256": word.sha256(directory / "response.json"),
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "scores": values,
            },
        )
        return {"scores": [999] * len(rows)}  # Authenticate actual files, not this return.


def overwrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_complete_inventory_and_four_full_epochs():
    records = cases()
    assert word.training_records({"records": records}) == records
    batches = word.batches_for(records)
    assert len(batches) == 216
    assert sum(len(batch["training_record_indices"]) for batch in batches) == 856
    for epoch in range(4):
        selected = [
            batch["training_record_indices"] for batch in batches if batch["epoch"] == epoch
        ]
        assert len(selected) == 54 and len(selected[-1]) == 2
        assert sorted(i for batch in selected for i in batch) == list(range(214))
    assert word.batches_for(records) == batches
    assert batches[:54] != batches[54:108]
    observed = [i for batch in batches for i in batch["training_record_indices"]]
    plan = {"batches": batches, "records": records, "planned_training_examples": 856}
    word.verify_coverage(plan, observed, 856, ["final"])
    for order, count, saved in (
        (observed[:-1], 855, ["partial"]),
        (observed[::-1], 856, ["final"]),
        (observed, 856, []),
    ):
        with pytest.raises(ValueError):
            word.verify_coverage(plan, order, count, saved)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(copy.deepcopy(rows[0])),
        lambda rows: rows[0]["record"]["original"]["row"].update(participant="P4"),
        lambda rows: rows[0]["record"]["original"]["row"].update(participant="P19"),
        lambda rows: rows[0]["record"]["original"]["row"].update(training_allowed=False),
        lambda rows: rows[0]["record"].update(ROI_lineage_training_allowed=False),
        lambda rows: rows[0]["record"]["normalized_probe"].update(decoded_frames=1),
    ],
)
def test_inventory_rejects_missing_duplicate_reserved_or_changed_inputs(mutation):
    records = cases()
    mutation(records)
    with pytest.raises(ValueError):
        word.training_records({"records": records})


def test_donors_skip_same_label_without_using_scores():
    records = cases(4)
    records[1]["record"]["original"]["row"].update(label="word0", reference="Word0")
    assert word.donor_indices(records) == [2, 2, 3, 0]
    records[2]["record"]["original"]["row"]["label"] = "word0"
    records[3]["record"]["original"]["row"]["label"] = "word0"
    with pytest.raises(ValueError):
        word.donor_indices(records)


def test_duplicate_donor_rollout_is_preserved_without_substitution():
    records, tokenizer = cases(3), Tokenizer()
    donors, pools = pools_for(records, tokenizer)
    donor = pools[0][-1]
    pools[0][-1] = {key: value for key, value in donor.items() if key != "donor_input_receipt"}
    pools[0][-1]["kind"] = "greedy"
    word.paired_inputs(records, donors, pools, tokenizer)
    pools[0][-1]["text"] = "tampered"
    with pytest.raises(ValueError):
        word.paired_inputs(records, donors, pools, tokenizer)


@pytest.fixture
def scored_batch(tmp_path):
    records, tokenizer = cases(4), Tokenizer()
    donors, pools = pools_for(records, tokenizer)
    paired = word.paired_inputs(records, donors, pools, tokenizer)
    directory = tmp_path / "batch"
    directory.mkdir()
    pacer = FakePacer()
    rewards = word.score_batch(records, donors, pools, tokenizer, directory, pacer, 7)
    assert rewards == [[10, 3]] * 4
    assert pacer.calls == 1
    return paired, directory


@pytest.mark.parametrize(
    "filename,change",
    [
        ("request.json", lambda value: value.update(model="fallback")),
        ("started.json", lambda value: value.update(request_sha256="wrong")),
        ("started.json", lambda value: value.update(automatic_retries=1)),
        ("response.json", lambda value: value.update(model="fallback")),
        ("response.json", lambda value: value.update(reasoning={"effort": "low"})),
        ("response.json", lambda value: value.update(status="incomplete")),
        ("scores.json", lambda value: value.update(response_sha256="wrong")),
        ("scores.json", lambda value: value["scores"].__setitem__(0, 9)),
    ],
)
def test_rewards_reject_actual_request_response_and_score_tampering(scored_batch, filename, change):
    paired, directory = scored_batch
    overwrite(directory / "request-000" / filename, change)
    with pytest.raises(ValueError):
        word.verified_rewards(paired, directory, 7)


def test_rewards_reject_changed_pair_mapping_or_missing_request(scored_batch):
    paired, directory = scored_batch
    changed = copy.deepcopy(paired)
    changed["cases"][0]["indices"] = [0, 0]
    with pytest.raises(ValueError):
        word.verified_rewards(changed, directory, 7)
    (directory / "request-extra").mkdir()
    with pytest.raises(ValueError):
        word.verified_rewards(paired, directory, 7)


def test_failed_calibration_preserves_raw_evidence_without_rewards(tmp_path):
    records, tokenizer = cases(4), Tokenizer()
    donors, pools = pools_for(records, tokenizer)
    pacer = FakePacer(fail_call=1)
    with pytest.raises(ValueError, match="calibration"):
        word.score_batch(records, donors, pools, tokenizer, tmp_path, pacer, 7)
    assert (tmp_path / "request-000/response.json").exists()
    assert not (tmp_path / "rewards.json").exists()
    assert pacer.calls == 1


def test_storage_keeps_eighty_GiB_floor_and_reserves_full_output(tmp_path):
    exact = sum(
        word.STORAGE[key]
        for key in ("physical_floor_bytes", "output_budget_bytes", "headroom_bytes")
    )
    word.storage(root=tmp_path, disk_usage=lambda root: SimpleNamespace(free=exact))
    with pytest.raises(RuntimeError):
        word.storage(root=tmp_path, disk_usage=lambda root: SimpleNamespace(free=exact - 1))
    with pytest.raises(RuntimeError):
        word.storage(
            word.STORAGE["output_budget_bytes"] + 1,
            root=tmp_path,
            disk_usage=lambda root: SimpleNamespace(free=exact),
        )
    output = tmp_path / "checkpoints/run"
    output.mkdir(parents=True)
    (output / "alias").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        word.storage(
            output=output, root=tmp_path, disk_usage=lambda root: SimpleNamespace(free=exact)
        )


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(2, 2)
        self.encoder = nn.Linear(2, 2).requires_grad_(False)

    def trainable_parameters(self):
        return {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def to(self, device):
        assert device == "mps"
        return self  # Control-flow fixture only; no Metal numerical assertion.


def test_actual_optimizer_guard_updates_only_selected_visual_values(monkeypatch):
    monkeypatch.setattr(word, "TENSORS", 2)
    model = TinyStudent()
    fixed = model.encoder.weight.detach().clone()
    before = model.vision.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.trainable_parameters().values(), lr=1e-5)
    (model.vision.weight.square().sum() + model.vision.bias.square().sum()).backward()
    norm = word.update_visual(model, optimizer, model.trainable_parameters())
    assert norm > 0 and not torch.equal(before, model.vision.weight)
    assert torch.equal(fixed, model.encoder.weight)
    assert all(parameter.grad is None for parameter in model.parameters())
    before = model.vision.weight.detach().clone()
    with pytest.raises(RuntimeError):
        word.update_visual(model, optimizer, model.trainable_parameters())
    assert torch.equal(before, model.vision.weight)


@pytest.fixture
def proof_gate(tmp_path, monkeypatch):
    root = tmp_path
    path = root / "runs/proof/plan.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    result = root / "checkpoints/proof/qualify/qualification.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}")
    descriptor = partial(artifact, root=root)
    monkeypatch.setattr(trainer, "PROOF_PLAN", path)
    monkeypatch.setattr(trainer, "PROOF", result)
    monkeypatch.setattr(trainer, "artifact", descriptor)
    monkeypatch.setattr(trainer, "checked_proof_plan", lambda path: {"real": "plan"})
    calls = []
    monkeypatch.setattr(
        trainer, "checked_qualification", lambda *args: calls.append(args) or {"actual": "proof"}
    )
    parent = result.parent.parent
    word.write(parent / "worker.json", {"pid": 123})
    word.write(parent / "exit.json", {"pid": 123, "exit_code": 0})
    complete = {
        "complete": True,
        "plan": descriptor(path),
        "qualification": descriptor(result),
        "optimizer_updates": 0,
        "automatic_retries": 0,
        "promotion_authorized": False,
    }
    word.write(parent / "complete.json", complete)
    return parent, calls


def test_optimizer_requires_actual_qualification_validator_and_parent(proof_gate):
    _, calls = proof_gate
    assert trainer.qualified() == ({"real": "plan"}, {"actual": "proof"})
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda parent: (parent / "complete.json").unlink(),
        lambda parent: overwrite(parent / "exit.json", lambda value: value.update(exit_code=1)),
        lambda parent: overwrite(parent / "exit.json", lambda value: value.update(pid=999)),
        lambda parent: overwrite(
            parent / "complete.json", lambda value: value.update(complete=False)
        ),
        lambda parent: overwrite(
            parent / "complete.json", lambda value: value.update(optimizer_updates=1)
        ),
    ],
)
def test_incomplete_failed_or_mismatched_proof_cannot_authorize_training(proof_gate, mutation):
    parent, _ = proof_gate
    mutation(parent)
    with pytest.raises((ValueError, FileNotFoundError)):
        trainer.qualified()


def test_freeze_without_qualification_creates_no_plan_or_optimizer(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    monkeypatch.setattr(trainer, "qualified", lambda: (_ for _ in ()).throw(ValueError("pending")))
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("optimizer constructed"))
    target = tmp_path / "runs/training/plan.json"
    with pytest.raises(ValueError, match="pending"):
        trainer.freeze(target)
    assert not target.parent.exists()


class FakeBudget:
    def __init__(self, path):
        pass

    def cpu_threads(self):
        return 2

    def duty_fraction(self):
        return 0.5


class FakeDuty:
    def __init__(self, *args):
        pass

    def run(self, function):
        return function()

    def receipt(self):
        return {"current_allowed_gpu_duty_fraction": 0.5, "current_cpu_threads": 2}


@pytest.fixture
def tiny_training(tmp_path, monkeypatch):
    import c5_certification_budget
    import terra_visual_learning

    monkeypatch.setattr(c5_certification_budget, "CertificationComputeWindow", FakeBudget)
    monkeypatch.setattr(c5_certification_budget, "WinnerDutyCycle", FakeDuty)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "manual_seed", lambda seed: None)
    monkeypatch.setattr(word, "TENSORS", 2)
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(word, "artifact", descriptor)
    monkeypatch.setattr(trainer, "artifact", descriptor)
    monkeypatch.setattr(word, "storage", lambda *a, **k: {})
    monkeypatch.setattr(trainer, "storage", lambda *a, **k: {})
    template = TinyStudent()
    models, optimizers = [], []

    def load_student(plan):
        model = copy.deepcopy(template)
        models.append(model)
        return model, Tokenizer()

    real_adam = torch.optim.AdamW

    def adam(*args, **kwargs):
        optimizer = real_adam(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(trainer, "load_student", load_student)
    monkeypatch.setattr(torch.optim, "AdamW", adam)
    monkeypatch.setattr(trainer, "mask_digest", lambda model: "fixed-mask")
    monkeypatch.setattr(
        trainer,
        "fixed_hashes",
        lambda model: {"encoder": trainer.state_digest(model.encoder.state_dict())},
    )
    records = cases(6)
    batches = word.batches_for(records)
    plan = {
        "records": records,
        "batches": batches,
        "donor_indices": word.donor_indices(records),
        "planned_training_examples": 24,
        "vision_initialization_sha256": trainer.state_digest(template.vision.state_dict()),
        "model_plan": {
            "resource_policy": "runs/resource.json",
            "reference": {"translator_checkpoint_sha256": "original"},
            "initial_reference_hashes": trainer.fixed_hashes(template),
            "expected_visual_gradient_names": list(template.trainable_parameters()),
        },
    }
    path = tmp_path / "runs/train/plan.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan))
    output = tmp_path / "checkpoints/train"
    output.mkdir(parents=True)
    monkeypatch.setattr(trainer, "checked", lambda path: plan)
    monkeypatch.setattr(
        trainer, "load_clip", lambda case: (torch.zeros(1, 2), torch.ones(1, 1, dtype=torch.bool))
    )
    history, divisors = [], []

    def rng_states():
        state = {
            "cpu": torch.get_rng_state().clone(),
            "mps": torch.tensor([1, 2], dtype=torch.uint8),
        }
        history.append(state)
        return state

    monkeypatch.setattr(trainer, "rng_states", rng_states)

    def candidate_pool(student, tokenizer, clip, mask, reference, **kwargs):
        assert kwargs["maximum_tokens"] == 512
        torch.rand(1)  # Failed rollouts must not replace the last committed RNG snapshot.
        return [
            {
                "tokens": tokenizer(reference, truncation=False).input_ids,
                "text": reference,
                "kind": "human_anchor",
                "termination": "eos",
                "truncated": False,
            }
        ]

    monkeypatch.setattr(terra_visual_learning, "candidate_pool", candidate_pool)

    def backward(model, clip, mask, labels, qualities, *, alpha, caption_weight, divisor):
        assert alpha == 0.005 and caption_weight == 0 and qualities == [10, 3]
        divisors.append(divisor)
        value = (model.vision.weight.square().sum() + model.vision.bias.square().sum()) / divisor
        value.backward()
        return {"expected_terra_error": 3.5, "gold_logits": torch.zeros(1)}

    monkeypatch.setattr(terra_visual_learning, "backward_candidate_risk", backward)
    pacer = FakePacer()
    monkeypatch.setattr(trainer, "RecoveryPacer", lambda: pacer)
    return SimpleNamespace(
        plan=plan,
        path=path,
        output=output,
        pacer=pacer,
        history=history,
        divisors=divisors,
        template=template,
        models=models,
        optimizers=optimizers,
    )


def test_tiny_training_joins_all_epochs_tail_divisors_and_real_checkpoints(tiny_training):
    from safetensors.torch import load_file

    env = tiny_training
    trainer.fit(env.path, env.output, env.plan)
    result = json.loads((env.output / "result.json").read_text())
    assert result["training_examples"] == 24 and result["optimizer_updates"] == 8
    assert len(result["checkpoints"]) == 4
    assert set(env.divisors) == {2, 4}
    steps = [json.loads(line) for line in (env.output / "steps.jsonl").read_text().splitlines()]
    assert [i for step in steps for i in step["training_record_indices"]] == [
        i for batch in env.plan["batches"] for i in batch["training_record_indices"]
    ]
    assert len(list(env.output.glob("batch-*/rewards.json"))) == 8
    checkpoint = env.output / "visual-after-000024.safetensors"
    values = load_file(str(checkpoint))
    assert set(values) == set(env.template.trainable_parameters())
    assert not torch.equal(values["vision.weight"], env.template.vision.weight)
    optimizer = load_file(str(checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")))
    assert optimizer["vision.weight.step"].item() == 8
    assert torch.equal(optimizer["rng.cpu"], env.history[-1]["cpu"])
    commit = json.loads(checkpoint.with_suffix(".json").read_text())
    assert commit["visual"] == trainer.artifact(checkpoint)
    assert commit["committed_batch_RNG"] is True


def test_tiny_training_skips_tied_batch_without_losing_inventory(tiny_training):
    env = tiny_training
    env.pacer.tie_calls = (1,)
    trainer.fit(env.path, env.output, env.plan)
    result = json.loads((env.output / "result.json").read_text())
    assert result["training_examples"] == 24 and result["optimizer_updates"] == 7
    first = json.loads((env.output / "steps.jsonl").read_text().splitlines()[0])
    assert first["contrasting_clips"] == 0 and first["optimizer_updates"] == 0
    assert all(row["expected_terra_error"] == 0 for row in first["before_update_risk"])


def test_tiny_training_calibration_stop_keeps_exact_committed_prefix_and_rng(tiny_training):
    from safetensors.torch import load_file

    env = tiny_training
    env.pacer.fail_call = 2
    with pytest.raises(ValueError, match="calibration"):
        trainer.fit(env.path, env.output, env.plan)
    stop = json.loads((env.output / "stopped.json").read_text())
    assert stop["training_examples"] == 4 and stop["optimizer_updates"] == 1
    assert stop["recovery_is_committed_prefix"] is True
    assert not stop["uncommitted_optimizer_mutation_possible"]
    assert not (env.output / "result.json").exists()
    assert len(env.divisors) == 4 and set(env.divisors) == {4}
    checkpoint = env.output / "visual-recovery-after-000004.safetensors"
    optimizer = load_file(str(checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")))
    assert torch.equal(optimizer["rng.cpu"], env.history[-1]["cpu"])
    assert not torch.equal(optimizer["rng.cpu"], torch.get_rng_state())
    assert optimizer["vision.weight.step"].item() == 1
    assert env.pacer.calls == 2  # The failed request is never retried.


def test_tiny_training_never_labels_uncommitted_update_as_recovery(tiny_training, monkeypatch):
    env = tiny_training
    real_update = trainer.update_visual

    def fail_after_mutation(*args, **kwargs):
        real_update(*args, **kwargs)
        raise RuntimeError("injected after optimizer mutation")

    monkeypatch.setattr(trainer, "update_visual", fail_after_mutation)
    with pytest.raises(RuntimeError, match="injected"):
        trainer.fit(env.path, env.output, env.plan)
    stop = json.loads((env.output / "stopped.json").read_text())
    assert stop["recovery"] is None and stop["uncommitted_optimizer_mutation_possible"]
    assert not list(env.output.glob("visual-*.safetensors"))


def test_tiny_training_preserves_original_error_when_recovery_save_fails(
    tiny_training, monkeypatch
):
    env = tiny_training
    env.pacer.fail_call = 2
    monkeypatch.setattr(
        trainer,
        "save_checkpoint",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("checkpoint unavailable")),
    )
    with pytest.raises(ValueError, match="calibration"):
        trainer.fit(env.path, env.output, env.plan)
    stop = json.loads((env.output / "stopped.json").read_text())
    assert "calibration" in stop["error"]
    assert stop["recovery"] is None and stop["recovery_error"] == "checkpoint unavailable"


def test_tiny_checkpoint_reproduces_the_next_actual_adam_update(tiny_training):
    from safetensors import safe_open
    from safetensors.torch import load_file
    from terra_training_recovery import restore_state

    env = tiny_training
    trainer.fit(env.path, env.output, env.plan)
    live, live_optimizer = env.models[0], env.optimizers[0]
    path = env.output / "visual-after-000024.safetensors"
    recovered = copy.deepcopy(live)
    saved = load_file(str(path))
    with torch.no_grad():
        for name, parameter in recovered.trainable_parameters().items():
            parameter.copy_(saved[name])
    optimizer = torch.optim.AdamW(
        recovered.trainable_parameters().values(),
        lr=word.RECIPE["learning_rate"],
        weight_decay=word.RECIPE["weight_decay"],
        foreach=False,
    )
    optimizer_path = path.with_name(path.stem + "-optimizer.safetensors")
    with safe_open(str(optimizer_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    restored_rng = []
    restore_state(
        recovered.trainable_parameters(),
        optimizer,
        load_file(str(optimizer_path)),
        metadata,
        updates=8,
        restore_rng=lambda *values: restored_rng.extend(values),
    )
    assert torch.equal(restored_rng[0], env.history[-1]["cpu"])
    for model, current_optimizer in ((live, live_optimizer), (recovered, optimizer)):
        (model.vision.weight.square().sum() + model.vision.bias.square().sum()).backward()
        current_optimizer.step()
    for name, parameter in live.trainable_parameters().items():
        torch.testing.assert_close(
            parameter, recovered.trainable_parameters()[name], rtol=0, atol=0
        )


def test_tiny_gradient_guard_failure_preserves_previous_committed_weights(
    tiny_training, monkeypatch
):
    env = tiny_training
    real_update = trainer.update_visual
    calls = 0

    def fail_before_second_mutation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected gradient guard")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(trainer, "update_visual", fail_before_second_mutation)
    with pytest.raises(RuntimeError, match="gradient guard"):
        trainer.fit(env.path, env.output, env.plan)
    stop = json.loads((env.output / "stopped.json").read_text())
    assert stop["training_examples"] == 4 and stop["optimizer_updates"] == 1
    assert stop["recovery_is_committed_prefix"] is True
    assert not stop["uncommitted_optimizer_mutation_possible"]
