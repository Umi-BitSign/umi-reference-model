from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import roi_semantic_resume as recovery
from benchmark_fleurs_dev import sha256
from train_tiny_rgb_feature_winner_full import optimizer_payload


def test_only_orchestration_may_change_when_reusing_real_gradient_proof():
    before = {
        "recipe": {"caption_weight": 0},
        "order": [2, 0, 1],
        "model_plan": {"proof": "same"},
        "source_sha256": {
            "tools/train_roi_semantic.py": "old",
            "tools/roi_semantic_student.py": "qualified",
        },
    }
    after = copy.deepcopy(before)
    after["source_sha256"]["tools/train_roi_semantic.py"] = "new"
    after["source_sha256"]["tools/roi_semantic_resume.py"] = "new"
    after["resume"] = {"training_examples": 72}
    recovery.compatible(before, after)
    for field in ("recipe", "order", "model_plan"):
        changed = copy.deepcopy(after)
        changed[field] = None
        with pytest.raises(ValueError):
            recovery.compatible(before, changed)
    after["source_sha256"]["tools/roi_semantic_student.py"] = "changed"
    with pytest.raises(ValueError):
        recovery.compatible(before, after)


def test_restore_preserves_weights_moments_rng_and_exact_next_step(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = torch.nn.Linear(3, 2)

        def trainable_parameters(self):
            return dict(self.named_parameters())

    torch.manual_seed(20)
    original = Student()
    optimizer = torch.optim.AdamW(original.parameters(), lr=3e-5, weight_decay=0.01, foreach=False)
    data = torch.randn(2, 3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        original.vision(data).square().sum().backward()
        optimizer.step()
    values, metadata = optimizer_payload(
        optimizer,
        original.trainable_parameters(),
        {
            "cpu": torch.get_rng_state(),
            "mps": torch.tensor([1, 2, 3], dtype=torch.uint8),
        },
    )
    visual, state = tmp_path / "visual.safetensors", tmp_path / "state.safetensors"
    save_file({k: p.detach().clone() for k, p in original.named_parameters()}, visual)
    save_file(values, state, metadata=metadata)
    monkeypatch.setattr(recovery, "ROOT", tmp_path)

    def artifact(path):
        return {"path": str(path.relative_to(tmp_path)), "sha256": sha256(path)}

    monkeypatch.setattr(recovery, "artifact", artifact)
    rng = []
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda value: rng.append(value.clone()))
    model = Student()
    resumed = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, foreach=False)
    plan = {
        "resume": {
            "visual": artifact(visual),
            "optimizer": artifact(state),
            "training_examples": 8,
            "optimizer_updates": 2,
        }
    }
    assert recovery.restore(plan, model, resumed) == (8, 2)
    assert len(rng) == 1 and rng[0].tolist() == [1, 2, 3]
    for instance, opt in ((original, optimizer), (model, resumed)):
        opt.zero_grad(set_to_none=True)
        instance.vision(data).square().sum().backward()
        opt.step()
    for a, b in zip(original.parameters(), model.parameters(), strict=True):
        assert torch.equal(a, b)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(optimizer.state[a][name], resumed.state[b][name])
    plan["resume"]["visual"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="bytes"):
        recovery.restore(plan, model, resumed)


def test_manual_recovery_cannot_create_a_retry_chain(tmp_path, monkeypatch):
    before = {"source_sha256": {}, "resume": {"training_examples": 72}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(before))
    monkeypatch.setattr(recovery, "verify_snapshot", lambda *args: None)
    with pytest.raises(ValueError, match="retry chain"):
        recovery.make_resume(path, tmp_path / "visual.safetensors", before)
