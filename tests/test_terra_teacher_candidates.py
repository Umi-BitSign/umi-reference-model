from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import terra_teacher_candidates as module
from benchmark_fleurs_dev import sha256
from terra_quality_training import minimum_risk


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return "native phrase" if tokens == [5, 1] else "different phrase"


@pytest.fixture(autouse=True)
def available_cache_space(monkeypatch):
    monkeypatch.setattr(module.shutil, "disk_usage", lambda root: SimpleNamespace(free=1024**4))


def approved_pair(tmp_path):
    path = tmp_path / "data/features.bin"
    path.parent.mkdir()
    path.write_bytes(b"audited-native-features")
    return {
        "feature_pretraining_allowed": True,
        "raw": {
            "source": "approved",
            "example_id": "clip",
            "role": "training",
            "training_allowed": True,
            "reference": "actual human caption",
        },
        "teacher": {
            "source": "approved",
            "id": "clip",
            "role": "training",
            "training_allowed": True,
            "reference": "actual human caption",
            "cache": "data/features.bin",
            "cache_sha256": sha256(path),
        },
        "header": {"sampled_frames": 3, "frame_indices_sha256": "canonical-three-frames"},
    }


def test_teacher_path_uses_native_features_without_reference_and_caches_exact_tokens(
    tmp_path, monkeypatch
):
    pair = approved_pair(tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module, "load_targets", lambda pair: {"face_features": np.zeros((3, 4), dtype=np.float32)}
    )
    calls = []

    class Encoder(torch.nn.Module):
        def forward(self, **kwargs):
            assert set(kwargs) == {"face_features", "return_dict"}
            calls.append("native_encoder")
            return SimpleNamespace(last_hidden_state=kwargs["face_features"])

    class Language(torch.nn.Module):
        config = SimpleNamespace(d_model=4)

        def generate(self, **kwargs):
            assert (
                "reference" not in kwargs and "labels" not in kwargs and "input_ids" not in kwargs
            )
            assert kwargs["num_beams"] == 1 and kwargs["do_sample"] is False
            assert kwargs["max_length"] == 513
            calls.append("generate")
            return torch.tensor([[0, 5, 1, 0]])

    student = torch.nn.Module()
    student.weight = torch.nn.Parameter(torch.zeros(1))
    student.semantic_encoder, student.language = Encoder(), Language()
    cache = module.TeacherTextCache(tmp_path / "runs/cache", workspace=tmp_path)
    first = module.teacher_candidate(
        student, Tokenizer(), pair, checkpoint_sha="pinned-C5", cache=cache
    )
    second = module.teacher_candidate(
        student, Tokenizer(), pair, checkpoint_sha="pinned-C5", cache=cache
    )
    assert (
        first
        == second
        == {
            "tokens": [5, 1],
            "truncated": False,
            "termination": "eos",
            "text": "native phrase",
            "kind": "c5_teacher",
        }
    )
    assert calls == ["native_encoder", "generate"]


def test_teacher_rejects_protected_role_before_sampler_or_cache_access(tmp_path, monkeypatch):
    pair = approved_pair(tmp_path)
    pair["raw"].update(role="open_development", training_allowed=False)
    monkeypatch.setattr(
        module, "candidate_pool", lambda *args, **kwargs: pytest.fail("sampler must not run")
    )
    with pytest.raises(ValueError, match="approved training"):
        module.teacher_augmented_pool(
            None,
            None,
            None,
            None,
            pair["raw"]["reference"],
            pair=pair,
            seed=317,
            checkpoint_sha="C5",
            cache=None,
        )


def test_teacher_pool_preserves_sampler_seed_and_deduplicates_tokens(tmp_path, monkeypatch):
    pair = approved_pair(tmp_path)
    base = [{"tokens": [3, 1], "kind": "human_anchor"}, {"tokens": [5, 1], "kind": "greedy"}]
    seeds = []

    def sample(*args, **kwargs):
        seeds.append(kwargs["seed"])
        assert kwargs["budget_factory"] is module.TrainingRolloutBudget
        return copy.deepcopy(base)

    monkeypatch.setattr(module, "candidate_pool", sample)
    monkeypatch.setattr(
        module,
        "teacher_candidate",
        lambda *args, **kwargs: {"tokens": [5, 1], "kind": "c5_teacher"},
    )
    args = (None, None, None, None, pair["raw"]["reference"])
    options = {"pair": pair, "seed": 317, "checkpoint_sha": "C5", "cache": None}
    assert module.teacher_augmented_pool(*args, **options) == base
    monkeypatch.setattr(
        module,
        "teacher_candidate",
        lambda *args, **kwargs: {"tokens": [6, 1], "kind": "c5_teacher"},
    )
    assert len(module.teacher_augmented_pool(*args, **options)) == 3
    assert seeds == [317, 317]


def test_teacher_cache_rejects_corruption_and_stays_in_owned_budget(tmp_path):
    pair = approved_pair(tmp_path)
    binding = module.generation_binding(pair, "C5")
    candidate = {
        "tokens": [5, 1],
        "truncated": False,
        "termination": "eos",
        "text": "native phrase",
        "kind": "c5_teacher",
    }
    cache = module.TeacherTextCache(tmp_path / "runs/cache", workspace=tmp_path)
    cache.put(binding, candidate)
    assert cache.get(binding, Tokenizer()) == candidate
    path = cache.root / f"{module.binding_key(binding)}.json"
    altered = json.loads(path.read_text())
    altered["candidate"]["text"] = "invented unbound text"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="actual token"):
        cache.get(binding, Tokenizer())
    limited = module.TeacherTextCache(
        tmp_path / "runs/tiny-cache", workspace=tmp_path, maximum_bytes=10
    )
    with pytest.raises(RuntimeError, match="byte budget"):
        limited.put(binding, candidate)
    with pytest.raises(ValueError, match="owned runs"):
        module.TeacherTextCache(tmp_path / "data", workspace=tmp_path)


def test_native_feature_revision_changes_teacher_cache_key(tmp_path):
    pair = approved_pair(tmp_path)
    original = module.binding_key(module.generation_binding(pair, "C5"))
    assert module.binding_key(module.generation_binding(pair, "other-checkpoint")) != original
    pair["teacher"]["cache_sha256"] = "different-native-feature-bytes"
    assert module.binding_key(module.generation_binding(pair, "C5")) != original


def test_terra_reinforces_good_teacher_and_suppresses_zero_score_teacher():
    logp = torch.full((4,), -3.0, requires_grad=True)
    positive = minimum_risk(logp, [10, 0, 0, 7])["coefficients"][-1]
    negative = minimum_risk(logp, [10, 0, 0, 0])["coefficients"][-1]
    assert positive < 0 < negative


def test_four_candidate_teacher_risk_matches_direct_model_autograd():
    from terra_quality_training import sequence_log_probability
    from terra_visual_learning import backward_candidate_risk

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.surface = torch.nn.Linear(3, 9)

        def forward(self, video, mask, labels):
            logits = self.surface(video).expand(1, labels.shape[1], 9)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
            return {"language": SimpleNamespace(logits=logits, loss=loss)}

    torch.manual_seed(317)
    direct = Student()
    released = copy.deepcopy(direct)
    video, mask = torch.randn(1, 1, 3), torch.ones(1, 1, dtype=torch.bool)
    labels = [
        torch.tensor([[3, 1]]),
        torch.tensor([[4, 5, 1]]),
        torch.tensor([[6, 1]]),
        torch.tensor([[7, 1]]),
    ]
    outputs = [direct(video, mask, value)["language"] for value in labels]
    logp = torch.stack(
        [
            sequence_log_probability(value.logits, label)
            for value, label in zip(outputs, labels, strict=True)
        ]
    )
    loss = minimum_risk(logp, [10, 0, 0, 7])["loss"] + 0.1 * outputs[0].loss
    loss.backward()
    actual = backward_candidate_risk(
        released, video, mask, labels, [10, 0, 0, 7], alpha=0.005, caption_weight=0.1
    )
    assert actual["loss"] == pytest.approx(float(loss), abs=1e-6)
    for expected, value in zip(direct.parameters(), released.parameters(), strict=True):
        assert torch.allclose(expected.grad, value.grad, atol=1e-7, rtol=1e-5)


def test_training_memory_budget_is_separate_from_unchanged_benchmark_budget():
    from evaluate_tiny_rgb_feature_winner import CPU_LIMITS, CpuBudget

    ids = torch.ones(1, 2, dtype=torch.long)
    between = 24 * 1024**3 + 1
    training = module.TrainingRolloutBudget(clock=lambda: 0, memory=lambda: between)
    assert not training(ids, None).any()
    benchmark = CpuBudget(clock=lambda: 0, memory=lambda: between)
    with pytest.raises(RuntimeError, match="resource policy"):
        benchmark(ids, None)
    assert CPU_LIMITS["maximum_process_peak_bytes"] == 24 * 1024**3
    assert module.TRAINING_ROLLOUT_LIMITS["maximum_process_peak_bytes"] == 32 * 1024**3
    exceeded = module.TrainingRolloutBudget(clock=lambda: 0, memory=lambda: 32 * 1024**3 + 1)
    with pytest.raises(RuntimeError, match="training CPU rollout") as caught:
        exceeded(ids, None)
    assert caught.value.terra_resource_limits == module.TRAINING_ROLLOUT_LIMITS


def test_base_sampler_keeps_benchmark_budget_unless_training_explicitly_injects_one():
    from evaluate_tiny_rgb_feature_winner import CpuBudget

    used = []

    class Language:
        def generate(self, **kwargs):
            used.append(type(kwargs["stopping_criteria"][0]))
            return torch.tensor([[0, 5, 1]])

    class Student:
        language = Language()

        def eval(self):
            return self

        def encode(self, video, mask):
            return torch.zeros(1, 2, 4), mask

    class Labels(Tokenizer):
        def __call__(self, reference, **kwargs):
            return SimpleNamespace(input_ids=[6, 1])

    video, mask = torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool)
    module.candidate_pool(Student(), Labels(), video, mask, "human", seed=317)
    module.candidate_pool(
        Student(),
        Labels(),
        video,
        mask,
        "human",
        seed=317,
        budget_factory=module.TrainingRolloutBudget,
    )
    assert used == [
        CpuBudget,
        CpuBudget,
        module.TrainingRolloutBudget,
        module.TrainingRolloutBudget,
    ]


def test_teacher_cache_reserves_physical_space_without_lowering_ingestion_guards(
    tmp_path, monkeypatch
):
    pair = approved_pair(tmp_path)
    binding = module.generation_binding(pair, "C5")
    candidate = {
        "tokens": [5, 1],
        "truncated": False,
        "termination": "eos",
        "text": "native phrase",
        "kind": "c5_teacher",
    }
    cache = module.TeacherTextCache(tmp_path / "runs/cache", workspace=tmp_path)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda root: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="physical floor"):
        cache.put(binding, candidate)
    assert not list(cache.root.iterdir())


def test_teacher_feature_paths_cannot_leave_data_subtree(tmp_path, monkeypatch):
    pair = approved_pair(tmp_path)
    pair["teacher"]["cache"] = "../protected-service-material"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="approved data"):
        module.teacher_candidate(None, None, pair, checkpoint_sha="C5", cache=None)


def test_visual_transfer_installs_exact_terra_weights_and_preserves_quality_gates(tmp_path):
    from safetensors.torch import save_file
    from terra_training_recovery import artifact

    source_path = tmp_path / "runs/source/plan.json"
    snapshot = source_path.with_name("source-snapshot")
    archived = snapshot / "tools/frozen_trainer.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("# immutable source snapshot\n")
    source = {
        "recipe": {
            "judge_model": "gpt-5.6-terra",
            "feature_distillation": False,
            "latent_distillation": False,
            "capacity": "large",
            "short_side": 192,
            "target_fps": 12.5,
            "max_frames": 512,
            "max_text_tokens": 512,
            "seed": 317,
        },
        "reference": {"translator_checkpoint_sha256": "fixed-C5"},
        "evaluation": [],
        "winner_control": [],
        "historical_controls": {},
        "gates": {"improvement": 0.05},
        "resource_policy": "unchanged",
        "resource_policy_sha256": "unchanged",
        "runtime_binding": {},
        "native_pairs_sha256": "approved-pairs",
        "proof_cases": [],
        "training_examples": 7523,
        "planned_training_examples": 30092,
        "source_sha256": {"tools/frozen_trainer.py": sha256(archived)},
    }
    source_path.write_text(json.dumps(source))
    (snapshot / "manifest.json").write_text(
        json.dumps({"plan_sha256": sha256(source_path), "source_sha256": source["source_sha256"]})
    )
    checkpoint = tmp_path / "checkpoints/source/visual-after-000004.safetensors"
    checkpoint.parent.mkdir(parents=True)
    values = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    save_file(
        values,
        str(checkpoint),
        metadata={
            "plan_sha256": sha256(source_path),
            "fixed_winner_checkpoint": "fixed-C5",
            "update": "4",
            "optimizer_updates": "1",
        },
    )
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {
                "complete": True,
                "visual": artifact(checkpoint, root=tmp_path),
                "training_examples": 4,
                "optimizer_updates": 1,
            }
        )
    )
    target = copy.deepcopy(source)
    target["recipe"]["teacher_candidates"] = 1
    expected = sha256(checkpoint)
    target["visual_initialization"] = module.make_visual_initialization(
        source_path, checkpoint, target, expected_sha256=expected, root=tmp_path
    )
    assert target["visual_initialization"]["optimizer_state_reused"] is False

    class Model:
        parameter = torch.nn.Parameter(torch.zeros(2, 3))

        def trainable_parameters(self):
            return {"weight": self.parameter}

    model = Model()
    module.install_visual_initialization(model, target, expected_sha256=expected, root=tmp_path)
    assert torch.equal(model.parameter, values["weight"])
    changed = copy.deepcopy(target)
    changed["gates"]["improvement"] = 0
    with pytest.raises(ValueError, match="quality gates"):
        module.checked_visual_initialization(changed, expected_sha256=expected, root=tmp_path)
    with pytest.raises(ValueError, match="bytes changed"):
        module.checked_visual_initialization(target, expected_sha256="wrong", root=tmp_path)
