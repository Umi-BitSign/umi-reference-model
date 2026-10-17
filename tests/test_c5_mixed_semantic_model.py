from __future__ import annotations

import copy
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_mixed_semantic_model as mixed
import test_c5_small_input_semantic as semantic_fixtures
import test_c5_word_evaluation as final_fixtures
import test_c5_word_training as training_fixtures
from terra_training_recovery import artifact


@pytest.fixture
def tiny_training(tmp_path, monkeypatch):
    return training_fixtures.tiny_training.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def completed_word_run(tiny_training, monkeypatch):
    env = final_fixtures.completed_word_run.__wrapped__(tiny_training, monkeypatch)
    monkeypatch.setattr(mixed, "ROOT", env.path.parents[2])
    monkeypatch.setattr(mixed, "artifact", partial(artifact, root=env.path.parents[2]))
    return env


def test_initializer_rejoins_the_complete_actual_tiny_word_training_lineage(completed_word_run):
    env = completed_word_run
    binding = mixed.word_initialization(env.path, env.output / "result.json")
    result = json.loads((env.output / "result.json").read_text())
    assert binding["initial_visual"] == result["visual"]
    assert binding["initial_visual_state_sha256"] == result["visual_final_sha256"]
    assert binding["original_model_plan"] == env.plan["model_plan"]
    assert binding["word_training_result"] == mixed.artifact(env.output / "result.json")
    assert binding["fresh_mixed_full_risk_qualification_required"] is True
    assert binding["optimizer_state_resumed"] is False
    assert mixed.checked_initialization(binding) == binding


def test_initializer_cannot_change_the_final_checkpoint_binding(completed_word_run):
    env = completed_word_run
    binding = mixed.word_initialization(env.path, env.output / "result.json")
    binding["initial_visual_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="actual final values"):
        mixed.checked_initialization(binding)


def test_unfinished_word_training_fails_before_model_construction(monkeypatch):
    monkeypatch.setattr(
        mixed,
        "checked_final",
        lambda *a: (_ for _ in ()).throw(ValueError("word training incomplete")),
    )
    monkeypatch.setattr(mixed, "load_original_student", lambda *a: pytest.fail("loaded a model"))
    with pytest.raises(ValueError, match="incomplete"):
        mixed.load_student(
            {
                "initialization": {
                    "word_training_plan": {"path": "runs/train/plan.json"},
                    "word_training_result": {"path": "checkpoints/train/result.json"},
                }
            }
        )


@pytest.fixture
def model_case(tmp_path, monkeypatch):
    """Tiny CPU tensors exercise loading, not C5 numerical or quality qualification."""
    from safetensors.torch import save_file

    model = semantic_fixtures.model()
    expected = copy.deepcopy(model)
    incoming = {}
    for index, (name, parameter) in enumerate(expected.trainable_parameters().items()):
        parameter.data.fill_(index + 2)
        incoming[name] = parameter.detach().clone()
    target = tmp_path / "checkpoints/word/visual-after-000856.safetensors"
    target.parent.mkdir(parents=True)
    save_file(incoming, str(target))
    monkeypatch.setattr(mixed, "ROOT", tmp_path)
    monkeypatch.setattr(mixed, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(mixed, "TENSORS", len(incoming))
    monkeypatch.setattr(
        mixed, "PARAMETERS", sum(p.numel() for p in model.trainable_parameters().values())
    )
    binding = {
        "original_model_plan": {
            "expected_visual_gradient_names": list(incoming),
            "pretrained_state_sha256": mixed.state_digest(model.vision.state_dict()),
            "initial_reference_hashes": mixed.fixed_hashes(model),
        },
        "initial_visual": mixed.artifact(target),
        "initial_visual_state_sha256": mixed.state_digest(expected.vision.state_dict()),
        "original_mask_sha256": mixed.mask_digest(model),
    }
    return model, binding, target, expected


def test_visual_install_preserves_parameter_identity_RNG_original_masks_and_fixed_backbones(
    model_case,
):
    model, binding, _, expected = model_case
    original = copy.deepcopy(model.state_dict())
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    mixed.install_visual(model, binding)
    mixed.require_model(model, {"initialization": binding})
    assert identities == {name: id(parameter) for name, parameter in model.named_parameters()}
    assert flags == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert torch.equal(rng, torch.get_rng_state())
    for name, parameter in model.state_dict().items():
        desired = expected.state_dict()[name] if name.startswith("vision.") else original[name]
        torch.testing.assert_close(parameter, desired, rtol=0, atol=0)
    assert not model.vision.hands.mask_token.requires_grad
    assert not model.vision.face.mask_token.requires_grad
    assert mixed.fixed_hashes(model) == binding["original_model_plan"]["initial_reference_hashes"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "nan", "state_hash"])
def test_invalid_incoming_visuals_are_rejected_before_any_model_mutation(model_case, mutation):
    from safetensors.torch import load_file, save_file

    model, binding, target, _ = model_case
    original = copy.deepcopy(model.state_dict())
    values = load_file(str(target))
    first = next(iter(values))
    if mutation == "missing":
        values.pop(first)
    elif mutation == "extra":
        values["vision.hands.mask_token"] = torch.ones(1, 384)
    elif mutation == "shape":
        values[first] = torch.ones(1)
    elif mutation == "dtype":
        values[first] = values[first].double()
    elif mutation == "nan":
        values[first].reshape(-1)[0] = torch.nan
    else:
        binding["initial_visual_state_sha256"] = "f" * 64
    save_file(values, str(target))
    binding["initial_visual"] = mixed.artifact(target)
    with pytest.raises(ValueError):
        mixed.install_visual(model, binding)
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


def test_changed_checkpoint_bytes_fail_before_safetensors_loading(model_case, monkeypatch):
    import safetensors.torch

    model, binding, target, _ = model_case
    original = copy.deepcopy(model.state_dict())
    target.write_bytes(b"changed")
    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda *a, **k: pytest.fail("loaded changed bytes")
    )
    with pytest.raises(ValueError, match="file changed"):
        mixed.install_visual(model, binding)
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


def test_nonCPU_model_fails_before_any_state_hash_or_checkpoint_read(model_case, monkeypatch):
    model, binding, _, _ = model_case
    model.to("meta")  # No GPU work or real C5 tensors are involved in this device guard fixture.
    monkeypatch.setattr(mixed, "_require_structure", lambda *a: pytest.fail("read nonCPU tensors"))
    monkeypatch.setattr(
        mixed, "artifact", lambda *a: pytest.fail("read checkpoint before device guard")
    )
    with pytest.raises(ValueError, match="CPU"):
        mixed.install_visual(model, binding)


@pytest.mark.parametrize(
    "mutation",
    [
        "encoder",
        "language",
        "mask",
        "trainable_mask",
        "frozen_visual",
        "train_mode",
        "visual_value",
        "side",
    ],
)
def test_original_model_scope_and_values_must_match_before_loading(model_case, mutation):
    model, binding, _, _ = model_case
    if mutation == "encoder":
        next(model.semantic_encoder.parameters()).data.add_(1)
    elif mutation == "language":
        next(model.language.parameters()).requires_grad_(True)
    elif mutation == "mask":
        model.vision.hands.mask_token.data.add_(1)
    elif mutation == "trainable_mask":
        model.vision.face.mask_token.requires_grad_(True)
    elif mutation == "frozen_visual":
        next(iter(model.trainable_parameters().values())).requires_grad_(False)
    elif mutation == "train_mode":
        model.vision.train()
    elif mutation == "visual_value":
        next(iter(model.trainable_parameters().values())).data.add_(1)
    else:
        model.vision.side = 64
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        mixed.install_visual(model, binding)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_full_model_is_constructed_only_after_initializer_revalidation(model_case, monkeypatch):
    model, binding, _, _ = model_case
    order = []

    def checked(incoming):
        assert incoming is binding
        order.append("checked")
        return incoming

    def original(plan):
        assert order == ["checked"] and plan == binding["original_model_plan"]
        order.append("constructed")
        return model, "original tokenizer"

    monkeypatch.setattr(mixed, "checked_initialization", checked)
    monkeypatch.setattr(mixed, "load_original_student", original)
    loaded, tokenizer = mixed.load_student({"initialization": binding})
    assert loaded is model and tokenizer == "original tokenizer"
    assert order == ["checked", "constructed"]
    mixed.require_model(loaded, {"initialization": binding})


def test_clip_adapter_retains_all149_frames_and_fixed_pose_without_padding(monkeypatch):
    values = {
        "pixels": np.zeros((149, 3, 112, 112, 3), dtype=np.uint8),
        "pose_features": np.arange(149 * 14, dtype=np.float32).reshape(149, 14),
    }
    values["pixels"][-1, 0] = 255
    bound = object()

    def read(incoming):
        assert incoming is bound
        return values

    monkeypatch.setattr(mixed, "load_verified_input", read)
    clip, mask = mixed.load_clip(bound)
    assert clip.pixels.shape == (149, 3, 3, 112, 112)
    assert mask.shape == (1, 149) and mask.dtype == torch.bool and mask.all()
    assert not clip.pixels.requires_grad and not clip.pose.requires_grad
    assert not torch.equal(clip.pixels[0, 0], clip.pixels[-1, 0])
    np.testing.assert_array_equal(clip.pose.numpy(), values["pose_features"])
    clip.pose.zero_()
    assert values["pose_features"].sum() > 0
