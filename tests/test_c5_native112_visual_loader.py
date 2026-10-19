from __future__ import annotations

import copy
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_native112_visual_loader as loader
from c5_patch_projection import LinearPatchProjection
from c5_small_input_vision import input_tensor
from test_export_c5_native112_visual import small_export  # noqa: F401


class PlainBackbone(nn.Module):
    """Real convolution, minimal state; no pretrained transformer or external asset."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 384, 14, stride=14)
        self.pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
        self.mask_token = nn.Parameter(torch.zeros(1, 384))

    def forward(self, pixels):
        return self.patch_embed.proj(pixels).mean((2, 3))


@pytest.fixture
def branches():
    torch.set_num_threads(1)
    models = {name: PlainBackbone() for name in ("hands", "face")}
    before = {branch: copy.deepcopy(model.state_dict()) for branch, model in models.items()}
    raw = {}
    for index, (branch, model) in enumerate(models.items()):
        state = {
            "backbone." + name: value.clone() + (0 if name == "mask_token" else (index + 1) / 100)
            for name, value in model.state_dict().items()
        }
        state["dino_head.ignored"] = torch.tensor([float("nan")])
        raw[branch] = state
    return SimpleNamespace(models=models, before=before, raw=raw)


def test_installed_values_frozen_masks_and_native_features_match_training_path(branches):
    f = branches
    raw_before = copy.deepcopy(f.raw)
    loader.prepare_backbones(f.models, f.raw)
    pixels = np.random.default_rng(719).integers(0, 256, (3, 112, 112, 3), dtype=np.uint8)
    training_pixels = input_tensor(np.repeat(pixels[:, None], 3, axis=1))[:, 0]
    for branch, model in f.models.items():
        assert isinstance(model.patch_embed.proj, LinearPatchProjection)
        assert all(not p.requires_grad for p in model.parameters())
        assert all(not m.training for m in model.modules())
        assert torch.equal(model.mask_token, f.before[branch]["mask_token"])
        for name, value in model.state_dict().items():
            assert torch.equal(value, f.raw[branch]["backbone." + name])
        reference = PlainBackbone()
        reference.load_state_dict(
            {
                k.removeprefix("backbone."): v
                for k, v in f.raw[branch].items()
                if k.startswith("backbone.")
            }
        )
        with torch.inference_mode():
            expected = reference(training_pixels)
            actual = model(training_pixels)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        for name, value in f.raw[branch].items():
            assert torch.equal(value.view(torch.int32), raw_before[branch][name].view(torch.int32))


@pytest.mark.parametrize(
    "fault",
    ["missing", "extra", "shape", "dtype", "nonfinite", "mask", "namespace", "mask_signed_zero"],
)
def test_bad_second_branch_is_rejected_before_either_backbone_is_mutated(branches, fault):
    f = branches
    values = f.raw["face"]
    key = "backbone.patch_embed.proj.bias"
    if fault == "missing":
        del values[key]
    elif fault == "extra":
        values["backbone.extra"] = torch.zeros(1)
    elif fault == "shape":
        values[key] = torch.zeros(383)
    elif fault == "dtype":
        values[key] = values[key].double()
    elif fault == "nonfinite":
        values[key][0] = float("inf")
    elif fault == "mask":
        values["backbone.mask_token"][0, 0] = 1
    elif fault == "namespace":
        values["alias.weight"] = torch.zeros(1)
    elif fault == "mask_signed_zero":
        values["backbone.mask_token"][0, 0] = -0.0
    with pytest.raises(ValueError):
        loader.prepare_backbones(f.models, f.raw)
    for branch, model in f.models.items():
        assert isinstance(model.patch_embed.proj, nn.Conv2d)
        for name, value in model.state_dict().items():
            assert torch.equal(value, f.before[branch][name])


def test_pair_loader_reads_files_and_produces_both_native_embedders(
    tmp_path, monkeypatch, branches
):
    f = branches
    monkeypatch.setattr(loader.exporter, "ROOT", tmp_path)
    files = {}
    for branch, state in f.raw.items():
        path = tmp_path / "checkpoints" / branch / "weights.safetensors"
        path.parent.mkdir(parents=True)
        save_file(state, str(path))
        files[branch] = path
    record = {"files": {b: loader.exporter.artifact(p) for b, p in files.items()}}
    monkeypatch.setattr(
        loader, "authenticate_export", lambda _: (record, {"synthetic": True}, files)
    )
    monkeypatch.setitem(
        sys.modules,
        "probe_c5_small_input",
        SimpleNamespace(
            legacy_models=lambda _: tuple(
                SimpleNamespace(model=f.models[b]) for b in ("hands", "face")
            )
        ),
    )
    actual = loader.load_visual_pair(
        tmp_path / "EXPORT.json", batch_context=nullcontext, batch_size=2
    )
    assert set(actual) == {"hands", "face"}
    assert actual["hands"].model is f.models["hands"]
    frames = np.random.default_rng(719).integers(0, 256, (3, 112, 112, 3), dtype=np.uint8)
    outputs = {
        branch: embedder.extract_embeddings_from_frames(frames)
        for branch, embedder in actual.items()
    }
    assert all(v.shape == (3, 384) and np.isfinite(v).all() for v in outputs.values())
    assert not np.array_equal(outputs["hands"], outputs["face"])


def test_incomplete_export_refuses_before_any_factory_or_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="owned complete visual export"):
        loader.load_visual_pair(
            tmp_path / "checkpoints/missing/EXPORT.json", batch_context=nullcontext
        )


@pytest.fixture
def exported(request, monkeypatch):
    """Actual tiny export transaction; complete-model admission is stubbed."""
    f = request.getfixturevalue("small_export")
    monkeypatch.setattr(loader, "ROOT", f.root)
    monkeypatch.setattr(loader, "qualified_shapes", lambda _: copy.deepcopy(f.shapes))
    evidence = {
        key: loader.exporter.artifact(path)
        for key, path in zip(
            ("training_plan", "training_result", "terminal_audit", "evaluation_complete"),
            f.inputs[:4],
            strict=True,
        )
    }
    evidence["evaluations"] = {
        kind: {"parent_complete": loader.exporter.artifact(path)}
        for kind, path in zip(("word", "sentence"), f.inputs[4:], strict=True)
    }
    calls = []

    def authenticate(*args):
        assert list(args) == f.inputs
        calls.append(args)
        return copy.deepcopy(evidence)

    monkeypatch.setattr(loader.exporter, "authenticate", authenticate)
    record = loader.exporter.run(*f.inputs, f.output)
    return SimpleNamespace(f=f, record=record, calls=calls, manifest=f.output / "EXPORT.json")


def test_actual_export_values_and_same_final_admission_are_rechecked(exported):
    e = exported
    record, plan, files = loader.authenticate_export(e.manifest)
    assert record == e.record
    assert len(e.calls) == 2
    assert plan == loader.exporter.read(e.f.inputs[0])["model_plan"]
    assert files == {b: e.f.output / n for b, n in loader.exporter.FILES.items()}


@pytest.mark.parametrize(
    "fault", ["trained_value", "frozen_mask", "wrong_report", "other_branch_path", "source_change"]
)
def test_export_loader_rejects_altered_values_and_bindings(exported, fault):
    from safetensors.torch import load_file

    e = exported
    record = copy.deepcopy(e.record)
    if fault in ("trained_value", "frozen_mask"):
        path = e.f.output / loader.exporter.FILES["face"]
        values = load_file(str(path))
        name = (
            "backbone.mask_token"
            if fault == "frozen_mask"
            else next(n for n in values if n.startswith("backbone.") and n != "backbone.mask_token")
        )
        values[name].reshape(-1)[0] += 1
        save_file(values, str(path))
        record["files"]["face"] = loader.exporter.artifact(path)
    elif fault == "wrong_report":
        record["verified_values"]["trained_tensors"] += 1
    elif fault == "other_branch_path":
        record["files"]["face"] = record["files"]["hands"]
    elif fault == "source_change":
        record["source_sha256"] = {}
    e.manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        loader.authenticate_export(e.manifest)
