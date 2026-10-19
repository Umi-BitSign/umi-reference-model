"""Load a fully gated research visual export into private native112 embedders.

This loader supplies frozen visual branches only. Full native crop/timeline,
feature/token/latency parity and independent semantic confirmation remain due.
"""

from __future__ import annotations

from pathlib import Path

import export_c5_native112_visual as exporter
from c5_native112_export_values import qualified_shapes, require, verify_visual_values

ROOT = Path(__file__).resolve().parents[1]


def authenticate_export(path):
    """Recheck the actual final, evaluations, exported values and source bindings."""
    path = path.absolute()
    require(
        path.resolve() == path
        and path.is_relative_to(ROOT / "checkpoints")
        and path.name == "EXPORT.json"
        and path.is_file(),
        "native112 loading requires an owned complete visual export",
    )
    record = exporter.read(path)
    require(
        record.get("schema") == "umi-studio-native112-trained-visual-export/1"
        and record.get("complete") is True
        and record.get("native_input_side") == 112
        and record.get("temporal_stride") == 2
        and record.get("source_sha256") == exporter.export_sources(),
        "native112 export schema, source or input policy changed",
    )
    evidence = record["evidence"]
    inputs = [
        exporter.bound(evidence[key])
        for key in (
            "training_plan",
            "training_result",
            "terminal_audit",
            "evaluation_complete",
        )
    ]
    # Evaluation descriptors belong to the authenticated export evidence.
    inputs.extend(
        exporter.bound(evidence["evaluations"][kind]["parent_complete"])
        for kind in ("word", "sentence")
    )
    require(exporter.authenticate(*inputs) == evidence, "native112 export admission changed")
    exporter.verify_evidence(evidence)
    plan, result = (exporter.read(inputs[i]) for i in (0, 1))
    shapes = qualified_shapes(exporter.read(exporter.bound(plan["model_plan"]["visual_scope"])))
    require(
        set(shapes) == set(plan["model_plan"]["expected_visual_gradient_names"]),
        "native112 export changed its full qualified visual scope",
    )
    from safetensors.numpy import load_file

    originals, exported, files = {}, {}, {}
    for branch, filename in exporter.FILES.items():
        raw_path = exporter.bound(record["files"][branch])
        require(raw_path == path.parent / filename, "native112 export branch path changed")
        original_path = exporter.bound(record["original_visual_files"][branch])
        expected = ROOT / plan["model_plan"]["reference"]["bundle_root"] / "model/models" / filename
        require(
            original_path == expected
            and exporter.sha(expected)
            == plan["model_plan"]["assets"][str(expected.relative_to(ROOT))],
            "native112 export original visual assets changed",
        )
        originals[branch] = load_file(str(original_path))
        exported[branch] = load_file(str(raw_path))
        files[branch] = raw_path
    trained_path = exporter.bound(result["visual"])
    measured = verify_visual_values(originals, load_file(str(trained_path)), exported, shapes)
    require(measured == record["verified_values"], "native112 exported values changed")
    return record, plan["model_plan"], files


def prepare_backbones(models, incoming):
    """Validate both private CPU backbones before installing either raw-file state."""
    import torch
    from c5_patch_projection import LinearPatchProjection
    from torch import nn

    require(set(models) == set(incoming) == {"hands", "face"}, "both visual branches required")
    require(models["hands"] is not models["face"], "visual branches cannot share a backbone")
    proposed = {}
    for branch, model in models.items():
        require(isinstance(model, nn.Module), "native112 loader requires private Torch backbones")
        existing = model.state_dict()
        values = incoming[branch]
        require(
            isinstance(values, dict)
            and values
            and all(
                isinstance(name, str) and name.startswith(("backbone.", "dino_head."))
                for name in values
            ),
            "native112 raw visual namespace changed",
        )
        state = {
            k.removeprefix("backbone."): v for k, v in values.items() if k.startswith("backbone.")
        }
        require(set(state) == set(existing), "native112 raw state is incomplete or contains extras")
        require(
            isinstance(model.patch_embed.proj, nn.Conv2d)
            and tuple(model.pos_embed.shape) == (1, 257, 384)
            and tuple(model.mask_token.shape) == (1, 384),
            "native112 loader requires the original unmodified DINO projection and positions",
        )
        # This constructor checks every qualified14px convolution attribute.
        LinearPatchProjection(model.patch_embed.proj)
        for name, value in state.items():
            require(
                isinstance(value, torch.Tensor)
                and value.device.type == existing[name].device.type == "cpu"
                and value.dtype == existing[name].dtype == torch.float32
                and value.shape == existing[name].shape
                and torch.isfinite(value).all(),
                "native112 raw state has changed shapes, dtype, device or nonfinite values",
            )
        require(
            torch.equal(
                state["mask_token"].view(torch.int32), existing["mask_token"].view(torch.int32)
            ),
            "native112 export changed the original frozen mask",
        )
        proposed[branch] = state
    for branch, model in models.items():
        model.load_state_dict(proposed[branch], strict=True)
        model.patch_embed.proj = LinearPatchProjection(model.patch_embed.proj)
        model.requires_grad_(False).eval()
    return models


def load_visual_pair(path, *, batch_context, batch_size=2):
    """Load on CPU; a future caller owns exclusive device use and parity checks."""
    require(callable(batch_context), "native112 loader requires a bounded execution context")
    require(type(batch_size) is int and batch_size > 0, "native112 batch size must be positive")
    record, plan, files = authenticate_export(path)
    from c5_native112_frame_embedder import Native112FrameEmbedder
    from probe_c5_small_input import legacy_models
    from safetensors.torch import load_file

    raw = {branch: load_file(str(target), device="cpu") for branch, target in files.items()}
    hands, face = legacy_models(plan)
    models = prepare_backbones({"hands": hands.model, "face": face.model}, raw)
    require(
        all(exporter.artifact(files[branch]) == record["files"][branch] for branch in files),
        "native112 export changed while loading",
    )
    return {
        branch: Native112FrameEmbedder(model, batch_context=batch_context, batch_size=batch_size)
        for branch, model in models.items()
    }
