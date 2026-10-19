from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import c5_native112_export_values as export  # noqa: E402


def fixture():
    originals, trained, shapes = {}, {}, {}
    for index, branch in enumerate(export.BRANCHES):
        originals[branch] = {
            "backbone.norm.bias": np.full(2, index, dtype=np.float32),
            "backbone.patch_embed.proj.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
            "backbone.mask_token": np.array([[-0.0, 5.0]], dtype=np.float32),
            "dino_head.last_layer.weight_g": np.full((2, 1), 37, dtype=np.float32),
        }
        for name in ("norm.bias", "patch_embed.proj.weight"):
            key = f"vision.{branch}.{name}"
            shapes[key] = originals[branch]["backbone." + name].shape
            trained[key] = np.full(shapes[key], 11 + index, dtype=np.float32)
    return originals, trained, shapes


def test_exact_raw_namespace_and_all_omitted_source_tensors_preserved(tmp_path):
    originals, trained, shapes = fixture()
    source = copy.deepcopy(originals)
    output = export.merge_visual_values(originals, trained, shapes)
    for index, branch in enumerate(export.BRANCHES):
        np.testing.assert_array_equal(output[branch]["backbone.norm.bias"], [11 + index] * 2)
        np.testing.assert_array_equal(
            output[branch]["backbone.patch_embed.proj.weight"],
            np.full((2, 2), 11 + index, dtype=np.float32),
        )
        for raw in ("backbone.mask_token", "dino_head.last_layer.weight_g"):
            assert output[branch][raw].tobytes() == source[branch][raw].tobytes()
        path = tmp_path / f"{branch}.safetensors"
        save_file(output[branch], str(path))
        output[branch] = load_file(str(path))
    report = export.verify_visual_values(originals, trained, output, shapes)
    assert report["trained_tensors"] == 4 and report["trained_values"] == 12
    assert report["runtime_or_semantic_equivalence_established"] is False
    assert all(row["changed_training_tensors"] == 2 for row in report["branches"].values())
    for branch in originals:
        for raw in originals[branch]:
            assert source[branch][raw].tobytes() == originals[branch][raw].tobytes()


def test_merge_does_not_alias_original_or_trained_arrays():
    originals, trained, shapes = fixture()
    output = export.merge_visual_values(originals, trained, shapes)
    output["face"]["backbone.mask_token"][:] = 99
    output["hands"]["backbone.norm.bias"][:] = 99
    assert originals["face"]["backbone.mask_token"][0, 1] == 5
    assert (trained["vision.hands.norm.bias"] == 12).all()


@pytest.mark.parametrize("fault", ["missing", "extra", "dtype", "nan", "shape", "mask"])
def test_invalid_training_tensor_inventory_cannot_be_merged(fault):
    originals, trained, shapes = fixture()
    key = "vision.face.norm.bias"
    if fault == "missing":
        trained.pop(key)
    elif fault == "extra":
        trained["language.weight"] = np.zeros(2, dtype=np.float32)
    elif fault == "dtype":
        trained[key] = trained[key].astype(np.float64)
    elif fault == "nan":
        trained[key][0] = np.nan
    elif fault == "shape":
        trained[key] = np.zeros((1, 2), dtype=np.float32)
    else:
        shapes["vision.face.mask_token"] = (1, 2)
        trained["vision.face.mask_token"] = originals["face"]["backbone.mask_token"].copy()
    with pytest.raises(ValueError):
        export.merge_visual_values(originals, trained, shapes)


@pytest.mark.parametrize("fault", ["mask", "head", "missing", "extra", "swap", "negative_zero"])
def test_independent_value_checker_rejects_serialized_state_corruption(fault):
    originals, trained, shapes = fixture()
    output = export.merge_visual_values(originals, trained, shapes)
    if fault == "mask":
        output["face"]["backbone.mask_token"][0, 1] += 1
    elif fault == "head":
        output["hands"]["dino_head.last_layer.weight_g"][0] += 1
    elif fault == "missing":
        del output["face"]["backbone.norm.bias"]
    elif fault == "extra":
        output["face"]["other.weight"] = np.zeros(2, dtype=np.float32)
    elif fault == "negative_zero":
        output["face"]["backbone.mask_token"][0, 0] = 0.0
    else:
        output["face"], output["hands"] = output["hands"], output["face"]
    with pytest.raises(ValueError):
        export.verify_visual_values(originals, trained, output, shapes)


def test_canonical_loader_alias_is_rejected_and_digest_is_order_independent():
    originals, trained, shapes = fixture()
    output = export.merge_visual_values(originals, trained, shapes)
    first = export.verify_visual_values(originals, trained, output, shapes)
    reordered = {branch: dict(reversed(list(values.items()))) for branch, values in output.items()}
    assert first == export.verify_visual_values(originals, trained, reordered, shapes)
    originals["face"]["norm.bias"] = originals["face"]["backbone.norm.bias"].copy()
    with pytest.raises(ValueError, match="namespace"):
        export.merge_visual_values(originals, trained, shapes)


def test_qualified_scope_rejects_partial_or_changed_parameter_counts():
    # Metadata fixture only; no real-model tensors or private assets are needed
    # to exercise the count/namespace guard in a public checkout.
    rows = [
        {"parameter": f"{branch}.parameter_{index}", "shape": [1]}
        for branch in export.BRANCHES
        for index in range(175)
    ]
    rows[0]["shape"] = [export.PARAMETERS - 349]
    scope = {
        "parameters": rows,
        "trainable_tensors": 350,
        "trainable_parameters": export.PARAMETERS,
    }
    shapes = export.qualified_shapes(scope)
    assert len(shapes) == 350 and all(name.startswith("vision.") for name in shapes)
    for change in ("partial", "duplicate", "shape", "bool_dimension"):
        changed = copy.deepcopy(scope)
        if change == "partial":
            changed["parameters"].pop()
        elif change == "duplicate":
            changed["parameters"][1] = changed["parameters"][0]
        elif change == "shape":
            changed["parameters"][0]["shape"][0] += 1
        else:
            changed["parameters"][0]["shape"][0] = True
        with pytest.raises(ValueError):
            export.qualified_shapes(changed)
