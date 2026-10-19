"""Pure CPU value mapping for a future eligible native112 C5 visual export.

No checkpoint loading, model construction, file writing or release decision is
provided here. A caller must separately authenticate complete training, semantic
quality and its export plan. Runtime numerical/token/latency checks remain due.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

BRANCHES = ("face", "hands")
TENSORS = 350
PARAMETERS = 43260672


def require(condition, message):
    if not condition:
        raise ValueError(message)


def qualified_shapes(scope):
    """Require the original complete 350-tensor, 43,260,672-value visual scope."""
    rows = scope["parameters"]
    require(
        len(rows) == TENSORS
        and type(scope["trainable_tensors"]) is int
        and type(scope["trainable_parameters"]) is int
        and scope["trainable_tensors"] == TENSORS
        and scope["trainable_parameters"] == PARAMETERS,
        "export needs the complete qualified visual scope",
    )
    shapes = {}
    for row in rows:
        name, shape = row["parameter"], row["shape"]
        require(
            isinstance(name, str)
            and name.startswith(tuple(branch + "." for branch in BRANCHES))
            and isinstance(shape, list)
            and shape
            and all(type(size) is int and size > 0 for size in shape)
            and name not in shapes,
            "invalid or duplicate qualified visual parameter",
        )
        require(name.split(".", 1)[1] != "mask_token", "frozen mask entered the trained scope")
        shapes[name] = tuple(shape)
    require(
        sum(math.prod(shape) for shape in shapes.values()) == PARAMETERS
        and all(
            sum(name.startswith(branch + ".") for name in shapes) == 175 for branch in BRANCHES
        ),
        "export visual parameter count or branch coverage changed",
    )
    return {"vision." + name: shape for name, shape in shapes.items()}


def export_mapping(shapes, raw_names):
    """Match each selected training key to one original released raw-file key.

    The generic mapping supports small synthetic fixtures. Production callers
    must obtain shapes through qualified_shapes using the bound real scope.
    """
    require(set(raw_names) == set(BRANCHES), "both original visual branches are required")
    require(
        shapes
        and all(
            isinstance(name, str) and name.startswith(tuple("vision." + b + "." for b in BRANCHES))
            for name in shapes
        ),
        "unexpected training namespace",
    )
    result = {}
    for branch in BRANCHES:
        names = list(raw_names[branch])
        require(names and len(names) == len(set(names)), "duplicate or empty raw tensor inventory")
        canonical = {}
        for name in names:
            require(isinstance(name, str), "raw tensor key must be text")
            if name.startswith("dino_head."):
                continue
            require(name.startswith("backbone."), "unexpected original raw tensor namespace")
            stripped = name.removeprefix("backbone.")
            require(stripped not in canonical, "ambiguous raw backbone mapping")
            canonical[stripped] = name
        prefix = "vision." + branch + "."
        selected = {name.removeprefix(prefix): name for name in shapes if name.startswith(prefix)}
        require(
            selected
            and "mask_token" not in selected
            and set(canonical) == set(selected) | {"mask_token"},
            "original backbone does not exactly cover the trained branch and frozen mask",
        )
        result[branch] = {training: canonical[name] for name, training in selected.items()}
    require(sum(map(len, result.values())) == len(shapes), "unmapped trained visual values")
    return result


def finite_f32(value):
    require(
        isinstance(value, np.ndarray) and value.dtype == np.float32 and np.isfinite(value).all(),
        "export values must be finite CPU FP32 arrays",
    )


def same_bits(first, second):
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and first.tobytes(order="C") == second.tobytes(order="C")
    )


def checked_values(originals, trained, shapes):
    mapping = export_mapping(
        shapes, {branch: values.keys() for branch, values in originals.items()}
    )
    require(set(trained) == set(shapes), "trained export cannot omit or add a tensor")
    for branch, table in originals.items():
        for value in table.values():
            finite_f32(value)
        for name, raw in mapping[branch].items():
            value = trained[name]
            finite_f32(value)
            require(
                value.shape == originals[branch][raw].shape == tuple(shapes[name]),
                "trained or original tensor shape differs from the qualified scope",
            )
    return mapping


def merge_visual_values(originals, trained, shapes):
    """Return independent raw-file arrays; preserve masks and ignored heads bitwise."""
    mapping = checked_values(originals, trained, shapes)
    output = {}
    for branch in BRANCHES:
        replacement = {raw: trained[name] for name, raw in mapping[branch].items()}
        output[branch] = {
            raw: np.array(replacement[raw] if raw in replacement else value, copy=True, order="C")
            for raw, value in originals[branch].items()
        }
    verify_visual_values(originals, trained, output, shapes)
    return output


def verify_visual_values(originals, trained, exported, shapes):
    """Check actual exported arrays against sources, including every unused key."""
    mapping = checked_values(originals, trained, shapes)
    require(set(exported) == set(BRANCHES), "export omitted or added a branch")
    report = {
        "trained_tensors": len(shapes),
        "trained_values": sum(map(math.prod, shapes.values())),
        "branches": {},
        "runtime_or_semantic_equivalence_established": False,
    }
    for branch in BRANCHES:
        original, actual = originals[branch], exported[branch]
        require(set(actual) == set(original), "export changed the original raw tensor inventory")
        trained_raw = {raw: name for name, raw in mapping[branch].items()}
        preserved, changed, h = [], 0, hashlib.sha256()
        for raw in sorted(original):
            finite_f32(actual[raw])
            expected = trained[trained_raw[raw]] if raw in trained_raw else original[raw]
            require(
                same_bits(actual[raw], expected),
                "export changed tensor values: " + branch + "/" + raw,
            )
            if raw in trained_raw:
                changed += not same_bits(actual[raw], original[raw])
            else:
                preserved.append(raw)
            metadata = json.dumps(
                [raw, str(actual[raw].dtype), list(actual[raw].shape)], separators=(",", ":")
            ).encode()
            h.update(len(metadata).to_bytes(8, "little"))
            h.update(metadata)
            raw_bytes = actual[raw].tobytes(order="C")
            h.update(len(raw_bytes).to_bytes(8, "little"))
            h.update(raw_bytes)
        report["branches"][branch] = {
            "mapped_training_tensors": len(trained_raw),
            "changed_training_tensors": changed,
            "preserved_original_raw_keys": preserved,
            "raw_tensor_state_sha256": h.hexdigest(),
        }
    return report
