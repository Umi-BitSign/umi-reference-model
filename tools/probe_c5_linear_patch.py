"""Qualify the evidenced linear patch projection with the original complete C5 probe."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import FunctionType

import probe_c5_small_input as original
import torch
from benchmark_fleurs_dev import sha256
from c5_patch_projection import LinearPatchProjection
from evaluate_c5_small_input import require_qualification
from evaluate_roi_short_translation import terminal
from launch_terra_visual_job import write
from probe_c5_patch_projection import checked as checked_diagnostic
from probe_tiny_rgb_capacity import compare, state_digest
from terra_training_recovery import artifact, verify_snapshot
from tiny_rgb_capacity_protocol import verified_comparison

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PLAN = ROOT / "runs/c5-patch-projection-isolation-v1/plan.json"
DIAGNOSTIC_RESULT = ROOT / "checkpoints/c5-patch-projection-isolation-v1/diagnosis.json"
POLICY = {
    **original.POLICY,
    "patch_projection": "original14px/stride14 weights; unfold plus linear; no resize or padding",
    "native_CPU_equivalence": "both cases, all350 gradients; original1e-4 CPU output limit",
    "MPS_im2col": "pinned runtime CPU fallback observed; no trainable input pixels",
}


def verified_diagnostic():
    checked_diagnostic(DIAGNOSTIC_PLAN)
    terminal(DIAGNOSTIC_RESULT)
    result = json.loads(DIAGNOSTIC_RESULT.read_text())
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != sha256(DIAGNOSTIC_PLAN)
        or result.get("corrected_first_case_passes") is not True
        or result.get("pretrained_values_unchanged") is not True
        or result.get("optimizer_updates") != 0
        or result.get("new_Terra_requests") != 0
        or result.get("full_component_numerics_qualified") is not False
    ):
        raise ValueError(
            "linear patch qualification requires the actual successful first-case diagnosis"
        )
    for name, equivalence in (
        ("full-backbone-CPU-equivalence.json", True),
        ("full-backbone-CPU-Metal.json", False),
    ):
        verified_comparison(
            json.loads(DIAGNOSTIC_RESULT.with_name(name).read_text()),
            350,
            cpu_equivalence=equivalence,
        )
    return result


def specification():
    base = original.specification()
    diagnostic = verified_diagnostic()
    sources = dict(base["source_sha256"])
    for name in (
        "probe_c5_linear_patch.py",
        "c5_patch_projection.py",
        "probe_c5_patch_projection.py",
        "evaluate_c5_small_input.py",
        "evaluate_roi_short_translation.py",
        "train_terra_c5_tail.py",
    ):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        **base,
        "schema": "umi-studio-C5-linear-patch-component-qualification/1",
        "policy": POLICY,
        "diagnostic_plan": artifact(DIAGNOSTIC_PLAN),
        "diagnostic_result": artifact(DIAGNOSTIC_RESULT),
        "expected_pretrained_state_sha256": diagnostic["pretrained_state_sha256"],
        "source_sha256": sources,
    }


def checked(path):
    plan = json.loads(path.read_text())
    if plan != specification():
        raise ValueError("linear patch qualification changed evidence, inputs, values or limits")
    verify_snapshot(path, plan)
    return plan


def linear_model(hands, face, *, side=112):
    if side not in (112, 224):
        raise ValueError("linear component qualification requires native112 and control224 only")
    if side == 112:
        for model in (hands, face):
            model.patch_embed.proj = LinearPatchProjection(model.patch_embed.proj)
    return original.SmallInputDino(hands, face, side=side)


def native_reference(model):
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("native projection reference requires CPU")
    result = copy.deepcopy(model)
    for name in ("hands", "face"):
        patch = getattr(result, name).patch_embed
        previous = patch.proj
        if not isinstance(previous, LinearPatchProjection):
            raise ValueError("native projection reference requires the qualified linear scope")
        with torch.random.fork_rng(devices=[]):
            conv = torch.nn.Conv2d(3, 384, 14, stride=14)
        conv.weight, conv.bias = previous.weight, previous.bias
        conv.register_forward_pre_hook(result._first_layer)
        patch.proj = conv
    if state_digest(result.state_dict()) != state_digest(model.state_dict()):
        raise ValueError("native CPU reference changed original parameter values")
    return result


def bound_run(output, plan):
    """Retain original numerical-loop bytecode and scientific comparison helpers.

    A private globals mapping supplies the evidenced model factory, immutable
    plan checker and an additional native-CPU comparison before each Metal case.
    The original module, crop/loss/backward functions and limits stay untouched.
    """
    cpu_reports = []

    def feature_probe(model, pixels):
        measured = original.feature_probe(model, pixels)
        if next(model.parameters()).device.type == "cpu":
            index = len(cpu_reports)
            row = plan["records"][index]["original_row"]
            native = original.feature_probe(native_reference(model), pixels)
            comparison = compare(
                native[2], measured[2], native[0], measured[0], native[1], measured[1]
            )
            frames = row["original_pair"]["header"]["sampled_frames"]
            report = {
                "case_index": index,
                "source": row["source"],
                "example_id": row["example_id"],
                "numerical_sample_positions": [0, frames - 1],
                "comparison": comparison,
            }
            write(output / f"native-CPU-equivalence-{len(cpu_reports):02d}.json", report)
            verified_comparison(comparison, 350, cpu_equivalence=True)
            cpu_reports.append(report)
        return measured

    def bound_write(path, value):
        if path.name == "qualification.json":
            if (
                len(cpu_reports) != 2
                or value.get("pretrained_state_sha256") != plan["expected_pretrained_state_sha256"]
            ):
                raise ValueError("linear qualification lacks both actual native CPU comparisons")
            value = {
                **value,
                "native_CPU_equivalence": cpu_reports,
                "patch_projection": POLICY["patch_projection"],
            }
        write(path, value)

    function = original.run
    return FunctionType(
        function.__code__,
        {
            **function.__globals__,
            "checked": checked,
            "SmallInputDino": linear_model,
            "feature_probe": feature_probe,
            "write": bound_write,
            "POLICY": POLICY,
        },
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )


def require_result(result, plan, plan_sha, output):
    require_qualification(result, plan, plan_sha)
    if (
        result.get("pretrained_state_sha256") != plan["expected_pretrained_state_sha256"]
        or result.get("patch_projection") != POLICY["patch_projection"]
        or len(result.get("native_CPU_equivalence", [])) != 2
    ):
        raise ValueError("linear qualification lost original weights or native CPU equivalence")
    for index, report in enumerate(result["native_CPU_equivalence"]):
        row = plan["records"][index]["original_row"]
        frames = row["original_pair"]["header"]["sampled_frames"]
        if (
            report.get("case_index") != index
            or report.get("source") != row["source"]
            or report.get("example_id") != row["example_id"]
            or report.get("numerical_sample_positions") != [0, frames - 1]
            or json.loads((output / f"native-CPU-equivalence-{index:02d}.json").read_text())
            != report
        ):
            raise ValueError("linear qualification differs from actual native CPU comparisons")
        verified_comparison(report["comparison"], 350, cpu_equivalence=True)
    for index, report in enumerate(result["comparisons"]):
        if json.loads((output / f"comparison-{index:02d}.json").read_text()) != report:
            raise ValueError("linear qualification differs from actual CPU/Metal comparisons")


def freeze(path):
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("linear patch qualification requires a fresh owned plan")
    storage()
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(json.dumps({"plan": artifact(path)}), flush=True)


def run(path, output):
    from train_terra_c5_tail import gpu_lease

    # The shared worker owns both terminal and failure receipts. Extra CPU
    # evidence is added before its successful qualification/exit commit.
    plan = checked(path)
    with gpu_lease():
        bound_run(output.resolve(), plan)(path, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "plan", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    if args.plan and args.output:
        run(args.plan, args.output)
    elif args.plan or args.output or not args.freeze:
        parser.error("choose --freeze or --plan/--output")
