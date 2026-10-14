"""Evaluate preserved raw-winner student weights without Metal generation graphs."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/tiny-rgb-winner-full-v1/plan.json"
CHECKPOINT = ROOT / "checkpoints/tiny-rgb-winner-full-v1/visual-after-002048.safetensors"
STEPS = CHECKPOINT.parent / "steps.jsonl"
SAMPLE = PARENT.parent / "owned-process-sample-2054.txt"
RECIPE = {
    "device": "cpu",
    "generation": "unchanged_winner_five_beam_2048",
    "optimizer_updates": 0,
    "checkpoint_update": 2048,
    "maximum_process_peak_bytes": 16 * 1024**3,
    "maximum_seconds_per_clip": 1200,
    "resource_failure": "raise_and_record_never_return_truncated_hypothesis",
}


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


class GenerationBudget:
    """A failing resource guard raises; successful decoding never stops early."""

    def __init__(self, *, clock=time.perf_counter, memory=peak_bytes):
        self.clock, self.memory = clock, memory
        self.started = clock()
        self.calls = 0

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        self.calls += 1
        if self.calls == 1 or self.calls % 64 == 0:
            if (
                self.clock() - self.started > RECIPE["maximum_seconds_per_clip"]
                or self.memory() > RECIPE["maximum_process_peak_bytes"]
            ):
                raise RuntimeError("CPU winner evaluation exceeded its recorded resource budget")
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def install_trainable_weights(student, values, metadata, parent_sha, reference_sha):
    import torch

    expected = student.trainable_parameters()
    if (
        metadata.get("plan_sha256") != parent_sha
        or metadata.get("fixed_winner_checkpoint") != reference_sha
        or metadata.get("update") != str(RECIPE["checkpoint_update"])
        or set(values) != set(expected)
        or any(
            value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or not torch.isfinite(value).all()
            for name, value in values.items()
        )
    ):
        raise ValueError("CPU student checkpoint topology/provenance/nonfinite weights differ")
    with torch.no_grad():
        for name, value in values.items():
            expected[name].copy_(value)


def sources():
    from benchmark_fleurs_dev import sha256
    from train_tiny_rgb_winner_full import sources as parent_sources

    return {**parent_sources(), "tools/evaluate_tiny_rgb_winner_cpu.py": sha256(Path(__file__))}


def parent_inputs():
    import math

    from train_tiny_rgb_winner_full import checked_plan

    parent, pairs = checked_plan(PARENT)
    rows = [json.loads(line) for line in STEPS.read_text().splitlines()]
    if (
        len(rows) != RECIPE["checkpoint_update"]
        or [row["update"] for row in rows] != list(range(1, len(rows) + 1))
        or any(
            row["gradient_tensors"] != 53
            or row["nonzero_gradient_tensors"] != 53
            or any(
                not math.isfinite(row[key])
                for key in ("loss", "caption_loss", "latent_loss", "feature_loss", "gradient_norm")
            )
            for row in rows
        )
    ):
        raise ValueError("CPU evaluation requires complete preserved finite 2048-update ledger")
    probe = min(
        pairs, key=lambda pair: (pair["header"]["sampled_frames"], pair["raw"]["example_id"])
    )
    return parent, probe


def checked_plan(path):
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_full_run import runtime_binding

    plan = json.loads(path.read_text())
    parent, probe = parent_inputs()
    if (
        plan["schema"] != "umi-studio-raw-winner-cpu-evaluation/1"
        or plan["recipe"] != RECIPE
        or plan["source_sha256"] != sources()
        or plan["runtime_binding"] != runtime_binding()
        or plan["parent_plan_sha256"] != sha256(PARENT)
        or plan["checkpoint_sha256"] != sha256(CHECKPOINT)
        or plan["step_ledger_sha256"] != sha256(STEPS)
        or plan["process_sample_sha256"] != sha256(SAMPLE)
        or plan["evaluation"] != parent["evaluation"]
        or plan["generation"] != parent["reference"]["generation"]
        or plan["training_runtime_probe"] != probe
    ):
        raise ValueError("CPU winner evaluation immutable bindings changed")
    snapshot = json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("CPU winner evaluation differs from source archive")
    return plan, parent


def freeze(path):
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from tiny_rgb_full_run import runtime_binding
    from train_tiny_rgb_capacity_cpu import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("CPU winner evaluation requires a new runs/ directory")
    parent, probe = parent_inputs()
    plan = {
        "schema": "umi-studio-raw-winner-cpu-evaluation/1",
        "recipe": RECIPE,
        "source_sha256": sources(),
        "runtime_binding": runtime_binding(),
        "parent_plan": str(PARENT.relative_to(ROOT)),
        "parent_plan_sha256": sha256(PARENT),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "step_ledger_sha256": sha256(STEPS),
        "process_sample_sha256": sha256(SAMPLE),
        "evaluation": parent["evaluation"],
        "generation": parent["reference"]["generation"],
        "training_runtime_probe": probe,
        "reason": "Owned Metal decode stopped with explicit terminal143 after first2048 "
        "updates; sample shows repeated MPSGraph compilation and101.3GiB footprint, "
        "with zero completed predictions. Original sources/logs/weights preserved.",
        "miner_modified": False,
        "promotion_authorized": False,
        "confirmation_decoded": False,
    }
    storage(len(json.dumps(plan).encode()) + 1024**2)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps({"plan_sha256": sha256(path), "evaluations": len(plan["evaluation"])}),
        flush=True,
    )


def run(path, output):
    import torch
    from benchmark_fleurs_dev import distance, normalize, sha256
    from c5_certification_budget import CertificationComputeWindow
    from c5_full_coverage_mps import comparison
    from c5_winner_language_decoder import load_winner_language
    from probe_tiny_rgb_capacity import state_digest
    from safetensors import safe_open
    from safetensors.torch import load_file
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_full_training_tools import decode_rgb_video_fast
    from tiny_rgb_winner_student import RawRgbWinnerStudent, winner_length_buckets
    from tiny_rgb_winner_student_protocol import BUDGET
    from train_tiny_rgb_capacity_cpu import storage
    from transformers import ByT5Tokenizer, StoppingCriteriaList

    plan, parent = checked_plan(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("CPU winner evaluation requires fresh private runs/ output")
    storage()
    output.mkdir(parents=True)
    budget = CertificationComputeWindow(BUDGET)
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    language, reference = load_winner_language()
    language_digest = state_digest(language.state_dict())
    student = RawRgbWinnerStudent(language).eval()
    with safe_open(CHECKPOINT, framework="pt", device="cpu") as file:
        metadata = file.metadata()
    install_trainable_weights(
        student,
        load_file(CHECKPOINT, device="cpu"),
        metadata,
        plan["parent_plan_sha256"],
        reference["translator_checkpoint_sha256"],
    )
    if (
        len(student.trainable_parameters()) != 53
        or sum(value.numel() for value in student.trainable_parameters().values()) != 6474334
    ):
        raise ValueError("CPU winner student trainable topology changed")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )

    def append(name, record):
        with output.joinpath(name).open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def decode(row, *, role):
        if sha256(ROOT / row["video"]) != row["video_sha256"]:
            raise ValueError("CPU winner raw video bytes changed")
        torch.set_num_threads(budget.cpu_threads())
        pixels, _, _ = decode_rgb_video_fast(
            row["video"],
            parent["recipe"]["short_side"],
            parent["recipe"]["target_fps"],
            parent["recipe"]["max_frames"],
        )
        video, mask, _ = winner_length_buckets(pixel_tensor(pixels)[None], None)
        started = time.perf_counter()
        try:
            tokens = student.generate(
                video,
                mask,
                **plan["generation"],
                early_stopping=True,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([GenerationBudget()]),
            )
        except Exception as error:
            append(
                "runtime-failures.jsonl",
                {
                    "example_id": row["example_id"],
                    "role": role,
                    "error": repr(error),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_bytes": peak_bytes(),
                    "partial_hypothesis_scored": False,
                },
            )
            raise
        result = {
            "example_id": row["example_id"],
            "role": role,
            "generated_tokens": tokens.shape[1],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_bytes": peak_bytes(),
            "cpu_threads": budget.cpu_threads(),
        }
        append("runtime.jsonl", result)
        print(json.dumps(result), flush=True)
        return tokenizer.decode(tokens[0], skip_special_tokens=True)

    probe = plan["training_runtime_probe"]["raw"]
    if probe["role"] != "training" or probe["training_allowed"] is not True:
        raise ValueError("CPU runtime probe cannot use protected identities")
    decode(probe, role="training_runtime_only")
    predictions = []
    for row in plan["evaluation"]:
        if row["training_allowed"] is not False or row["role"] != "open_development":
            raise ValueError("CPU quality evaluation role changed")
        hypothesis = decode(row, role="open_development")
        # References are consulted only after generation; never condition inference.
        words = len(normalize(row["reference"]))
        edits = distance(normalize(row["reference"]), normalize(hypothesis))
        prediction = {
            **{key: row[key] for key in ("example_id", "domain", "signer", "source_group")},
            "checkpoint_examples": RECIPE["checkpoint_update"],
            "model": "raw_winner_student_cpu",
            "role": "open_development",
            "training_allowed": False,
            "hypothesis": hypothesis,
            "reference_words": words,
            "edits": edits,
            "wer": edits / words,
            "input_policy": "raw_aspect192_native25fps_even_frames",
            "execution_device": "cpu",
        }
        predictions.append(prediction)
        append("evaluation-results.jsonl", prediction)
        storage()
    primary = comparison(parent["winner_control"], predictions)
    original = comparison(
        parent["historical_controls"]["original_checkpoint_common_inputs"], predictions
    )
    selected = comparison(parent["historical_controls"]["selected_parent"], predictions)
    original_guard = all(
        values["mean_wer"] <= values["parent_mean_wer"]
        and values["word_weighted_wer"] <= values["parent_word_weighted_wer"]
        for values in original["domains"].values()
    )
    if state_digest(student.language.state_dict()) != language_digest:
        raise ValueError("CPU evaluation changed fixed winner language")
    checked_plan(path)
    result = {
        "complete": True,
        "plan_sha256": sha256(path),
        "checkpoint_sha256": sha256(CHECKPOINT),
        **primary,
        "vs_historical_original": original,
        "vs_historical_selected": selected,
        "passes_original_baseline_guard": original_guard,
        "eligible_for_native_composite_validation": primary["passes_open_development_gate"]
        and original_guard,
        "peak_process_bytes": peak_bytes(),
        "optimizer_updates": 0,
        "language_changed": False,
        "complete_composite_verified": False,
        "promotion_authorized": False,
        "miner_modified": False,
        "confirmation_decoded": False,
    }
    output.joinpath("decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    for name, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[name] = value
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
        if args.output_dir:
            run(args.freeze, args.output_dir)
    elif args.plan and args.output_dir:
        run(args.plan, args.output_dir)
    else:
        parser.error("use --freeze or --plan with --output-dir")


if __name__ == "__main__":
    main()
