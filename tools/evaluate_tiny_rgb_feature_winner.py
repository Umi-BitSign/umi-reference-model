"""Isolated CPU scoring of the full-backbone student's immutable checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_LIMITS = {"maximum_process_peak_bytes": 24 * 1024**3, "maximum_seconds_per_clip": 1200}


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


class CpuBudget:
    def __init__(self, *, clock=time.perf_counter, memory=peak_bytes):
        self.clock, self.memory, self.calls = clock, memory, 0
        self.started = clock()

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        self.calls += 1
        if self.calls == 1 or self.calls % 64 == 0:
            if (
                self.clock() - self.started > CPU_LIMITS["maximum_seconds_per_clip"]
                or self.memory() > CPU_LIMITS["maximum_process_peak_bytes"]
            ):
                raise RuntimeError("full-backbone CPU scoring exceeded its resource policy")
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def install_weights(student, values, metadata, *, parent_sha, reference_sha, update):
    import torch

    expected = student.trainable_parameters()
    if (
        metadata.get("plan_sha256") != parent_sha
        or metadata.get("fixed_winner_checkpoint") != reference_sha
        or metadata.get("update") != str(update)
        or set(values) != set(expected)
        or any(
            value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or not torch.isfinite(value).all()
            for name, value in values.items()
        )
    ):
        raise ValueError("feature winner checkpoint source/topology/finite values differ")
    with torch.no_grad():
        for name, value in values.items():
            expected[name].copy_(value)


def run(path, checkpoint, update, output):
    import torch
    from audit_winner_student_prefix import quality_bounds
    from benchmark_fleurs_dev import distance, normalize, sha256
    from c5_certification_budget import CertificationComputeWindow
    from c5_full_coverage_mps import comparison
    from c5_winner_language_decoder import load_winner_language
    from probe_tiny_rgb_capacity import state_digest
    from safetensors import safe_open
    from safetensors.torch import load_file
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_feature_winner_student import RawFeatureWinnerStudent
    from tiny_rgb_full_training_tools import decode_rgb_video_fast
    from tiny_rgb_winner_student import load_frozen_winner_encoder, winner_length_buckets
    from tiny_rgb_winner_student_protocol import BUDGET
    from train_tiny_rgb_capacity_cpu import storage
    from train_tiny_rgb_feature_winner_full import checked_plan
    from transformers import ByT5Tokenizer, StoppingCriteriaList

    plan, _ = checked_plan(path)
    output, checkpoint = output.resolve(), checkpoint.resolve()
    if (
        output.exists()
        or not output.is_relative_to(ROOT / "checkpoints")
        or not checkpoint.is_relative_to(ROOT / "checkpoints")
        or update not in plan["recipe"]["checks_after_updates"]
    ):
        raise ValueError("feature CPU scoring requires a fresh bound checkpoint result path")
    storage()
    output.mkdir(parents=True)
    budget = CertificationComputeWindow(BUDGET)
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    language, reference = load_winner_language()
    encoder = load_frozen_winner_encoder(reference)
    student = RawFeatureWinnerStudent(encoder, language).eval()
    fixed_before = {
        "language": state_digest(language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }
    with safe_open(checkpoint, framework="pt", device="cpu") as file:
        metadata = file.metadata()
    install_weights(
        student,
        load_file(checkpoint, device="cpu"),
        metadata,
        parent_sha=sha256(path),
        reference_sha=reference["translator_checkpoint_sha256"],
        update=update,
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )

    def append(name, record):
        with output.joinpath(name).open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    predictions, bounds = [], None
    for row in plan["evaluation"]:
        if (
            row["training_allowed"] is not False
            or row["role"] != "open_development"
            or sha256(ROOT / row["video"]) != row["video_sha256"]
        ):
            raise ValueError("feature CPU scoring cannot change protected roles/video bytes")
        torch.set_num_threads(budget.cpu_threads())
        pixels, _, _ = decode_rgb_video_fast(
            row["video"],
            plan["recipe"]["short_side"],
            plan["recipe"]["target_fps"],
            plan["recipe"]["max_frames"],
        )
        video, mask, _ = winner_length_buckets(pixel_tensor(pixels)[None], None)
        started = time.perf_counter()
        try:
            tokens = student.generate(
                video,
                mask,
                **reference["generation"],
                early_stopping=True,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([CpuBudget()]),
            )
        except Exception as error:
            append(
                "runtime-failures.jsonl",
                {
                    "example_id": row["example_id"],
                    "error": repr(error),
                    "partial_hypothesis_scored": False,
                    "peak_bytes": peak_bytes(),
                },
            )
            raise
        runtime = {
            "example_id": row["example_id"],
            "generated_tokens": tokens.shape[1],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_bytes": peak_bytes(),
            "cpu_threads": budget.cpu_threads(),
        }
        append("runtime.jsonl", runtime)
        print(json.dumps(runtime), flush=True)
        hypothesis = tokenizer.decode(tokens[0], skip_special_tokens=True)
        # Consult caption references only after raw-only generation is complete.
        words = len(normalize(row["reference"]))
        edits = distance(normalize(row["reference"]), normalize(hypothesis))
        prediction = {
            **{key: row[key] for key in ("example_id", "domain", "signer", "source_group")},
            "checkpoint_examples": update,
            "model": "raw_feature_winner_student_cpu",
            "role": "open_development",
            "training_allowed": False,
            "hypothesis": hypothesis,
            "reference_words": words,
            "edits": edits,
            "wer": edits / words,
            "execution_device": "cpu",
        }
        predictions.append(prediction)
        append("evaluation-results.jsonl", prediction)
        bounds = quality_bounds(plan["evaluation"], plan["winner_control"], predictions)
        if bounds["full_quality_gate_mathematically_impossible"]:
            break
        storage()
    complete = len(predictions) == len(plan["evaluation"])
    result = {
        "evaluation_task_complete": True,
        "complete_benchmark_score": complete,
        "plan_sha256": sha256(path),
        "checkpoint_sha256": sha256(checkpoint),
        "evaluated_clips": len(predictions),
        "quality_lower_bounds": bounds,
        "passes_open_development_gate": False,
        "eligible_for_native_composite_validation": False,
        "optimizer_updates": 0,
        "promotion_authorized": False,
        "complete_composite_verified": False,
        "miner_modified": False,
        "confirmation_decoded": False,
    }
    if complete:
        primary = comparison(plan["winner_control"], predictions)
        original = comparison(
            plan["historical_controls"]["original_checkpoint_common_inputs"], predictions
        )
        selected = comparison(plan["historical_controls"]["selected_parent"], predictions)
        original_guard = all(
            value["mean_wer"] <= value["parent_mean_wer"]
            and value["word_weighted_wer"] <= value["parent_word_weighted_wer"]
            for value in original["domains"].values()
        )
        result.update(
            {
                **primary,
                "vs_historical_original": original,
                "vs_historical_selected": selected,
                "passes_original_baseline_guard": original_guard,
                "eligible_for_native_composite_validation": primary["passes_open_development_gate"]
                and original_guard,
            }
        )
    else:
        result["historical_original_lower_bounds"] = quality_bounds(
            plan["evaluation"],
            plan["historical_controls"]["original_checkpoint_common_inputs"],
            predictions,
        )
        result["historical_selected_lower_bounds"] = quality_bounds(
            plan["evaluation"], plan["historical_controls"]["selected_parent"], predictions
        )
        if not bounds["full_quality_gate_mathematically_impossible"]:
            raise ValueError("incomplete CPU result lacks a proven frozen-gate rejection")
    if fixed_before != {
        "language": state_digest(language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }:
        raise RuntimeError("CPU evaluation changed fixed winner sequence/language")
    result["frozen_backbones_changed"] = False
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.checkpoint, args.update, args.output_dir)


if __name__ == "__main__":
    main()
