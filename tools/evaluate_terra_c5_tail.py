"""Frozen C5/WER gates for native-feature decoder-tail Terra checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def decision(plan, predictions, *, plan_sha, checkpoint_sha):
    """The same complete/prefix C5 and historical gates as the visual evaluator."""
    from audit_winner_student_prefix import quality_bounds
    from c5_full_coverage_mps import comparison

    bounds = quality_bounds(plan["evaluation"], plan["winner_control"], predictions)
    complete = len(predictions) == len(plan["evaluation"])
    result = {
        "evaluation_task_complete": True,
        "complete_benchmark_score": complete,
        "plan_sha256": plan_sha,
        "checkpoint_sha256": checkpoint_sha,
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
    return result


def run(path, checkpoint, update, output):
    import numpy as np
    import torch
    from audit_winner_student_prefix import quality_bounds
    from benchmark_fleurs_dev import distance, normalize, sha256
    from c5_certification_budget import CertificationComputeWindow
    from c5_terra_tail import fixed_hashes, load_model
    from cache_fleurs_encoder import checked_stream_features
    from evaluate_terra_visual import install_weights
    from evaluate_tiny_rgb_feature_winner import CpuBudget, peak_bytes
    from probe_tiny_rgb_capacity import state_digest
    from safetensors import safe_open
    from safetensors.torch import load_file
    from terra_training_storage import storage
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from train_terra_c5_tail import checked_plan
    from transformers import ByT5Tokenizer, StoppingCriteriaList
    from transformers.modeling_outputs import BaseModelOutput

    plan, _ = checked_plan(path)
    checkpoint, output = checkpoint.resolve(), output.resolve()
    if (
        output.exists()
        or not output.is_relative_to(ROOT / "checkpoints")
        or not checkpoint.is_relative_to(ROOT / "checkpoints")
        or update not in plan["recipe"]["checks_after_examples"]
    ):
        raise ValueError("C5 tail CPU evaluation requires a fresh bound checkpoint output")
    storage()
    output.mkdir(parents=True)
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    model, reference = load_model()
    encoder = load_frozen_winner_encoder(reference).requires_grad_(False).eval()
    fixed = fixed_hashes(model, encoder)
    with safe_open(checkpoint, framework="pt", device="cpu") as stream:
        metadata = stream.metadata()
    install_weights(
        model,
        load_file(checkpoint, device="cpu"),
        metadata,
        parent_sha=sha256(path),
        reference_sha=reference["translator_checkpoint_sha256"],
        update=update,
    )
    initial_policy = state_digest(model.policy_parameters())
    model.requires_grad_(False)
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )

    def append(name, row):
        with (output / name).open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    predictions = []
    for row in plan["evaluation"]:
        target = (ROOT / row["cache"]).resolve()
        if (
            row["role"] != "open_development"
            or row["training_allowed"] is not False
            or not target.is_relative_to(ROOT / "data")
            or sha256(target) != row["cache_sha256"]
        ):
            raise ValueError("native C5 CPU evaluation changed protected roles/cache bytes")
        torch.set_num_threads(budget.cpu_threads())
        with np.load(target, allow_pickle=False) as cache:
            arrays = checked_stream_features(
                {name: cache[name] for name in cache.files}, len(cache["face_features"])
            )
        started = time.perf_counter()
        try:
            with torch.no_grad():
                memory = encoder(
                    **{name: torch.from_numpy(value)[None] for name, value in arrays.items()},
                    return_dict=True,
                ).last_hidden_state
                tokens = model.language.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                    **reference["generation"],
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
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
        hypothesis = tokenizer.decode(tokens[0], skip_special_tokens=True)
        words = len(normalize(row["reference"]))
        edits = distance(normalize(row["reference"]), normalize(hypothesis))
        prediction = {
            **{key: row[key] for key in ("example_id", "domain", "signer", "source_group")},
            "checkpoint_examples": update,
            "model": "terra_native_C5_tail_cpu",
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
        print(json.dumps(runtime), flush=True)
        bounds = quality_bounds(plan["evaluation"], plan["winner_control"], predictions)
        if bounds["full_quality_gate_mathematically_impossible"]:
            break
        storage()
    result = decision(plan, predictions, plan_sha=sha256(path), checkpoint_sha=sha256(checkpoint))
    if (
        fixed_hashes(model, encoder) != fixed
        or state_digest(model.policy_parameters()) != initial_policy
    ):
        raise RuntimeError("native C5 CPU evaluation changed model tensor values")
    result["frozen_backbones_changed"] = False
    (output / "decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    for key, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ.setdefault(key, value)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.checkpoint, args.update, args.output_dir)


if __name__ == "__main__":
    main()
