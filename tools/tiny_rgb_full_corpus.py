"""Freeze all eligible RGB training identities and separate already-open quality controls."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from c5_experiment_sources import archive_sources
from c5_full_coverage_mps import GATES, load_training
from c5_stationary_schedule import mixture, stationary_schedule
from tiny_rgb_fit import ROOT, SEED, SIDES, checked_backend, private_path

FULL_RECIPE = {
    "visual_learning_rate": 3e-4,
    "language_learning_rate": 5e-5,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "gradient_accumulation": 4,
    "warmup_updates": 256,
    "minimum_learning_rate_ratio": 0.1,
    "domain_mass": {"how2sign": 0.60, "fleurs": 0.20, "asl_stem": 0.15, "openasl": 0.05},
    "checkpoint_after_exposures": [2048, 8192, 16384, 32768, 48800],
    "target_fps": 12.5,
    "max_frames": 512,
    "max_text_tokens": 256,
    "activation_implementation": "portable_exact_gelu",
    "augmentation": "none",
    "generation": {"num_beams": 3, "max_new_tokens": 256, "do_sample": False},
    "maximum_process_mps_bytes": 6 * 1024**3,
    "minimum_free_bytes": 80 * 1024**3,
    "data_budget_bytes": 4 * 1024**3,
}


def full_schedule(rows: list[dict]) -> list[dict]:
    if any(r.get("role") != "training" or r.get("training_allowed") is not True for r in rows):
        raise ValueError("full RGB training cannot accept an evaluation or protected identity")
    prepared = [{**r, "id": r["example_id"]} for r in rows]
    return stationary_schedule(prepared, FULL_RECIPE["domain_mass"], SEED, maximum_exposures=48800)


def fit_passes(decision: dict, plan_sha: str) -> dict[int, dict]:
    if (
        decision["plan_sha256"] != plan_sha
        or not decision["complete"]
        or decision["training_examples_per_arm"] != 24
        or decision["optimizer_updates_per_arm"] != 6144
        or decision["held_out_quality_verified"]
        or decision["selected_model_changed"]
    ):
        raise ValueError("full RGB planning requires the completed matched256 fit control")
    final_rows = [r for r in decision["checks"] if r["epoch"] == 256]
    final = {r["short_side"]: r for r in final_rows}
    if (
        len(final_rows) != 2
        or set(final) != set(SIDES)
        or not all(
            r["passes_training_fit_gate"]
            and math.isfinite(r["training_mean_wer"])
            and 0 <= r["training_mean_wer"] <= 0.25
            and r["training_exact_translations"] >= 12
            and math.isfinite(r["wrong_video_wer_increase"])
            and r["wrong_video_wer_increase"] >= 0.10
            and not r["held_out_quality_verified"]
            and not r["promotion_authorized"]
            for r in final.values()
        )
    ):
        raise ValueError("both full RGB resolution arms must first pass the original fit gate")
    return final


def validate_split(training: list[dict], evaluation: list[dict]) -> None:
    identities = {(r["source"], r["example_id"]) for r in training}
    groups = {r["source_group"] for r in training}
    references = {normalize(r["reference"]) for r in training}
    if len(identities) != len(training):
        raise ValueError("full RGB training identities repeat")
    if any(r["role"] != "training" or r["training_allowed"] is not True for r in training):
        raise ValueError("full RGB training contains protected material")
    if len({r["example_id"] for r in evaluation}) != len(evaluation) or any(
        r["role"] != "open_development"
        or r["training_allowed"] is not False
        or r["source_group"] in groups
        or normalize(r["reference"]) in references
        or (r["source"], r["example_id"]) in identities
        for r in evaluation
    ):
        raise ValueError("full RGB source/reference split isolation failed")


def open_rgb_evaluation() -> tuple[list[dict], dict[str, str]]:
    """Bind only the existing142 open cases to the exact clips used by fixed controls."""
    fresh_path = ROOT / "runs/c5-fresh-face-benchmark-v1/plan.json"
    pilot_path = ROOT / "runs/c5-face-learning-pilot-v1/plan.json"
    control_path = ROOT / "runs/c5-full-coverage-mps-v1/plan.json"
    fresh, pilot, control = (
        json.loads(p.read_text()) for p in (fresh_path, pilot_path, control_path)
    )
    receipt_path = fresh_path.with_name("prepared.jsonl")
    prepared = {r["example_id"]: r for r in read_rows(receipt_path)}
    raw = {}
    for row in fresh["examples"]:
        identity = row["sentence_name"]
        receipt = prepared[identity]
        if receipt["training_allowed"] or receipt["plan_sha256"] != sha256(fresh_path):
            raise ValueError("open RGB preparation source receipt changed")
        video = f"{fresh['data_dir']}/clips-25fps/{identity}.mp4"
        raw[identity] = {
            "video": video,
            "video_sha256": receipt["clip_sha256"],
            "video_fps": 25.0,
            "source": "how2sign_fresh_open",
        }
    for row in pilot["examples"]:
        if row["role"] == "pilot_source_holdout":
            raw[row["example_id"]] = {
                "video": row["clip"],
                "video_sha256": row["clip_sha256"],
                "video_fps": row["clip_fps"],
                "source": row["source"],
            }
    evaluation = []
    for row in control["evaluation"]:
        if row["domain"] == "fleurs":
            binding = {
                "video": row["clip"],
                "video_sha256": row["clip_sha256"],
                "video_fps": 25.0,
                "source": "fleurs_open",
            }
        else:
            binding = raw[row["example_id"]]
        if sha256(private_path(binding["video"], "data")) != binding["video_sha256"]:
            raise ValueError("open RGB clip differs from its fixed-control source")
        evaluation.append(
            {
                **{
                    k: row[k]
                    for k in ("example_id", "domain", "signer", "source_group", "reference")
                },
                **binding,
                "role": "open_development",
                "training_allowed": False,
            }
        )
    if len(evaluation) != 142:
        raise ValueError("open RGB evaluation count differs from fixed controls")
    dependencies = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in (fresh_path, pilot_path, control_path, receipt_path)
    }
    return evaluation, dependencies


def read_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines()]


def frozen_controls(evaluation: list[dict]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    path = ROOT / "checkpoints/c5-full-coverage-mps-v1/evaluation-results.jsonl"
    original = read_rows(path)
    models = sorted({r["model"] for r in original if r["checkpoint_examples"] == 0})
    if len(models) != 2:
        raise ValueError("fixed control identities are incomplete")
    outputs = {}
    by_id = {r["example_id"]: r for r in evaluation}
    for name in models:
        rows = [r for r in original if r["model"] == name and r["checkpoint_examples"] == 0]
        if len(rows) != 142 or {r["example_id"] for r in rows} != set(by_id):
            raise ValueError("fixed controls differ from raw RGB evaluation identities")
        for row in rows:
            reference = normalize(by_id[row["example_id"]]["reference"])
            if row["reference_words"] != len(reference) or row["edits"] != distance(
                reference, normalize(row["hypothesis"])
            ):
                raise ValueError("fixed control scoring/reference binding differs")
        outputs[name] = rows
    return outputs, {str(path.relative_to(ROOT)): sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-plan", type=Path, required=True)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("full RGB planning needs a new private runs/ directory")
    fit_plan_path = args.fit_plan.resolve()
    fit_plan = json.loads(fit_plan_path.read_text())
    decision_path = args.fit_output.resolve() / "decision.json"
    fit = fit_passes(json.loads(decision_path.read_text()), sha256(fit_plan_path))
    corpus = ROOT / "runs/c5-all-source-corpus-v2"
    eligible = load_training(corpus / "audit.json", corpus / "training-streams.jsonl")
    raw_dir = ROOT / "runs/tiny-rgb-corpus-v1"
    audit_path, inventory = raw_dir / "audit.json", raw_dir / "training-inventory.jsonl"
    raw_audit = json.loads(audit_path.read_text())
    if sha256(inventory) != raw_audit["training_inventory_sha256"]:
        raise ValueError("full RGB raw corpus audit changed")
    training = read_rows(inventory)
    if {(r["source"], r["example_id"]) for r in training} != {
        (r["source"], r["id"]) for r in eligible
    }:
        raise ValueError("full RGB corpus differs from all protected eligible identities")
    evaluation, dependencies = open_rgb_evaluation()
    validate_split(training, evaluation)
    controls, control_dependencies = frozen_controls(evaluation)
    dependencies.update(control_dependencies)
    dependencies.update(raw_audit["source_inventory_sha256"])
    for path in (fit_plan_path, decision_path, inventory, audit_path):
        dependencies[str(path.relative_to(ROOT))] = sha256(path)
    for name, digest in {**dependencies, **fit_plan["language_asset_sha256"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("full RGB planning dependency changed")
    backend = checked_backend(fit_plan["examples"], ROOT / fit_plan["backend"]["original_plan"])
    if backend != fit_plan["backend"]:
        raise ValueError("full RGB planning numerical proof differs from fit")
    schedule = full_schedule(training)
    if len(schedule) != 48800 or len({(r["source"], r["id"]) for r in schedule}) != 7529:
        raise ValueError("full RGB balanced replay did not cover every unique training clip")
    output.mkdir(parents=True)
    schedule_path = output / "schedule.jsonl"
    with schedule_path.open("x") as stream:
        for row in schedule:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    source_names = (
        "tiny_rgb_full_corpus.py",
        "tiny_rgb_fit.py",
        "tiny_rgb_data.py",
        "tiny_rgb_model.py",
        "tiny_rgb_language_model.py",
        "tiny_rgb_mps_safe.py",
        "c5_stationary_schedule.py",
        "c5_all_source_training.py",
        "c5_coverage_schedule.py",
        "c5_full_coverage_mps.py",
        "audit_c5_all_source_corpus.py",
        "c5_gpu_budget.py",
    )
    plan = {
        "schema": "umi-studio-tiny-rgb-full-corpus-proposal/1",
        "planning_only": True,
        "seed": SEED,
        "short_sides": list(SIDES),
        "recipe": FULL_RECIPE,
        "gates": GATES,
        "training_inventory": str(inventory.relative_to(ROOT)),
        "training_examples": len(training),
        "planned_exposures_per_arm": len(schedule),
        "planned_updates_per_arm": len(schedule) // 4,
        "schedule_sha256": sha256(schedule_path),
        "evaluation": evaluation,
        "fixed_controls": controls,
        "fit_gate_evidence": fit,
        "numerical_backend": backend,
        "language_path": fit_plan["language_path"],
        "language": fit_plan["language"],
        "language_asset_sha256": fit_plan["language_asset_sha256"],
        "visual_config": fit_plan["visual_config"],
        "initial_diagnostic_weights_reused": False,
        "pretrained_visual_weights": False,
        "learned_feature_inputs": False,
        "c5_teacher_targets": False,
        "dependency_sha256": dependencies,
        "source_sha256": {f"tools/{name}": sha256(ROOT / "tools" / name) for name in source_names},
        "domain_counts": dict(Counter(r["source"] for r in training)),
        "mixture": {
            "all": mixture(schedule),
            "first512": mixture(schedule[:512]),
            "last512": mixture(schedule[-512:]),
        },
        "maximum_training_repeat_count": max(
            Counter((r["source"], r["id"]) for r in schedule).values()
        ),
        "qualification": (
            "Planning-only full raw-pixel experiment, not a training launch or quality result. "
            "All7529 unique protected-eligible clips are included with stationary domain/signer "
            "replay and unit exposure weights. Minority repetitions can overfit. Complete matched "
            "12.5fps aspect-preserving128/192 RGB; generic pretrained language only. Existing142 "
            "open cases remain source/reference-disjoint and are never trained on. Fixed C5 "
            "controls are earlier verified predictions on these exact clip bytes, not untouched "
            "full-rate runtime or official scores. Source-held-out gates and independent new "
            "confirmation must pass before promotion. A bounded streaming trainer, complete "
            "resource/runtime bindings, frame/target validation and preflight remain required."
        ),
    }
    plan_path = output / "proposal.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(plan_path, output / "source-snapshot")
    print(
        json.dumps(
            {
                "proposal_sha256": sha256(plan_path),
                "training_examples": len(training),
                "exposures_per_arm": len(schedule),
                "evaluation_examples": len(evaluation),
                "planning_only": True,
            }
        )
    )


if __name__ == "__main__":
    main()
