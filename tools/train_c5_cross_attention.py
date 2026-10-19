"""Fresh public q/k alignment training on the clean complete expanded corpus.

Four fixed full epochs, authentic graded risk on517 clips, declared balanced
caption/retention auxiliaries on all3041 clips, and one final checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from c5_expanded_training_admission import require_training_row
from c5_timing_feature_data import (
    ROOT,
    array_metadata,
    artifact,
    bound,
    owned,
    read,
    require,
    sha,
    terminal,
)
from launch_terra_visual_job import identity, now, write

WIDTHS = {
    "face_features": 384,
    "left_hand_features": 384,
    "right_hand_features": 384,
    "pose_features": 14,
}
DATA = ROOT / "runs/c5-expanded-native-feature-audit-v3/audit.json"
RISK = ROOT / "runs/c5-expanded-risk-materials-v1/report.json"
PROOF = ROOT / "checkpoints/c5-cross-attention-alignment-proof-v3/qualification.json"
ADMISSION = ROOT / "runs/c5-expanded-training-admission-v1/scope.json"
RECIPE = {
    "epochs": 4,
    "clips_per_epoch": 3041,
    "risk_clips": 517,
    "batch_size": 8,
    "planned_exposures": 12164,
    "planned_optimizer_updates": 1524,
    "shuffle_seed": 20261019,
    "rank": 8,
    "adapter_seed": 20261019,
    "optimizer": "AdamW",
    "learning_rate": 0.0005,
    "warmup_updates": 32,
    "cosine_final_fraction": 0.2,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "gradient_norm_limit": 1.0,
    "risk_alpha": 1.0,
    "risk_objective": "mean authentic expected error across517 eligible original pools",
    "risk_sample_weight": 3041 / 517,
    "caption_weight": 0.1,
    "retention_weight": 0.01,
    "retention_temperature": 1.0,
    "auxiliary_family_mass": {"word": 0.25, "STEM_phrase": 0.25, "How2Sign": 0.25, "OpenASL": 0.25},
    "auxiliary_sample_weight": "3041/(4*family_clip_count); all clips once per epoch",
    "ungraded_supplemental_risk": "none; only declared human-caption and frozen-original KL losses",
    "candidate_refresh": "none; exact authentic source candidates excluding14 protected donors",
    "parent_model_or_optimizer_inheritance": False,
    "memory": "fresh full public encoder outputs for every clip before the first update",
    "known_risk_inputs": "exact original fresh native features for all517 graded inputs",
    "selection": "only final after all four whole epochs; no development checkpoint selection",
    "checkpoint_every_updates": 32,
    "CPU_threads": 2,
    "GPU_use": False,
    "maximum_output_bytes": 6 * 1024**3,
    "physical_free_floor_bytes": 84 * 1024**3,
    "maximum_process_bytes": 32 * 1024**3,
    "new_Terra_requests": 0,
    "held_out_quality_established": False,
    "promotion_authorized": False,
}


def family(record):
    if record["kind"] == "word":
        return "word"
    source = record["row"]["source"]
    if source.startswith("asl_stem"):
        return "STEM_phrase"
    if source == "how2sign_precut_expansion":
        return "How2Sign"
    require(source == "openasl_scale", "alignment auxiliary has an unknown source family")
    return "OpenASL"


def batches():
    result = []
    for epoch in range(1, 5):
        order = list(range(3041))
        random.Random(RECIPE["shuffle_seed"] + epoch - 1).shuffle(order)
        result.extend({"epoch": epoch, "indices": order[o : o + 8]} for o in range(0, 3041, 8))
    return result


def rate(number):
    if number <= RECIPE["warmup_updates"]:
        return RECIPE["learning_rate"] * number / RECIPE["warmup_updates"]
    progress = (number - RECIPE["warmup_updates"]) / (1524 - RECIPE["warmup_updates"])
    return RECIPE["learning_rate"] * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * progress)))


def hashes(values):
    return {
        name: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, v in values.items()
    }


def specification():
    data, risk, proof, admission = (read(p) for p in (DATA, RISK, PROOF, ADMISSION))
    require(
        sha(DATA) == "a31c34adda082d041b2ab0d66a66d027dddd75bf204de9c9e1835c972ed017dd"
        and data["complete"] is True
        and data["raw_feature_training_data_admitted"] is True
        and data["clips"] == 3041,
        "training needs the complete clean native feature audit",
    )
    authority = terminal(PROOF.parent)
    poll_path = ROOT / "runs/c5-cross-attention-alignment-proof-v3/terminal-tool-poll.json"
    poll, worker = read(poll_path), read(bound(authority["worker.json"]))
    require(
        proof["complete"] is True
        and proof["worker"] == authority["worker.json"]
        and poll["actual_owned_pid"] == worker["pid"]
        and poll["tool_result"]["exit_code"] == 0
        and "session_id" not in poll["tool_result"]
        and len(proof["comparisons"]) == 16
        and proof["configuration"]["trainable_parameters"] == 221184
        and proof["original_public_values_unchanged"] is True
        and proof["optimizer_updates"] == 0,
        "training needs actual terminal original-public gradient qualification",
    )
    proof_plan = read(bound(proof["plan"]))
    require(
        Path(worker["argv"][1]).absolute() == bound(proof_plan["program"]),
        "qualified worker program changed",
    )
    require(
        all(
            sha(ROOT / "tools" / name) == digest
            for name, digest in proof_plan["source_sha256"].items()
        ),
        "qualified alignment source changed",
    )
    expected_comparisons = {
        (i, state, graded)
        for i in (0, 1, 214, 215)
        for state in ("zero_identity", "fixed_nonzero_output_factors")
        for graded in (True, False)
    }
    require(
        {(c["index"], c["state"], c["graded"]) for c in proof["comparisons"]}
        == expected_comparisons,
        "qualified gradient coverage changed",
    )
    for comparison in proof["comparisons"]:
        require(
            abs(comparison["actual_objective"]["loss"] - comparison["reference_loss"]) <= 1e-5,
            "qualified full-graph loss differs",
        )
        with np.load(bound(comparison["gradients"]), allow_pickle=False) as archive:
            require(
                set(archive.files)
                == {
                    prefix + name
                    for name in comparison["metrics"]
                    for prefix in ("actual.", "reference.")
                },
                "qualified gradient inventory changed",
            )
            for name, metrics in comparison["metrics"].items():
                actual, reference = archive["actual." + name], archive["reference." + name]
                require(
                    actual.dtype == reference.dtype == np.float32
                    and actual.shape == reference.shape
                    and np.isfinite(actual).all()
                    and np.isfinite(reference).all()
                    and np.allclose(actual, reference, rtol=0.005, atol=1e-7)
                    and float(np.max(np.abs(actual - reference)))
                    == metrics["maximum_absolute_error"],
                    "qualified full-autograd gradients changed",
                )
    for name in ("program",):
        bound(data[name])
        bound(risk[name])
    require(
        risk["complete"] is True
        and risk["clean_risk_clips"] == 517
        and risk["excluded_protected_donor_targets"] == 14
        and risk["admission"] == artifact(ADMISSION),
        "training needs the exact clean authenticated risk materials",
    )
    for asset in risk["all_original_grading_assets"]:
        bound(asset)
    known = {r["index"]: r for r in read(bound(risk["records"]))}
    protected_groups = set(admission["protected_groups"])
    protected_references = {
        tuple(key) for key in admission["protected_sentence_references_normalized"]
    }
    records = read(bound(data["records"]))
    for record in records:
        require_training_row(record["row"], protected_groups, protected_references)
        index = record["current_risk_index"]
        if index is not None:
            current = known[index]["record"]
            require(
                record["row"] == current["row"]
                and record["header"] == current["header"]
                and all(
                    record["visibility_arrays"][name] == current["arrays"][name]
                    for name in record["visibility_arrays"]
                ),
                "risk features lost their original exact source/grid join",
            )
            record["features"] = current["features"]
            record["arrays"] = {name: current["arrays"][name] for name in WIDTHS}
    require(
        len(records) == 3041 and sum(r["current_risk_index"] is not None for r in records) == 517,
        "complete alignment training inventory changed",
    )
    counts = dict(Counter(family(r) for r in records))
    require(
        set(counts) == set(RECIPE["auxiliary_family_mass"]),
        "alignment auxiliary lost a source family",
    )
    return {
        "schema": "umi-studio-cross-attention-alignment-training-plan/1",
        "program": artifact(Path(__file__).absolute()),
        "recipe": RECIPE,
        "data_audit": artifact(DATA),
        "risk_materials": artifact(RISK),
        "admission": artifact(ADMISSION),
        "gradient_qualification": artifact(PROOF),
        "gradient_qualification_poll": artifact(poll_path),
        "model_plan": proof_plan["materials"]["model_plan"],
        "configuration": proof["configuration"],
        "original_controls": proof["controls"],
        "records": records,
        "risk_records": risk["records"],
        "family_counts": counts,
        "batches": batches(),
        "parameter_order": [
            f"language.decoder.block.{layer}.layer.1.EncDecAttention.{projection}.{part}.weight"
            for layer in range(6)
            for projection in ("q", "k")
            for part in ("input_projection", "output_projection")
        ],
        "source_sha256": {
            name: sha(ROOT / "tools" / name)
            for name in (
                "train_c5_cross_attention.py",
                "c5_cross_attention_alignment.py",
                "c5_cross_attention_risk.py",
                "c5_expanded_training_admission.py",
                "c5_expanded_risk_materials.py",
                "c5_timing_feature_data.py",
                "c5_retention.py",
                "c5_terra_tail.py",
                "terra_quality_training.py",
                "evaluate_c5_words.py",
                "evaluate_roi_short_translation.py",
                "c5_winner_language_decoder.py",
                "train_terra_c5_tail.py",
                "prepare_c5_timing_features.py",
            )
        },
    }


def checked(path):
    plan = read(path)
    require(
        plan["program"] == artifact(Path(__file__).absolute())
        and plan["recipe"] == RECIPE
        and plan["batches"] == batches()
        and all(
            sha(ROOT / "tools" / name) == digest for name, digest in plan["source_sha256"].items()
        ),
        "alignment frozen source or recipe changed",
    )
    for name in (
        "data_audit",
        "risk_materials",
        "admission",
        "gradient_qualification",
        "gradient_qualification_poll",
        "risk_records",
    ):
        bound(plan[name])
    return plan


def guard(output):
    import shutil

    from evaluate_tiny_rgb_feature_winner import peak_bytes

    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) if output.exists() else 0
    remaining = RECIPE["maximum_output_bytes"] - size
    require(
        remaining >= 0
        and shutil.disk_usage(ROOT).free
        >= RECIPE["physical_free_floor_bytes"] + remaining + 128 * 1024**2
        and peak_bytes() <= RECIPE["maximum_process_bytes"],
        "alignment training resource allowance exceeded",
    )


def run(path, output):
    import torch
    from c5_cross_attention_alignment import CrossAttentionAlignment
    from c5_cross_attention_risk import backward_alignment_risk
    from c5_terra_tail import candidate_labels
    from evaluate_c5_words import models
    from evaluate_roi_short_translation import decode
    from prepare_c5_timing_features import require_environment
    from probe_tiny_rgb_capacity import state_digest
    from train_terra_c5_tail import gpu_lease

    require_environment()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan = checked(path)
    owned(output, "checkpoints")
    require(
        not output.exists() and not path.with_name("execution-claim.json").exists(),
        "fresh one-use alignment training required",
    )
    guard(output)
    output.mkdir()
    (output / "memory").mkdir()
    (output / "steps").mkdir()
    (output / "recovery").mkdir()
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    write(
        path.with_name("execution-claim.json"),
        {"plan": artifact(path), "worker": artifact(output / "worker.json"), "claimed_at": now()},
    )
    code = 1
    try:
        with gpu_lease():
            original = plan["model_plan"]
            encoder, language, tokenizer = models(
                {
                    "reference": original["reference"],
                    "initial_reference_hashes": original["initial_reference_hashes"],
                }
            )
            initial = {
                "encoder": state_digest(encoder.state_dict()),
                "language": state_digest(language.state_dict()),
            }
            memory_assets, controls = [], {c["index"]: c for c in plan["original_controls"]}
            for index, record in enumerate(plan["records"]):
                guard(output)
                with np.load(bound(record["features"]), allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in WIDTHS}
                require(
                    {name: array_metadata(value) for name, value in arrays.items()}
                    == record["arrays"],
                    "training native feature values changed",
                )
                with torch.no_grad():
                    memory = encoder(
                        **{name: torch.from_numpy(value)[None] for name, value in arrays.items()},
                        return_dict=True,
                    ).last_hidden_state
                require(
                    memory.shape == (1, record["header"]["sampled_frames"], 1536)
                    and torch.isfinite(memory).all(),
                    "training original memory lost complete source positions",
                )
                if record["current_risk_index"] in controls:
                    require(
                        array_metadata(memory[0].numpy())
                        == controls[record["current_risk_index"]]["memory_metadata"],
                        "training memory differs from actual original-public qualification",
                    )
                target = output / "memory" / f"memory-{index:04d}.npy"
                with target.open("xb") as destination:
                    np.save(destination, memory[0].numpy(), allow_pickle=False)
                memory_assets.append(artifact(target))
                if (index + 1) % 100 == 0 or index == 3040:
                    print(
                        json.dumps(
                            {"stage": "fresh_original_memory", "clips": index + 1, "of": 3041}
                        ),
                        flush=True,
                    )
            write(
                output / "memory.json",
                {
                    "complete": True,
                    "plan": artifact(path),
                    "files": memory_assets,
                    "old_cached_semantic_states_used": False,
                },
            )
            model = CrossAttentionAlignment(language, rank=8, seed=20261019)
            selected = model.trainable_parameters()
            require(
                model.configuration() == plan["configuration"]
                and list(selected) == plan["parameter_order"],
                "alignment optimizer scope/configuration changed",
            )
            for index, record in enumerate(plan["records"]):
                if record["current_risk_index"] in controls:
                    with torch.no_grad():
                        memory = torch.from_numpy(
                            np.load(bound(memory_assets[index]), allow_pickle=False)
                        )[None]
                        prediction = decode(
                            model.language, tokenizer, original["reference"]["generation"], memory
                        )
                        require(
                            list(prediction) == controls[record["current_risk_index"]]["baseline"],
                            "fresh training zero adapter changed complete native generation",
                        )
            known = {r["index"]: r for r in read(bound(plan["risk_records"]))}
            targets = []
            for record in plan["records"]:
                if record["current_risk_index"] is not None:
                    material = known[record["current_risk_index"]]
                    candidates, quality = material["candidates"], material["quality"]
                else:
                    candidates = [
                        {
                            "kind": "human_anchor",
                            "text": record["row"]["reference"],
                            "tokens": tokenizer(
                                record["row"]["reference"], truncation=False
                            ).input_ids,
                            "truncated": False,
                            "termination": "eos",
                        }
                    ]
                    quality = None
                require(
                    tokenizer(record["row"]["reference"], truncation=False).input_ids
                    == candidates[0]["tokens"],
                    "alignment human target tokens changed",
                )
                targets.append((candidate_labels(candidates), quality))
            write(
                output / "optimizer-claim.json",
                {
                    "plan": artifact(path),
                    "worker": artifact(output / "worker.json"),
                    "claimed_at": now(),
                    "all3041_fresh_original_memories_complete": True,
                    "all_four_complete_generation_controls_passed": True,
                    "initial_state_hashes": hashes(selected),
                },
            )
            optimizer = torch.optim.AdamW(
                list(selected.values()),
                lr=RECIPE["learning_rate"],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
            )
            steps, exposures = [], 0
            for number, batch in enumerate(plan["batches"], start=1):
                guard(output)
                optimizer.zero_grad(set_to_none=True)
                before, samples = hashes(selected), []
                for index in batch["indices"]:
                    record = plan["records"][index]
                    memory = torch.from_numpy(
                        np.load(bound(memory_assets[index]), allow_pickle=False)
                    )[None]
                    labels, quality = targets[index]
                    auxiliary_scale = 3041 / (4 * plan["family_counts"][family(record)])
                    details = backward_alignment_risk(
                        model,
                        memory,
                        torch.ones(memory.shape[:2], dtype=torch.bool),
                        labels,
                        quality,
                        risk_weight=RECIPE["risk_sample_weight"] if quality is not None else 0.0,
                        caption_weight=0.1 * auxiliary_scale,
                        retention_weight=0.01 * auxiliary_scale,
                        divisor=len(batch["indices"]),
                    )
                    samples.append(
                        {
                            "index": index,
                            "risk_record_index": record["current_risk_index"],
                            "family": family(record),
                            "auxiliary_scale": auxiliary_scale,
                            "objective": details,
                        }
                    )
                require(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in selected.values()
                    ),
                    "alignment update lacks complete finite gradients",
                )
                gradient_path = output / "steps" / f"gradient-{number:04d}.npz"
                with gradient_path.open("xb") as destination:
                    np.savez(
                        destination,
                        **{name: p.grad.detach().numpy().copy() for name, p in selected.items()},
                    )
                norm = float(torch.nn.utils.clip_grad_norm_(list(selected.values()), 1.0))
                require(math.isfinite(norm), "alignment gradient norm overflowed")
                learning_rate = rate(number)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                model.require_scope()
                exposures += len(batch["indices"])
                step_path = output / "steps" / f"step-{number:04d}.json"
                write(
                    step_path,
                    {
                        "step": number,
                        "batch": batch,
                        "exposures": exposures,
                        "before": before,
                        "after": hashes(selected),
                        "gradient_before_clip": artifact(gradient_path),
                        "gradient_norm_before_clip": norm,
                        "learning_rate": learning_rate,
                        "samples": samples,
                    },
                )
                steps.append(artifact(step_path))
                if number % 32 == 0 or number == 1524:
                    target = output / "recovery" / f"recovery-{number:04d}.pt"
                    torch.save(
                        {
                            "plan": artifact(path),
                            "step": number,
                            "exposures": exposures,
                            "alignment": {name: p.detach().clone() for name, p in selected.items()},
                            "optimizer": optimizer.state_dict(),
                        },
                        target,
                    )
                if number % 10 == 0 or number == 1524:
                    print(
                        json.dumps(
                            {
                                "stage": "alignment_training",
                                "step": number,
                                "of": 1524,
                                "epoch": batch["epoch"],
                                "exposures": exposures,
                                "last_batch_mean_loss": sum(s["objective"]["loss"] for s in samples)
                                / len(samples),
                            }
                        ),
                        flush=True,
                    )
                if number % 381 == 0:
                    require(plan == checked(path), "frozen training sources changed during epoch")
            require(
                exposures == 12164
                and len(steps) == 1524
                and initial
                == {
                    "encoder": state_digest(encoder.state_dict()),
                    "language": state_digest(model.frozen_state()),
                },
                "alignment final is incomplete or changed original C5 values",
            )
            final = output / "final-adapter.pt"
            torch.save(
                {
                    "schema": "umi-studio-cross-attention-alignment-final/1",
                    "configuration": model.configuration(),
                    "alignment": {name: p.detach().clone() for name, p in selected.items()},
                    "plan": artifact(path),
                    "steps": 1524,
                    "exposures": 12164,
                },
                final,
            )
            optimizer_path = output / "final-optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
            write(
                output / "result.json",
                {
                    "schema": "umi-studio-complete-cross-attention-alignment-training/1",
                    "complete": True,
                    "plan": artifact(path),
                    "worker": artifact(output / "worker.json"),
                    "adapter": artifact(final),
                    "optimizer": artifact(optimizer_path),
                    "memory": artifact(output / "memory.json"),
                    "steps": steps,
                    "exposures": exposures,
                    "optimizer_updates": 1524,
                    "completed_epochs": 4,
                    "training_clips": 3041,
                    "risk_clips": 517,
                    "configuration": model.configuration(),
                    "final_state_hashes": hashes(selected),
                    "original_public_values_unchanged": True,
                    "new_Terra_requests": 0,
                    "quality_gain_established": False,
                    "trained_native_runtime_qualified": False,
                    "independent_confirmation_consumed": False,
                },
            )
        code = 0
    except BaseException as error:
        write(
            output / "failure.json",
            {"error_type": type(error).__name__, "message": str(error), "automatic_retries": 0},
        )
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=lambda p: Path(p).absolute())
    parser.add_argument("--plan", type=lambda p: Path(p).absolute())
    parser.add_argument("--output", type=lambda p: Path(p).absolute())
    args = parser.parse_args()
    if args.freeze:
        owned(args.freeze, "runs")
        require(not args.freeze.parent.exists(), "fresh alignment plan directory required")
        plan = specification()
        args.freeze.parent.mkdir()
        write(args.freeze, plan)
    if args.output:
        require(args.plan is not None, "training output requires an explicit frozen plan")
        run(args.plan, args.output)
    elif not args.freeze:
        parser.error("freeze a plan or run its fresh training output")


if __name__ == "__main__":
    main()
