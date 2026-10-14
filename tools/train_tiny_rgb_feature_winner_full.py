"""Full-corpus appearance learning through the complete frozen winning sequence model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROOF_PLAN = ROOT / "runs/tiny-rgb-feature-winner-preflight-v1/plan.json"
PROOF = PROOF_PLAN.parent / "proof/receipt.json"
RGB_PARENT = ROOT / "runs/tiny-rgb-full-v3-buckets/plan.json"
WINNER_RESULTS = ROOT / "checkpoints/c5-winner-full-v2-early/evaluation-results.jsonl"
RECIPE = {
    "seed": 317,
    "epochs": 4,
    "learning_rate": 3e-4,
    "warmup_updates": 128,
    "minimum_lr_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "gradient_accumulation": 1,
    "latent_coefficient": 0.5,
    "feature_coefficient": 1.0,
    "capacity": "large",
    "retained_native_semantic_encoder": True,
    "evaluation_device": "isolated_cpu_subprocess",
    "optimizer_state_saved_at_each_check": True,
    "short_side": 192,
    "target_fps": 12.5,
    "max_frames": 512,
    "max_text_tokens": 512,
    "checks_after_updates": [2048, 8192, 16384, 30092],
    "sampler": "each_native_identity_once_per_epoch_no_minority_repeats",
    "augmentation": "none",
    "teacher_states": "fresh_winner_online_no_disk_cache",
    "initial_diagnostic_weights_reused": False,
    "frozen_language": True,
}


def lr_factor(update, total=30092):
    if update < RECIPE["warmup_updates"]:
        return (update + 1) / RECIPE["warmup_updates"]
    progress = min(1.0, (update - RECIPE["warmup_updates"]) / (total - RECIPE["warmup_updates"]))
    return (
        RECIPE["minimum_lr_ratio"]
        + (1 - RECIPE["minimum_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
    )


def order_key(pair, epoch):
    row = pair["raw"]
    return hashlib.sha256(
        f"winner-raw317:{epoch}:{row['source']}:{row['example_id']}".encode()
    ).hexdigest()


def optimizer_payload(optimizer, parameters, rng_states):
    """Copy serializable state without moving live parameters or optimizer tensors."""
    values = {
        f"rng.{name}": value.detach().cpu().contiguous() for name, value in rng_states.items()
    }
    for name, parameter in parameters.items():
        state = optimizer.state[parameter]
        for field in ("step", "exp_avg", "exp_avg_sq"):
            values[f"{name}.{field}"] = state[field].detach().cpu().contiguous()
    metadata = {
        "parameter_order": json.dumps(list(parameters)),
        "groups_without_parameters": json.dumps(
            [
                {key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups
            ]
        ),
    }
    return values, metadata


def sources():
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_feature_winner_protocol import sources as proof_sources

    return {
        **proof_sources(),
        "tools/train_tiny_rgb_feature_winner_full.py": sha256(Path(__file__)),
        "tools/evaluate_tiny_rgb_feature_winner.py": sha256(
            ROOT / "tools/evaluate_tiny_rgb_feature_winner.py"
        ),
        "tools/audit_winner_student_prefix.py": sha256(
            ROOT / "tools/audit_winner_student_prefix.py"
        ),
    }


def checked_proof():
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_capacity_protocol import verified_comparison
    from tiny_rgb_feature_winner_protocol import RECIPE as qualified_recipe
    from tiny_rgb_feature_winner_protocol import sources as qualified_sources

    qualification = json.loads(PROOF_PLAN.read_text())
    if (
        qualification["recipe"] != qualified_recipe
        or qualification["source_sha256"] != qualified_sources()
    ):
        raise ValueError("feature winner numerical qualification implementation changed")
    proof = json.loads(PROOF.read_text())
    records = [json.loads(s) for s in PROOF.with_name("comparisons.jsonl").read_text().splitlines()]
    if (
        proof["plan_sha256"] != sha256(PROOF_PLAN)
        or proof["numerics_passed"] is not True
        or proof["complete"] is not True
        or proof["optimizer_updates"] != 0
        or proof["language_weights_changed"] is not False
        or len(proof["real_cases"]) != 6
        or proof["trainable_parameter_tensors"] != 90
        or len(records) != 24
        or proof["semantic_encoder_weights_changed"] is not False
        or proof["teacher_weights_trainable"] is not False
        or proof["weights_saved"] is not False
    ):
        raise ValueError("winner raw full training requires complete unchanged numerical proof")
    stages = {
        "frozen_winner_teacher_forward",
        "native_cpu_equivalence",
        "cpu_padding",
        "cpu_mps_comparison",
    }
    expected = {
        (case["source"], case["example_id"], stage)
        for case in proof["real_cases"]
        for stage in stages
    }
    if (
        len(expected) != 24
        or {(row["source"], row["example_id"], row["stage"]) for row in records} != expected
    ):
        raise ValueError("winner raw numerical proof case/stage coverage differs")
    for row in records:
        if row["optimizer_updates"] != 0 or row["training_only"] is not True:
            raise ValueError("winner raw numerical proof scope differs")
        if "comparison" in row:
            verified_comparison(
                row["comparison"], 90, cpu_equivalence=row["stage"] == "native_cpu_equivalence"
            )
        elif (
            not math.isfinite(row["maximum_absolute_difference"])
            or not 0 <= row["maximum_absolute_difference"] <= 1e-3
        ):
            raise ValueError("winner raw semantic teacher forward proof differs")
    return proof


def inputs():
    from benchmark_fleurs_dev import distance, normalize
    from tiny_rgb_full_corpus import validate_split
    from tiny_rgb_winner_student_protocol import checked_corpus

    corpus, pairs = checked_corpus()
    parent = json.loads(RGB_PARENT.read_text())
    evaluation = parent["evaluation"]
    validate_split([p["raw"] for p in pairs], evaluation)
    winner = [
        r
        for r in (json.loads(s) for s in WINNER_RESULTS.read_text().splitlines())
        if r["model"] == "winner_matched_translator_control" and r["checkpoint_examples"] == 0
    ]
    by_id = {r["example_id"]: r for r in evaluation}
    if len(winner) != 142 or len(by_id) != 142 or {r["example_id"] for r in winner} != set(by_id):
        raise ValueError("winner raw open control coverage differs")
    for row in winner:
        reference = normalize(by_id[row["example_id"]]["reference"])
        if row["reference_words"] != len(reference) or row["edits"] != distance(
            reference, normalize(row["hypothesis"])
        ):
            raise ValueError("winner raw fixed control metrics/reference changed")
    return corpus, pairs, evaluation, winner, parent["fixed_controls"]


def checked_plan(path):
    from benchmark_fleurs_dev import sha256
    from c5_full_coverage_mps import GATES
    from c5_winner_reference import load_reference
    from tiny_rgb_full_run import runtime_binding
    from tiny_rgb_winner_student_protocol import BUDGET

    plan = json.loads(path.read_text())
    if (
        plan["schema"] != "umi-studio-feature-winner-full-training/1"
        or plan["recipe"] != RECIPE
        or plan["source_sha256"] != sources()
        or plan["runtime_binding"] != runtime_binding()
        or plan["gates"] != GATES
        or plan["reference"] != load_reference()
        or plan["resource_policy_sha256"] != sha256(BUDGET)
        or plan["proof_sha256"] != sha256(PROOF)
    ):
        raise ValueError("winner raw full plan/source/runtime/quality policy changed")
    checked_proof()
    for name, digest in plan["dependency_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("winner raw full frozen dependency changed")
    corpus, pairs, evaluation, winner, historical = inputs()
    if (
        plan["native_pairs_sha256"] != corpus["native_pairs_sha256"]
        or plan["training_examples"] != len(pairs)
        or plan["planned_updates"] != len(pairs) * RECIPE["epochs"]
        or evaluation != plan["evaluation"]
        or winner != plan["winner_control"]
        or historical != plan["historical_controls"]
    ):
        raise ValueError("winner raw full protected selection/control changed")
    snapshot = json.loads(path.with_name("source-snapshot").joinpath("manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("winner raw full plan differs from source archive")
    return plan, pairs


def freeze(path):
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from c5_full_coverage_mps import GATES
    from c5_winner_reference import load_reference
    from tiny_rgb_full_run import runtime_binding
    from tiny_rgb_winner_student_protocol import BUDGET
    from train_tiny_rgb_capacity_cpu import storage
    from transformers import ByT5Tokenizer

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("winner raw full training requires a new runs/ plan")
    checked_proof()
    corpus, pairs, evaluation, winner, historical = inputs()
    reference = load_reference()
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    lengths = [len(tokenizer(p["raw"]["reference"], truncation=False).input_ids) for p in pairs]
    if max(lengths) > RECIPE["max_text_tokens"]:
        raise ValueError("winner raw full target audit cannot truncate references")
    dependencies = dict(corpus["dependency_sha256"])
    for file in (
        PROOF_PLAN,
        PROOF,
        PROOF.with_name("comparisons.jsonl"),
        RGB_PARENT,
        WINNER_RESULTS,
    ):
        dependencies[str(file.relative_to(ROOT))] = sha256(file)
    for row in evaluation:
        dependencies[row["video"]] = row["video_sha256"]
    plan = {
        "schema": "umi-studio-feature-winner-full-training/1",
        "recipe": RECIPE,
        "gates": GATES,
        "reference": reference,
        "source_sha256": sources(),
        "runtime_binding": runtime_binding(),
        "dependency_sha256": dependencies,
        "proof_sha256": sha256(PROOF),
        "native_pairs": corpus["native_pairs"],
        "native_pairs_sha256": corpus["native_pairs_sha256"],
        "training_examples": len(pairs),
        "planned_updates": len(pairs) * RECIPE["epochs"],
        "maximum_complete_byte_target": max(lengths),
        "evaluation": evaluation,
        "winner_control": winner,
        "historical_controls": historical,
        "resource_policy": str(BUDGET.relative_to(ROOT)),
        "resource_policy_sha256": sha256(BUDGET),
        "miner_modified": False,
        "confirmation_decoded": False,
        "promotion_authorized": False,
        "qualification": "Fresh large raw192 appearance; complete fixed winner sequence/language. "
        "HumanCE, winning encoder online states and coefficient1 native feature targets; "
        "no preloading full RGB corpus or storing large latent caches. All7523 identities "
        "once per epoch. Old142 common-input translator controls remain an already-open "
        "development benchmark, not native complete-composite/official quality. Passing "
        "gates still requires native FS/router and new independent confirmation.",
    }
    storage(len(json.dumps(plan).encode()) + 1024**2)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, path.with_name("source-snapshot"))
    checked_plan(path)
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "training_clips": len(pairs),
                "updates": plan["planned_updates"],
                "max_byte_target": max(lengths),
            }
        ),
        flush=True,
    )


def run(path, output):
    import torch
    from benchmark_fleurs_dev import sha256
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_winner_language_decoder import load_winner_language
    from probe_tiny_rgb_capacity import state_digest
    from safetensors.torch import save_file
    from tiny_rgb_data import pixel_tensor
    from tiny_rgb_feature_winner_student import RawFeatureWinnerStudent
    from tiny_rgb_full_training_tools import decode_rgb_video_fast
    from tiny_rgb_visual_teacher_data import load_targets
    from tiny_rgb_winner_student import (
        load_frozen_winner_encoder,
        student_objective,
        winner_length_buckets,
    )
    from tiny_rgb_winner_student_protocol import BUDGET
    from train_tiny_rgb_capacity_cpu import storage
    from transformers import ByT5Tokenizer

    plan, pairs = checked_plan(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("winner raw full run needs fresh checkpoint output")
    storage()
    output.mkdir(parents=True)
    budget = CertificationComputeWindow(BUDGET)
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("winner raw full training requires qualified Metal")
    duty = WinnerDutyCycle(torch, budget)
    language, reference = load_winner_language()
    language_digest = state_digest(language.state_dict())
    teacher = load_frozen_winner_encoder(reference)
    encoder_digest = state_digest(teacher.state_dict())
    student = RawFeatureWinnerStudent(teacher, language)
    torch.manual_seed(RECIPE["seed"])
    duty.run(lambda: student.to("mps"))
    duty.run(lambda: teacher.to("mps"))
    if (
        len(student.trainable_parameters()) != 90
        or sum(p.numel() for p in student.trainable_parameters().values()) != 17526750
    ):
        raise RuntimeError("winner raw full optimizer topology changed")
    optimizer = torch.optim.AdamW(
        student.trainable_parameters().values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
        foreach=False,
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    frozen_versions = {name: p._version for name, p in student.language.named_parameters()}
    teacher_versions = {name: p._version for name, p in teacher.named_parameters()}
    output.joinpath("run.json").write_text(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "recipe": RECIPE,
                "language_sha256": language_digest,
                "sequence_encoder_sha256": encoder_digest,
                "miner_mode": "user_authorized_maintenance",
                "miner_modified": False,
            },
            indent=2,
        )
        + "\n"
    )

    def append(name, record):
        with output.joinpath(name).open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def check_frozen():
        for module, versions in ((student.language, frozen_versions), (teacher, teacher_versions)):
            if any(
                p.grad is not None or p.requires_grad or p._version != versions[name]
                for name, p in module.named_parameters()
            ):
                raise RuntimeError("winner raw full changed fixed language/teacher tensors")

    def evaluate(update, checkpoint):
        import subprocess

        result_dir = output / f"cpu-evaluation-{update:06d}"
        log_path = output / f"cpu-evaluation-{update:06d}.log"
        # Keep all live optimizer parameters on Metal. A separate CPU process
        # loads only the saved appearance weights plus verified frozen backbones.
        with log_path.open("x") as stream:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/evaluate_tiny_rgb_feature_winner.py"),
                    "--plan",
                    str(path),
                    "--checkpoint",
                    str(checkpoint),
                    "--update",
                    str(update),
                    "--output-dir",
                    str(result_dir),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        decision = json.loads((result_dir / "decision.json").read_text())
        if decision["checkpoint_sha256"] != sha256(checkpoint):
            raise RuntimeError("CPU feature winner evaluated different appearance weights")
        return decision

    def save_optimizer(checkpoint, update):
        values, state_metadata = optimizer_payload(
            optimizer,
            student.trainable_parameters(),
            {"cpu": torch.get_rng_state(), "mps": torch.mps.get_rng_state()},
        )
        destination = checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")
        save_file(
            values,
            str(destination),
            metadata={
                "plan_sha256": sha256(path),
                "update": str(update),
                **state_metadata,
            },
        )
        return {
            "optimizer_checkpoint": str(destination.relative_to(ROOT)),
            "optimizer_checkpoint_sha256": sha256(destination),
        }

    updates, seen, decisions = 0, set(), []
    for epoch in range(1, RECIPE["epochs"] + 1):
        for pair in sorted(pairs, key=lambda pair, epoch=epoch: order_key(pair, epoch)):
            raw = pair["raw"]
            student.train()
            if sha256(ROOT / raw["video"]) != raw["video_sha256"]:
                raise RuntimeError("winner raw training video changed")
            pixels, indices, info = decode_rgb_video_fast(
                raw["video"], RECIPE["short_side"], RECIPE["target_fps"], RECIPE["max_frames"]
            )
            if (
                len(pixels) != pair["header"]["sampled_frames"]
                or hashlib.sha256(indices.tobytes()).hexdigest()
                != pair["header"]["frame_indices_sha256"]
                or info["source_shape"] != pair["header"]["source_shape"]
            ):
                raise RuntimeError("winner raw full decoded native frame timeline changed")
            labels = tokenizer(raw["reference"], return_tensors="pt", truncation=False).input_ids
            if labels.shape[1] > RECIPE["max_text_tokens"]:
                raise RuntimeError("winner raw full never truncates human captions")
            video, mask, target_labels = winner_length_buckets(pixel_tensor(pixels)[None], labels)
            features = {
                name: torch.from_numpy(value)[None] for name, value in load_targets(pair).items()
            }
            rate = RECIPE["learning_rate"] * lr_factor(updates, plan["planned_updates"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            frames = len(pixels)

            def update(
                video=video,
                mask=mask,
                target_labels=target_labels,
                features=features,
                frames=frames,
            ):
                feature_gpu = {name: value.to("mps") for name, value in features.items()}
                with torch.no_grad():
                    latent = teacher(**feature_gpu, return_dict=True).last_hidden_state
                if latent.shape != (1, frames, 1536) or not torch.isfinite(latent).all():
                    raise RuntimeError("winner raw full teacher state is invalid")
                latent_padded = latent.new_zeros(1, video.shape[1], 1536)
                latent_padded[:, : latent.shape[1]] = latent
                padded_features = {}
                for name, value in feature_gpu.items():
                    padded_features[name] = value.new_zeros(1, video.shape[1], value.shape[-1])
                    padded_features[name][:, : value.shape[1]] = value
                optimizer.zero_grad(set_to_none=True)
                predicted = student(video.to("mps"), mask.to("mps"), target_labels.to("mps"))
                objective = student_objective(
                    predicted,
                    latent_padded,
                    padded_features,
                    latent_coefficient=RECIPE["latent_coefficient"],
                    feature_coefficient=RECIPE["feature_coefficient"],
                )
                if not torch.isfinite(objective["loss"]):
                    raise RuntimeError("winner raw full objective is nonfinite")
                objective["loss"].backward()
                norms = []
                for name, parameter in student.trainable_parameters().items():
                    if parameter.grad is None:
                        raise RuntimeError(f"winner raw student lacks gradient: {name}")
                    norms.append(parameter.grad.float().norm())
                values = torch.stack(norms).detach().cpu().tolist()
                if len(values) != 90 or any(not math.isfinite(value) for value in values):
                    raise RuntimeError("winner raw full gradients are nonfinite")
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        student.trainable_parameters().values(), RECIPE["gradient_clip"]
                    )
                )
                if not math.isfinite(norm):
                    raise RuntimeError("winner raw full clipped norm is nonfinite")
                check_frozen()
                optimizer.step()
                return {name: float(value) for name, value in objective.items()} | {
                    "gradient_norm": norm,
                    "gradient_tensors": 90,
                    "nonzero_gradient_tensors": sum(value > 0 for value in values),
                }

            measured = duty.run(update)
            updates += 1
            seen.add((raw["source"], raw["example_id"]))
            append(
                "steps.jsonl",
                {
                    "update": updates,
                    "epoch": epoch,
                    "source": raw["source"],
                    "example_id": raw["example_id"],
                    "learning_rate": rate,
                    **measured,
                },
            )
            torch.mps.empty_cache()
            if updates % 128 == 0:
                storage()
                print(
                    json.dumps(
                        {
                            "updates": updates,
                            "epoch": epoch,
                            "unique_clips": len(seen),
                            "last_objective": measured,
                            **duty.receipt(),
                        }
                    ),
                    flush=True,
                )
            if updates not in RECIPE["checks_after_updates"]:
                continue
            storage(17526750 * 12 + 1024**2)
            checkpoint = output / f"visual-after-{updates:06d}.safetensors"
            save_file(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in student.trainable_parameters().items()
                },
                str(checkpoint),
                metadata={
                    "plan_sha256": sha256(path),
                    "update": str(updates),
                    "fixed_winner_checkpoint": reference["translator_checkpoint_sha256"],
                },
            )
            optimizer_receipt = save_optimizer(checkpoint, updates)
            decision = {
                **evaluate(updates, checkpoint),
                **optimizer_receipt,
                "update": updates,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "unique_training_clips": len(seen),
            }
            check_frozen()
            decisions.append(decision)
            output.joinpath("open-decisions.json").write_text(
                json.dumps(decisions, indent=2) + "\n"
            )
            print(json.dumps(decision), flush=True)
            if state_digest(student.language.state_dict()) != language_digest:
                raise RuntimeError("winner raw full checkpoint changed fixed language weights")
            if state_digest(teacher.state_dict()) != encoder_digest:
                raise RuntimeError("feature winner checkpoint changed fixed sequence encoder")
    if updates != plan["planned_updates"] or len(seen) != 7523:
        raise RuntimeError("winner raw full training failed complete corpus/horizon coverage")
    check_frozen()
    if state_digest(teacher.state_dict()) != encoder_digest:
        raise RuntimeError("feature winner final state changed fixed sequence encoder")
    checked_plan(path)
    result = {
        "schema": "umi-studio-feature-winner-full-decision/1",
        "complete": True,
        "plan_sha256": sha256(path),
        "optimizer_updates": updates,
        "unique_training_clips": len(seen),
        "evaluations": decisions,
        "resource_receipt": duty.receipt(),
        "frozen_language_changed": False,
        "frozen_sequence_encoder_changed": False,
        "selected_serving_model_changed": False,
        "complete_composite_verified": False,
        "miner_modified": False,
        "confirmation_decoded": False,
    }
    output.joinpath("decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"complete": True, "decision_sha256": sha256(output / "decision.json")}),
        flush=True,
    )


def main():
    for name, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
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
