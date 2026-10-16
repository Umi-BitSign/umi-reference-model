"""Fine-tune only C5's final two cross-attention blocks using actual Terra errors."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/terra-visual-training-v8-detached-recovery/plan.json"
PARENT_SHA = "0ef51ebe28902a0e4b9f316e5ae2b83eb5ed1a6c4f58eeab0139a92acac73daf"
NATIVE = ROOT / "runs/c5-winner-full-v2-early/plan.json"
NATIVE_SHA = "c3270ed287990d3fa838f546e30af66ee6095fd90e3e8051657ca716e40e56f1"
CONTROL = ROOT / "runs/c5-terra-tail-control-v1"
JUDGE = ROOT / "checkpoints/terra-visual-training-v7-teacher-qualification"
PREDECESSOR = ROOT / "runs/terra-visual-training-v8-training-job"
RECIPE = {
    "seed": 419,
    "epochs": 1,
    "batch_clips": 8,
    "learning_rate": 1e-5,
    "warmup_examples": 128,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "risk_alpha": 0.005,
    "caption_weight": 0.1,
    "judge_model": "gpt-5.6-terra",
    "judge_reasoning_effort": "medium",
    "semantic_error": "10 minus actual Terra quality",
    "trainable_parameters": 9437184,
    "trainable_tensors": 8,
    "trainable_scope": "existing q/k/v/o weights in decoder blocks4 and5 only",
    "candidate_pool": "complete human anchor plus deduplicated reference-blind current greedy",
    "generated_candidates": 1,
    "training_generation": {
        "max_length": 513,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
    },
    "training_token_horizon": 512,
    "maximum_frames": 512,
    "checks_after_examples": [128, 512, 2048, 4096, 7523],
    "qualification": (
        "new actual eight-tensor CPU/Metal proof; authenticated identical C5 text scores"
    ),
    "feature_or_latent_distillation": False,
    "custom_raw_visual_checkpoint_loaded": False,
    "native_visual_and_sequence_encoder_fixed": True,
    "fingerspelling_and_router_weights_fixed": True,
    "graceful_checkpoint_signal": "SIGUSR1",
    "training_cpu_limits": {
        "maximum_process_peak_bytes": 24 * 1024**3,
        "maximum_seconds_per_clip": 1200,
    },
    "checkpoint_serialization": (
        "legacy visual fields/filenames contain only eight decoder-tail tensors"
    ),
}


def artifact(path):
    from terra_training_recovery import artifact as checked_artifact

    return checked_artifact(path, root=ROOT)


def predecessor():
    """Never overlap the previous owned GPU trainer, including bounded qualification."""
    worker_path, exit_path = PREDECESSOR / "worker.json", PREDECESSOR / "exit.json"
    if not exit_path.exists():
        raise ValueError(
            "the previous owned trainer must have an authoritative terminal receipt first"
        )
    worker, terminal = (json.loads(p.read_text()) for p in (worker_path, exit_path))
    if terminal["pid"] != worker["pid"] or type(terminal.get("exit_code")) is not int:
        raise ValueError("previous trainer terminal receipt differs from its owned identity")
    # Check start/UID before command, without inspecting a reused unrelated PID's command.
    metadata = [
        subprocess.run(
            ["ps", "-p", str(worker["pid"]), "-o", key + "="], capture_output=True, text=True
        ).stdout.strip()
        for key in ("uid", "lstart")
    ]
    if metadata == [str(worker["uid"]), worker["ps_started_local"]]:
        raise RuntimeError("the previous owned GPU worker is still present")
    return {"worker": artifact(worker_path), "terminal": artifact(exit_path)}


def sources(base):
    from benchmark_fleurs_dev import sha256

    result = dict(base["source_sha256"])
    for name in (
        "c5_terra_tail.py",
        "train_terra_c5_tail.py",
        "evaluate_terra_c5_tail.py",
        "launch_terra_c5_tail_job.py",
        "launch_terra_visual_job.py",
        "cache_fleurs_encoder.py",
    ):
        result[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return result


def specification():
    from benchmark_fleurs_dev import sha256
    from evaluate_tiny_rgb_feature_winner import CPU_LIMITS
    from terra_teacher_candidates import DECODE
    from terra_training_recovery import verify_snapshot
    from train_terra_visual import checked_plan as parent_plan

    if RECIPE["training_generation"] != DECODE or RECIPE["training_cpu_limits"] != CPU_LIMITS:
        raise ValueError("native C5 training generation or CPU resource policy differs")
    if sha256(PARENT) != PARENT_SHA or sha256(NATIVE) != NATIVE_SHA:
        raise ValueError("native C5 parent data/control plans changed")
    base, pairs = parent_plan(PARENT)
    native = json.loads(NATIVE.read_text())
    verify_snapshot(NATIVE, native)
    if native["reference"] != base["reference"] or native["gates"] != base["gates"]:
        raise ValueError("native C5 baseline reference/quality gates changed")
    raw = {r["example_id"]: r for r in base["evaluation"]}
    evaluation = native["evaluation"]
    if len(evaluation) != 142 or {r["example_id"] for r in evaluation} != set(raw):
        raise ValueError("native C5 evaluation coverage changed")
    for row in evaluation:
        if any(
            row[k] != raw[row["example_id"]][k]
            for k in ("reference", "domain", "signer", "source_group", "role", "training_allowed")
        ):
            raise ValueError("native C5 evaluation labels/identities/roles changed")
    receipt_path, predictions_path, probe_path = (
        CONTROL / n for n in ("receipt.json", "predictions.jsonl", "probe.py")
    )
    receipt = json.loads(receipt_path.read_text())
    predictions = [json.loads(s) for s in predictions_path.read_text().splitlines()]
    expected = {r["example_id"]: r for r in base["winner_control"]}
    if (
        receipt.get("complete") is not True
        or receipt.get("exact_text_matches") != 142
        or receipt.get("all_baseline_weights_and_buffers_unchanged") is not True
        or receipt["parent_sha256"] != NATIVE_SHA
        or receipt["probe_sha256"] != sha256(probe_path)
        or len(predictions) != 142
        or {r["example_id"] for r in predictions} != set(expected)
        or any(
            r["hypothesis"] != expected[r["example_id"]]["hypothesis"]
            or r["edits"] != expected[r["example_id"]]["edits"]
            for r in predictions
        )
        or len(pairs) != RECIPE["checks_after_examples"][-1]
        or any(p.get("feature_pretraining_allowed") is not True for p in pairs)
        or any(not 0 < p["header"]["sampled_frames"] <= RECIPE["maximum_frames"] for p in pairs)
    ):
        raise ValueError(
            "native C5 exact frozen-control reproduction or training permissions failed"
        )
    judge = {
        key: artifact(JUDGE / filename)
        for key, filename in (
            ("receipt", "qualification.json"),
            ("inputs", "qualification-inputs.jsonl"),
            ("request", "qualification-terra/request.json"),
            ("response", "qualification-terra/response.json"),
        )
    }
    result = {
        "schema": "umi-studio-terra-c5-tail-training/1",
        "recipe": RECIPE,
        "parent_plan_sha256": PARENT_SHA,
        "native_evaluation_plan_sha256": NATIVE_SHA,
        "source_sha256": sources(base),
        "training_examples": len(pairs),
        "planned_training_examples": len(pairs) * RECIPE["epochs"],
        "evaluation": evaluation,
        "proof_cases": base["proof_cases"],
        "judge_source": judge,
        "predecessor": predecessor(),
        "baseline_control": {
            "receipt": artifact(receipt_path),
            "predictions": artifact(predictions_path),
            "source": artifact(probe_path),
        },
        "initial_reference_hashes": receipt["initial_hashes"],
        "training_only_judge_requests": True,
        "owner_approved_terra_training_cost": True,
        "promotion_authorized": False,
        "miner_modified": False,
        "confirmation_decoded": False,
        **{
            k: base[k]
            for k in (
                "reference",
                "winner_control",
                "historical_controls",
                "gates",
                "resource_policy",
                "resource_policy_sha256",
                "runtime_binding",
                "native_pairs_sha256",
            )
        },
    }
    return result, pairs


def checked_plan(path):
    from benchmark_fleurs_dev import sha256
    from c5_terra_tail import existing_judge_cases
    from terra_training_recovery import checked_resume, verify_snapshot

    plan = json.loads(path.read_text())
    expected, pairs = specification()
    if {k: v for k, v in plan.items() if k != "resume"} != expected:
        raise ValueError("C5 Terra tail data/source/recipe/control binding changed")
    verify_snapshot(path, plan)
    if sha256(ROOT / plan["resource_policy"]) != plan["resource_policy_sha256"]:
        raise ValueError("C5 Terra resource policy changed")
    existing_judge_cases(plan)
    if "resume" in plan:
        checked_resume(plan)
    return plan, pairs


def freeze(path, *, resume_plan=None, resume_checkpoint=None, qualification=None):
    from benchmark_fleurs_dev import sha256
    from c5_experiment_sources import archive_sources
    from terra_training_recovery import make_resume
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("C5 Terra requires a fresh owned runs/ plan")
    plan, pairs = specification()
    if resume_plan is not None or resume_checkpoint is not None:
        if any(x is None for x in (resume_plan, resume_checkpoint, qualification)):
            raise ValueError(
                "C5 Terra resume requires source plan, full checkpoint and qualification"
            )
        plan["resume"] = make_resume(resume_plan, resume_checkpoint, qualification, plan)
    storage(1024**2)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, path.with_name("source-snapshot"))
    checked_plan(path)
    print(json.dumps({"plan_sha256": sha256(path), "training_examples": len(pairs)}), flush=True)


def append(output, name, value):
    with (output / name).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def gpu_lease():
    path = ROOT / "runs/terra-c5-tail-gpu.lock"
    if path.is_symlink():
        raise ValueError("native C5 GPU lease refuses links")
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another owned C5 Terra GPU job holds the lease") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def checked_qualification(receipt, path, initial, fixed, plan):
    from benchmark_fleurs_dev import sha256
    from c5_terra_tail import existing_judge_cases
    from tiny_rgb_capacity_protocol import verified_comparison

    result = json.loads(receipt.read_text())
    if (
        result.get("complete") is not True
        or result.get("numerics_passed") is not True
        or result.get("plan_sha256") != sha256(path)
        or result.get("initial_visual_sha256") != initial
        or result.get("fixed_backbones") != fixed
        or result.get("optimizer_updates") != 0
        or result.get("caption_weight_during_proof") != 0
    ):
        raise ValueError("C5 tail qualification differs from the actual initial policy")
    comparisons = receipt.with_name("qualification-comparisons.jsonl")
    inputs = receipt.with_name("qualification-inputs.json")
    if (
        sha256(comparisons) != result["comparisons_sha256"]
        or sha256(inputs) != result["inputs_sha256"]
    ):
        raise ValueError("C5 tail numerical proof evidence changed")
    data = json.loads(inputs.read_text())
    if data["pairs"] != plan["proof_cases"] or data["cases"] != existing_judge_cases(plan):
        raise ValueError("C5 tail gradients used different inputs or actual Terra scores")
    rows = [json.loads(s) for s in comparisons.read_text().splitlines()]
    if [(r["source"], r["example_id"]) for r in rows] != [
        (p["raw"]["source"], p["raw"]["example_id"]) for p in plan["proof_cases"]
    ]:
        raise ValueError("C5 tail numerical coverage differs")
    for row, case in zip(rows, data["cases"], strict=True):
        if (
            row["quality"] != case["quality"]
            or row["caption_weight"] != 0
            or row["terra_gradient_coefficient_l1"] <= 0
        ):
            raise ValueError("C5 tail proof lacks actual Terra-only gradients")
        verified_comparison(row["comparison"], RECIPE["trainable_tensors"])
    return result


def evaluate_checkpoint(path, checkpoint, examples, output):
    from benchmark_fleurs_dev import sha256

    evaluation = output / f"cpu-evaluation-{examples:06d}"
    with output.joinpath(f"cpu-evaluation-{examples:06d}.log").open("x") as stream:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/evaluate_terra_c5_tail.py"),
                "--plan",
                str(path),
                "--checkpoint",
                str(checkpoint),
                "--update",
                str(examples),
                "--output-dir",
                str(evaluation),
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    result = json.loads(evaluation.joinpath("decision.json").read_text())
    if result["checkpoint_sha256"] != sha256(checkpoint):
        raise RuntimeError("native C5 evaluator loaded a different checkpoint")
    return result


def missing_resume_evaluation(resumed, decisions):
    return (
        resumed is not None
        and resumed["training_examples"] in RECIPE["checks_after_examples"]
        and not any(d["checkpoint_sha256"] == resumed["visual"]["sha256"] for d in decisions)
    )


def run(path, output, *, qualification_only=False, qualification=None, restore_only=False):
    import torch
    from benchmark_fleurs_dev import sha256
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import (
        candidate_labels,
        candidate_pool,
        existing_judge_cases,
        fixed_hashes,
        gradients,
        load_model,
        native_memory,
        padded_memory,
        synchronize_policy,
    )
    from probe_tiny_rgb_capacity import compare, state_digest
    from terra_training_recovery import (
        CheckpointRequest,
        checked_resume,
        record_preupdate_failure,
        restore_checkpoint,
        save_checkpoint,
    )
    from terra_training_storage import storage
    from terra_visual_learning import backward_candidate_risk
    from tiny_rgb_capacity_protocol import verified_comparison
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from train_terra_visual import score_pools
    from transformers import ByT5Tokenizer

    plan, pairs = checked_plan(path)
    resumed = checked_resume(plan) if "resume" in plan else None
    if resumed:
        if qualification is not None or qualification_only:
            raise ValueError("C5 tail recovery uses only its frozen qualification")
        qualification = resumed["qualification"]
    if restore_only and not resumed:
        raise ValueError("C5 tail restore-only requires a complete recovery plan")
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("C5 Terra requires a fresh checkpoint subtree")
    storage(215000000)
    output.mkdir(parents=True)
    stop = CheckpointRequest()
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    duty = WinnerDutyCycle(torch, budget)
    with gpu_lease():
        torch.manual_seed(RECIPE["seed"])
        cpu, reference = load_model()
        encoder = load_frozen_winner_encoder(reference).requires_grad_(False).eval()
        actual = {
            "language": state_digest(cpu.language.state_dict()),
            "encoder": state_digest(encoder.state_dict()),
        }
        if actual != plan["initial_reference_hashes"]:
            raise ValueError("C5 native baseline values differ from the exact142-control path")
        initial, fixed = state_digest(cpu.policy_parameters()), fixed_hashes(cpu, encoder)
        gpu, gpu_reference = load_model()
        if (
            gpu_reference != reference
            or state_digest(gpu.language.state_dict()) != actual["language"]
        ):
            raise ValueError("C5 CPU/Metal initial language tensors differ")
        if not torch.backends.mps.is_available():
            raise RuntimeError("C5 Terra requires qualified Metal")
        gpu = duty.run(lambda: gpu.to("mps"))
        if state_digest(gpu.policy_parameters()) != initial or fixed_hashes(gpu, encoder) != fixed:
            raise ValueError("C5 policy transfer changed baseline tensor values")
        tokenizer = ByT5Tokenizer.from_pretrained(
            ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
        )
        (output / "run.json").write_text(
            json.dumps(
                {
                    "plan_sha256": sha256(path),
                    "initial_visual_sha256": initial,
                    "legacy_visual_field_contains_decoder_tail_policy": True,
                    "fixed_backbones": fixed,
                    "initial_full_C5_hashes": actual,
                    "trainable_parameters": RECIPE["trainable_parameters"],
                    "trainable_tensors": 8,
                    "resume": plan.get("resume"),
                    "new_visual_weights_loaded": False,
                    "miner_modified": False,
                    "promotion_authorized": False,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps({"stage": "native_C5_tail_loaded", "initial_policy_sha256": initial}),
            flush=True,
        )
        if qualification is None:
            cases = existing_judge_cases(plan)
            (output / "qualification-inputs.json").write_text(
                json.dumps({"pairs": plan["proof_cases"], "cases": cases}, indent=2) + "\n"
            )
            for pair, case in zip(plan["proof_cases"], cases, strict=True):
                memory = native_memory(encoder, pair, reference["translator_checkpoint_sha256"])
                if candidate_pool(cpu, tokenizer, memory, pair["raw"]["reference"]) != case["pool"]:
                    raise ValueError(
                        "actual baseline C5 greedy output differs from authenticated Terra text"
                    )
                memory, mask = padded_memory(memory)
                labels = candidate_labels(case["pool"])
                cpu.zero_grad(set_to_none=True)
                gpu.zero_grad(set_to_none=True)
                before = backward_candidate_risk(
                    cpu,
                    memory,
                    mask,
                    labels,
                    case["quality"],
                    alpha=RECIPE["risk_alpha"],
                    caption_weight=0,
                )
                old = gradients(cpu)

                def metal_case(memory=memory, mask=mask, labels=labels, quality=case["quality"]):
                    result = backward_candidate_risk(
                        gpu,
                        memory,
                        mask,
                        labels,
                        quality,
                        alpha=RECIPE["risk_alpha"],
                        caption_weight=0,
                    )
                    return result, gradients(gpu)

                after, new = duty.run(metal_case)
                comparison = compare(
                    old,
                    new,
                    before["loss"],
                    after["loss"],
                    before["gold_logits"],
                    after["gold_logits"],
                )
                row = {
                    "source": pair["raw"]["source"],
                    "example_id": pair["raw"]["example_id"],
                    "quality": case["quality"],
                    "caption_weight": 0,
                    "terra_gradient_coefficient_l1": before["terra_gradient_coefficient_l1"],
                    "comparison": comparison,
                }
                append(output, "qualification-comparisons.jsonl", row)
                worst = verified_comparison(comparison, 8)
                if before["terra_gradient_coefficient_l1"] <= 0:
                    raise RuntimeError("C5 tail actual semantic error has no gradient")
                print(
                    json.dumps(
                        {
                            "stage": "native_C5_tail_qualified",
                            "example_id": row["example_id"],
                            "worst_relative_gradient_error": worst,
                        }
                    ),
                    flush=True,
                )
                del old, new
                cpu.zero_grad(set_to_none=True)
                gpu.zero_grad(set_to_none=True)
                torch.mps.empty_cache()
            if (
                fixed_hashes(cpu, encoder) != fixed
                or fixed_hashes(gpu, encoder) != fixed
                or state_digest(gpu.policy_parameters()) != initial
            ):
                raise RuntimeError("C5 tail qualification changed initial weights")
            (output / "qualification.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "numerics_passed": True,
                        "plan_sha256": sha256(path),
                        "initial_visual_sha256": initial,
                        "fixed_backbones": fixed,
                        "comparisons_sha256": sha256(output / "qualification-comparisons.jsonl"),
                        "inputs_sha256": sha256(output / "qualification-inputs.json"),
                        "optimizer_updates": 0,
                        "caption_weight_during_proof": 0,
                        "actual_Terra_only_gradients": True,
                        "resource_receipt": duty.receipt(),
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            proof_path = resumed["qualification_plan"] if resumed else path
            checked_qualification(qualification, proof_path, initial, fixed, plan)
            (output / "qualification-reuse.json").write_text(
                json.dumps(
                    {
                        "receipt": str(qualification),
                        "sha256": sha256(qualification),
                        "qualification_plan_sha256": sha256(proof_path),
                        "current_plan_sha256": sha256(path),
                    },
                    indent=2,
                )
                + "\n"
            )
        if qualification_only:
            print(
                json.dumps({"stage": "qualification_complete", "optimizer_updates": 0}), flush=True
            )
            return
        cpu.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            gpu.trainable_parameters().values(),
            lr=RECIPE["learning_rate"],
            weight_decay=RECIPE["weight_decay"],
            foreach=False,
        )
        examples, updates, decisions = 0, 0, []
        (output / "steps.jsonl").touch(exist_ok=False)
        if resumed:

            def restore_rng(a, b):
                torch.set_rng_state(a)
                torch.mps.set_rng_state(b)

            examples, updates = duty.run(
                lambda: restore_checkpoint(
                    plan,
                    resumed,
                    gpu.trainable_parameters(),
                    optimizer,
                    initial_visual=initial,
                    fixed=fixed,
                    restore_rng=restore_rng,
                )
            )
            if "source_decisions" in resumed:
                decisions = json.loads(resumed["source_decisions"].read_text())
                (output / "open-decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
            (output / "resume-restored.json").write_text(
                json.dumps(
                    {
                        "training_examples": examples,
                        "optimizer_updates": updates,
                        "restore_only": restore_only,
                        "new_Terra_requests": 0,
                    },
                    indent=2,
                )
                + "\n"
            )
            if restore_only:
                return
        versions = {name: p._version for name, p in gpu.frozen_parameters().items()}
        ordered = sorted(
            pairs,
            key=lambda p: hashlib.sha256(
                f"c5-terra-tail:{RECIPE['seed']}:{p['raw']['source']}:{p['raw']['example_id']}".encode()
            ).hexdigest(),
        )

        def save(recovery=False):
            return save_checkpoint(
                gpu,
                optimizer,
                output,
                path,
                reference["translator_checkpoint_sha256"],
                examples,
                updates,
                recovery=recovery,
            )

        if missing_resume_evaluation(plan.get("resume"), decisions):
            # A prior evaluator may have failed after the checkpoint was saved.
            # Rebind those exact restored weights to this plan, then finish that
            # scheduled check before allowing another optimizer update.
            checkpoint = save()
            decisions.append(evaluate_checkpoint(path, checkpoint, examples, output))
            (output / "open-decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")

        while examples < len(ordered):
            if stop.requested:
                checkpoint = save(recovery=True) if updates else None
                (output / "requested-stop.json").write_text(
                    json.dumps(
                        {
                            "requested_stop": True,
                            "signal": "SIGUSR1",
                            "training_examples": examples,
                            "optimizer_updates": updates,
                            "checkpoint": str(checkpoint) if checkpoint else None,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                return
            next_check = next(n for n in RECIPE["checks_after_examples"] if n > examples)
            count = min(RECIPE["batch_clips"], len(ordered) - examples, next_check - examples)
            batch = ordered[examples : examples + count]
            synchronize_policy(gpu, cpu)
            policy_sha = state_digest(gpu.policy_parameters())
            if state_digest(cpu.policy_parameters()) != policy_sha:
                raise RuntimeError("native C5 CPU rollout and Metal policy differ")
            memories, pools = [], []
            stage = "native_rollout"
            receipt_path = output / "terra" / f"batch-{updates + 1:06d}"
            try:
                for pair in batch:
                    memory = native_memory(encoder, pair, reference["translator_checkpoint_sha256"])
                    pools.append(candidate_pool(cpu, tokenizer, memory, pair["raw"]["reference"]))
                    memories.append(memory)
                stage = "terra_scoring"
                quality, judge = score_pools(batch, pools, receipt_path)
            except (RuntimeError, ValueError, OSError) as error:
                record_preupdate_failure(
                    output,
                    stage=stage,
                    error=error,
                    examples=examples,
                    updates=updates,
                    save=lambda: save(recovery=True).relative_to(ROOT),
                )
                raise
            append(
                output,
                "candidate-pools.jsonl",
                {
                    "optimizer_update": updates + 1,
                    "policy_sha256": policy_sha,
                    "identities": [
                        {
                            k: p["raw"][k]
                            for k in ("source", "example_id", "role", "training_allowed")
                        }
                        for p in batch
                    ],
                    "pools": pools,
                    "qualities": quality,
                    "judge": judge,
                },
            )
            optimizer.zero_grad(set_to_none=True)
            warmup = min(1.0, (examples + count) / RECIPE["warmup_examples"])
            progress = max(
                0.0,
                (examples - RECIPE["warmup_examples"]) / (len(ordered) - RECIPE["warmup_examples"]),
            )
            rate = (
                RECIPE["learning_rate"]
                * warmup
                * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            measured = []
            for memory, pool, qs in zip(memories, pools, quality, strict=True):
                memory, mask = padded_memory(memory)
                labels = candidate_labels(pool)
                result = duty.run(
                    lambda memory=memory,
                    mask=mask,
                    labels=labels,
                    qs=qs,
                    count=count: backward_candidate_risk(
                        gpu,
                        memory,
                        mask,
                        labels,
                        qs,
                        alpha=RECIPE["risk_alpha"],
                        caption_weight=RECIPE["caption_weight"],
                        divisor=count,
                    )
                )
                result.pop("gold_logits")
                measured.append(result)
                torch.mps.empty_cache()

            def step():
                if any(
                    p.requires_grad or p.grad is not None or p._version != versions[name]
                    for name, p in gpu.frozen_parameters().items()
                ):
                    raise RuntimeError(
                        "native C5 optimization crossed the frozen language boundary"
                    )
                norms = [
                    float(p.grad.norm()) if p.grad is not None else math.nan
                    for p in gpu.trainable_parameters().values()
                ]
                if len(norms) != 8 or any(not math.isfinite(v) for v in norms):
                    raise RuntimeError("native C5 tail lacks all eight finite optimizer gradients")
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        gpu.trainable_parameters().values(), RECIPE["gradient_clip"]
                    )
                )
                if not math.isfinite(norm):
                    raise RuntimeError("native C5 optimizer norm is nonfinite")
                optimizer.step()
                return {
                    "gradient_tensors": len(norms),
                    "nonzero_gradient_tensors": sum(v > 0 for v in norms),
                    "gradient_norm": norm,
                }

            gradient = duty.run(step)
            examples += count
            updates += 1
            append(
                output,
                "steps.jsonl",
                {
                    "training_examples": examples,
                    "optimizer_update": updates,
                    "batch_clips": count,
                    "learning_rate": rate,
                    "measurements": measured,
                    **gradient,
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "native_C5_terra_training",
                        "training_examples": examples,
                        "optimizer_updates": updates,
                        **gradient,
                        **duty.receipt(),
                    }
                ),
                flush=True,
            )
            storage(215000000, output=output)
            if examples in RECIPE["checks_after_examples"]:
                checkpoint = save()
                if fixed_hashes(gpu, encoder) != fixed:
                    raise RuntimeError("C5 tail training changed frozen weights")
                decisions.append(evaluate_checkpoint(path, checkpoint, examples, output))
                (output / "open-decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
        checked_plan(path)
        if fixed_hashes(gpu, encoder) != fixed:
            raise RuntimeError("C5 tail final frozen values changed")
        (output / "decision.json").write_text(
            json.dumps(
                {
                    "training_complete": True,
                    "training_examples": examples,
                    "optimizer_updates": updates,
                    "evaluations": decisions,
                    "promotion_authorized": False,
                    "confirmation_decoded": False,
                    "miner_modified": False,
                },
                indent=2,
            )
            + "\n"
        )


def main():
    for key, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ.setdefault(key, value)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-plan", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--restore-only", action="store_true")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--qualification-only", action="store_true")
    stages.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(
            args.freeze,
            resume_plan=args.resume_plan,
            resume_checkpoint=args.resume_checkpoint,
            qualification=args.qualification,
        )
    elif args.plan and args.output_dir:
        run(
            args.plan,
            args.output_dir,
            qualification_only=args.qualification_only,
            qualification=args.qualification,
            restore_only=args.restore_only,
        )
    else:
        parser.error("provide --freeze or --plan with --output-dir")


if __name__ == "__main__":
    main()
