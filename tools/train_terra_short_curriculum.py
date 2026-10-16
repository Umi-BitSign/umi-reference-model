"""Fresh-C5, no-download short-phrase pilot with semantic-only checkpoint decisions."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from launch_terra_visual_job import identity, now, write
from prepare_terra_short_curriculum import ROOT
from prepare_terra_short_curriculum import checked as checked_data
from terra_short_curriculum_learning import (
    RECIPE,
    JudgePacer,
    comparison_items,
    order,
    score_comparison,
    semantic_decision,
)

PROOF_PLAN = ROOT / "runs/terra-c5-tail-training-v1/plan.json"
PROOF = ROOT / "checkpoints/terra-c5-tail-qualification-v1/qualification.json"


def artifact(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT / "runs") and not path.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("short pilot artifacts must remain in owned runs/checkpoints")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def specification(data_path, control_path):
    data = checked_data(data_path)
    control = json.loads(control_path.read_text())
    terminal = json.loads(control_path.with_name("exit.json").read_text())
    development = [r for r in data["records"] if r["role"] == "development"]
    if (
        control.get("complete") is not True
        or control.get("all_weights_unchanged") is not True
        or control.get("optimizer_updates") != 0
        or control.get("GPU_use") is not False
        or control.get("plan_sha256") != sha256(data_path)
        or control.get("original_C5_generation") != data["reference"]["generation"]
        or control.get("clips") != len(development)
        or terminal.get("exit_code") != 0
    ):
        raise ValueError("short pilot requires a complete unchanged-C5 control")
    comparison_items(development, control["predictions"], control["predictions"])
    original = json.loads(PROOF_PLAN.read_text())
    proof = json.loads(PROOF.read_text())
    if (
        original["reference"] != data["reference"]
        or proof.get("complete") is not True
        or proof.get("plan_sha256") != sha256(PROOF_PLAN)
        or proof.get("actual_Terra_only_gradients") is not True
        or proof.get("numerics_passed") is not True
        or proof.get("optimizer_updates") != 0
        or proof.get("caption_weight_during_proof") != 0
    ):
        raise ValueError("short pilot cannot change the qualified C5 component")
    sources = {
        **data["source_sha256"],
        **{
            f"tools/{name}": sha256(ROOT / "tools" / name)
            for name in (
                "train_terra_short_curriculum.py",
                "terra_short_curriculum_learning.py",
                "audit_native_terra_semantics.py",
                "terra_semantic_selection.py",
            )
        },
    }
    return {
        "schema": "umi-studio-terra-short-pilot/1",
        "recipe": RECIPE,
        "data": artifact(data_path),
        "control": artifact(control_path),
        "control_terminal": artifact(control_path.with_name("exit.json")),
        "qualified_component_plan": artifact(PROOF_PLAN),
        "qualified_component_proof": artifact(PROOF),
        "source_sha256": sources,
        "reference": data["reference"],
        "resource_policy": data["resource_policy"],
        "resource_policy_sha256": data["resource_policy_sha256"],
        "initial_reference_hashes": original["initial_reference_hashes"],
        "training_identities": data["training_clips"],
        "planned_training_examples": data["training_clips"] * RECIPE["epochs"],
        "checks_after_examples": [data["training_clips"] * e for e in RECIPE["check_after_epochs"]],
        "stage_scope": "two_or_three_word_full_clips_with_newly_reserved_STEM_development",
        "qualification_scope": (
            "unchanged native C5 eight-tensor Terra-loss component, original six real cases"
        ),
        "WER_is_diagnostic_only": True,
        "new_data_downloads": False,
        "promotion_authorized": False,
        "miner_or_codex_lb_inspection": False,
    }


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = json.loads(path.read_text())
    expected = specification(ROOT / plan["data"]["path"], ROOT / plan["control"]["path"])
    if plan != expected:
        raise ValueError("short pilot source/data/recipe/control binding changed")
    for name, digest in plan["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("short pilot component implementation changed")
    verify_snapshot(path, plan)
    return plan


def freeze(path, data, control):
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("short pilot requires a fresh owned runs/ plan")
    storage()
    plan = specification(data, control)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "training_examples": plan["planned_training_examples"],
                "checks": plan["checks_after_examples"],
            }
        ),
        flush=True,
    )


def append(output, name, row):
    with (output / name).open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def evaluate(cpu, encoder, tokenizer, reference, rows, baseline, output, pacer):
    import torch
    from c5_terra_tail import native_memory
    from evaluate_tiny_rgb_feature_winner import CpuBudget
    from transformers import StoppingCriteriaList
    from transformers.modeling_outputs import BaseModelOutput

    output.mkdir()
    predictions = []
    for row in rows:
        if row["training_allowed"] is not False or row["role"] != "development":
            raise ValueError("short pilot cannot score training rows as development")
        with torch.no_grad():
            memory = native_memory(
                encoder, row["original_pair"], reference["translator_checkpoint_sha256"]
            )
            tokens = cpu.language.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                **reference["generation"],
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([CpuBudget()]),
            )
        hypothesis = tokenizer.decode(tokens[0], skip_special_tokens=True)
        words = len(normalize(row["reference"]))
        edits = distance(normalize(row["reference"]), normalize(hypothesis))
        result = {
            **{
                k: row[k]
                for k in (
                    "source",
                    "example_id",
                    "source_group",
                    "signer",
                    "reference",
                    "role",
                    "training_allowed",
                )
            },
            "hypothesis": hypothesis,
            "reference_words": words,
            "edits": edits,
            "WER_diagnostic": edits / words,
        }
        predictions.append(result)
        write(output / f"case-{len(predictions):04d}.json", result)
    comparison = comparison_items(rows, baseline, predictions)
    scores = score_comparison(comparison, output / "semantic", pacer)
    result = semantic_decision(comparison, scores, baseline, predictions)
    write(output / "decision.json", result)
    print(json.dumps({"stage": "short_semantic_checkpoint", **result}), flush=True)
    return result


def fit(path, output, *, qualification_only=False):
    import torch
    from terra_training_storage import storage
    from train_terra_c5_tail import gpu_lease

    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("short pilot requires a fresh owned checkpoint directory")
    storage(215000000)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            _fit(torch, path, output, plan, qualification_only=qualification_only)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def _fit(torch, path, output, plan, *, qualification_only=False):
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import (
        candidate_labels,
        candidate_pool,
        fixed_hashes,
        load_model,
        native_memory,
        padded_memory,
        synchronize_policy,
    )
    from probe_tiny_rgb_capacity import state_digest
    from terra_training_recovery import CheckpointRequest, record_preupdate_failure, save_checkpoint
    from terra_training_storage import storage
    from terra_visual_learning import backward_candidate_risk
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from train_terra_c5_tail import checked_qualification
    from transformers import ByT5Tokenizer

    data = checked_data(ROOT / plan["data"]["path"])
    control = json.loads((ROOT / plan["control"]["path"]).read_text())["predictions"]
    rows = [r for r in data["records"] if r["role"] == "development"]
    ordered = order(data["records"])
    if len(ordered) != plan["planned_training_examples"]:
        raise ValueError("short pilot sampling coverage differs")
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if budget.duty_fraction() != 0.5 or budget.cpu_threads() != 2:
        raise ValueError("short pilot requires the current 50% GPU/two-CPU policy")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(RECIPE["seed"])
    stop, pacer = CheckpointRequest(), JudgePacer()
    duty = WinnerDutyCycle(torch, budget)
    cpu, reference = load_model()
    encoder = load_frozen_winner_encoder(reference).requires_grad_(False).eval()
    initial = state_digest(cpu.policy_parameters())
    fixed = fixed_hashes(cpu, encoder)
    actual = {
        "language": state_digest(cpu.language.state_dict()),
        "encoder": state_digest(encoder.state_dict()),
    }
    if reference != plan["reference"] or actual != plan["initial_reference_hashes"]:
        raise ValueError("short pilot did not initialize from original C5")
    proof_plan = ROOT / plan["qualified_component_plan"]["path"]
    checked_qualification(
        ROOT / plan["qualified_component_proof"]["path"],
        proof_plan,
        initial,
        fixed,
        json.loads(proof_plan.read_text()),
    )
    cpu.requires_grad_(False)
    gpu, gpu_reference = load_model()
    if gpu_reference != reference or state_digest(gpu.policy_parameters()) != initial:
        raise ValueError("short CPU/Metal initialization differs")
    if not torch.backends.mps.is_available():
        raise RuntimeError("short pilot requires the qualified Metal backend")
    gpu = duty.run(lambda: gpu.to("mps"))
    if fixed_hashes(gpu, encoder) != fixed or state_digest(gpu.policy_parameters()) != initial:
        raise ValueError("short pilot Metal transfer changed C5 values")
    versions = {name: p._version for name, p in gpu.frozen_parameters().items()}
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    optimizer = torch.optim.AdamW(
        gpu.trainable_parameters().values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
        foreach=False,
    )
    write(
        output / "run.json",
        {
            "plan_sha256": sha256(path),
            "initial_visual_sha256": initial,
            "fixed_backbones": fixed,
            "initial_full_C5_hashes": actual,
            "qualification_reused": plan["qualified_component_proof"],
            "qualification_scope": plan["qualification_scope"],
            "resume": None,
            "legacy_visual_fields_contain_eight_decoder_tail_tensors": True,
            "new_word_videos_downloaded": 0,
            "promotion_authorized": False,
        },
    )
    if qualification_only:
        write(
            output / "initialization-check.json",
            {
                "complete": True,
                "plan_sha256": sha256(path),
                "original_C5_CPU_and_Metal_values_verified": True,
                "existing_actual_Terra_gradient_proof_revalidated": plan[
                    "qualified_component_proof"
                ],
                "new_curriculum_gradient_proof_claimed": False,
                "optimizer_updates": 0,
                "new_Terra_requests": 0,
                "resource_receipt": duty.receipt(),
            },
        )
        return
    examples = updates = 0
    decisions = []

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

    while examples < len(ordered):
        if stop.requested:
            checkpoint = save(recovery=True) if updates else None
            write(
                output / "requested-stop.json",
                {
                    "training_examples": examples,
                    "optimizer_updates": updates,
                    "checkpoint": str(checkpoint),
                    "requested_stop": True,
                },
            )
            return
        next_check = next(n for n in plan["checks_after_examples"] if n > examples)
        count = min(RECIPE["batch_clips"], next_check - examples)
        batch = ordered[examples : examples + count]
        synchronize_policy(gpu, cpu)
        memories, pools = [], []
        stage = "native_rollout"
        try:
            for pair in batch:
                memory = native_memory(encoder, pair, reference["translator_checkpoint_sha256"])
                pools.append(candidate_pool(cpu, tokenizer, memory, pair["raw"]["reference"]))
                memories.append(memory)
            stage = "terra_scoring"
            request = [
                {"reference": pair["raw"]["reference"], "candidate": candidate["text"]}
                for pair, pool in zip(batch, pools, strict=True)
                for candidate in pool
            ]
            judge = pacer.score(request, output / "terra" / f"batch-{updates + 1:06d}")
            qualities, offset = [], 0
            for pool in pools:
                quality = judge["scores"][offset : offset + len(pool)]
                if quality[0] != 10:
                    raise RuntimeError("Terra did not score the exact human anchor ten")
                qualities.append(quality)
                offset += len(pool)
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
                "identities": [
                    {"source": p["raw"]["source"], "example_id": p["raw"]["example_id"]}
                    for p in batch
                ],
                "pools": pools,
                "qualities": qualities,
                "judge": judge,
            },
        )
        optimizer.zero_grad(set_to_none=True)
        rate = (
            RECIPE["learning_rate"]
            * min(1, (examples + count) / RECIPE["warmup_examples"])
            * (0.1 + 0.9 * (1 + math.cos(math.pi * examples / len(ordered))) / 2)
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        measured = []
        for memory, pool, quality in zip(memories, pools, qualities, strict=True):
            memory, mask = padded_memory(memory)
            labels = candidate_labels(pool)
            result = duty.run(
                lambda memory=memory,
                mask=mask,
                labels=labels,
                quality=quality,
                count=count: backward_candidate_risk(
                    gpu,
                    memory,
                    mask,
                    labels,
                    quality,
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
                raise RuntimeError("short pilot crossed its frozen parameter boundary")
            norms = [
                float(p.grad.norm()) if p.grad is not None else math.nan
                for p in gpu.trainable_parameters().values()
            ]
            if len(norms) != 8 or any(not math.isfinite(v) for v in norms):
                raise RuntimeError("short pilot lacks eight finite gradients")
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    gpu.trainable_parameters().values(), RECIPE["gradient_clip"]
                )
            )
            if not math.isfinite(norm):
                raise RuntimeError("short pilot optimizer norm is nonfinite")
            optimizer.step()
            return {
                "gradient_tensors": len(norms),
                "nonzero_gradient_tensors": sum(v > 0 for v in norms),
                "gradient_norm": norm,
            }

        gradient = duty.run(step)
        examples, updates = examples + count, updates + 1
        append(
            output,
            "steps.jsonl",
            {
                "training_examples": examples,
                "optimizer_update": updates,
                "learning_rate": rate,
                "measurements": measured,
                **gradient,
            },
        )
        print(
            json.dumps(
                {
                    "stage": "short_Terra_training",
                    "training_examples": examples,
                    "optimizer_updates": updates,
                    **gradient,
                    **duty.receipt(),
                }
            ),
            flush=True,
        )
        storage(215000000, output=output)
        if examples in plan["checks_after_examples"]:
            checkpoint = save()
            if fixed_hashes(gpu, encoder) != fixed:
                raise RuntimeError("short pilot changed frozen C5 weights")
            synchronize_policy(gpu, cpu)
            result = evaluate(
                cpu,
                encoder,
                tokenizer,
                reference,
                rows,
                control,
                output / f"semantic-{examples:06d}",
                pacer,
            )
            result.update({"checkpoint_sha256": sha256(checkpoint), "training_examples": examples})
            decisions.append(result)
            append(output, "semantic-decisions.jsonl", result)
            if result["stop_for_semantic_decline"] or result["stop_for_failed_calibration"]:
                break
    checked(path)
    if fixed_hashes(gpu, encoder) != fixed or fixed_hashes(cpu, encoder) != fixed:
        raise RuntimeError("short pilot final frozen hashes changed")
    write(
        output / "decision.json",
        {
            "training_complete": examples == len(ordered),
            "training_examples": examples,
            "optimizer_updates": updates,
            "semantic_decisions": decisions,
            "semantic_guard_stop": examples != len(ordered),
            "WER_veto_used": False,
            "full_composite_or_competitive_win_claimed": False,
            "promotion_authorized": False,
            "resource_receipt": duty.receipt(),
        },
    )


def main():
    for name, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ.setdefault(name, value)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--qualification-only", action="store_true")
    args = parser.parse_args()
    if args.freeze and args.data and args.control and not args.plan:
        freeze(args.freeze, args.data, args.control)
    elif args.plan and args.output_dir and not args.freeze:
        fit(args.plan, args.output_dir, qualification_only=args.qualification_only)
    else:
        parser.error("provide --freeze with --data/--control OR --plan with --output-dir")


if __name__ == "__main__":
    main()
