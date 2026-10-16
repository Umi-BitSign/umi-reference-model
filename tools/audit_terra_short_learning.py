"""Compare C5 and rejected944-update policy without optimization or new judging."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import identity, now, write
from prepare_terra_short_curriculum import ROOT
from prepare_terra_short_curriculum import checked as checked_data
from terra_frozen_short import RECIPE, checked_rewards, key, pair
from terra_learning_diagnostics import (
    conditioning,
    gradient_balance,
    retrieval_cases,
    stratum,
    sum_equivalence,
)
from terra_training_recovery import artifact, verify_snapshot
from train_terra_short_curriculum import append

PARENT = ROOT / "runs/terra-frozen-short-training-v1/plan.json"
PARENT_SHA = "a5005d916f563abe69f205aa4bb98cf2feb7811f3135f28dc435532ebcd56204"
CANDIDATE = ROOT / "checkpoints/terra-frozen-short-training-v1/visual-after-007424.safetensors"
CANDIDATE_SHA = "e1c06e5705640beb98854837dc2722a31ea8c6d5826644eb3d03a519cdc03156"


def specification():
    if sha256(PARENT) != PARENT_SHA or sha256(CANDIDATE) != CANDIDATE_SHA:
        raise ValueError("short audit parent or rejected candidate changed")
    parent = json.loads(PARENT.read_text())
    verify_snapshot(PARENT, parent)
    for name, digest in parent["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("short audit qualified scientific component changed")
    data = checked_data(ROOT / parent["data"]["path"])
    rewards = checked_rewards(data)
    if parent["data"] != artifact(ROOT / parent["data"]["path"]):
        raise ValueError("short audit uses different data bytes")
    output = CANDIDATE.parent
    terminal, worker, completion = [
        json.loads((output / name).read_text())
        for name in ("exit.json", "worker.json", "decision.json")
    ]
    if (
        terminal["pid"] != worker["pid"]
        or terminal["exit_code"] != 0
        or completion["training_complete"] is not True
        or completion["optimizer_updates"] != 944
        or completion["training_examples"] != 7424
        or completion["semantic_decisions"][-1]["checkpoint_sha256"] != CANDIDATE_SHA
        or any(r["passes_semantic_research_gate"] for r in completion["semantic_decisions"])
    ):
        raise ValueError("short audit needs the terminal, rejected candidate")
    sources = dict(parent["source_sha256"])
    for name in ("terra_learning_diagnostics.py", "audit_terra_short_learning.py"):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-terra-short-learning-audit/1",
        "parent": artifact(PARENT),
        "candidate": artifact(CANDIDATE),
        "data": parent["data"],
        "historical_training_rewards": parent["historical_training_rewards"],
        "candidate_terminal": artifact(output / "exit.json"),
        "candidate_completion": artifact(output / "decision.json"),
        "candidate_worker": artifact(output / "worker.json"),
        "source_sha256": sources,
        "reference": parent["reference"],
        "initial_reference_hashes": parent["initial_reference_hashes"],
        "resource_policy": parent["resource_policy"],
        "resource_policy_sha256": parent["resource_policy_sha256"],
        "qualified_component_plan": parent["qualified_component_plan"],
        "qualified_component_proof": parent["qualified_component_proof"],
        "gradient_identities": [key(r["data_row"]) for r in rewards],
        "conditioning_identities": [key(r["data_row"]) for r in retrieval_cases(rewards)],
        "risk_alpha": RECIPE["risk_alpha"],
        "auxiliary_CE_weight": RECIPE["caption_weight"],
        "new_Terra_requests": 0,
        "optimizer_updates": 0,
        "development_labels_or_outputs_used": False,
        "on_policy_reward_claimed": False,
        "is_model_quality_or_promotion_gate": False,
        "GPU_duty_fraction": 0.5,
        "CPU_threads": 2,
    }


def checked(path):
    plan = json.loads(path.read_text())
    # JSON encodes tuple identities as lists.
    if plan != json.loads(json.dumps(specification())):
        raise ValueError("short diagnostic recipe, data, sources or candidate changed")
    verify_snapshot(path, plan)
    if sha256(ROOT / plan["resource_policy"]) != plan["resource_policy_sha256"]:
        raise ValueError("short diagnostic resource policy changed")
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("short diagnostic requires a fresh owned plan")
    storage(2 * 1024**2)
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "training_gradient_cases": len(plan["gradient_identities"]),
                "conditioning_cases": len(plan["conditioning_identities"]),
            }
        ),
        flush=True,
    )


def run(path, output):
    from terra_training_storage import storage
    from train_terra_c5_tail import gpu_lease

    plan = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("short diagnostic needs a fresh owned output")
    storage(2 * 1024**2)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            fit(path, output, plan)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def fit(path, output, plan):
    import torch
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import (
        candidate_labels,
        fixed_hashes,
        load_model,
        native_memory,
        padded_memory,
    )
    from probe_tiny_rgb_capacity import state_digest
    from safetensors import safe_open
    from safetensors.torch import load_file
    from terra_visual_learning import backward_candidate_risk
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from train_terra_c5_tail import checked_qualification
    from transformers import ByT5Tokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if budget.duty_fraction() != 0.5 or budget.cpu_threads() != 2:
        raise ValueError("short diagnostic requires50% GPU/two CPU threads")
    duty = WinnerDutyCycle(torch, budget)
    data = checked_data(ROOT / plan["data"]["path"])
    rewards = checked_rewards(data)
    model, reference = load_model()
    encoder = load_frozen_winner_encoder(reference).requires_grad_(False).eval()
    initial, fixed = state_digest(model.policy_parameters()), fixed_hashes(model, encoder)
    if (
        reference != plan["reference"]
        or {
            "language": state_digest(model.language.state_dict()),
            "encoder": state_digest(encoder.state_dict()),
        }
        != plan["initial_reference_hashes"]
    ):
        raise ValueError("short diagnostic did not load original C5")
    proof_plan = ROOT / plan["qualified_component_plan"]["path"]
    checked_qualification(
        ROOT / plan["qualified_component_proof"]["path"],
        proof_plan,
        initial,
        fixed,
        json.loads(proof_plan.read_text()),
    )
    if not torch.backends.mps.is_available():
        raise RuntimeError("short diagnostic requires the qualified Metal backend")
    original = {n: p.detach().cpu().clone() for n, p in model.policy_parameters().items()}
    model = duty.run(lambda: model.to("mps"))
    if state_digest(model.policy_parameters()) != initial or fixed_hashes(model, encoder) != fixed:
        raise ValueError("short diagnostic Metal transfer changed baseline values")
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    prepared, cache_bytes = {}, 0
    for row in rewards:
        labels = candidate_labels(row["pool"])
        if any(
            tokenizer.decode(
                c["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            != c["text"]
            for c in row["pool"]
        ):
            raise ValueError("short diagnostic tokens differ from their actual judged texts")
        memory, mask = padded_memory(
            native_memory(encoder, pair(row), reference["translator_checkpoint_sha256"])
        )
        cache_bytes += memory.numel() * memory.element_size() + mask.numel() * mask.element_size()
        if cache_bytes > RECIPE["native_memory_CPU_cache_budget_bytes"]:
            raise RuntimeError("short diagnostic CPU cache exceeds128MiB")
        prepared[key(row["data_row"])] = (memory, mask, labels)

    def gradients():
        if any(p.grad is not None for p in model.frozen_parameters().values()):
            raise RuntimeError("short diagnostic crossed the frozen parameter boundary")
        result = {}
        for name, p in model.trainable_parameters().items():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError("short diagnostic lacks finite component gradients")
            result[name] = p.grad.detach().cpu().clone()
        if len(result) != 8:
            raise RuntimeError("short diagnostic gradient scope differs from eight tensors")
        return result

    def risk(row, caption_weight):
        memory, mask, labels = prepared[key(row["data_row"])]
        model.zero_grad(set_to_none=True)
        result = backward_candidate_risk(
            model,
            memory,
            mask,
            labels,
            row["quality"],
            alpha=plan["risk_alpha"],
            caption_weight=caption_weight,
        )
        result.pop("gold_logits")
        return result, gradients()

    def auxiliary(row):
        memory, mask, labels = prepared[key(row["data_row"])]
        model.zero_grad(set_to_none=True)
        loss = model(memory.to("mps"), mask.to("mps"), labels[0].to("mps"))["language"].loss
        (plan["auxiliary_CE_weight"] * loss).backward()
        return gradients()

    results = {}
    for model_name, policy in (("C5", original), ("rejected944", None)):
        if policy is None:
            with safe_open(str(CANDIDATE), framework="pt", device="cpu") as file:
                metadata = file.metadata()
            if (
                metadata.get("plan_sha256") != PARENT_SHA
                or metadata.get("optimizer_updates") != "944"
                or metadata.get("update") != "7424"
                or metadata.get("fixed_winner_checkpoint")
                != reference["translator_checkpoint_sha256"]
            ):
                raise ValueError("rejected policy checkpoint metadata differs")
            policy = load_file(str(CANDIDATE), device="cpu")
        parameters = model.policy_parameters()
        if policy.keys() != parameters.keys() or any(
            policy[n].shape != p.shape or policy[n].dtype != p.dtype for n, p in parameters.items()
        ):
            raise ValueError("short diagnostic policy tensor scope or dtype differs")
        with torch.no_grad():
            for name, p in parameters.items():
                duty.run(lambda p=p, value=policy[name]: p.copy_(value.to("mps")))
        policy_digest = state_digest(parameters)
        rows, accum_sem, accum_aux = [], {}, {}
        verified = 0
        for row in rewards:
            measurement, semantic = duty.run(lambda row=row: risk(row, 0))
            aux = duty.run(lambda row=row: auxiliary(row))
            balance = gradient_balance(semantic, aux)
            if stratum(row) == "contrast" and verified < 2:
                _, combined = duty.run(lambda row=row: risk(row, plan["auxiliary_CE_weight"]))
                balance["combined_loss_gradient_equivalence_max_relative_error"] = sum_equivalence(
                    semantic, aux, combined
                )
                verified += 1
                del combined
            for name in semantic:
                if name not in accum_sem:
                    accum_sem[name], accum_aux[name] = semantic[name].double(), aux[name].double()
                else:
                    accum_sem[name].add_(semantic[name])
                    accum_aux[name].add_(aux[name])
            result = {
                "identity": key(row["data_row"]),
                "stratum": stratum(row),
                "historical_quality": row["quality"],
                "measurement": measurement,
                **balance,
            }
            rows.append(result)
            append(output, f"{model_name}-gradients.jsonl", result)
            del semantic, aux
            torch.mps.empty_cache()
        if verified != 2:
            raise RuntimeError("short diagnostic lacks actual component-sum gradient checks")
        for values in (accum_sem, accum_aux):
            for tensor in values.values():
                tensor.div_(len(rewards))
        aggregate = gradient_balance(accum_sem, accum_aux)
        del accum_sem, accum_aux
        selected = retrieval_cases(rewards)
        matrix = []
        with torch.no_grad():
            for caption_row in selected:
                labels = prepared[key(caption_row["data_row"])][2][0]
                losses = []
                for video_row in selected:
                    memory, mask, _ = prepared[key(video_row["data_row"])]
                    value = duty.run(
                        lambda memory=memory, mask=mask, labels=labels: float(
                            model(memory.to("mps"), mask.to("mps"), labels.to("mps"))[
                                "language"
                            ].loss
                        )
                    )
                    losses.append(value)
                matrix.append(losses)
        model.zero_grad(set_to_none=True)
        if state_digest(parameters) != policy_digest or fixed_hashes(model, encoder) != fixed:
            raise RuntimeError("short diagnostic changed policy values or frozen backbones")
        contrasts = [r for r in rows if r["stratum"] == "contrast"]
        results[model_name] = {
            "gradient_cases": len(rows),
            "semantic_zero_gradient_cases": sum(r["semantic_gradient_is_zero"] for r in rows),
            "contrast_median_semantic_to_weighted_auxiliary_norm_ratio": statistics.median(
                r["semantic_to_weighted_auxiliary_norm_ratio"] for r in contrasts
            ),
            "contrast_median_gradient_cosine": statistics.median(
                r["gradient_cosine"] for r in contrasts
            ),
            "mean_human_caption_CE": statistics.mean(
                r["measurement"]["caption_loss"] for r in rows
            ),
            "mean_historical_candidate_semantic_risk": statistics.mean(
                r["measurement"]["expected_terra_error"] for r in rows
            ),
            "full_training_mean_gradient_balance": aggregate,
            "conditioning": conditioning(matrix),
            "parameters_unchanged_during_measurement": True,
        }
        print(
            json.dumps(
                {
                    "stage": "short_learning_audit",
                    "model": model_name,
                    "gradient_cases": len(rows),
                    "aggregate": aggregate,
                    "conditioning": {
                        k: v
                        for k, v in results[model_name]["conditioning"].items()
                        if not isinstance(v, list)
                    },
                }
            ),
            flush=True,
        )
    checked(path)
    write(
        output / "receipt.json",
        {
            "complete": True,
            "plan_sha256": sha256(path),
            "models": results,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
            "development_used": False,
            "training_only_diagnostic_not_generalization_or_model_selection": True,
            "all_policy_values_unchanged_during_each_measurement": True,
            "resource_receipt": duty.receipt(),
            "CPU_cache_bytes": cache_bytes,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.output:
        run(args.plan, args.output)
    else:
        parser.error("provide --freeze or --plan and --output")


if __name__ == "__main__":
    main()
