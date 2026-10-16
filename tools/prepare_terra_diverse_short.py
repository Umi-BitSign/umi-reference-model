"""Add reference-blind frozen-C5 samples to authenticated training-only pools."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import sha256
from launch_terra_visual_job import identity, now, write
from prepare_terra_short_curriculum import ROOT
from prepare_terra_short_curriculum import checked as checked_data
from terra_frozen_short import checked_rewards, key, pair
from terra_training_recovery import artifact, verify_snapshot
from train_terra_frozen_short import checked as checked_parent

PARENT = ROOT / "runs/terra-frozen-short-training-v1/plan.json"
GENERATION = {
    "max_length": 513,
    "num_beams": 1,
    "do_sample": True,
    "num_return_sequences": 4,
    "temperature": 0.8,
    "top_k": 0,
    "top_p": 0.95,
    "use_cache": True,
}


def case_seed(row):
    payload = f"diverse-short-v1:{row['source']}:{row['example_id']}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)


def merge_pool(original, sampled):
    """Preserve original judgments; additions remain ungraded unless text is identical."""
    row = copy.deepcopy(original)
    row["original_pool_size"] = len(row["pool"])
    tokens = {tuple(c["tokens"]) for c in row["pool"]}
    known = {c["text"]: q for c, q in zip(row["pool"], row["quality"], strict=True)}
    for candidate in sampled:
        identity_key = tuple(candidate["tokens"])
        if identity_key not in tokens:
            row["pool"].append(copy.deepcopy(candidate))
            row["quality"].append(known.get(candidate["text"]))
            tokens.add(identity_key)
    return row


def specification():
    parent = checked_parent(PARENT)
    data = checked_data(ROOT / parent["data"]["path"])
    rewards = checked_rewards(data)
    sources = dict(parent["source_sha256"])
    sources["tools/prepare_terra_diverse_short.py"] = sha256(Path(__file__))
    return {
        "schema": "umi-studio-terra-diverse-short-generation/1",
        "parent": artifact(PARENT),
        "data": parent["data"],
        "original_rewards": parent["historical_training_rewards"],
        "source_sha256": sources,
        "reference": parent["reference"],
        "initial_reference_hashes": parent["initial_reference_hashes"],
        "resource_policy": parent["resource_policy"],
        "resource_policy_sha256": parent["resource_policy_sha256"],
        "generation": GENERATION,
        "seed": "SHA256 of version, source and example_id, excluding reference",
        "training_identities": [key(r["data_row"]) for r in rewards],
        "original_candidate_pools_and_scores_preserved": True,
        "sampled_policy": "original C5, never rejected local weights",
        "reference_conditions_generation": False,
        "new_Terra_requests": 0,
        "optimizer_updates": 0,
        "all_new_scores_must_be_authenticated_before_training": True,
        "heldout_material_used": False,
        "sampling_is_not_a_model_quality_gain": True,
        "GPU_duty_fraction": 0.5,
        "CPU_threads": 2,
    }, rewards


def checked(path):
    plan = json.loads(path.read_text())
    expected, rewards = specification()
    if plan != json.loads(json.dumps(expected)):
        raise ValueError("diverse short generation data, sources or sampling profile changed")
    verify_snapshot(path, plan)
    if sha256(ROOT / plan["resource_policy"]) != plan["resource_policy_sha256"]:
        raise ValueError("diverse short resource policy changed")
    return plan, rewards


def freeze(path):
    from c5_experiment_sources import archive_sources
    from terra_training_storage import storage

    path = path.resolve()
    if path.parent.exists() or not path.is_relative_to(ROOT / "runs"):
        raise ValueError("diverse short generation needs a fresh owned plan")
    storage(5 * 1024**2)
    plan, _ = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {"plan_sha256": sha256(path), "training_clips": len(plan["training_identities"])}
        ),
        flush=True,
    )


def run(path, output):
    from terra_training_storage import storage
    from train_terra_c5_tail import gpu_lease

    plan, rewards = checked(path)
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("diverse short generation needs a fresh owned output")
    storage(5 * 1024**2)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        with gpu_lease():
            generate(path, output, plan, rewards)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def generate(path, output, plan, rewards):
    import torch
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_terra_tail import candidate_labels, fixed_hashes, load_model, native_memory
    from probe_tiny_rgb_capacity import state_digest
    from terra_teacher_candidates import TrainingRolloutBudget
    from terra_visual_learning import bounded_candidate
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from transformers import ByT5Tokenizer, StoppingCriteriaList
    from transformers.modeling_outputs import BaseModelOutput

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    if budget.duty_fraction() != 0.5 or budget.cpu_threads() != 2:
        raise ValueError("diverse short generation requires50% GPU/two CPU threads")
    duty = WinnerDutyCycle(torch, budget)
    model, reference = load_model()
    encoder = load_frozen_winner_encoder(reference).requires_grad_(False).eval()
    if (
        reference != plan["reference"]
        or {
            "language": state_digest(model.language.state_dict()),
            "encoder": state_digest(encoder.state_dict()),
        }
        != plan["initial_reference_hashes"]
    ):
        raise ValueError("diverse short generation must initialize from original C5")
    fixed, initial = fixed_hashes(model, encoder), state_digest(model.policy_parameters())
    model.requires_grad_(False)
    if not torch.backends.mps.is_available():
        raise RuntimeError("diverse short generation requires qualified Metal")
    model = duty.run(lambda: model.to("mps"))
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    records = []
    with torch.no_grad():
        for original in rewards:
            memory = native_memory(
                encoder, pair(original), reference["translator_checkpoint_sha256"]
            )
            seed = case_seed(original["data_row"])

            def sample(memory=memory, seed=seed):
                torch.manual_seed(seed)
                torch.mps.manual_seed(seed)
                return model.language.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=memory.to("mps")),
                    attention_mask=torch.ones(memory.shape[:2], dtype=torch.long, device="mps"),
                    **plan["generation"],
                    stopping_criteria=StoppingCriteriaList([TrainingRolloutBudget()]),
                ).cpu()

            observed = duty.run(sample)
            if observed.ndim != 2 or len(observed) != 4:
                raise ValueError("diverse short generator did not emit four observed samples")
            samples = []
            for i, tokens in enumerate(observed):
                candidate = bounded_candidate(tokens, 512)
                samples.append(
                    {
                        **candidate,
                        "text": tokenizer.decode(
                            candidate["tokens"],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        "kind": "sampled_original_C5",
                        "sample_index": i,
                    }
                )
            result = merge_pool(original, samples)
            candidate_labels(result["pool"])
            result["new_sample_origin"] = {
                "plan_sha256": sha256(path),
                "seed": seed,
                "original_C5_policy_digest": initial,
                "observed_samples": samples,
            }
            records.append(result)
            write(output / f"case-{len(records):04d}.json", result)
            print(
                json.dumps(
                    {
                        "stage": "diverse_short_generation",
                        "clips": len(records),
                        "pool_size": len(result["pool"]),
                        "ungraded_candidates": result["quality"].count(None),
                    }
                ),
                flush=True,
            )
            torch.mps.empty_cache()
    if fixed_hashes(model, encoder) != fixed or state_digest(model.policy_parameters()) != initial:
        raise RuntimeError("diverse short generation changed original C5 values")
    checked(path)
    write(
        output / "pending-pools.json",
        {
            "schema": "umi-studio-terra-diverse-short-ungraded-pools/1",
            "plan_sha256": sha256(path),
            "original_rewards": plan["original_rewards"],
            "records": records,
            "training_identities": len(records),
            "new_unique_token_candidates": sum(
                len(r["pool"]) - r["original_pool_size"] for r in records
            ),
            "ungraded_candidate_positions": sum(r["quality"].count(None) for r in records),
            "model_training_ready": False,
            "new_Terra_requests": 0,
            "optimizer_updates": 0,
            "original_C5_and_frozen_backbones_unchanged": True,
            "heldout_material_used": False,
            "resource_receipt": duty.receipt(),
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
