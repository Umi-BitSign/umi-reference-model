"""Train stationary replay after validating every reused fixed-parent teacher tensor."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    for key, value in {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    from benchmark_fleurs_dev import distance, normalize, sha256
    from c5_face_learning_pilot import BASELINE, BUNDLE, ROOT, read_rows
    from c5_full_coverage_mps import comparison, data_path, learning_rate_factor, storage
    from c5_gpu_budget import ApprovedGpuWindow, windowed_duty_cycle
    from c5_retention import teacher_kl
    from c5_stationary_mps import GATES, RECIPE, checked_plan, read_teacher
    from cache_fleurs_encoder import checked_stream_features
    from interpolate_visual_adapter import DIRECT_NAMES
    from train_c5_face_learning_pilot import (
        DutyCycle,
        build,
        configure,
        environment,
        visual_gradient_audit,
    )
    from train_c5_stem_continuation import stride2_streams
    from train_c5_stem_projection_frozen import complete_adapter_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backend-probe", type=Path)
    args = parser.parse_args()
    if bool(args.output_dir) == bool(args.backend_probe):
        parser.error("choose a new checkpoint directory or a training-only backend probe")
    path = args.plan.resolve()
    plan, rows, unique, teachers = checked_plan(path)
    environment()
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors.torch import save_file
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    configure(torch)
    budget = ApprovedGpuWindow(ROOT / plan["gpu_budget_policy"])
    duty = windowed_duty_cycle(DutyCycle, budget)(torch)
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BUNDLE / "models/byt5_base"), local_files_only=True
    )

    def inputs(row):
        bound = {**row, "cache": str(data_path(row["cache"]))}
        if row.get("sidecar"):
            bound["sidecar"] = str(data_path(row["sidecar"]))
        return {
            name: torch.from_numpy(value).unsqueeze(0).to("mps")
            for name, value in stride2_streams(bound).items()
        }

    def target(row):
        return torch.from_numpy(read_teacher(teachers[(row["source"], row["id"])])).to("mps")

    def fixed_prefix(network, row):
        labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to("mps")
        teacher = target(row)
        logits = network(**inputs(row), labels=labels, use_cache=False, return_dict=True).logits
        kl = teacher_kl(logits, teacher, labels, RECIPE["retention_temperature"])
        maximum = (logits - teacher).abs().max()
        if (
            not torch.isfinite(logits).all()
            or not torch.isfinite(kl)
            or abs(float(kl)) > RECIPE["teacher_maximum_initial_kl"]
            or float(maximum) > RECIPE["teacher_maximum_absolute_difference"]
        ):
            raise RuntimeError(
                f"reused teacher fails fixed-parent parity: {row['source']}:{row['id']}"
            )
        return logits, labels, float(kl), float(maximum)

    if args.backend_probe:
        output = args.backend_probe.resolve()
        if output.exists() or not output.is_relative_to(ROOT / "runs"):
            raise ValueError("stationary probe requires a new private runs/ output")
        model, selected, trainable = build(torch, plan["seed"], duty)
        cases = {
            (r["source"], r["id"]): r
            for r in [
                rows[0],
                max(
                    unique.values(),
                    key=lambda r: r["frames"]
                    if r.get("stored_temporal_stride") == 2
                    else (r["frames"] + 1) // 2,
                ),
                max(unique.values(), key=lambda r: len(r["reference"].encode())),
            ]
        }
        results = []
        for row in cases.values():

            def backward(row=row):
                model.zero_grad(set_to_none=True)
                logits, labels, initial_kl, maximum = fixed_prefix(model, row)
                ce = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(),
                    label_smoothing=RECIPE["label_smoothing"],
                )
                loss = ce + RECIPE["retention_coefficient"] * teacher_kl(
                    logits, target(row), labels, RECIPE["retention_temperature"]
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("stationary preflight objective is nonfinite")
                loss.backward()
                norms = visual_gradient_audit(torch, trainable, selected)
                norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("stationary preflight norm is nonfinite")
                return float(loss), float(norm), norms, initial_kl, maximum

            loss, norm, norms, initial_kl, maximum = duty.run(backward)
            results.append(
                {
                    "source": row["source"],
                    "example_id": row["id"],
                    "loss": loss,
                    "gradient_norm": norm,
                    "visual_gradient_norms": norms,
                    "initial_teacher_kl": initial_kl,
                    "teacher_maximum_difference": maximum,
                }
            )
        receipt = {
            "schema": "umi-studio-c5-stationary-mps-probe/1",
            "plan_sha256": sha256(path),
            "cases": results,
            "optimizer_updates": 0,
            "weights_saved": False,
            "maximum_allocated_bytes": duty.maximum_allocated_bytes,
            "gpu_resource_policy": budget.record,
            "miner_changed": False,
        }
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt), flush=True)
        return

    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("stationary training requires a new private checkpoint directory")
    evaluation = read_rows(ROOT / plan["evaluation_features"])
    parent_plan = json.loads((ROOT / plan["parent_plan"]).read_text())
    expected = {r["example_id"]: r for r in parent_plan["evaluation"]}
    if len(evaluation) != 142 or len({r["example_id"] for r in evaluation}) != 142:
        raise ValueError("stationary open evaluation coverage differs")
    prior = read_rows(ROOT / plan["prior_control_predictions"])
    prior_controls = {
        (r["model"], r["example_id"]): r["hypothesis"]
        for r in prior
        if r["model"] in ("selected_parent", "original_checkpoint_common_inputs")
    }
    if len(prior_controls) != 284:
        raise ValueError("prior fixed-control prediction coverage differs")
    output.mkdir()
    run = {
        "schema": "umi-studio-c5-stationary-mps-run/1",
        "plan_sha256": sha256(path),
        "recipe": RECIPE,
        "parent_checkpoint_sha256": plan["assets"]["selected_model"]["sha256"],
        "gpu_resource_policy": budget.record,
        "initial_gpu_duty_fraction": budget.duty_fraction(),
        "miner_health_policy": plan["miner_health_policy"],
        "miner_changed": False,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run": run}), flush=True)

    def evaluate(network, count, label):
        records = []
        for row in evaluation:

            def decode(row=row):
                target_path = data_path(row["cache"])
                if sha256(target_path) != row["cache_sha256"]:
                    raise ValueError("stationary open evaluation tensor changed")
                with np.load(target_path, allow_pickle=False) as cache:
                    arrays = {name: cache[name] for name in cache.files}
                arrays = checked_stream_features(arrays, len(arrays["face_features"]))
                with torch.no_grad():
                    hidden = network.encoder(
                        **{
                            name: torch.from_numpy(value).unsqueeze(0).to("mps")
                            for name, value in arrays.items()
                        },
                        return_dict=True,
                    ).last_hidden_state
                    tokens = GenerationMixin.generate(
                        network,
                        encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                            last_hidden_state=hidden
                        ),
                        max_length=RECIPE["generation_max_length"],
                        num_beams=RECIPE["generation_num_beams"],
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                return tokenizer.decode(tokens[0], skip_special_tokens=True)

            hypothesis = duty.run(decode)
            if count == 0 and prior_controls.get((label, row["example_id"])) != hypothesis:
                raise RuntimeError(
                    "stationary fixed control differs from the prior verified MPS control"
                )
            if (
                count == 0
                and label == "selected_parent"
                and row["domain"] == "fleurs"
                and hypothesis != expected[row["example_id"]]["expected_parent_cpu_hypothesis"]
            ):
                raise RuntimeError("selected FLEURS CPU/MPS parity differs")
            words = len(normalize(row["reference"]))
            edits = distance(normalize(row["reference"]), normalize(hypothesis))
            record = {
                "example_id": row["example_id"],
                "domain": row["domain"],
                "source_group": row["source_group"],
                "signer": row["signer"],
                "checkpoint_examples": count,
                "model": label,
                "hypothesis": hypothesis,
                "reference_words": words,
                "edits": edits,
                "wer": edits / words,
                "role": "open_development",
                "input_policy": plan["evaluation_input_policy"],
            }
            records.append(record)
            with (output / "evaluation-results.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return records

    original = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(BASELINE / "models/checkpoint-11625"), local_files_only=True
    )
    original.requires_grad_(False)
    original.eval()
    duty.run(lambda network=original: network.to("mps"))
    original_records = evaluate(original, 0, "original_checkpoint_common_inputs")
    duty.run(lambda network=original: network.to("cpu"))
    del original
    gc.collect()
    torch.mps.empty_cache()
    model, selected, trainable = build(torch, plan["seed"], duty)
    frozen_direct = duty.run(
        lambda: {name: selected[name].detach().cpu().clone() for name in DIRECT_NAMES}
    )
    parent_records = evaluate(model, 0, "selected_parent")
    print(
        json.dumps(
            {
                "fixed_control_parity_cases": 284,
                "fleurs_parent_cpu_parity": True,
                "parent_vs_original_common_inputs": comparison(original_records, parent_records),
            }
        ),
        flush=True,
    )

    maximum_difference = 0.0
    maximum_kl = 0.0
    for index, row in enumerate(unique.values(), 1):
        storage(plan)

        def verify(row=row):
            with torch.no_grad():
                _, _, kl, difference = fixed_prefix(model, row)
            return abs(kl), difference

        kl, difference = duty.run(verify)
        maximum_difference = max(maximum_difference, difference)
        maximum_kl = max(maximum_kl, kl)
        if index % 512 == 0:
            print(f"verified training-only parent teacher {index}/{len(unique)}", flush=True)
    verified = {
        "schema": "umi-studio-c5-stationary-teacher-validation/1",
        "plan_sha256": sha256(path),
        "unique_training_prefixes": len(unique),
        "maximum_absolute_difference": maximum_difference,
        "maximum_absolute_kl": maximum_kl,
        "optimizer_updates_before_validation": 0,
        "parent_checkpoint_sha256": plan["assets"]["selected_model"]["sha256"],
        "training_only": True,
        "all_teachers_verified": True,
    }
    (output / "teacher-validation.json").write_text(
        json.dumps(verified, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(verified), flush=True)
    groups = [
        {
            "params": [v for n, v in trainable.items() if "signhubert" in n],
            "lr": RECIPE["visual_learning_rate"],
        },
        {
            "params": [v for n, v in trainable.items() if n.startswith("decoder.")],
            "lr": RECIPE["decoder_learning_rate"],
        },
    ]
    if sum(len(g["params"]) for g in groups) != len(trainable):
        raise ValueError("stationary optimizer parameter partition differs")
    optimizer = torch.optim.AdamW(groups, weight_decay=RECIPE["weight_decay"])
    total = plan["optimizer_updates"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, total)
    )
    decisions = []
    seen = set()
    accumulation = RECIPE["gradient_accumulation"]
    for offset in range(0, len(rows), accumulation):
        storage(plan)
        batch = rows[offset : offset + accumulation]

        def update(batch=batch):
            optimizer.zero_grad(set_to_none=True)
            ce_values, kl_values = [], []
            for row in batch:
                if (
                    row["role"] != "training"
                    or not row["training_allowed"]
                    or row["loss_weight"] != 1.0
                ):
                    raise RuntimeError(
                        "stationary optimizer received another role or objective weight"
                    )
                labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to("mps")
                logits = model(
                    **inputs(row), labels=labels, use_cache=False, return_dict=True
                ).logits
                ce = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(),
                    label_smoothing=RECIPE["label_smoothing"],
                )
                kl = teacher_kl(logits, target(row), labels, RECIPE["retention_temperature"])
                if not torch.isfinite(ce + kl) or float(kl) < -1e-5:
                    raise RuntimeError("stationary objective is nonfinite")
                ((ce + RECIPE["retention_coefficient"] * kl) / len(batch)).backward()
                ce_values.append(float(ce))
                kl_values.append(float(kl))
            norms = visual_gradient_audit(torch, trainable, selected)
            norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("stationary gradient norm is nonfinite")
            optimizer.step()
            scheduler.step()
            if any(
                not torch.equal(selected[n].detach().cpu(), frozen_direct[n]) for n in DIRECT_NAMES
            ):
                raise RuntimeError("frozen stationary projection changed")
            return ce_values, kl_values, float(norm), norms

        ce, kl, norm, norms = duty.run(update)
        seen.update((r["source"], r["id"]) for r in batch)
        count = offset + len(batch)
        record = {
            "examples": count,
            "unique_training_examples_seen": len(seen),
            "update": offset // accumulation + 1,
            "example_ids": [r["id"] for r in batch],
            "sources": [r["source"] for r in batch],
            "signers": [r["signer"] for r in batch],
            "repeat_cycles": [r["repeat_cycle"] for r in batch],
            "ce": ce,
            "kl": kl,
            "gradient_norm": norm,
            "visual_gradient_norms": norms,
            "mps_active_seconds": duty.active_seconds,
            "mps_idle_seconds": duty.idle_seconds,
            "current_allowed_gpu_duty_fraction": budget.duty_fraction(),
        }
        with (output / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if (offset // accumulation + 1) % 128 == 0:
            print(
                f"trained {count}/{len(rows)} update={offset // accumulation + 1}/{total} "
                f"unique={len(seen)}/{len(unique)}",
                flush=True,
            )
        if count not in RECIPE["checkpoint_after_examples"]:
            continue
        target_path = output / f"stationary-after-{count:06d}.safetensors"
        state = duty.run(lambda: complete_adapter_state(selected))
        save_file(
            state,
            str(target_path),
            metadata={
                "schema": "umi-studio-c5-merged-parent-visual-lora/1",
                "plan_sha256": sha256(path),
                "parent_checkpoint_sha256": plan["assets"]["selected_model"]["sha256"],
                "rank": "16",
                "alpha": "32.0",
                "visual_last_layers": "4",
                "examples": str(count),
                "direct_parameters_frozen": "true",
                "objective_weights": "unit_per_exposure",
            },
        )
        results = evaluate(model, count, "candidate")
        decision = {
            "examples": count,
            "unique_training_examples_seen": len(seen),
            "checkpoint_sha256": sha256(target_path),
            **comparison(parent_records, results),
            "vs_original_common_inputs": comparison(original_records, results),
        }
        original_domains = decision["vs_original_common_inputs"]["domains"]
        original_safe = all(
            v["mean_wer"] - v["parent_mean_wer"] <= GATES["maximum_original_mean_delta"]
            and v["word_weighted_wer"] - v["parent_word_weighted_wer"]
            <= GATES["maximum_original_word_weighted_delta"]
            for v in original_domains.values()
        )
        decision["passes_original_baseline_guard"] = original_safe
        decision["passes_open_development_gate"] &= original_safe
        decisions.append(decision)
        (output / "open-decisions.json").write_text(
            json.dumps(decisions, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(decision), flush=True)
    if seen != set(unique):
        raise RuntimeError("stationary training ended without full unique coverage")
    passing = [r for r in decisions if r["passes_open_development_gate"]]
    result = {
        "schema": "umi-studio-c5-stationary-mps-decision/1",
        "plan_sha256": sha256(path),
        "evaluations": decisions,
        "unique_training_examples_seen": len(seen),
        "training_exposures": len(rows),
        "optimizer_updates": total,
        "selected_open_checkpoint": min(
            passing, key=lambda r: (r["domains"]["how2sign"]["mean_wer"], r["examples"])
        )
        if passing
        else None,
        "decision": "verify_runtime_then_freeze_new_independent_confirmation"
        if passing
        else "reject_stationary_replay_no_open_generalization_gain",
        "mps_active_seconds": duty.active_seconds,
        "mps_idle_seconds": duty.idle_seconds,
        "maximum_allocated_bytes": duty.maximum_allocated_bytes,
        "gpu_resource_policy": budget.record,
        "confirmation_decoded": False,
        "promoted": False,
        "miner_changed": False,
        "qualification": plan["qualification"],
    }
    (output / "decision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
