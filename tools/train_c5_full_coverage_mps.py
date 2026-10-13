"""Train one protected full-corpus pass with cached frozen-parent KL on Metal."""

from __future__ import annotations

import argparse
import gc
import json
import math
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
    from c5_full_coverage_mps import (
        GATES,
        RECIPE,
        checked_plan,
        comparison,
        data_path,
        learning_rate_factor,
        storage,
    )
    from c5_gpu_budget import ApprovedGpuWindow, windowed_duty_cycle
    from c5_retention import teacher_kl
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
        parser.error("choose new checkpoint output or a training-only backend probe")
    path = args.plan.resolve()
    plan = checked_plan(path)
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
    rows = read_rows(path.with_name("schedule.jsonl"))
    if len(rows) != plan["training_examples"] or any(
        r["role"] != "training" or not r["training_allowed"] for r in rows
    ):
        raise ValueError("frozen training schedule includes another role")
    training = {(r["source"], r["id"]): r for r in rows}
    if len(training) != len(rows):
        raise ValueError("frozen schedule repeats a training example")

    def inputs(row):
        bound = {**row, "cache": str(data_path(row["cache"]))}
        if row.get("sidecar"):
            bound["sidecar"] = str(data_path(row["sidecar"]))
        arrays = stride2_streams(bound)
        return {
            name: torch.from_numpy(value).unsqueeze(0).to("mps") for name, value in arrays.items()
        }

    if args.backend_probe:
        model, selected, trainable = build(torch, plan["seed"], duty)
        output = args.backend_probe.resolve()
        if output.exists() or not output.is_relative_to(ROOT / "runs"):
            raise ValueError("backend probe requires a new private runs/ output")

        def key(row):
            return row["source"], row["id"]

        cases = {
            key(r): r
            for r in [
                rows[0],
                max(
                    rows,
                    key=lambda r: r["frames"]
                    if r.get("stored_temporal_stride") == 2
                    else (r["frames"] + 1) // 2,
                ),
                max(rows, key=lambda r: len(r["reference"].encode())),
            ]
        }
        results = []
        for row in cases.values():

            def backward(row=row):
                model.zero_grad(set_to_none=True)
                labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to("mps")
                logits = model(
                    **inputs(row), labels=labels, use_cache=False, return_dict=True
                ).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), labels.flatten(), label_smoothing=0.1
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite full-coverage preflight loss")
                loss.backward()
                norms = visual_gradient_audit(torch, trainable, selected)
                gradient = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
                if not torch.isfinite(gradient):
                    raise RuntimeError("nonfinite full-coverage preflight norm")
                return float(loss), float(gradient), norms

            loss, gradient, norms = duty.run(backward)
            results.append(
                {
                    "source": row["source"],
                    "example_id": row["id"],
                    "loss": loss,
                    "gradient_norm": gradient,
                    "visual_gradient_norms": norms,
                }
            )
        result = {
            "schema": "umi-studio-c5-full-coverage-mps-probe/1",
            "plan_sha256": sha256(path),
            "cases": results,
            "optimizer_updates": 0,
            "weights_saved": False,
            "miner_changed": False,
            "mps_active_seconds": duty.active_seconds,
            "mps_idle_seconds": duty.idle_seconds,
            "maximum_allocated_bytes": duty.maximum_allocated_bytes,
            "resource_policy": budget.record,
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result))
        return

    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("training requires a new private checkpoint directory")
    evaluation_path = path.with_name("evaluation-features.jsonl")
    evaluation = read_rows(evaluation_path)
    if len(evaluation) != 142 or len({r["example_id"] for r in evaluation}) != 142:
        raise ValueError("open evaluation feature preparation incomplete")
    expected = {r["example_id"]: r for r in plan["evaluation"]}
    for row in evaluation:
        if (
            any(row[k] != value for k, value in expected[row["example_id"]].items())
            or row["training_allowed"]
        ):
            raise ValueError("open evaluation role or identity changed")
    output.mkdir()
    data = ROOT / plan["data_dir"]
    teacher_dir = data / "teacher-logits"
    teacher_dir.mkdir()
    run = {
        "schema": "umi-studio-c5-full-coverage-mps-run/1",
        "plan_sha256": sha256(path),
        "recipe": RECIPE,
        "parent_checkpoint_sha256": plan["assets"]["selected_model"]["sha256"],
        "gpu_resource_policy": budget.record,
        "initial_gpu_duty_fraction": budget.duty_fraction(),
        "evaluation_features_sha256": sha256(evaluation_path),
        "scientific_recipe_unchanged": True,
        "miner_health_policy": "user_authorized_maintenance",
        "miner_changed": False,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run": run}), flush=True)

    def evaluate(network, step, label):
        records = []
        for row in evaluation:

            def decode(row=row):
                target = data_path(row["cache"])
                if sha256(target) != row["cache_sha256"]:
                    raise ValueError("open evaluation feature cache changed")
                with np.load(target, allow_pickle=False) as cache:
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
                        max_length=2048,
                        num_beams=5,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                return tokenizer.decode(tokens[0], skip_special_tokens=True)

            hypothesis = duty.run(decode)
            words = len(normalize(row["reference"]))
            edits = distance(normalize(row["reference"]), normalize(hypothesis))
            record = {
                "example_id": row["example_id"],
                "domain": row["domain"],
                "source_group": row["source_group"],
                "signer": row["signer"],
                "checkpoint_examples": step,
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
    parent_fleurs = [r for r in parent_records if r["domain"] == "fleurs"]
    parity = [
        r["hypothesis"] == expected[r["example_id"]]["expected_parent_cpu_hypothesis"]
        for r in parent_fleurs
    ]
    if not all(parity):
        raise RuntimeError("selected MPS FLEURS control differs from the verified CPU runtime")
    print(
        json.dumps(
            {
                "parent_vs_original_common_inputs": comparison(original_records, parent_records),
                "fleurs_parent_cpu_parity": all(parity),
            }
        ),
        flush=True,
    )

    teacher_paths = {}
    for index, row in enumerate(rows):
        storage(plan)

        def logits(row=row):
            with torch.no_grad():
                labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to("mps")
                return (
                    model(**inputs(row), labels=labels, use_cache=False, return_dict=True)
                    .logits.cpu()
                    .numpy()
                    .astype(np.float32)
                )

        values = duty.run(logits)
        target = teacher_dir / f"{index:04d}.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, logits=values)
        teacher_paths[(row["source"], row["id"])] = (target, sha256(target))
        if (index + 1) % 256 == 0:
            print(f"teacher training-only {index + 1}/{len(rows)}", flush=True)
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
        raise ValueError("full-coverage optimizer partition differs")
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    total = math.ceil(len(rows) / 4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, total)
    )
    decisions = []
    for offset in range(0, len(rows), 4):
        storage(plan)
        batch = rows[offset : offset + 4]

        def update(batch=batch, offset=offset):
            optimizer.zero_grad(set_to_none=True)
            ce_values = []
            kl_values = []
            for row in batch:
                if row["role"] != "training" or not row["training_allowed"]:
                    raise RuntimeError("evaluation example reached optimizer")
                labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to("mps")
                target, digest = teacher_paths[(row["source"], row["id"])]
                if sha256(target) != digest:
                    raise ValueError("frozen training-prefix teacher logits changed")
                with np.load(target, allow_pickle=False) as cache:
                    teacher = torch.from_numpy(cache["logits"]).to("mps")
                logits = model(
                    **inputs(row), labels=labels, use_cache=False, return_dict=True
                ).logits
                ce = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), labels.flatten(), label_smoothing=0.1
                )
                kl = teacher_kl(logits, teacher, labels, 2.0)
                if not torch.isfinite(ce + kl) or float(kl) < -1e-5:
                    raise RuntimeError("full-coverage objective is nonfinite")
                if offset == 0 and abs(float(kl)) > 1e-5:
                    raise RuntimeError("starting teacher and student differ")
                ((ce + kl) * row["loss_weight"] / len(batch)).backward()
                ce_values.append(float(ce))
                kl_values.append(float(kl))
            norms = visual_gradient_audit(torch, trainable, selected)
            norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("full-coverage gradient norm nonfinite")
            optimizer.step()
            scheduler.step()
            if any(
                not torch.equal(selected[n].detach().cpu(), frozen_direct[n]) for n in DIRECT_NAMES
            ):
                raise RuntimeError("frozen full-coverage projection changed")
            return ce_values, kl_values, float(norm), norms

        ce, kl, norm, norms = duty.run(update)
        count = offset + len(batch)
        record = {
            "examples": count,
            "update": offset // 4 + 1,
            "example_ids": [r["id"] for r in batch],
            "sources": [r["source"] for r in batch],
            "signers": [r["signer"] for r in batch],
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
        if (offset // 4 + 1) % 64 == 0:
            print(f"trained {count}/{len(rows)} update={offset // 4 + 1}/{total}", flush=True)
        if count in RECIPE["checkpoint_after_examples"]:
            target = output / f"coverage-after-{count:06d}.safetensors"
            state = duty.run(lambda: complete_adapter_state(selected))
            save_file(
                state,
                str(target),
                metadata={
                    "schema": "umi-studio-c5-merged-parent-visual-lora/1",
                    "plan_sha256": sha256(path),
                    "parent_checkpoint_sha256": plan["assets"]["selected_model"]["sha256"],
                    "rank": "16",
                    "alpha": "32.0",
                    "visual_last_layers": "4",
                    "examples": str(count),
                    "direct_parameters_frozen": "true",
                },
            )
            results = evaluate(model, count, "candidate")
            decision = {
                "examples": count,
                "checkpoint_sha256": sha256(target),
                **comparison(parent_records, results),
                "vs_original_common_inputs": comparison(original_records, results),
            }
            original_domains = decision["vs_original_common_inputs"]["domains"]
            original_safe = all(
                values["mean_wer"] - values["parent_mean_wer"]
                <= GATES["maximum_original_mean_delta"]
                and values["word_weighted_wer"] - values["parent_word_weighted_wer"]
                <= GATES["maximum_original_word_weighted_delta"]
                for values in original_domains.values()
            )
            decision["passes_original_baseline_guard"] = original_safe
            decision["passes_open_development_gate"] &= original_safe
            decisions.append(decision)
            (output / "open-decisions.json").write_text(
                json.dumps(decisions, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(decision), flush=True)
    passing = [r for r in decisions if r["passes_open_development_gate"]]
    result = {
        "schema": "umi-studio-c5-full-coverage-mps-decision/1",
        "plan_sha256": sha256(path),
        "evaluations": decisions,
        "selected_open_checkpoint": min(
            passing, key=lambda r: (r["domains"]["how2sign"]["mean_wer"], r["examples"])
        )
        if passing
        else None,
        "decision": "verify_runtime_then_freeze_new_independent_confirmation"
        if passing
        else "reject_full_coverage_no_open_generalization_gain",
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
