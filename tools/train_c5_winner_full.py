"""Train all functional C5-winner translator weights with frozen-parent retention."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from c5_full_coverage_mps import comparison, load_training
from c5_winner_reference import ROOT, load_reference
from c5_winner_training import CORPUS, RECIPE, checked_plan, lr_factor, private_data, storage
from tiny_rgb_full_corpus import read_rows


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def configure_model(model):
    from c5_winner_mps_safe import replace_fairseq_exact_gelu, replace_native_layer_norms
    from probe_c5_winner_decoder import trainable_full
    from tiny_rgb_mps_safe import replace_native_gelu

    counts = (
        replace_fairseq_exact_gelu(model),
        replace_native_gelu(model),
        replace_native_layer_norms(model),
    )
    if counts != (12, 1, 30):
        raise RuntimeError("winner training numerical topology differs from the qualified proof")
    return trainable_full(model)


def main():
    for key, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = checked_plan(args.plan)
    reference = load_reference()
    model_root = ROOT / reference["bundle_root"] / "model"
    sys.path.insert(0, str(model_root / "vendor"))
    sys.path.insert(0, str(model_root / "vendor/demo"))
    os.environ["SHUBERT_DINOV2_SOURCE"] = str(model_root / "vendor/dinov2-source")
    import numpy as np
    import torch
    from c5_certification_budget import CertificationComputeWindow, WinnerDutyCycle
    from c5_retention import teacher_kl
    from cache_fleurs_encoder import checked_stream_features
    from inference import SignLanguageByT5ForConditionalGeneration
    from probe_c5_winner_decoder import decoder_gradient_guard
    from safetensors.torch import save_file
    from train_c5_stem_continuation import stride2_streams
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    budget = CertificationComputeWindow(ROOT / plan["resource_policy"])
    torch.set_num_threads(budget.cpu_threads())
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("winner full training requires native Metal")
    torch.manual_seed(plan["seed"])
    duty = WinnerDutyCycle(torch, budget)
    output = args.output_dir.resolve()
    data = ROOT / plan["data_dir"]
    if output.exists() or data.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("winner training requires new checkpoint and derived-data directories")
    output.mkdir(parents=True)
    teacher_directory = data / "teacher"
    teacher_directory.mkdir(parents=True)
    training = load_training(CORPUS / "audit.json", CORPUS / "training-streams.jsonl")
    training_by_id = {(row["source"], row["id"]): row for row in training}
    schedule = read_rows(args.plan.with_name("schedule.jsonl"))
    tokenizer = ByT5Tokenizer.from_pretrained(
        model_root / "models/byt5_base", local_files_only=True
    )
    labels = {
        identity: tokenizer(row["reference"], return_tensors="pt", truncation=False).input_ids
        for identity, row in training_by_id.items()
    }
    if any(value.shape[1] > RECIPE["max_text_tokens"] for value in labels.values()):
        raise ValueError("winner training target exceeds its bound; never truncate")
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        model_root / "models/checkpoint-11625", local_files_only=True
    )
    selected = configure_model(model)
    if (
        sum(value.numel() for value in selected.values()) != plan["parameter_counts"]["trainable"]
        or len(selected) != 456
    ):
        raise ValueError("winner functional optimizer topology changed")
    duty.run(lambda: model.to("mps"))
    selected = {name: value for name, value in model.named_parameters() if value.requires_grad}
    output.joinpath("launch.json").write_text(
        json.dumps(
            {
                "plan_sha256": sha256(args.plan),
                "base_model": reference["model_bundle_sha256"],
                "resource_policy": budget.record,
                "stage": "initial_matched_winner_control",
                "miner_modified": False,
                "optimizer_updates": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    def inputs(row):
        bound = {**row, "cache": str(private_data(row["cache"]))}
        if row.get("sidecar"):
            bound["sidecar"] = str(private_data(row["sidecar"]))
        arrays = stride2_streams(bound)
        return {
            name: torch.from_numpy(value).unsqueeze(0).to("mps") for name, value in arrays.items()
        }

    def evaluate(exposures, label):
        model.eval()
        predictions = []
        for row in plan["evaluation"]:
            target = private_data(row["cache"])
            if sha256(target) != row["cache_sha256"]:
                raise ValueError("winner evaluation feature cache changed")
            with np.load(target, allow_pickle=False) as cache:
                arrays = checked_stream_features(
                    {name: cache[name] for name in cache.files}, len(cache["face_features"])
                )

            def decode(arrays=arrays):
                with torch.no_grad():
                    hidden = model.encoder(
                        **{
                            name: torch.from_numpy(value).unsqueeze(0).to("mps")
                            for name, value in arrays.items()
                        },
                        return_dict=True,
                    ).last_hidden_state
                    tokens = GenerationMixin.generate(
                        model,
                        encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                            last_hidden_state=hidden
                        ),
                        **reference["generation"],
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    return tokenizer.decode(tokens[0].cpu(), skip_special_tokens=True)

            hypothesis = duty.run(decode)
            torch.mps.empty_cache()
            words = len(normalize(row["reference"]))
            edits = distance(normalize(row["reference"]), normalize(hypothesis))
            record = {
                **{key: row[key] for key in ("example_id", "domain", "signer", "source_group")},
                "role": "open_development",
                "training_allowed": False,
                "model": label,
                "checkpoint_examples": exposures,
                "hypothesis": hypothesis,
                "reference_words": words,
                "edits": edits,
                "wer": edits / words,
                "input_policy": RECIPE["evaluation_input_policy"],
            }
            predictions.append(record)
            append(output / "evaluation-results.jsonl", record)
        return predictions

    parent_predictions = evaluate(0, "winner_matched_translator_control")
    parent_summary = comparison(parent_predictions, parent_predictions)
    output.joinpath("winner-control.json").write_text(
        json.dumps(
            {
                "model_bundle_sha256": reference["model_bundle_sha256"],
                "domains": parent_summary["domains"],
                "predictions_sha256": sha256(output / "evaluation-results.jsonl"),
                "full_composite_or_official_score": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps({"stage": "winner_control_complete", "domains": parent_summary["domains"]}),
        flush=True,
    )
    teacher_paths = {}
    model.eval()
    for index, row in enumerate(training):
        prospective = labels[(row["source"], row["id"])].shape[1] * 384 * 4 + 1024**2
        storage(plan, prospective, prospective)
        identity = row["source"], row["id"]

        def teacher(row=row, identity=identity):
            with torch.no_grad():
                target = labels[identity].to("mps")
                result = model(**inputs(row), labels=target, use_cache=False, return_dict=True)
                return result.logits.cpu().numpy().astype(np.float32)

        logits = duty.run(teacher)
        torch.mps.empty_cache()
        path = teacher_directory / f"{index:05d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, logits=logits)
        teacher_paths[identity] = (path, sha256(path))
        append(
            output / "teacher.jsonl",
            {
                "source": row["source"],
                "example_id": row["id"],
                "role": "training",
                "checkpoint_sha256": reference["translator_checkpoint_sha256"],
                "path": str(path.relative_to(ROOT)),
                "sha256": teacher_paths[identity][1],
            },
        )
        if (index + 1) % 512 == 0:
            print(
                json.dumps(
                    {
                        "stage": "training_prefix_teacher",
                        "complete": index + 1,
                        "total": len(training),
                        **duty.receipt(),
                    }
                ),
                flush=True,
            )
    optimizer = torch.optim.AdamW(
        selected.values(),
        lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"],
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: lr_factor(update, plan["planned_updates"])
    )
    run_record = {
        "schema": "umi-studio-c5-winner-full-execution/1",
        "plan_sha256": sha256(args.plan),
        "base_model": reference["model_bundle_sha256"],
        "parameter_counts": plan["parameter_counts"],
        "teacher_receipt_sha256": sha256(output / "teacher.jsonl"),
        "winner_control_sha256": sha256(output / "winner-control.json"),
        "resource_policy": budget.record,
        "miner_modified": False,
        "confirmation_decoded": False,
    }
    output.joinpath("run.json").write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n")
    seen, decisions = set(), []
    for start in range(0, len(schedule), RECIPE["gradient_accumulation"]):
        group = schedule[start : start + RECIPE["gradient_accumulation"]]
        optimizer.zero_grad(set_to_none=True)
        losses, divergences = [], []
        for exposure in group:
            identity = exposure["source"], exposure["example_id"]
            row = training_by_id[identity]
            teacher_path, digest = teacher_paths[identity]
            if sha256(teacher_path) != digest:
                raise ValueError("winner frozen training-prefix teacher changed")
            with np.load(teacher_path, allow_pickle=False) as payload:
                frozen = payload["logits"].copy()

            divisor = len(group)

            def backward(row=row, identity=identity, frozen=frozen, divisor=divisor):
                model.train()
                target = labels[identity].to("mps")
                result = model(**inputs(row), labels=target, use_cache=False, return_dict=True)
                kl = teacher_kl(
                    result.logits,
                    torch.from_numpy(frozen).to("mps"),
                    target,
                    RECIPE["retention_temperature"],
                )
                loss = result.loss + RECIPE["retention_coefficient"] * kl
                if not torch.isfinite(loss):
                    raise RuntimeError("winner training loss is nonfinite")
                (loss / divisor).backward()
                values = float(result.loss), float(kl)
                del result, target, loss, kl
                torch.mps.synchronize()
                torch.mps.empty_cache()
                return values

            ce, kl = duty.run(backward)
            losses.append(ce)
            divergences.append(kl)
            seen.add(identity)

        def update():
            coverage = decoder_gradient_guard(model, selected, torch)
            norm = float(torch.nn.utils.clip_grad_norm_(selected.values(), RECIPE["gradient_clip"]))
            if not math.isfinite(norm):
                raise RuntimeError("winner aggregate gradient norm is nonfinite")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.mps.synchronize()
            torch.mps.empty_cache()
            return {"gradient_coverage": coverage, "gradient_norm": norm}

        gradient = duty.run(update)
        exposures = start + len(group)
        record = {
            "exposures": exposures,
            "update": exposures // 8,
            "unique_training_examples_seen": len(seen),
            "mean_ce": statistics.mean(losses),
            "mean_training_prefix_kl": statistics.mean(divergences),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **gradient,
            **duty.receipt(),
        }
        append(output / "steps.jsonl", record)
        if exposures % 512 == 0:
            print(json.dumps(record), flush=True)
        if exposures not in RECIPE["checkpoint_after_exposures"]:
            continue
        storage(plan, reference["verified_local_intake"]["archive_bytes"])
        checkpoint = output / f"winner-full-after-{exposures:06d}.safetensors"

        def save(checkpoint=checkpoint, exposures=exposures):
            # Preserve the publisher's duplicate tied aliases for faithful HF reloads.
            state = {
                name: value.detach().cpu().contiguous().clone()
                for name, value in model.state_dict().items()
            }
            save_file(
                state,
                str(checkpoint),
                metadata={
                    "schema": "umi-studio-c5-winner-full-translator/1",
                    "plan_sha256": sha256(args.plan),
                    "parent_checkpoint_sha256": reference["translator_checkpoint_sha256"],
                    "exposures": str(exposures),
                },
            )

        duty.run(save)
        gc.collect()
        predictions = evaluate(exposures, "winner_finetuned_translator")
        decision = {
            "exposures": exposures,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            **comparison(parent_predictions, predictions),
            "vs_historical_selected": comparison(
                plan["historical_controls"]["selected_parent"], predictions
            ),
            "vs_historical_original": comparison(
                plan["historical_controls"]["original_checkpoint_common_inputs"], predictions
            ),
            "complete_composite_verified": False,
            "promotion_authorized": False,
        }
        decisions.append(decision)
        output.joinpath("open-decisions.json").write_text(
            json.dumps(decisions, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(decision), flush=True)
    if len(seen) != 7529:
        raise RuntimeError("winner full training failed unique eligible-corpus coverage")
    checked_plan(args.plan)
    final = {
        "schema": "umi-studio-c5-winner-full-decision/1",
        "complete": True,
        "plan_sha256": sha256(args.plan),
        "evaluations": decisions,
        "requires_native_composite_validation": any(
            row["passes_open_development_gate"] for row in decisions
        ),
        "unique_training_examples": len(seen),
        "selected_serving_model_changed": False,
        "miner_modified": False,
        "confirmation_decoded": False,
        **duty.receipt(),
    }
    output.joinpath("decision.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"complete": True, "decision_sha256": sha256(output / "decision.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
