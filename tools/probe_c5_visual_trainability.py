"""Compare two fixed learning rates on repeated, training-only OpenASL clips."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from interpolate_visual_adapter import DIRECT_NAMES

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if not args.miner_maintenance_mode or args.output_dir.exists():
        parser.error("trainability probe requires maintenance mode and a new output")
    plan = json.loads(args.plan.read_text())
    for key, relative in (
        ("index_sha256", plan["index"]),
        ("partition_sha256", plan["partition"]),
        ("starting_adapter_sha256", plan["starting_adapter"]),
        ("baseline_results_sha256", plan["baseline_results"]),
    ):
        if sha256(ROOT / relative) != plan[key]:
            raise ValueError(f"trainability probe input changed: {relative}")
    if (
        plan["schema"] != "umi-studio-c5-visual-trainability-plan/1"
        or plan["trainer_source_sha256"] != sha256(Path(__file__))
        or plan["adapter_implementation_sha256"] != sha256(ROOT / "tools/shubert_lora.py")
    ):
        raise ValueError("trainability probe implementation changed")
    rows = [json.loads(line) for line in (ROOT / plan["index"]).read_text().splitlines()]
    partition = json.loads((ROOT / plan["partition"]).read_text())
    allowed = {row["example_id"] for row in partition["training_records"]}
    if len(rows) != 24 or any(row["example_id"] not in allowed for row in rows):
        raise ValueError("trainability probe includes another split")
    baseline_rows = [json.loads(line) for line in (ROOT / plan["baseline_results"]).read_text().splitlines()]
    baseline_by_id = {row["example_id"]: row for row in baseline_rows}
    if set(baseline_by_id) != {row["clip_id"] for row in rows}:
        raise ValueError("trainability comparator examples differ")
    for row in baseline_rows:
        if len(row["hypothesis"].encode("utf-8")) >= plan["max_length"] - 2:
            raise ValueError("comparator hypothesis exceeds the probe generation budget")
    for key, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(BASELINE / "run/hf"), "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(BASELINE / "vendor"))
    sys.path.insert(0, str(BASELINE / "vendor/demo"))
    import numpy as np
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors.torch import load_file, save_file
    from shubert_visual_lora import install_visual_lora, load_visual_adapter
    from train_c5_stem_projection_frozen import complete_adapter_state, freeze_direct_parameters
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    torch.set_num_threads(4)
    tokenizer = ByT5Tokenizer.from_pretrained(str(BASELINE / "models/byt5_base"), local_files_only=True)
    examples = []
    for row in rows:
        cache = ROOT / plan["cache_dir"] / f"{row['clip_id']}.npz"
        if sha256(cache) != row["encoder_cache_sha256"]:
            raise ValueError("trainability cache changed")
        with np.load(cache, allow_pickle=False) as payload:
            features = {
                name: torch.from_numpy(payload[name]).unsqueeze(0)
                for name in ("face_features", "left_hand_features", "right_hand_features", "pose_features")
            }
        labels = tokenizer(row["reference"], return_tensors="pt").input_ids
        examples.append((row, features, labels))
    starting_weights = load_file(str(ROOT / plan["starting_adapter"]), device="cpu")
    args.output_dir.mkdir(parents=True)
    results = []
    for arm in plan["arms"]:
        torch.manual_seed(plan["seed"])
        model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
            str(BASELINE / "models/checkpoint-11625"), local_files_only=True
        ).eval()
        selected = install_visual_lora(model, 16, 32.0, 4)
        load_visual_adapter(model, starting_weights)
        trainable = freeze_direct_parameters(selected)
        direct = {name: selected[name].detach().clone() for name in DIRECT_NAMES}
        optimizer = torch.optim.AdamW(trainable.values(), lr=arm["learning_rate"], weight_decay=0.01)
        for epoch in range(1, plan["epochs"] + 1):
            order = list(range(len(examples)))
            random.Random(plan["seed"] + epoch).shuffle(order)
            losses = []
            for offset in range(0, len(order), 4):
                batch = order[offset : offset + 4]
                optimizer.zero_grad(set_to_none=True)
                for index in batch:
                    row, features, labels = examples[index]
                    output = model(**features, labels=labels, use_cache=False, return_dict=True)
                    loss = torch.nn.functional.cross_entropy(
                        output.logits.reshape(-1, output.logits.shape[-1]), labels.reshape(-1),
                        label_smoothing=0.1,
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite trainability loss")
                    (loss / len(batch)).backward()
                    losses.append(float(loss.detach()))
                visual = [parameter for name, parameter in trainable.items() if "signhubert" in name]
                if len(visual) != 16 or any(parameter.grad is None for parameter in visual):
                    raise RuntimeError("trainability visual gradient disconnected")
                norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("non-finite trainability gradient")
                optimizer.step()
                if any(not torch.equal(selected[name], value) for name, value in direct.items()):
                    raise RuntimeError("trainability direct projection changed")
            if epoch not in plan["evaluate_after_epochs"]:
                continue
            target = args.output_dir / f"{arm['name']}-epoch-{epoch:02d}.safetensors"
            save_file(complete_adapter_state(selected), str(target), metadata={
                "schema": "umi-studio-shubert-visual-lora/1", "rank": "16", "alpha": "32.0",
                "visual_last_layers": "4", "direct_parameters_frozen": "true",
                "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
                "plan_sha256": sha256(args.plan), "diagnostic_only": "true",
                "adapter_implementation_sha256": plan["adapter_implementation_sha256"],
            })
            records = []
            with torch.inference_mode():
                for row, features, _ in examples:
                    memory = model.encoder(**features, return_dict=True).last_hidden_state
                    ids = GenerationMixin.generate(
                        model,
                        encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=memory),
                        max_length=plan["max_length"], num_beams=5, early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    )
                    hypothesis = tokenizer.decode(ids[0], skip_special_tokens=True)
                    reference = normalize(row["reference"])
                    wer = distance(reference, normalize(hypothesis)) / len(reference)
                    baseline = baseline_by_id[row["clip_id"]]
                    records.append({
                        "example_id": row["clip_id"], "reference": row["reference"],
                        "hypothesis": hypothesis, "wer": wer,
                        "baseline_wer": baseline["single_reference_wer"],
                        "baseline_hypothesis": baseline["hypothesis"],
                    })
            baseline_mean = sum(row["baseline_wer"] for row in records) / len(records)
            candidate_mean = sum(row["wer"] for row in records) / len(records)
            improved = sum(row["wer"] < row["baseline_wer"] for row in records)
            summary = {
                "arm": arm["name"], "learning_rate": arm["learning_rate"], "epoch": epoch,
                "mean_loss": sum(losses) / len(losses),
                "baseline_mean_wer": baseline_mean, "candidate_mean_wer": candidate_mean,
                "mean_wer_change": candidate_mean - baseline_mean,
                "improved": improved, "exact": sum(row["wer"] == 0 for row in records),
                "distinct_hypotheses": len({row["hypothesis"] for row in records}),
                "checkpoint_sha256": sha256(target),
                "passes_training_fit": (
                    candidate_mean - baseline_mean <= plan["gate"]["maximum_mean_wer_change"]
                    and improved >= plan["gate"]["minimum_improved_examples"]
                ),
            }
            results.append(summary)
            (args.output_dir / f"{arm['name']}-epoch-{epoch:02d}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
            )
            with (args.output_dir / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(summary, sort_keys=True) + "\n")
            print(json.dumps(summary), flush=True)
        del model, optimizer, selected, trainable
    decision = {
        "schema": "umi-studio-c5-visual-trainability-decision/1", "plan_sha256": sha256(args.plan),
        "arms": results,
        "decision": "design_full_data_run" if any(row["passes_training_fit"] for row in results) else "training_fit_failed",
        "diagnostic_only": True, "confirmation_used": False, "miner_changed": False,
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
