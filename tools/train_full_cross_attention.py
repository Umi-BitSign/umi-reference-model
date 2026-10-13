"""Fine-tune only ByT5 decoder cross-attention on frozen public encoder states.

The full aligned-data plan fixes the corpus and recipe before tuning evaluation.
This offline worker checks the active miner before and after every training step.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from train_mixed_lora import load_examples


def check_inputs(plan: dict, fleurs_index: Path, how2sign_index: Path) -> dict:
    if (
        plan.get("schema") != "umi-studio-full-cross-attention-plan/1"
        or plan.get("training_examples") != {"fleurs": 313, "how2sign": 1024, "flores2m": 0}
        or plan.get("trainable_scope")
        != "decoder cross-attention q, k, v and o projection weights only"
        or plan.get("fleurs_repeat") != 1
        or plan.get("how2sign_repeat") != 1
        or plan.get("seed") != 17
        or sha256(fleurs_index) != plan["fleurs_training_index_sha256"]
        or sha256(how2sign_index) != plan["how2sign_training_index_sha256"]
    ):
        raise ValueError("training inputs differ from the full-data cross-attention plan")
    rows = [json.loads(line) for line in how2sign_index.read_text().splitlines()]
    if any(row["selection_sha256"] != plan["how2sign_selection_sha256"] for row in rows):
        raise ValueError("aligned training index uses another frozen selection")
    expected_recipe = {
        "id": "full_cross_attention_e1",
        "epochs": 1,
        "learning_rate": 0.00003,
        "weight_decay": 0.1,
        "label_smoothing": 0.2,
    }
    if plan["recipes"] != [expected_recipe]:
        raise ValueError("unexpected full-data cross-attention recipe")
    return expected_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleurs-index", type=Path, required=True)
    parser.add_argument("--fleurs-cache-dir", type=Path, required=True)
    parser.add_argument("--how2sign-index", type=Path, required=True)
    parser.add_argument("--how2sign-cache-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if baseline != workspace / "artifacts/baseline-6fe8df59":
        parser.error("trainer requires the independent frozen baseline")
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        workspace / "checkpoints"
    ):
        parser.error("new output directory must stay under workspace checkpoints")
    plan = json.loads(args.plan.read_text())
    recipe = check_inputs(plan, args.fleurs_index, args.how2sign_index)
    fleurs = load_examples(args.fleurs_index, args.fleurs_cache_dir, "fleurs", baseline)
    how2sign = load_examples(args.how2sign_index, args.how2sign_cache_dir, "how2sign", baseline)
    if len(fleurs) != 313 or len(how2sign) != 1024:
        raise ValueError("full training cache counts differ from frozen plan")
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(baseline / "run/hf"),
        "TORCH_HOME": str(baseline / "run/torch"),
        "MPLCONFIGDIR": str(baseline / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(baseline / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(baseline / "vendor"))
    sys.path.insert(0, str(baseline / "vendor/demo"))
    import numpy as np
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors.torch import save_file
    from shubert_cross_attention import cross_attention_parameters, train_cross_attention
    from transformers import ByT5Tokenizer

    torch.set_num_threads(2)
    torch.manual_seed(plan["seed"])
    checkpoint = baseline / "models/checkpoint-11625"
    checkpoint_hash = sha256(checkpoint / "model.safetensors")
    revision = miner_revision(args.miner_health_url, None)
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    trainable = train_cross_attention(model)
    model.to("cpu")
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(baseline / "models/byt5_base"), local_files_only=True
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
    )
    revision = miner_revision(args.miner_health_url, revision)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "schema": "umi-studio-full-cross-attention-training/1",
        "plan_sha256": sha256(args.plan),
        "recipe_id": recipe["id"],
        "fleurs_index_sha256": sha256(args.fleurs_index),
        "how2sign_index_sha256": sha256(args.how2sign_index),
        "baseline_checkpoint_sha256": checkpoint_hash,
        "trainer_source_sha256": sha256(Path(__file__)),
        "cross_attention_source_sha256": sha256(
            Path(__file__).with_name("shubert_cross_attention.py")
        ),
        "fleurs_example_ids": [row["id"] for row in fleurs],
        "how2sign_sentence_names": [row["id"] for row in how2sign],
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "miner_model_revision": revision,
    }
    with (args.output_dir / "run.json").open("x", encoding="utf-8") as output:
        json.dump(run, output, sort_keys=True, indent=2)
        output.write("\n")
    ledger = args.output_dir / "steps.jsonl"
    step = 0
    for epoch in range(1, recipe["epochs"] + 1):
        ordered = fleurs * plan["fleurs_repeat"] + how2sign * plan["how2sign_repeat"]
        random.Random(plan["seed"] + epoch).shuffle(ordered)
        model.train()
        for row in ordered:
            revision = miner_revision(args.miner_health_url, revision)
            with np.load(row["cache"], allow_pickle=False) as payload:
                state = payload["encoder_hidden_state"]
            if list(state.shape) != row["shape"] or state.dtype != np.float32:
                raise ValueError(f"training cache shape or dtype differs: {row['cache']}")
            hidden = torch.from_numpy(state).unsqueeze(0)
            labels = tokenizer(row["reference"], return_tensors="pt").input_ids
            if labels.shape[1] > 512:
                raise ValueError("training reference exceeds 512 ByT5 tokens")
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(
                encoder_outputs=(hidden,),
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            loss = torch.nn.functional.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                label_smoothing=recipe["label_smoothing"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite cross-attention gradient")
            optimizer.step()
            revision = miner_revision(args.miner_health_url, revision)
            step += 1
            record = {
                "epoch": epoch,
                "step": step,
                "source": row["source"],
                "example_id": row["id"],
                "loss": float(loss.detach().cpu()),
                "gradient_norm_before_clipping": float(gradient_norm.detach().cpu()),
                "seconds": time.perf_counter() - started,
                "miner_model_revision": revision,
            }
            with ledger.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
            if step % 50 == 0:
                print(
                    f"epoch={epoch} step={step} source={row['source']} loss={record['loss']:.4f}",
                    flush=True,
                )
        weights = {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in cross_attention_parameters(model).items()
        }
        checkpoint_file = args.output_dir / f"cross-attention-epoch{epoch:04d}.safetensors"
        temporary = checkpoint_file.with_suffix(".partial.safetensors")
        metadata = {
            "schema": "umi-studio-shubert-cross-attention/1",
            "plan_sha256": run["plan_sha256"],
            "recipe_id": recipe["id"],
            "fleurs_index_sha256": run["fleurs_index_sha256"],
            "how2sign_index_sha256": run["how2sign_index_sha256"],
            "baseline_checkpoint_sha256": checkpoint_hash,
            "trainer_source_sha256": run["trainer_source_sha256"],
            "cross_attention_source_sha256": run["cross_attention_source_sha256"],
            "epoch": str(epoch),
        }
        save_file(weights, str(temporary), metadata=metadata)
        temporary.rename(checkpoint_file)
        print(f"saved {checkpoint_file.name} sha256={sha256(checkpoint_file)}", flush=True)


if __name__ == "__main__":
    main()
