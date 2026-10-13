"""Fine-tune the complete ByT5 decoder on frozen public encoder states."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from freeze_full_decoder_plan import RECIPE
from train_mixed_lora import load_examples


def check_inputs(plan: dict, fleurs_index: Path, how2sign_index: Path) -> dict:
    if (
        plan.get("schema") != "umi-studio-full-decoder-plan/1"
        or plan.get("training_examples") != {"fleurs": 313, "how2sign": 1024, "flores2m": 0}
        or plan.get("trainable_scope") != "complete ByT5 decoder, decoder embedding and output head"
        or plan.get("fleurs_repeat") != 1
        or plan.get("how2sign_repeat") != 1
        or plan.get("seed") != 17
        or plan.get("recipes") != [RECIPE]
        or sha256(fleurs_index) != plan["fleurs_training_index_sha256"]
        or sha256(how2sign_index) != plan["how2sign_training_index_sha256"]
    ):
        raise ValueError("training inputs differ from the frozen full-decoder plan")
    rows = [json.loads(line) for line in how2sign_index.read_text().splitlines()]
    if any(row["selection_sha256"] != plan["how2sign_selection_sha256"] for row in rows):
        raise ValueError("How2Sign index uses another frozen selection")
    return RECIPE


def scheduled_examples(fleurs: list[dict], how2sign: list[dict], seed: int) -> list[dict]:
    if len(fleurs) != 313 or len(how2sign) != 1024:
        raise ValueError("full decoder training requires all frozen source examples")
    ordered = fleurs + how2sign
    random.Random(seed + 1).shuffle(ordered)
    return ordered


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
    ordered = scheduled_examples(fleurs, how2sign, plan["seed"])
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
    from shubert_full_decoder import full_decoder_state, train_full_decoder
    from transformers import ByT5Tokenizer

    torch.set_num_threads(2)
    torch.manual_seed(plan["seed"])
    checkpoint = baseline / "models/checkpoint-11625"
    checkpoint_hash = sha256(checkpoint / "model.safetensors")
    revision = miner_revision(args.miner_health_url, None)
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    selected = train_full_decoder(model)
    model.to("cpu")
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(baseline / "models/byt5_base"), local_files_only=True
    )
    optimizer = torch.optim.AdamW(
        selected.values(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]
    )
    revision = miner_revision(args.miner_health_url, revision)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "schema": "umi-studio-full-decoder-training/1",
        "plan_sha256": sha256(args.plan),
        "recipe_id": recipe["id"],
        "fleurs_index_sha256": sha256(args.fleurs_index),
        "how2sign_index_sha256": sha256(args.how2sign_index),
        "baseline_checkpoint_sha256": checkpoint_hash,
        "trainer_source_sha256": sha256(Path(__file__)),
        "decoder_scope_source_sha256": sha256(Path(__file__).with_name("shubert_full_decoder.py")),
        "fleurs_example_ids": [row["id"] for row in fleurs],
        "how2sign_sentence_names": [row["id"] for row in how2sign],
        "trainable_parameter_count": sum(parameter.numel() for parameter in selected.values()),
        "miner_model_revision": revision,
    }
    with (args.output_dir / "run.json").open("x", encoding="utf-8") as output:
        json.dump(run, output, sort_keys=True, indent=2)
        output.write("\n")
    ledger = args.output_dir / "steps.jsonl"
    accumulation = recipe["gradient_accumulation"]
    milestones = set(recipe["checkpoint_after_examples"])
    example_step = 0
    optimizer_step = 0
    model.train()
    for offset in range(0, len(ordered), accumulation):
        batch = ordered[offset : offset + accumulation]
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for row in batch:
            revision = miner_revision(args.miner_health_url, revision)
            with np.load(row["cache"], allow_pickle=False) as payload:
                state = payload["encoder_hidden_state"]
            if list(state.shape) != row["shape"] or state.dtype != np.float32:
                raise ValueError(f"training cache shape or dtype differs: {row['cache']}")
            hidden = torch.from_numpy(state).unsqueeze(0)
            labels = tokenizer(row["reference"], return_tensors="pt").input_ids
            if labels.shape[1] > 512:
                raise ValueError("training reference exceeds 512 ByT5 tokens")
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
                raise RuntimeError("non-finite full-decoder training loss")
            (loss / len(batch)).backward()
            losses.append(float(loss.detach().cpu()))
            revision = miner_revision(args.miner_health_url, revision)
        gradient_norm = torch.nn.utils.clip_grad_norm_(selected.values(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite full-decoder gradient")
        revision = miner_revision(args.miner_health_url, revision)
        optimizer.step()
        revision = miner_revision(args.miner_health_url, revision)
        example_step += len(batch)
        optimizer_step += 1
        record = {
            "epoch": 1,
            "example_step": example_step,
            "optimizer_step": optimizer_step,
            "sources": [row["source"] for row in batch],
            "example_ids": [row["id"] for row in batch],
            "losses": losses,
            "gradient_norm_before_clipping": float(gradient_norm.detach().cpu()),
            "seconds": time.perf_counter() - started,
            "miner_model_revision": revision,
        }
        with ledger.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if example_step in milestones:
            weights = full_decoder_state(model)
            path = args.output_dir / f"full-decoder-after-{example_step:06d}.safetensors"
            temporary = path.with_suffix(".partial.safetensors")
            if temporary.exists():
                raise RuntimeError("unfinished full-decoder checkpoint needs inspection")
            metadata = {
                "schema": "umi-studio-shubert-full-decoder/1",
                "plan_sha256": run["plan_sha256"],
                "recipe_id": recipe["id"],
                "fleurs_index_sha256": run["fleurs_index_sha256"],
                "how2sign_index_sha256": run["how2sign_index_sha256"],
                "baseline_checkpoint_sha256": checkpoint_hash,
                "trainer_source_sha256": run["trainer_source_sha256"],
                "decoder_scope_source_sha256": run["decoder_scope_source_sha256"],
                "example_step": str(example_step),
                "optimizer_step": str(optimizer_step),
            }
            save_file(weights, str(temporary), metadata=metadata)
            temporary.rename(path)
            revision = miner_revision(args.miner_health_url, revision)
            print(f"saved {path.name} sha256={sha256(path)}", flush=True)
        elif optimizer_step % 25 == 0:
            print(
                f"examples={example_step} updates={optimizer_step} "
                f"mean_loss={sum(losses) / len(losses):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
