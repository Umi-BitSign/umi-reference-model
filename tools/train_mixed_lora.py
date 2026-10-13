"""Train a SHuBERT/ByT5 LoRA adapter with FLEURS and public replay data.

Both inputs are public source-training caches from this independent workspace.
No validation, confirmation, test, miner, or cohort material enters training.
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


def load_examples(index: Path, cache_dir: Path, source: str, baseline: Path) -> list[dict]:
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    if not rows:
        raise ValueError(f"empty {source} training index")
    identity_hash = sha256(baseline / "MODEL_IDENTITY.json")
    runtime_hash = sha256(baseline / "runtime.py")
    examples = []
    identifiers = set()
    for row in rows:
        if source == "fleurs":
            if row["split"] != "training" or row["signer"] != 3:
                raise ValueError("FLEURS index contains a non-training or non-signer-3 row")
            identifier = row["example_id"]
        elif source == "how2sign":
            if (
                row["schema"] != "umi-studio-how2sign-encoder-cache-example/1"
                or row["split"] != "training"
                or row["signer"] not in (1, 2, 3, 4, 5, 8, 9, 11)
            ):
                raise ValueError("How2Sign index contains a non-training row")
            identifier = row["sentence_name"]
        elif source == "flores2m":
            if (
                row["schema"] != "umi-studio-flores2m-encoder-cache-example/1"
                or row["split"] != "training"
            ):
                raise ValueError("2M-Flores index contains a non-training row")
            identifier = row["example_id"]
        else:
            raise ValueError(f"unexpected training source: {source}")
        if (
            identifier in identifiers
            or row["baseline_identity_sha256"] != identity_hash
            or row["baseline_runtime_sha256"] != runtime_hash
            or row["clip_fps"] != 25
        ):
            raise ValueError(f"{source} training index is duplicated or differs from baseline")
        identifiers.add(identifier)
        cache = cache_dir / f"{identifier}.npz"
        if not cache.is_file() or sha256(cache) != row["encoder_cache_sha256"]:
            raise ValueError(f"training cache missing or altered: {cache}")
        examples.append(
            {
                "id": identifier,
                "source": source,
                "reference": row["reference"],
                "cache": cache,
                "cache_sha256": row["encoder_cache_sha256"],
                "shape": row["encoder_shape"],
            }
        )
    return examples


def check_source_plan(
    plan: dict, fleurs_index: Path, how2sign_index: Path, flores2m_index: Path | None
) -> None:
    schema = plan.get("schema")
    if schema not in {"umi-studio-mixed-pilot-plan/1", "umi-studio-aligned-mixed-plan/1"}:
        raise ValueError("unexpected mixed training plan schema")
    if sha256(fleurs_index) != plan["fleurs_training_index_sha256"]:
        raise ValueError("FLEURS training index differs from frozen plan")
    if schema == "umi-studio-mixed-pilot-plan/1" and flores2m_index is None:
        raise ValueError("original mixed pilot requires 2M-Flores training data")
    if schema == "umi-studio-aligned-mixed-plan/1":
        if flores2m_index is not None or plan["training_examples"]["flores2m"] != 0:
            raise ValueError("aligned mixed plan excludes 2M-Flores")
        if sha256(how2sign_index) != plan["how2sign_training_index_sha256"]:
            raise ValueError("aligned How2Sign cache index differs from frozen plan")
    checks = [(how2sign_index, plan["how2sign_selection_sha256"])]
    if flores2m_index is not None:
        checks.append((flores2m_index, plan["flores2m_selection_sha256"]))
    for index, expected_hash in checks:
        if any(
            json.loads(line)["selection_sha256"] != expected_hash
            for line in index.read_text().splitlines()
        ):
            raise ValueError("training cache selection differs from the frozen plan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleurs-index", type=Path, required=True)
    parser.add_argument("--fleurs-cache-dir", type=Path, required=True)
    parser.add_argument("--how2sign-index", type=Path, required=True)
    parser.add_argument("--how2sign-cache-dir", type=Path, required=True)
    parser.add_argument("--flores2m-index", type=Path)
    parser.add_argument("--flores2m-cache-dir", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--recipe-id", required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--fleurs-repeat", type=int, default=2)
    parser.add_argument("--how2sign-repeat", type=int, default=1)
    parser.add_argument("--flores2m-repeat", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument(
        "--targets", choices=("cross_qv", "decoder_attention", "decoder_all"), required=True
    )
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if (
        min(
            args.fleurs_repeat,
            args.how2sign_repeat,
            args.flores2m_repeat,
            args.epochs,
            args.rank,
        )
        < 1
        or args.alpha <= 0
        or args.learning_rate <= 0
        or args.weight_decay < 0
        or not 0 <= args.label_smoothing <= 0.2
    ):
        parser.error("invalid training counts or LoRA hyperparameters")
    if (args.flores2m_index is None) != (args.flores2m_cache_dir is None):
        parser.error("2M-Flores index and cache directory must be supplied together")
    plan = json.loads(args.plan.read_text())
    recipes = {recipe["id"]: recipe for recipe in plan.get("recipes", [])}
    if (
        plan.get("schema")
        not in {"umi-studio-mixed-pilot-plan/1", "umi-studio-aligned-mixed-plan/1"}
        or args.recipe_id not in recipes
    ):
        raise ValueError("training plan or recipe ID differs from the frozen run")
    recipe = recipes[args.recipe_id]
    for name in (
        "targets",
        "epochs",
        "learning_rate",
        "weight_decay",
        "label_smoothing",
    ):
        if getattr(args, name) != recipe[name]:
            raise ValueError(f"{name} differs from the frozen recipe")
    for name in ("seed", "rank", "alpha", "fleurs_repeat", "how2sign_repeat", "flores2m_repeat"):
        if getattr(args, name) != plan[name]:
            raise ValueError(f"{name} differs from the frozen plan")
    check_source_plan(plan, args.fleurs_index, args.how2sign_index, args.flores2m_index)
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if baseline != workspace / "artifacts/baseline-6fe8df59":
        parser.error("trainer requires the independent frozen baseline")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    if not args.output_dir.resolve().is_relative_to(workspace / "checkpoints"):
        parser.error("output must stay under this workspace's checkpoints directory")
    fleurs = load_examples(args.fleurs_index, args.fleurs_cache_dir, "fleurs", baseline)
    how2sign = load_examples(args.how2sign_index, args.how2sign_cache_dir, "how2sign", baseline)
    flores2m = (
        load_examples(args.flores2m_index, args.flores2m_cache_dir, "flores2m", baseline)
        if args.flores2m_index is not None
        else []
    )
    if (
        len(fleurs) != plan["training_examples"]["fleurs"]
        or len(how2sign) != plan["training_examples"]["how2sign"]
        or len(flores2m) != plan["training_examples"]["flores2m"]
    ):
        raise ValueError("training cache counts differ from the frozen plan")
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
    from shubert_lora import adapter_state, install_lora
    from transformers import ByT5Tokenizer

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    checkpoint = baseline / "models/checkpoint-11625"
    checkpoint_hash = sha256(checkpoint / "model.safetensors")
    revision = miner_revision(args.miner_health_url, None)
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    trainable = install_lora(model, args.rank, args.alpha, args.targets)
    model.to("cpu")
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(baseline / "models/byt5_base"), local_files_only=True
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    revision = miner_revision(args.miner_health_url, revision)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "schema": "umi-studio-mixed-lora-training/1",
        "plan_sha256": sha256(args.plan),
        "recipe_id": args.recipe_id,
        "fleurs_index_sha256": sha256(args.fleurs_index),
        "how2sign_index_sha256": sha256(args.how2sign_index),
        "flores2m_index_sha256": (
            sha256(args.flores2m_index) if args.flores2m_index is not None else None
        ),
        "baseline_checkpoint_sha256": checkpoint_hash,
        "trainer_source_sha256": sha256(Path(__file__)),
        "lora_source_sha256": sha256(Path(__file__).with_name("shubert_lora.py")),
        "fleurs_example_ids": [row["id"] for row in fleurs],
        "how2sign_sentence_names": [row["id"] for row in how2sign],
        "flores2m_example_ids": [row["id"] for row in flores2m],
        "fleurs_repeat": args.fleurs_repeat,
        "how2sign_repeat": args.how2sign_repeat,
        "flores2m_repeat": args.flores2m_repeat if flores2m else 0,
        "epochs": args.epochs,
        "rank": args.rank,
        "alpha": args.alpha,
        "targets": args.targets,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "miner_model_revision": revision,
    }
    with (args.output_dir / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(run, stream, sort_keys=True, indent=2)
        stream.write("\n")
    ledger = args.output_dir / "steps.jsonl"
    step = 0
    for epoch in range(1, args.epochs + 1):
        ordered = (
            fleurs * args.fleurs_repeat
            + how2sign * args.how2sign_repeat
            + flores2m * args.flores2m_repeat
        )
        random.Random(args.seed + epoch).shuffle(ordered)
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
            loss = output.loss
            if args.label_smoothing > 0:
                loss = torch.nn.functional.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    labels.reshape(-1),
                    label_smoothing=args.label_smoothing,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite LoRA gradient")
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
            with ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step} source={row['source']} loss={record['loss']:.4f}",
                    flush=True,
                )
        adapter = args.output_dir / f"adapter-epoch{epoch:04d}.safetensors"
        temporary = adapter.with_suffix(".partial.safetensors")
        metadata = {
            "schema": "umi-studio-shubert-lora/1",
            "plan_sha256": run["plan_sha256"],
            "recipe_id": args.recipe_id,
            "fleurs_index_sha256": run["fleurs_index_sha256"],
            "how2sign_index_sha256": run["how2sign_index_sha256"],
            "flores2m_index_sha256": run["flores2m_index_sha256"] or "none",
            "baseline_checkpoint_sha256": checkpoint_hash,
            "trainer_source_sha256": run["trainer_source_sha256"],
            "lora_source_sha256": run["lora_source_sha256"],
            "rank": str(args.rank),
            "alpha": str(args.alpha),
            "targets": args.targets,
            "epoch": str(epoch),
            "weight_decay": str(args.weight_decay),
            "label_smoothing": str(args.label_smoothing),
        }
        save_file(adapter_state(model), str(temporary), metadata=metadata)
        temporary.rename(adapter)
        print(f"saved {adapter.name} sha256={sha256(adapter)}", flush=True)


if __name__ == "__main__":
    main()
