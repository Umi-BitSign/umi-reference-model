"""Fine-tune SHuBERT/ByT5 decoder cross-attention on frozen FLEURS states.

Only source-split training states are accepted. This writes an adapter and an
auditable step ledger, not a serving bundle. Run at low priority so the active
miner retains capacity.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument(
        "--targets", choices=("cross_qv", "decoder_attention", "decoder_all"), default="cross_qv"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pilot-selection", type=Path)
    args = parser.parse_args()
    if (
        args.max_examples < 1
        or args.epochs < 1
        or args.rank < 1
        or args.alpha <= 0
        or args.learning_rate <= 0
        or args.weight_decay < 0
        or not 0 <= args.label_smoothing <= 0.2
    ):
        parser.error("training counts and LoRA hyperparameters must be positive")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    if not args.cache_dir.is_dir() or not args.baseline.is_dir():
        parser.error("cache and independent baseline directories must exist")
    workspace = Path(__file__).resolve().parents[1]
    if args.baseline.resolve() != workspace / "artifacts/baseline-6fe8df59":
        parser.error("trainer requires the independent frozen baseline in this workspace")
    if not args.output_dir.resolve().is_relative_to(workspace / "checkpoints"):
        parser.error("trainer output must stay under this workspace's checkpoints directory")
    index_hash = sha256(args.index)
    rows = [json.loads(line) for line in args.index.read_text().splitlines()]
    if not rows or any(row["split"] != "training" or row["signer"] != 3 for row in rows):
        raise ValueError("trainer accepts only the signer-3 public training cache")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("training cache index contains duplicate examples")
    pilot_selection_hash = None
    if args.pilot_selection is not None:
        pilot_selection_hash = sha256(args.pilot_selection)
        pilot = json.loads(args.pilot_selection.read_bytes())
        identifiers = pilot.get("all_example_ids")
        if (
            pilot.get("schema") != "umi-studio-fleurs-training-pilot/1"
            or not isinstance(identifiers, list)
            or len(identifiers) != args.max_examples
            or any(not isinstance(identifier, str) for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or any(row["selection_sha256"] != pilot.get("selection_sha256") for row in rows)
        ):
            raise ValueError("pilot selection differs from the training cache")
        by_id = {row["example_id"]: row for row in rows}
        if set(identifiers) != by_id.keys():
            raise ValueError("training cache does not exactly match pilot selection")
        rows = [by_id[example_id] for example_id in identifiers]
    else:
        rows = rows[: args.max_examples]
    baseline = args.baseline.resolve()
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(baseline / "run/hf"),
        "TORCH_HOME": str(baseline / "run/torch"),
        "MPLCONFIGDIR": str(baseline / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(baseline / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
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
    for row in rows:
        if (
            row["baseline_identity_sha256"] != sha256(baseline / "MODEL_IDENTITY.json")
            or row["baseline_runtime_sha256"] != sha256(baseline / "runtime.py")
            or row["clip_fps"] != 25
        ):
            raise ValueError("training cache does not match baseline or 25-fps transform")
        cache = args.cache_dir / f"{row['example_id']}.npz"
        if sha256(cache) != row["encoder_cache_sha256"]:
            raise ValueError(f"training cache is missing or altered: {cache}")
    revision = miner_revision(args.miner_health_url, None)
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    trainable = install_lora(model, args.rank, args.alpha, args.targets)
    model.to(args.device)
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(baseline / "models/byt5_base"), local_files_only=True
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer_source_hash = sha256(Path(__file__))
    lora_source_hash = sha256(Path(__file__).with_name("shubert_lora.py"))
    revision = miner_revision(args.miner_health_url, revision)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = {
        "schema": "umi-studio-fleurs-lora-training/1",
        "index_sha256": index_hash,
        "baseline_checkpoint_sha256": checkpoint_hash,
        "trainer_source_sha256": trainer_source_hash,
        "lora_source_sha256": lora_source_hash,
        "training_example_ids": [row["example_id"] for row in rows],
        "pilot_selection_sha256": pilot_selection_hash,
        "device": args.device,
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
    with (args.output_dir / "run.json").open("x", encoding="utf-8") as output:
        json.dump(run, output, sort_keys=True, indent=2)
        output.write("\n")
    ledger = args.output_dir / "steps.jsonl"
    step = 0
    for epoch in range(1, args.epochs + 1):
        ordered = list(rows)
        random.Random(args.seed + epoch).shuffle(ordered)
        model.train()
        for row in ordered:
            revision = miner_revision(args.miner_health_url, revision)
            cache = args.cache_dir / f"{row['example_id']}.npz"
            with np.load(cache, allow_pickle=False) as payload:
                state = payload["encoder_hidden_state"]
            if list(state.shape) != row["encoder_shape"] or state.dtype != np.float32:
                raise ValueError(f"training cache shape or dtype differs from index: {cache}")
            hidden = torch.from_numpy(state).unsqueeze(0).to(args.device)
            labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to(args.device)
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
            elapsed = time.perf_counter() - started
            revision = miner_revision(args.miner_health_url, revision)
            step += 1
            record = {
                "epoch": epoch,
                "step": step,
                "example_id": row["example_id"],
                "loss": float(loss.detach().cpu()),
                "gradient_norm_before_clipping": float(gradient_norm.detach().cpu()),
                "seconds": elapsed,
                "miner_model_revision": revision,
            }
            with ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(
                f"epoch={epoch} step={step} loss={record['loss']:.4f} seconds={elapsed:.1f}",
                flush=True,
            )
        adapter = args.output_dir / f"adapter-epoch{epoch:04d}.safetensors"
        temporary = adapter.with_suffix(".partial.safetensors")
        metadata = {
            "schema": "umi-studio-shubert-lora/1",
            "index_sha256": index_hash,
            "pilot_selection_sha256": pilot_selection_hash or "none",
            "baseline_checkpoint_sha256": checkpoint_hash,
            "trainer_source_sha256": trainer_source_hash,
            "lora_source_sha256": lora_source_hash,
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
