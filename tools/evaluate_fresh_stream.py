"""Evaluate a fresh stream checkpoint on its fit gate or frozen open tuning."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256


def selection_rows(inventory: Path, selection: Path, root: Path) -> list[dict]:
    from fresh_stream_data import load_inventory

    rows = load_inventory(inventory, root)
    by_id = {(row["source"], row["example_id"]): row for row in rows}
    payload = json.loads(selection.read_text())
    if payload.get("inventory_sha256") != sha256(inventory):
        raise ValueError("fit selection inventory differs")
    return [by_id[(item["source"], item["example_id"])] for item in payload["examples"]]


def tuning_rows(root: Path) -> list[dict]:
    from fresh_stream_data import SCHEMA

    raw = [json.loads(line) for line in (root / "runs/fresh-lowres-v1/tuning-inventory-v2.jsonl").read_text().splitlines()]
    metadata = {(row["source"], row["example_id"]): row for row in raw}
    layouts = {
        "how2sign": (
            root / "runs/how2sign-validation-v1/stream-encoder-cache.jsonl",
            root / "data/how2sign/validation-stream-encoder-cache-25fps",
        ),
        "fleurs": (
            root / "runs/fleurs-dev-v1/tuning-stream-encoder-cache.jsonl",
            root / "data/fleurs-asl/encoder-cache-25fps/tuning-stream",
        ),
    }
    rows = []
    for source, (index, cache_dir) in layouts.items():
        for item in [json.loads(line) for line in index.read_text().splitlines()]:
            example_id = item.get("sentence_name", item.get("example_id"))
            key = (source, example_id)
            if key not in metadata:
                continue
            meta = metadata[key]
            cache = cache_dir / f"{example_id}.npz"
            rows.append({
                "schema": SCHEMA,
                "split": "tuning",
                "source": source,
                "source_group": meta["source_group"],
                "signer": meta["signer"],
                "example_id": example_id,
                "reference": item["reference"],
                "cache_path": cache,
                "cache_sha256": item["encoder_cache_sha256"],
                "sidecar_path": None,
                "sidecar_sha256": None,
                "frames": item["encoder_shape"][0],
                "stored_temporal_stride": 1,
            })
    if len(rows) != 23 or {row["source"] for row in rows} != {"how2sign", "fleurs"}:
        raise ValueError("fresh stream tuning rows differ")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--open-tuning", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--language-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint_file or args.checkpoint_dir / "trainable.safetensors"
    run_path = args.checkpoint_dir / "run.json"
    if (
        args.output.exists()
        or not args.output.resolve().is_relative_to((root / "runs").resolve())
        or not checkpoint.is_file()
        or not checkpoint.resolve().is_relative_to(args.checkpoint_dir.resolve())
        or args.open_tuning == (args.selection is not None)
        or not args.miner_maintenance_mode
        or not 1 <= args.num_beams <= 8
    ):
        parser.error("invalid fresh stream evaluation arguments")
    for key, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(root / "artifacts/fresh-stream-v1/huggingface"),
        "TORCH_HOME": str(root / "artifacts/fresh-stream-v1/torch"),
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    import torch
    from fresh_stream_data import load_stream_tensor
    from fresh_stream_model import (
        FreshStreamTranslator,
        configure_alignment_training,
        configure_decoder_training,
    )
    from safetensors import safe_open
    from safetensors.torch import load_file
    from train_fresh_lowres import language_artifact_receipt
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    torch.set_num_threads(2)
    run = json.loads(run_path.read_text())
    receipt = language_artifact_receipt(root, args.language_model)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if (
        run.get("schema") != "umi-studio-fresh-stream-training/1"
        or run.get("inventory_sha256") != sha256(args.inventory)
        or run.get("language_model") != receipt
        or run.get("model_source_sha256") != sha256(Path(__file__).with_name("fresh_stream_model.py"))
        or run.get("data_source_sha256") != sha256(Path(__file__).with_name("fresh_stream_data.py"))
        or metadata.get("schema") != "umi-studio-fresh-stream-checkpoint/1"
        or metadata.get("run_sha256") != sha256(run_path)
    ):
        raise ValueError("fresh stream checkpoint provenance differs")
    language_model = T5ForConditionalGeneration.from_pretrained(args.language_model, local_files_only=True)
    model = FreshStreamTranslator(language_model)
    expected = (
        configure_alignment_training(model)
        if run.get("training_stage") == "alignment"
        else configure_decoder_training(model)
    )
    state = load_file(str(checkpoint), device="cpu")
    if set(state) != set(expected):
        raise ValueError("fresh stream checkpoint topology differs")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, local_files_only=True)
    rows = tuning_rows(root) if args.open_tuning else selection_rows(args.inventory, args.selection, root)
    records = []
    for row in rows:
        values = load_stream_tensor(row)
        streams = torch.from_numpy(values).unsqueeze(0).to(args.device)
        mask = torch.ones((1, values.shape[0]), dtype=torch.bool, device=args.device)
        tokens = model.generate(
            streams, mask, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        hypothesis = tokenizer.decode(tokens[0], skip_special_tokens=True)
        reference_words = normalize(row["reference"])
        hypothesis_words = normalize(hypothesis)
        records.append({
            "source": row["source"], "source_group": row["source_group"],
            "signer": row["signer"], "example_id": row["example_id"],
            "reference": row["reference"], "hypothesis": hypothesis,
            "frames": values.shape[0],
            "wer": distance(reference_words, hypothesis_words) / len(reference_words),
        })
        print(f"decoded {row['source']} {row['example_id']}: {hypothesis!r}", flush=True)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["source"]].append(row)
    summaries = {
        source: {
            "examples": len(items),
            "mean_single_reference_wer": sum(item["wer"] for item in items) / len(items),
            "exact_hypotheses": sum(item["wer"] == 0 for item in items),
            "distinct_hypotheses": len({item["hypothesis"] for item in items}),
        }
        for source, items in sorted(grouped.items())
    }
    result = {
        "schema": "umi-studio-fresh-stream-evaluation/1",
        "split": "open_tuning" if args.open_tuning else "training_fit",
        "confirmation_evaluated": False,
        "run_sha256": sha256(run_path), "checkpoint_sha256": sha256(checkpoint),
        "inventory_sha256": sha256(args.inventory),
        "selection_sha256": sha256(args.selection) if args.selection else None,
        "miner_health_policy": "user_authorized_maintenance",
        "summaries": summaries, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, indent=2)
            output.write("\n"); output.flush(); os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
