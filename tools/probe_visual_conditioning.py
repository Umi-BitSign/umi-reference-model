"""Test baseline video sensitivity by swapping training-only encoder states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import distance, miner_revision, normalize, sha256

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
SIGNERS = (1, 2, 3, 5, 8)
ALL_TRAINING_SIGNERS = {1, 2, 3, 4, 5, 8, 9, 11}


def choose_rows(rows: list[dict]) -> list[dict]:
    if not rows or any(
        row.get("schema") != "umi-studio-how2sign-encoder-cache-example/1"
        or row.get("split") != "training"
        or row.get("signer") not in ALL_TRAINING_SIGNERS
        for row in rows
    ):
        raise ValueError("visual conditioning probe needs training-only How2Sign caches")
    selected = []
    for signer in SIGNERS:
        candidates = [
            row for row in rows
            if row["signer"] == signer
            and 60 <= row["encoder_shape"][0] <= 150
            and 5 <= len(normalize(row["reference"])) <= 18
        ]
        if not candidates:
            raise ValueError(f"visual conditioning probe lacks signer {signer}")
        selected.append(min(candidates, key=lambda row: row["sentence_name"]))
    if len({row["sentence_name"] for row in selected}) != len(SIGNERS):
        raise ValueError("visual conditioning probe repeats a training clip")
    return selected


def summarize(records: list[dict]) -> dict:
    if len(records) != len(SIGNERS):
        raise ValueError("visual conditioning probe has an incomplete pair set")
    return {
        "pairs": len(records),
        "changed_hypotheses": sum(
            row["own_hypothesis"] != row["swapped_hypothesis"] for row in records
        ),
        "mean_hypothesis_word_edits": sum(
            row["hypothesis_word_edits"] for row in records
        ) / len(records),
        "mean_own_wer": sum(row["own_wer"] for row in records) / len(records),
        "mean_swapped_wer": sum(row["swapped_wer"] for row in records) / len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.index.resolve().is_relative_to(ROOT / "runs")
        or not args.cache_dir.resolve().is_relative_to(ROOT / "data")
        or not args.output.resolve().is_relative_to(ROOT / "runs")
    ):
        parser.error("visual conditioning probe needs frozen training inputs and a new ledger")
    rows = [json.loads(line) for line in args.index.read_text().splitlines()]
    selected = choose_rows(rows)
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(BASELINE / "run/hf"),
        "TORCH_HOME": str(BASELINE / "run/torch"),
        "MPLCONFIGDIR": str(BASELINE / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(BASELINE / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(BASELINE / "vendor"))
    sys.path.insert(0, str(BASELINE / "vendor/demo"))
    import numpy as np
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    torch.set_num_threads(2)
    revision = miner_revision(args.miner_health_url, None)
    checkpoint = BASELINE / "models/checkpoint-11625"
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BASELINE / "models/byt5_base"), local_files_only=True
    )
    model.to("cpu")
    model.eval()
    states = {}
    for row in selected:
        cache = args.cache_dir / f"{row['sentence_name']}.npz"
        if sha256(cache) != row["encoder_cache_sha256"]:
            raise ValueError(f"visual conditioning cache differs: {cache}")
        with np.load(cache, allow_pickle=False) as payload:
            state = payload["encoder_hidden_state"]
        if list(state.shape) != row["encoder_shape"] or state.dtype != np.float32:
            raise ValueError(f"visual conditioning cache shape differs: {cache}")
        states[row["sentence_name"]] = torch.from_numpy(state).unsqueeze(0)
    def decode(state):
        nonlocal revision
        revision = miner_revision(args.miner_health_url, revision)
        with torch.no_grad():
            ids = GenerationMixin.generate(
                model,
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=state
                ),
                max_length=2048,
                num_beams=5,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        revision = miner_revision(args.miner_health_url, revision)
        return tokenizer.decode(ids[0], skip_special_tokens=True)
    records = []
    for position, row in enumerate(selected):
        other = selected[(position + 1) % len(selected)]
        own = decode(states[row["sentence_name"]])
        swapped = decode(states[other["sentence_name"]])
        reference_words = normalize(row["reference"])
        own_words = normalize(own)
        swapped_words = normalize(swapped)
        records.append({
            "signer": row["signer"],
            "example_id": row["sentence_name"],
            "source_cache_sha256": row["encoder_cache_sha256"],
            "swapped_from_example_id": other["sentence_name"],
            "swapped_from_cache_sha256": other["encoder_cache_sha256"],
            "reference": row["reference"],
            "own_hypothesis": own,
            "swapped_hypothesis": swapped,
            "own_wer": distance(reference_words, own_words) / len(reference_words),
            "swapped_wer": distance(reference_words, swapped_words) / len(reference_words),
            "hypothesis_word_edits": distance(own_words, swapped_words),
        })
        print(f"signer={row['signer']} changed={own != swapped}", flush=True)
    result = {
        "schema": "umi-studio-visual-conditioning-probe/1",
        "split": "training",
        "diagnostic_only": True,
        "index_sha256": sha256(args.index),
        "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "probe_source_sha256": sha256(Path(__file__)),
        "miner_model_revision": revision,
        "selection_rule": "lexicographically first 60-150 frame, 5-18 word clip per signer",
        "swap_rule": "cyclic next signer in the fixed 1,2,3,5,8 order",
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished visual conditioning probe needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, sort_keys=True, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
