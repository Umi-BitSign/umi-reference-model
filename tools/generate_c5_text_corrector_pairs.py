"""Decode a frozen, training-only mixed schedule for a text-correction pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from freeze_c5_anchored_projection_plan import (
    BASELINE,
    RECIPE,
    ROOT,
    frozen_rows,
    load_examples,
    load_expansion_examples,
    load_openasl_examples,
    select_schedule,
)
from train_c5_stem_continuation import stride2_streams


def load_frozen_schedule(plan_path: Path) -> tuple[dict, list[dict]]:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "umi-studio-c5-anchored-projection-plan/1":
        raise ValueError("text-corrector pairs require the anchored training-only plan")
    replay, stem, source_hashes = load_examples()
    expansion, expansion_hashes = load_expansion_examples()
    openasl, openasl_hashes = load_openasl_examples()
    if {**source_hashes, **expansion_hashes, **openasl_hashes} != plan["source_hashes"]:
        raise ValueError("text-corrector source indexes changed")
    schedule = select_schedule(replay, stem, expansion, openasl)
    schedule_path = plan_path.with_name("schedule.jsonl")
    if (
        sha256(schedule_path) != plan["schedule_sha256"]
        or [json.loads(line) for line in schedule_path.read_text().splitlines()]
        != frozen_rows(schedule)
    ):
        raise ValueError("text-corrector source schedule changed")
    return plan, schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1_024)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if (
        not args.miner_maintenance_mode
        or not args.plan.is_file()
        or args.limit < 1
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("pair generation requires maintenance mode and a runs/ output")
    plan, schedule = load_frozen_schedule(args.plan)
    if args.limit > len(schedule):
        parser.error("limit exceeds the frozen schedule")
    plan_hash = sha256(args.plan)
    starting_adapter = ROOT / plan["starting_adapter"]
    if sha256(starting_adapter) != plan["starting_adapter_sha256"]:
        raise ValueError("starting adapter changed")

    completed: dict[tuple[str, str], dict] = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            key = (row["source"], row["example_id"])
            if (
                row.get("schema") != "umi-studio-c5-text-corrector-pair/1"
                or row.get("source_plan_sha256") != plan_hash
                or row.get("starting_adapter_sha256") != plan["starting_adapter_sha256"]
                or key in completed
            ):
                raise ValueError("existing text-corrector pairs use other inputs")
            completed[key] = row

    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(BASELINE / "run/hf"),
        "TORCH_HOME": str(BASELINE / "run/torch"),
        "MPLCONFIGDIR": str(BASELINE / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(BASELINE / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(BASELINE / "vendor"))
    sys.path.insert(0, str(BASELINE / "vendor/demo"))
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors import safe_open
    from safetensors.torch import load_file
    from shubert_visual_lora import install_visual_lora, load_visual_adapter
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    torch.set_num_threads(4)
    checkpoint = BASELINE / "models/checkpoint-11625"
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    install_visual_lora(
        model, RECIPE["rank"], RECIPE["alpha"], RECIPE["visual_last_layers"]
    )
    with safe_open(str(starting_adapter), framework="pt", device="cpu") as source:
        metadata = source.metadata()
    if (
        metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("baseline_checkpoint_sha256") != plan["baseline_checkpoint_sha256"]
    ):
        raise ValueError("starting adapter metadata changed")
    load_visual_adapter(model, load_file(str(starting_adapter), device="cpu"))
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BASELINE / "models/byt5_base"), local_files_only=True
    )
    model.to("cpu").eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    for position, row in enumerate(schedule[: args.limit]):
        key = (row["source"], row["id"])
        if key in completed:
            continue
        started = time.perf_counter()
        streams = stride2_streams(row)
        with torch.inference_mode():
            hidden = model.encoder(
                **{
                    name: torch.from_numpy(value).unsqueeze(0)
                    for name, value in streams.items()
                },
                return_dict=True,
            ).last_hidden_state
            ids = GenerationMixin.generate(
                model,
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden
                ),
                max_length=256,
                num_beams=5,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        hypothesis = tokenizer.decode(ids[0], skip_special_tokens=True)
        reference_words = normalize(row["reference"])
        edits = distance(reference_words, normalize(hypothesis))
        record = {
            "schema": "umi-studio-c5-text-corrector-pair/1",
            "position": position,
            "source": row["source"],
            "example_id": row["id"],
            "signer": row["signer"],
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "single_reference_wer": edits / len(reference_words),
            "single_reference_word_edits": edits,
            "reference_word_count": len(reference_words),
            "temporal_stride": 2,
            "source_plan_sha256": plan_hash,
            "starting_adapter_sha256": plan["starting_adapter_sha256"],
            "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
            "active_seconds": time.perf_counter() - started,
            "miner_health_policy": "user_authorized_maintenance",
        }
        with args.output.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        generated += 1
        print(
            f"{position + 1}/{args.limit} {row['source']} {row['id']} "
            f"WER={record['single_reference_wer']:.3f}",
            flush=True,
        )
    print(json.dumps({"completed": len(completed) + generated, "generated": generated}))


if __name__ == "__main__":
    main()
