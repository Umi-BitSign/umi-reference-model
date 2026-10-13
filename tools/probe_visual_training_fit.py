"""Compare baseline and projection-ablated LoRA on examples seen during training."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import distance, miner_revision, normalize, sha256
from freeze_visual_lora_plan import SOURCE_LAYOUT
from train_visual_lora import training_streams

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
STAGE_PLAN = ROOT / "runs/visual-lora-stage1-v1/plan.json"
STAGE_STEPS = ROOT / "checkpoints/visual-lora-stage1-v1/steps.jsonl"
ABLATION_PLAN = ROOT / "runs/visual-lora-projection-ablation-v1/plan.json"
ABLATION_DECISION = ROOT / "runs/visual-lora-projection-ablation-v1/tuning-decision.json"
ADAPTER = ROOT / "checkpoints/visual-lora-projection-ablation-v1/adapter.safetensors"
QUOTAS = {
    "fleurs": {3: 4},
    "how2sign_original": {signer: 2 for signer in (1, 2, 3, 5, 8)},
    "how2sign_new": {signer: 2 for signer in (1, 2, 3, 5, 8)},
    "how2sign_raw": {signer: 1 for signer in (1, 2, 3, 4, 5, 8, 9, 11)},
}


def seen_training_ids(path: Path) -> dict[str, set[str]]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != 1024 or [row.get("example_step") for row in records] != list(
        range(4, 4097, 4)
    ):
        raise ValueError("training-fit probe needs the complete stage-one ledger")
    seen = defaultdict(set)
    for row in records:
        sources, identifiers = row["sources"], row["example_ids"]
        if len(sources) != 4 or len(identifiers) != 4:
            raise ValueError("stage-one ledger has an incomplete optimizer batch")
        for source, identifier in zip(sources, identifiers, strict=True):
            if source not in QUOTAS:
                raise ValueError("stage-one ledger has an unknown training source")
            seen[source].add(identifier)
    return seen


def choose_rows(source_rows: dict[str, list[dict]], seen: dict[str, set[str]]) -> list[dict]:
    selected = []
    for source, quotas in QUOTAS.items():
        rows = source_rows[source]
        if len(rows) != SOURCE_LAYOUT[source]["expected"] or any(
            row.get("split") != "training"
            or row.get("schema")
            != (
                "umi-studio-fleurs-dev-encoder-cache-example/1"
                if source == "fleurs"
                else "umi-studio-how2sign-encoder-cache-example/1"
            )
            for row in rows
        ):
            raise ValueError(f"training-fit {source} source index differs")
        for signer, count in quotas.items():
            eligible = [
                row for row in rows
                if row["signer"] == signer
                and row.get("example_id", row.get("sentence_name")) in seen[source]
                and 60 <= row["encoder_shape"][0] <= 150
                and 5 <= len(normalize(row["reference"])) <= 18
            ]
            if len(eligible) < count:
                raise ValueError(f"training-fit {source} lacks signer {signer}")
            selected.extend(
                {"source": source, **row}
                for row in sorted(
                    eligible,
                    key=lambda item: item.get("example_id", item.get("sentence_name")),
                )[:count]
            )
    if len(selected) != 32 or len({
        (row["source"], row.get("example_id", row.get("sentence_name")))
        for row in selected
    }) != len(selected):
        raise ValueError("training-fit probe selection is incomplete or repeated")
    return selected


def summarize(records: list[dict]) -> dict:
    if len(records) != 32:
        raise ValueError("training-fit result is incomplete")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["source"]].append(row)
        if row["source"] != "fleurs":
            grouped["how2sign_all"].append(row)
    return {
        source: {
            "examples": len(rows),
            "baseline_mean_wer": statistics.fmean(row["baseline_wer"] for row in rows),
            "candidate_mean_wer": statistics.fmean(row["candidate_wer"] for row in rows),
            "candidate_minus_baseline_mean_wer": statistics.fmean(
                row["candidate_wer"] - row["baseline_wer"] for row in rows
            ),
        }
        for source, rows in sorted(grouped.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("training-fit probe needs a new run-ledger output")
    stage = json.loads(STAGE_PLAN.read_text())
    ablation = json.loads(ABLATION_PLAN.read_text())
    decision = json.loads(ABLATION_DECISION.read_text())
    checkpoint = BASELINE / "models/checkpoint-11625"
    checkpoint_hash = sha256(checkpoint / "model.safetensors")
    if (
        stage.get("schema") != "umi-studio-visual-lora-plan/1"
        or stage.get("baseline_checkpoint_sha256") != checkpoint_hash
        or ablation.get("parent_plan_sha256") != sha256(STAGE_PLAN)
        or ablation.get("baseline_checkpoint_sha256") != checkpoint_hash
        or decision.get("plan_sha256") != sha256(ABLATION_PLAN)
        or decision.get("candidate_adapter_sha256") != sha256(ADAPTER)
        or decision.get("decision") != "retain_frozen_baseline"
        or decision.get("confirmation_evaluated") is not False
    ):
        raise ValueError("training-fit probe source model provenance differs")
    source_rows = {}
    for source, layout in SOURCE_LAYOUT.items():
        index = ROOT / layout["index"]
        if sha256(index) != stage["sources"][source]["index_sha256"]:
            raise ValueError(f"training-fit {source} index changed")
        if "sidecar_index" in layout and sha256(ROOT / layout["sidecar_index"]) != stage[
            "sources"
        ][source]["sidecar_index_sha256"]:
            raise ValueError(f"training-fit {source} sidecar index changed")
        source_rows[source] = [json.loads(line) for line in index.read_text().splitlines()]
    chosen = choose_rows(source_rows, seen_training_ids(STAGE_STEPS))
    sidecar_maps = {}
    for source, layout in SOURCE_LAYOUT.items():
        if "sidecar_index" in layout:
            sidecar_maps[source] = {
                row["sentence_name"]: row
                for row in map(
                    json.loads, (ROOT / layout["sidecar_index"]).read_text().splitlines()
                )
            }
    examples = []
    for row in chosen:
        source = row["source"]
        identifier = row.get("example_id", row.get("sentence_name"))
        layout = SOURCE_LAYOUT[source]
        cache = ROOT / layout["cache_dir"] / f"{identifier}.npz"
        sidecar = sidecar_maps.get(source, {}).get(identifier)
        sidecar_path = (
            ROOT / layout["sidecar_dir"] / f"{identifier}.npz" if sidecar else None
        )
        if sha256(cache) != row["encoder_cache_sha256"] or (
            sidecar_path is not None
            and sha256(sidecar_path) != sidecar["stream_sha256"]
        ):
            raise ValueError(f"training-fit example cache differs: {identifier}")
        examples.append({
            "id": identifier,
            "source": source,
            "signer": row["signer"],
            "reference": row["reference"],
            "frames": row["encoder_shape"][0],
            "cache": str(cache),
            "cache_sha256": row["encoder_cache_sha256"],
            "sidecar": str(sidecar_path) if sidecar_path else None,
            "sidecar_sha256": sidecar["stream_sha256"] if sidecar else None,
        })
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
    from safetensors import safe_open
    from safetensors.torch import load_file
    from shubert_visual_lora import install_visual_lora, load_visual_adapter
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    torch.set_num_threads(2)
    revision = miner_revision(args.miner_health_url, stage["miner_model_revision"])
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BASELINE / "models/byt5_base"), local_files_only=True
    )
    model.to("cpu")
    model.eval()
    def decode(hidden):
        nonlocal revision
        revision = miner_revision(args.miner_health_url, revision)
        with torch.no_grad():
            ids = GenerationMixin.generate(
                model,
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden
                ),
                max_length=2048,
                num_beams=5,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        revision = miner_revision(args.miner_health_url, revision)
        return tokenizer.decode(ids[0], skip_special_tokens=True)
    baseline_hypotheses = {}
    for row in examples:
        revision = miner_revision(args.miner_health_url, revision)
        with np.load(row["cache"], allow_pickle=False) as payload:
            state = payload["encoder_hidden_state"]
        if state.shape != (row["frames"], 1536) or state.dtype != np.float32:
            raise ValueError(f"training-fit baseline state differs: {row['id']}")
        baseline_hypotheses[(row["source"], row["id"])] = decode(
            torch.from_numpy(state).unsqueeze(0)
        )
        print(f"baseline {row['source']} {row['id']}", flush=True)
    with safe_open(str(ADAPTER), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if (
        metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("plan_sha256") != sha256(ABLATION_PLAN)
        or metadata.get("baseline_checkpoint_sha256") != checkpoint_hash
        or metadata.get("ablation") != "baseline_visual_projection"
    ):
        raise ValueError("training-fit ablation adapter metadata differs")
    install_visual_lora(
        model,
        int(metadata["rank"]),
        float(metadata["alpha"]),
        int(metadata["visual_last_layers"]),
    )
    load_visual_adapter(model, load_file(str(ADAPTER), device="cpu"))
    model.eval()
    records = []
    for row in examples:
        revision = miner_revision(args.miner_health_url, revision)
        streams = training_streams(row)
        with torch.no_grad():
            hidden = model.encoder(
                **{
                    name: torch.from_numpy(value).unsqueeze(0)
                    for name, value in streams.items()
                },
                return_dict=True,
            ).last_hidden_state
        if hidden.shape != (1, row["frames"], 1536) or not torch.isfinite(hidden).all():
            raise ValueError(f"training-fit candidate state differs: {row['id']}")
        candidate = decode(hidden)
        baseline = baseline_hypotheses[(row["source"], row["id"])]
        reference_words = normalize(row["reference"])
        records.append({
            "source": row["source"],
            "signer": row["signer"],
            "example_id": row["id"],
            "cache_sha256": row["cache_sha256"],
            "sidecar_sha256": row["sidecar_sha256"],
            "reference": row["reference"],
            "baseline_hypothesis": baseline,
            "candidate_hypothesis": candidate,
            "baseline_wer": distance(reference_words, normalize(baseline)) / len(reference_words),
            "candidate_wer": distance(reference_words, normalize(candidate)) / len(reference_words),
        })
        print(f"candidate {row['source']} {row['id']}", flush=True)
    result = {
        "schema": "umi-studio-visual-training-fit-probe/1",
        "split": "training",
        "diagnostic_only": True,
        "selection_rule": (
            "Among stage-one exposures, choose the first 60-150 frame, 5-18 word "
            "examples by ID at the fixed source and signer quotas in the probe source."
        ),
        "stage_plan_sha256": sha256(STAGE_PLAN),
        "stage_steps_sha256": sha256(STAGE_STEPS),
        "ablation_plan_sha256": sha256(ABLATION_PLAN),
        "ablation_decision_sha256": sha256(ABLATION_DECISION),
        "baseline_checkpoint_sha256": checkpoint_hash,
        "adapter_sha256": sha256(ADAPTER),
        "probe_source_sha256": sha256(Path(__file__)),
        "miner_model_revision": revision,
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished training-fit probe needs inspection")
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
