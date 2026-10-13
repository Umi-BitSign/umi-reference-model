"""Check a fixed beam length-penalty grid on cached training examples."""

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

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
STAGE_PLAN = ROOT / "runs/visual-lora-stage1-v1/plan.json"
TRAINING_FIT = ROOT / "runs/visual-training-fit-probe-v1/result.json"
PENALTIES = (1.0, 0.7, 1.3, 1.6)
GATE = {
    "how2sign_max_mean_wer_change": -0.05,
    "fleurs_max_mean_wer_change": 0.03,
    "equal_domain_max_mean_wer_change": -0.025,
    "how2sign_min_improved_signers": 3,
}


def summarize(records: list[dict]) -> dict:
    if len(records) != 32 or len({(r["source"], r["example_id"]) for r in records}) != 32:
        raise ValueError("decode-length probe needs all 32 distinct examples")
    result = {}
    for penalty in PENALTIES:
        key = str(penalty)
        fleurs = [row for row in records if row["source"] == "fleurs"]
        how2sign = [row for row in records if row["source"] != "fleurs"]
        h_delta = statistics.fmean(row["decodes"][key]["wer"] - row["baseline_wer"] for row in how2sign)
        f_delta = statistics.fmean(row["decodes"][key]["wer"] - row["baseline_wer"] for row in fleurs)
        signer_deltas = {
            str(signer): statistics.fmean(
                row["decodes"][key]["wer"] - row["baseline_wer"]
                for row in how2sign if row["signer"] == signer
            )
            for signer in (1, 2, 3, 5, 8)
        }
        if any(not any(row["signer"] == signer for row in how2sign) for signer in (1, 2, 3, 5, 8)):
            raise ValueError("decode-length probe lacks a primary How2Sign signer")
        result[key] = {
            "how2sign_mean_wer": statistics.fmean(row["decodes"][key]["wer"] for row in how2sign),
            "fleurs_mean_wer": statistics.fmean(row["decodes"][key]["wer"] for row in fleurs),
            "how2sign_mean_wer_change": h_delta,
            "fleurs_mean_wer_change": f_delta,
            "equal_domain_mean_wer_change": (h_delta + f_delta) / 2,
            "how2sign_signer_mean_wer_changes": signer_deltas,
            "how2sign_signers_improved": sum(delta < 0 for delta in signer_deltas.values()),
            "how2sign_mean_hypothesis_words": statistics.fmean(
                len(normalize(row["decodes"][key]["hypothesis"])) for row in how2sign
            ),
        }
    return result


def choose(summary: dict) -> float | None:
    passing = []
    for penalty in PENALTIES:
        if penalty == 1.0:
            continue
        row = summary[str(penalty)]
        if (
            row["how2sign_mean_wer_change"] <= GATE["how2sign_max_mean_wer_change"]
            and row["fleurs_mean_wer_change"] <= GATE["fleurs_max_mean_wer_change"]
            and row["equal_domain_mean_wer_change"] <= GATE["equal_domain_max_mean_wer_change"]
            and row["how2sign_signers_improved"] >= GATE["how2sign_min_improved_signers"]
        ):
            passing.append(penalty)
    return min(passing, key=lambda penalty: (summary[str(penalty)]["equal_domain_mean_wer_change"], penalty)) if passing else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("decode-length probe needs a new run-ledger output")
    stage = json.loads(STAGE_PLAN.read_text())
    fit = json.loads(TRAINING_FIT.read_text())
    checkpoint = BASELINE / "models/checkpoint-11625"
    checkpoint_hash = sha256(checkpoint / "model.safetensors")
    if (
        stage.get("schema") != "umi-studio-visual-lora-plan/1"
        or stage.get("baseline_checkpoint_sha256") != checkpoint_hash
        or fit.get("schema") != "umi-studio-visual-training-fit-probe/1"
        or fit.get("split") != "training"
        or fit.get("stage_plan_sha256") != sha256(STAGE_PLAN)
        or len(fit.get("records", [])) != 32
    ):
        raise ValueError("decode-length probe source provenance differs")
    indexed = {}
    for source in SOURCE_LAYOUT:
        index = ROOT / SOURCE_LAYOUT[source]["index"]
        if sha256(index) != stage["sources"][source]["index_sha256"]:
            raise ValueError(f"decode-length source index changed: {source}")
        for source_row in map(json.loads, index.read_text().splitlines()):
            example_id = source_row.get("example_id", source_row.get("sentence_name"))
            indexed[(source, example_id)] = source_row
    examples = []
    for row in fit["records"]:
        source, example_id = row["source"], row["example_id"]
        source_row = indexed[(source, example_id)]
        if (
            source_row["encoder_cache_sha256"] != row["cache_sha256"]
            or source_row["reference"] != row["reference"]
        ):
            raise ValueError(f"decode-length source example changed: {example_id}")
        cache = ROOT / SOURCE_LAYOUT[source]["cache_dir"] / f"{example_id}.npz"
        if sha256(cache) != row["cache_sha256"]:
            raise ValueError(f"decode-length cache changed: {example_id}")
        examples.append({**row, "cache": str(cache), "frames": source_row["encoder_shape"][0]})
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
    revision = miner_revision(args.miner_health_url, stage["miner_model_revision"])
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    ).to("cpu").eval()
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BASELINE / "models/byt5_base"), local_files_only=True
    )
    records = []
    for row in examples:
        with np.load(row["cache"], allow_pickle=False) as payload:
            state = payload["encoder_hidden_state"]
        if state.shape != (row["frames"], 1536) or state.dtype != np.float32:
            raise ValueError(f"decode-length state changed: {row['example_id']}")
        hidden = torch.from_numpy(state).unsqueeze(0)
        reference = normalize(row["reference"])
        decodes = {}
        for penalty in PENALTIES:
            revision = miner_revision(args.miner_health_url, revision)
            with torch.no_grad():
                ids = GenerationMixin.generate(
                    model,
                    encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden),
                    max_length=2048,
                    num_beams=5,
                    length_penalty=penalty,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            revision = miner_revision(args.miner_health_url, revision)
            hypothesis = tokenizer.decode(ids[0], skip_special_tokens=True)
            wer = distance(reference, normalize(hypothesis)) / len(reference)
            if penalty == 1.0 and (
                hypothesis != row["baseline_hypothesis"] or abs(wer - row["baseline_wer"]) > 1e-12
            ):
                raise ValueError(f"decode-length baseline reproduction failed: {row['example_id']}")
            decodes[str(penalty)] = {"hypothesis": hypothesis, "wer": wer}
        records.append({
            "source": row["source"],
            "signer": row["signer"],
            "example_id": row["example_id"],
            "cache_sha256": row["cache_sha256"],
            "reference": row["reference"],
            "baseline_wer": row["baseline_wer"],
            "decodes": decodes,
        })
        print(f"decoded {len(records)}/{len(examples)} training examples", flush=True)
    summary = summarize(records)
    selected = choose(summary)
    result = {
        "schema": "umi-studio-decode-length-probe/1",
        "split": "training",
        "diagnostic_only": True,
        "stage_plan_sha256": sha256(STAGE_PLAN),
        "training_fit_sha256": sha256(TRAINING_FIT),
        "probe_source_sha256": sha256(Path(__file__)),
        "baseline_checkpoint_sha256": checkpoint_hash,
        "miner_model_revision": revision,
        "penalties": PENALTIES,
        "gate": GATE,
        "selected_penalty_for_frozen_tuning": selected,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished decode-length probe needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"selected": selected, "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
