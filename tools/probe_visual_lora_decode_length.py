"""Screen longer-output beam penalties on training-only visual LoRA examples."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from benchmark_fleurs_dev import distance, miner_revision, normalize, sha256
from freeze_visual_lora_plan import SOURCE_LAYOUT
from train_visual_lora import training_streams

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
STAGE_PLAN = ROOT / "runs/visual-lora-stage1-v1/plan.json"
TRAINING_FIT = ROOT / "runs/visual-training-fit-probe-v1/result.json"
CORRECTIVE_PLAN = ROOT / "runs/visual-lora-corrective-v1/plan.json"
CORRECTIVE_DECISION = ROOT / "runs/visual-lora-corrective-v1/tuning-decision.json"
ADAPTER = (
    ROOT
    / "checkpoints/visual-lora-corrective-v1/visual-lora-after-000384.safetensors"
)
PENALTIES = (1.0, 1.1, 1.2, 1.3)
SCREEN = {
    "how2sign_max_mean_wer_change": -0.015,
    "fleurs_max_mean_wer_change": 0.02,
    "equal_domain_max_mean_wer_change": -0.005,
    "how2sign_min_improved_signers": 3,
}


def summarize(records: list[dict]) -> dict:
    if len(records) != 32:
        raise ValueError("decode probe needs all 32 training examples")
    result = {}
    for penalty in PENALTIES:
        key = str(penalty)
        fleurs = [row for row in records if row["source"] == "fleurs"]
        how2sign = [row for row in records if row["source"] != "fleurs"]
        h_delta = statistics.fmean(
            row["decodes"][key]["wer"] - row["decodes"]["1.0"]["wer"]
            for row in how2sign
        )
        f_delta = statistics.fmean(
            row["decodes"][key]["wer"] - row["decodes"]["1.0"]["wer"]
            for row in fleurs
        )
        signer_deltas = {
            str(signer): statistics.fmean(
                row["decodes"][key]["wer"]
                - row["decodes"]["1.0"]["wer"]
                for row in how2sign
                if row["signer"] == signer
            )
            for signer in (1, 2, 3, 5, 8)
        }
        if any(
            not any(row["signer"] == signer for row in how2sign)
            for signer in (1, 2, 3, 5, 8)
        ):
            raise ValueError("decode probe lacks a primary How2Sign signer")
        result[key] = {
            "how2sign_mean_wer": statistics.fmean(
                row["decodes"][key]["wer"] for row in how2sign
            ),
            "fleurs_mean_wer": statistics.fmean(
                row["decodes"][key]["wer"] for row in fleurs
            ),
            "how2sign_mean_wer_change": h_delta,
            "fleurs_mean_wer_change": f_delta,
            "equal_domain_mean_wer_change": (h_delta + f_delta) / 2,
            "how2sign_signer_mean_wer_changes": signer_deltas,
            "how2sign_signers_improved": sum(
                delta < 0 for delta in signer_deltas.values()
            ),
            "how2sign_mean_hypothesis_words": statistics.fmean(
                len(normalize(row["decodes"][key]["hypothesis"]))
                for row in how2sign
            ),
        }
    return result


def choose(summary: dict) -> float | None:
    passing = []
    for penalty in PENALTIES[1:]:
        row = summary[str(penalty)]
        if (
            row["how2sign_mean_wer_change"]
            <= SCREEN["how2sign_max_mean_wer_change"]
            and row["fleurs_mean_wer_change"]
            <= SCREEN["fleurs_max_mean_wer_change"]
            and row["equal_domain_mean_wer_change"]
            <= SCREEN["equal_domain_max_mean_wer_change"]
            and row["how2sign_signers_improved"]
            >= SCREEN["how2sign_min_improved_signers"]
        ):
            passing.append(penalty)
    if not passing:
        return None
    return min(
        passing,
        key=lambda value: (
            summary[str(value)]["equal_domain_mean_wer_change"],
            value,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url")
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if args.miner_maintenance_mode == bool(args.miner_health_url):
        parser.error(
            "choose exactly one of --miner-health-url or --miner-maintenance-mode"
        )
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("decode probe needs a new run-ledger output")

    stage = json.loads(STAGE_PLAN.read_text())
    fit = json.loads(TRAINING_FIT.read_text())
    corrective = json.loads(CORRECTIVE_PLAN.read_text())
    decision = json.loads(CORRECTIVE_DECISION.read_text())
    checkpoint = BASELINE / "models/checkpoint-11625"
    checkpoint_hash = sha256(checkpoint / "model.safetensors")
    adapter_hash = sha256(ADAPTER)
    if (
        stage.get("schema") != "umi-studio-visual-lora-plan/1"
        or stage.get("baseline_checkpoint_sha256") != checkpoint_hash
        or fit.get("schema") != "umi-studio-visual-training-fit-probe/1"
        or fit.get("split") != "training"
        or len(fit.get("records", [])) != 32
        or corrective.get("schema")
        != "umi-studio-visual-lora-corrective-plan/1"
        or corrective.get("baseline_checkpoint_sha256") != checkpoint_hash
        or decision.get("plan_sha256") != sha256(CORRECTIVE_PLAN)
        or decision.get("candidate_adapter_sha256") != adapter_hash
        or decision.get("confirmation_evaluated") is not False
    ):
        raise ValueError("decode probe source provenance differs")

    indexed = {}
    sidecars = {}
    for source, layout in SOURCE_LAYOUT.items():
        index = ROOT / layout["index"]
        if sha256(index) != stage["sources"][source]["index_sha256"]:
            raise ValueError(f"decode probe source index changed: {source}")
        for source_row in map(json.loads, index.read_text().splitlines()):
            example_id = source_row.get(
                "example_id", source_row.get("sentence_name")
            )
            indexed[(source, example_id)] = source_row
        if "sidecar_index" in layout:
            sidecar_index = ROOT / layout["sidecar_index"]
            if (
                sha256(sidecar_index)
                != stage["sources"][source]["sidecar_index_sha256"]
            ):
                raise ValueError(f"decode probe sidecar index changed: {source}")
            sidecars[source] = {
                row["sentence_name"]: row
                for row in map(json.loads, sidecar_index.read_text().splitlines())
            }

    examples = []
    for selected in fit["records"]:
        source, example_id = selected["source"], selected["example_id"]
        row = indexed[(source, example_id)]
        layout = SOURCE_LAYOUT[source]
        cache = ROOT / layout["cache_dir"] / f"{example_id}.npz"
        sidecar = sidecars.get(source, {}).get(example_id)
        sidecar_path = (
            ROOT / layout["sidecar_dir"] / f"{example_id}.npz"
            if sidecar is not None
            else None
        )
        if (
            row["reference"] != selected["reference"]
            or sha256(cache) != selected["cache_sha256"]
            or (
                sidecar_path is not None
                and sha256(sidecar_path) != selected["sidecar_sha256"]
            )
        ):
            raise ValueError(f"decode probe example changed: {example_id}")
        examples.append(
            {
                "source": source,
                "id": example_id,
                "signer": selected["signer"],
                "reference": selected["reference"],
                "frames": row["encoder_shape"][0],
                "cache": str(cache),
                "cache_sha256": selected["cache_sha256"],
                "sidecar": str(sidecar_path) if sidecar_path else None,
                "sidecar_sha256": selected["sidecar_sha256"],
            }
        )

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
    import torch
    from inference import SignLanguageByT5ForConditionalGeneration
    from safetensors import safe_open
    from safetensors.torch import load_file
    from shubert_visual_lora import install_visual_lora, load_visual_adapter
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

    torch.set_num_threads(2)
    revision = (
        None
        if args.miner_maintenance_mode
        else miner_revision(args.miner_health_url, None)
    )
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    with safe_open(str(ADAPTER), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if (
        metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("baseline_checkpoint_sha256") != checkpoint_hash
        or metadata.get("plan_sha256") != sha256(CORRECTIVE_PLAN)
        or metadata.get("example_step") != "384"
    ):
        raise ValueError("decode probe adapter metadata differs")
    install_visual_lora(
        model,
        int(metadata["rank"]),
        float(metadata["alpha"]),
        int(metadata["visual_last_layers"]),
    )
    load_visual_adapter(model, load_file(str(ADAPTER), device="cpu"))
    tokenizer = ByT5Tokenizer.from_pretrained(
        str(BASELINE / "models/byt5_base"), local_files_only=True
    )
    model.to("cpu").eval()

    def check_miner() -> None:
        nonlocal revision
        if not args.miner_maintenance_mode:
            revision = miner_revision(args.miner_health_url, revision)

    records = []
    for row in examples:
        check_miner()
        streams = training_streams(row)
        with torch.no_grad():
            hidden = model.encoder(
                **{
                    name: torch.from_numpy(value).unsqueeze(0)
                    for name, value in streams.items()
                },
                return_dict=True,
            ).last_hidden_state
        if hidden.shape != (1, row["frames"], 1536):
            raise ValueError(f"decode probe encoder shape changed: {row['id']}")
        decodes = {}
        reference = normalize(row["reference"])
        for penalty in PENALTIES:
            check_miner()
            with torch.no_grad():
                ids = GenerationMixin.generate(
                    model,
                    encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                        last_hidden_state=hidden
                    ),
                    max_length=2048,
                    num_beams=5,
                    length_penalty=penalty,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            hypothesis = tokenizer.decode(ids[0], skip_special_tokens=True)
            decodes[str(penalty)] = {
                "hypothesis": hypothesis,
                "wer": distance(reference, normalize(hypothesis)) / len(reference),
            }
        records.append(
            {
                "source": row["source"],
                "signer": row["signer"],
                "example_id": row["id"],
                "reference": row["reference"],
                "decodes": decodes,
            }
        )
        print(f"decoded {len(records)}/{len(examples)} training examples", flush=True)

    summary = summarize(records)
    selected = choose(summary)
    result = {
        "schema": "umi-studio-visual-lora-decode-length-probe/1",
        "split": "training",
        "diagnostic_only": True,
        "confirmation_evaluated": False,
        "stage_plan_sha256": sha256(STAGE_PLAN),
        "training_fit_sha256": sha256(TRAINING_FIT),
        "corrective_plan_sha256": sha256(CORRECTIVE_PLAN),
        "corrective_decision_sha256": sha256(CORRECTIVE_DECISION),
        "adapter_sha256": adapter_hash,
        "probe_source_sha256": sha256(Path(__file__)),
        "miner_model_revision": revision,
        "miner_health_policy": (
            "user_authorized_maintenance"
            if args.miner_maintenance_mode
            else "required_stable_revision"
        ),
        "penalties": PENALTIES,
        "screen": SCREEN,
        "selected_penalty_for_frozen_tuning": selected,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
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
