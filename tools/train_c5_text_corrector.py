"""Fine-tune and select a compact T5 text corrector on frozen C5 errors."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from freeze_c5_text_corrector_plan import RECIPE, ROOT, SEED

PREFIX = "correct sign transcript: "


def checked_rows(path: Path, expected_hash: str) -> list[dict]:
    if sha256(path) != expected_hash:
        raise ValueError(f"frozen text-corrector rows changed: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or len({(row["source"], row["example_id"]) for row in rows}) != len(rows):
        raise ValueError("frozen text-corrector rows are empty or duplicated")
    return rows


def metric_summary(rows: list[dict], hypotheses: list[str]) -> dict:
    if len(rows) != len(hypotheses):
        raise ValueError("text-corrector evaluation lengths differ")
    by_source = defaultdict(list)
    examples = []
    for row, hypothesis in zip(rows, hypotheses, strict=True):
        reference = normalize(row["reference"])
        edits = distance(reference, normalize(hypothesis))
        candidate_wer = edits / len(reference)
        item = {
            "source": row["source"],
            "example_id": row["example_id"],
            "reference": row["reference"],
            "input_hypothesis": row["hypothesis"],
            "corrected_hypothesis": hypothesis,
            "baseline_wer": row["single_reference_wer"],
            "corrected_wer": candidate_wer,
            "wer_change": candidate_wer - row["single_reference_wer"],
        }
        examples.append(item)
        by_source[row["source"]].append(item)
    sources = {}
    for source, items in sorted(by_source.items()):
        baseline = sum(item["baseline_wer"] for item in items) / len(items)
        corrected = sum(item["corrected_wer"] for item in items) / len(items)
        sources[source] = {
            "examples": len(items),
            "baseline_mean_wer": baseline,
            "corrected_mean_wer": corrected,
            "mean_wer_change": corrected - baseline,
            "improved": sum(item["wer_change"] < 0 for item in items),
            "tied": sum(item["wer_change"] == 0 for item in items),
            "regressed": sum(item["wer_change"] > 0 for item in items),
        }
    baseline = sum(item["baseline_wer"] for item in examples) / len(examples)
    corrected = sum(item["corrected_wer"] for item in examples) / len(examples)
    return {
        "examples": len(examples),
        "baseline_mean_wer": baseline,
        "corrected_mean_wer": corrected,
        "mean_wer_change": corrected - baseline,
        "improved": sum(item["wer_change"] < 0 for item in examples),
        "tied": sum(item["wer_change"] == 0 for item in examples),
        "regressed": sum(item["wer_change"] > 0 for item in examples),
        "sources": sources,
        "records": examples,
    }


def duty_sleep(active_seconds: float, active_fraction: float) -> None:
    idle = active_seconds * (1.0 / active_fraction - 1.0)
    if idle > 0:
        time.sleep(idle)


def generate_texts(model, tokenizer, texts: list[str], batch_size: int, device: str) -> list[str]:
    import torch

    outputs = []
    for offset in range(0, len(texts), batch_size):
        started = time.perf_counter()
        batch = tokenizer(
            texts[offset : offset + batch_size],
            padding=True,
            truncation=True,
            max_length=RECIPE["max_input_tokens"],
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            tokens = model.generate(
                **batch,
                max_new_tokens=RECIPE["max_target_tokens"],
                num_beams=RECIPE["generation_beams"],
                early_stopping=True,
            )
        outputs.extend(tokenizer.batch_decode(tokens, skip_special_tokens=True))
        duty_sleep(time.perf_counter() - started, RECIPE["mps_active_fraction"])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if (
        not args.miner_maintenance_mode
        or not args.plan.is_file()
        or args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to((ROOT / "checkpoints").resolve())
    ):
        parser.error("training requires maintenance mode and a new checkpoints/ output")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema") != "umi-studio-c5-text-corrector-plan/1"
        or plan.get("recipe") != RECIPE
        or plan.get("seed") != SEED
        or plan.get("trainer_source_sha256") != sha256(Path(__file__))
    ):
        raise ValueError("text-corrector plan differs from the trainer")
    train_rows = checked_rows(args.plan.with_name("train.jsonl"), plan["train_sha256"])
    holdout_rows = checked_rows(args.plan.with_name("holdout.jsonl"), plan["holdout_sha256"])
    if len(train_rows) != plan["train_examples"] or len(holdout_rows) != plan["holdout_examples"]:
        raise ValueError("text-corrector split counts changed")
    model_path = ROOT / RECIPE["pretrained_model"]
    if not model_path.is_dir():
        raise ValueError("local T5-small checkpoint is unavailable")
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    }.items():
        os.environ[key] = value
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    if not torch.backends.mps.is_available():
        raise RuntimeError("the frozen text-corrector recipe requires MPS")
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = RECIPE["device"]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"]
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir()
    run = {
        "schema": "umi-studio-c5-text-corrector-training/1",
        "plan_sha256": sha256(args.plan),
        "recipe": RECIPE,
        "train_examples": len(train_rows),
        "holdout_examples": len(holdout_rows),
        "trainable_parameters": sum(value.numel() for value in model.parameters()),
        "miner_health_policy": "user_authorized_maintenance",
        "miner_changed": False,
        "trainer_source_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    training_examples = []
    for row in train_rows:
        training_examples.append((PREFIX + row["hypothesis"], row["reference"], "correction"))
        for _ in range(RECIPE["identity_examples_per_correction"]):
            training_examples.append((PREFIX + row["reference"], row["reference"], "identity"))
    accumulation = RECIPE["gradient_accumulation"]
    epoch_summaries = []
    for epoch in range(1, RECIPE["epochs"] + 1):
        order = list(range(len(training_examples)))
        random.Random(SEED + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        kinds = defaultdict(int)
        for batch_number, offset in enumerate(range(0, len(order), RECIPE["batch_size"]), 1):
            started = time.perf_counter()
            indexes = order[offset : offset + RECIPE["batch_size"]]
            batch_rows = [training_examples[index] for index in indexes]
            inputs = tokenizer(
                [item[0] for item in batch_rows],
                padding=True,
                truncation=True,
                max_length=RECIPE["max_input_tokens"],
                return_tensors="pt",
            ).to(device)
            labels = tokenizer(
                text_target=[item[1] for item in batch_rows],
                padding=True,
                truncation=True,
                max_length=RECIPE["max_target_tokens"],
                return_tensors="pt",
            ).input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            result = model(**inputs, labels=labels)
            if not torch.isfinite(result.loss):
                raise RuntimeError("non-finite text-corrector loss")
            (result.loss / accumulation).backward()
            losses.append(float(result.loss.detach().cpu()))
            for item in batch_rows:
                kinds[item[2]] += 1
            final_batch = offset + RECIPE["batch_size"] >= len(order)
            if batch_number % accumulation == 0 or final_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            duty_sleep(time.perf_counter() - started, RECIPE["mps_active_fraction"])
        checkpoint = args.output_dir / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        model.eval()
        corrected = generate_texts(
            model,
            tokenizer,
            [PREFIX + row["hypothesis"] for row in holdout_rows],
            RECIPE["batch_size"],
            device,
        )
        identity = generate_texts(
            model,
            tokenizer,
            [PREFIX + row["reference"] for row in holdout_rows],
            RECIPE["batch_size"],
            device,
        )
        metrics = metric_summary(holdout_rows, corrected)
        metrics["identity_changed"] = sum(
            normalize(output) != normalize(row["reference"])
            for row, output in zip(holdout_rows, identity, strict=True)
        )
        metrics["identity_change_rate"] = metrics["identity_changed"] / len(holdout_rows)
        summary = {
            "epoch": epoch,
            "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
            "mean_training_loss": sum(losses) / len(losses),
            "training_kinds": dict(kinds),
            "holdout": {key: value for key, value in metrics.items() if key != "records"},
        }
        epoch_summaries.append(summary)
        (args.output_dir / f"holdout-epoch-{epoch}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in metrics["records"])
        )
        with (args.output_dir / "epochs.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(summary, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        print(
            f"epoch={epoch} loss={summary['mean_training_loss']:.4f} "
            f"holdout_delta={metrics['mean_wer_change']:+.4f} "
            f"identity_change={metrics['identity_change_rate']:.3f}",
            flush=True,
        )
    gate = plan["selection"]
    eligible = []
    for summary in epoch_summaries:
        holdout = summary["holdout"]
        max_source_regression = max(
            source["mean_wer_change"] for source in holdout["sources"].values()
        )
        improved_sources = sum(
            source["mean_wer_change"] < 0 for source in holdout["sources"].values()
        )
        if (
            holdout["mean_wer_change"] <= gate["maximum_holdout_mean_wer_change"]
            and max_source_regression <= gate["maximum_source_mean_wer_regression"]
            and improved_sources >= gate["minimum_sources_improved"]
        ):
            eligible.append(summary)
    selected = min(
        eligible,
        key=lambda row: (
            row["holdout"]["corrected_mean_wer"],
            row["holdout"]["identity_change_rate"],
            row["epoch"],
        ),
        default=None,
    )
    decision = {
        "schema": "umi-studio-c5-text-corrector-selection/1",
        "decision": "advance_to_open_screen" if selected else "reject",
        "plan_sha256": sha256(args.plan),
        "selected_epoch": selected["epoch"] if selected else None,
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        "epochs": epoch_summaries,
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"decision": decision["decision"], "selected_epoch": decision["selected_epoch"]}))


if __name__ == "__main__":
    main()
