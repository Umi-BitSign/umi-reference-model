"""Matched face-learning arms with frozen-parent KL and bounded MPS duty cycle."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from benchmark_fleurs_dev import distance, normalize, sha256
from c5_face_learning_pilot import ARMS, BASELINE, BUNDLE, GATES, RECIPE, ROOT, SIGNERS, balanced_epoch, checked_plan, read_rows, storage


def environment() -> None:
    for name, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "HF_HOME": str(BASELINE / "run/hf"), "TORCH_HOME": str(BASELINE / "run/torch"), "MPLCONFIGDIR": str(BASELINE / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(BUNDLE / "vendor/dinov2-source"), "PYTORCH_ENABLE_MPS_FALLBACK": "1"}.items(): os.environ[name] = value
    sys.path.insert(0, str(BUNDLE / "vendor/demo")); sys.path.insert(0, str(BUNDLE / "vendor"))


class DutyCycle:
    def __init__(self, torch, *, device="mps", clock=time.perf_counter, sleep=time.sleep):
        self.torch = torch; self.device = device; self.clock = clock; self.sleep = sleep
        self.active_seconds = 0.; self.idle_seconds = 0.; self.maximum_allocated_bytes = 0

    def run(self, operation):
        if self.device == "mps": self.torch.mps.synchronize()
        started = self.clock()
        try:
            return operation()
        finally:
            if self.device == "mps": self.torch.mps.synchronize()
            seconds = self.clock() - started; self.active_seconds += seconds
            if self.device == "mps":
                self.maximum_allocated_bytes = max(self.maximum_allocated_bytes, self.torch.mps.current_allocated_memory())
                self.sleep(seconds); self.idle_seconds += seconds
                if self.maximum_allocated_bytes > RECIPE["maximum_process_mps_bytes"]:
                    raise RuntimeError("MPS process memory guard exceeded")


def configure(torch):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available(): raise RuntimeError("the frozen pilot requires MPS")
    if hasattr(torch.mps, "recommended_max_memory"):
        fraction = min(.5, RECIPE["maximum_process_mps_bytes"] / torch.mps.recommended_max_memory())
        torch.mps.set_per_process_memory_fraction(fraction)


def build(torch, seed: int, duty: DutyCycle):
    from inference import SignLanguageByT5ForConditionalGeneration
    from shubert_visual_lora import install_visual_lora
    from train_c5_stem_projection_frozen import freeze_direct_parameters
    torch.manual_seed(seed)
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(str(BUNDLE / "models/checkpoint-11625"), local_files_only=True)
    selected = install_visual_lora(model, RECIPE["rank"], RECIPE["alpha"], RECIPE["visual_last_layers"])
    trainable = freeze_direct_parameters(selected)
    duty.run(lambda: model.to("mps")); model.eval()
    return model, selected, trainable


def visual_gradient_audit(torch, trainable: dict, selected: dict) -> dict:
    from interpolate_visual_adapter import DIRECT_NAMES
    visual = {name: float(value.grad.norm()) if value.grad is not None else None for name, value in trainable.items() if "signhubert" in name}
    if len(visual) != 16 or any(value is None or not math.isfinite(value) for value in visual.values()):
        raise RuntimeError("MPS visual adapter gradient coverage is incomplete or nonfinite")
    if any(selected[name].grad is not None for name in DIRECT_NAMES): raise RuntimeError("frozen projection received gradients")
    return visual


def probe(output: Path) -> None:
    if output.exists() or not output.resolve().is_relative_to(ROOT / "runs"): raise ValueError("MPS probe needs a new runs/ output")
    environment()
    import torch
    from train_c5_stem_continuation import stride2_streams
    from transformers import ByT5Tokenizer
    configure(torch); duty = DutyCycle(torch)
    rows = read_rows(ROOT / "runs/c5-all-source-corpus-v1/training-streams.jsonl")
    row = next(r for r in rows if r["source"].startswith("how2sign") and r["signer"] == 2 and r["frames"] <= 256)
    bound = {**row, "cache": str(ROOT / row["cache"])}
    if row.get("sidecar"): bound["sidecar"] = str(ROOT / row["sidecar"])
    arrays = stride2_streams(bound)
    tokenizer = ByT5Tokenizer.from_pretrained(str(BUNDLE / "models/byt5_base"), local_files_only=True)
    model, selected, trainable = build(torch, 281, duty)
    def backward():
        features = {name: torch.from_numpy(value).unsqueeze(0).to("mps") for name, value in arrays.items()}
        labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to("mps")
        result = model(**features, labels=labels, use_cache=False, return_dict=True)
        loss = torch.nn.functional.cross_entropy(result.logits.flatten(0, 1), labels.flatten(), label_smoothing=.1)
        if not torch.isfinite(loss): raise RuntimeError("MPS probe loss is nonfinite")
        loss.backward(); norms = visual_gradient_audit(torch, trainable, selected)
        gradient = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.)
        if not torch.isfinite(gradient): raise RuntimeError("MPS probe gradient is nonfinite")
        return float(loss), float(gradient), norms
    loss, gradient, norms = duty.run(backward)
    positives = sum(value > 0 for name, value in norms.items() if ".lora_b." in name)
    if positives != 8: raise RuntimeError("an MPS visual B projection received no gradient")
    result = {"schema": "umi-studio-c5-face-learning-mps-probe/1", "passes": True, "example_id": row["id"],
        "role": "eligible_training_only", "input_policy": "existing_native224_stride2_backend_diagnostic_only",
        "parent_checkpoint_sha256": sha256(BUNDLE / "models/checkpoint-11625/model.safetensors"),
        "loss": loss, "gradient_norm": gradient, "visual_gradient_norms": norms,
        "mps_active_seconds": duty.active_seconds, "mps_idle_seconds": duty.idle_seconds,
        "maximum_allocated_bytes": duty.maximum_allocated_bytes, "optimizer_updates": 0,
        "weights_saved": False, "miner_changed": False}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result))


def load_features(row: dict, arm: str) -> dict:
    path = ROOT / row["features"][arm]["path"]
    if sha256(path) != row["features"][arm]["sha256"]: raise ValueError("pilot feature cache changed")
    from cache_fleurs_encoder import checked_stream_features
    with np.load(path, allow_pickle=False) as payload: values = {name: payload[name] for name in payload.files}
    return checked_stream_features(values, row["frames"])


def learning_rate_factor(step: int, total: int) -> float:
    warmup = RECIPE["warmup_updates"]
    if step < warmup: return (step + 1) / warmup
    progress = min(1., (step - warmup) / (total - warmup))
    floor = RECIPE["minimum_learning_rate_ratio"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def comparison(parent: list[dict], candidate: list[dict]) -> dict:
    before = {r["example_id"]: r for r in parent}; after = {r["example_id"]: r for r in candidate}
    if set(before) != set(after) or len(before) != 64 or len(after) != 64: raise ValueError("pilot evaluation coverage differs")
    by_source = {}; signer_deltas = {}
    for key in before:
        row = before[key]; delta = after[key]["wer"] - row["wer"]
        by_source.setdefault(row["source_group"], []).append(delta)
        signer_deltas.setdefault(str(row["signer"]), []).append(delta)
    actor_delta = {s: statistics.fmean(values) for s, values in signer_deltas.items()}
    sources = sorted(by_source); sums = np.array([sum(by_source[s]) for s in sources]); counts = np.array([len(by_source[s]) for s in sources])
    sample = np.random.default_rng(GATES["bootstrap_seed"]).integers(0, len(sources), size=(GATES["bootstrap_samples"], len(sources)))
    interval = np.quantile(sums[sample].sum(axis=1) / counts[sample].sum(axis=1), [.025, .975]).tolist()
    parent_mean = statistics.fmean(r["wer"] for r in parent); mean = statistics.fmean(r["wer"] for r in candidate)
    parent_weighted = sum(r["edits"] for r in parent) / sum(r["reference_words"] for r in parent)
    weighted = sum(r["edits"] for r in candidate) / sum(r["reference_words"] for r in candidate)
    relative = 1 - mean / parent_mean if parent_mean else 0.
    passes = relative >= .05 and weighted <= parent_weighted and sum(d < 0 for d in actor_delta.values()) >= 3 and max(actor_delta.values()) <= .05 and interval[1] <= 0 and all(r["hypothesis"].strip() for r in candidate)
    return {"mean_wer": mean, "parent_mean_wer_same_recovered_inputs": parent_mean,
        "relative_mean_wer_improvement": relative, "word_weighted_wer": weighted, "parent_word_weighted_wer": parent_weighted,
        "signer_mean_deltas": actor_delta, "source_bootstrap_95_interval": interval, "passes_diagnostic_gate": passes}


def train(plan_path: Path, output: Path) -> None:
    plan = checked_plan(plan_path)
    if output.exists() or not output.resolve().is_relative_to(ROOT / "checkpoints"): raise ValueError("pilot training requires a new checkpoint directory")
    receipts = read_rows(plan_path.parent / "features.jsonl")
    expected = {r["example_id"]: r for r in plan["examples"]}
    if len(receipts) != 384 or len({r["example_id"] for r in receipts}) != 384: raise ValueError("pilot feature preparation is incomplete")
    for row in receipts:
        if row["plan_sha256"] != sha256(plan_path) or any(row[k] != value for k, value in expected[row["example_id"]].items()): raise ValueError("pilot prepared role or identity differs")
    training = [r for r in receipts if r["training_allowed"] and r["role"] == "training"]
    holdout = [r for r in receipts if not r["training_allowed"] and r["role"] == "pilot_source_holdout"]
    if len(training) != 320 or len(holdout) != 64 or {r["source_group"] for r in training} & {r["source_group"] for r in holdout}:
        raise ValueError("pilot training/diagnostic source separation failed")
    environment()
    import torch
    from c5_retention import teacher_kl
    from interpolate_visual_adapter import DIRECT_NAMES
    from safetensors.torch import save_file
    from train_c5_stem_projection_frozen import complete_adapter_state
    from transformers import ByT5Tokenizer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
    configure(torch); duty = DutyCycle(torch)
    tokenizer = ByT5Tokenizer.from_pretrained(str(BUNDLE / "models/byt5_base"), local_files_only=True)
    output.mkdir(); data = ROOT / plan["data_dir"]; (data / "teacher-logits").mkdir()
    (output / "run.json").write_text(json.dumps({"schema": "umi-studio-c5-face-learning-pilot-run/1", "plan_sha256": sha256(plan_path),
        "parent_checkpoint_sha256": plan["asset_sha256"]["models/checkpoint-11625/model.safetensors"], "recipe": RECIPE,
        "training_examples": 320, "holdout_examples": 64, "miner_health_policy": "user_authorized_maintenance", "miner_changed": False}, indent=2, sort_keys=True) + "\n")
    parent_records = None; evaluations = []

    def inputs(row, arm):
        arrays = load_features(row, arm)
        return {name: torch.from_numpy(value).unsqueeze(0).to("mps") for name, value in arrays.items()}

    def evaluate(model, epoch, arm):
        result = []
        for row in holdout:
            def decode():
                with torch.no_grad():
                    hidden = model.encoder(**inputs(row, ARMS[1]), return_dict=True).last_hidden_state
                    ids = GenerationMixin.generate(model, encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden),
                        max_length=2048, num_beams=5, early_stopping=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    return tokenizer.decode(ids[0], skip_special_tokens=True)
            hypothesis = duty.run(decode); words = len(normalize(row["reference"])); edits = distance(normalize(row["reference"]), normalize(hypothesis))
            record = {"example_id": row["example_id"], "source_group": row["source_group"], "signer": row["signer"], "role": row["role"],
                      "arm": arm, "epoch": epoch, "input_policy": ARMS[1], "hypothesis": hypothesis, "reference_words": words, "edits": edits, "wer": edits / words}
            result.append(record)
            with (output / "evaluation-results.jsonl").open("a") as stream: stream.write(json.dumps(record, sort_keys=True) + "\n")
        return result

    for arm in ARMS:
        model, selected, trainable = build(torch, plan["seed"], duty)
        frozen_direct = duty.run(lambda: {name: selected[name].detach().cpu().clone() for name in DIRECT_NAMES})
        initial = evaluate(model, 0, arm)
        if parent_records is None: parent_records = initial
        elif [r["hypothesis"] for r in initial] != [r["hypothesis"] for r in parent_records]: raise RuntimeError("matched arms start from different predictions")
        teacher_paths = {}
        for index, row in enumerate(training):
            storage(plan)
            def frozen_logits():
                with torch.no_grad():
                    labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to("mps")
                    return model(**inputs(row, arm), labels=labels, use_cache=False, return_dict=True).logits.cpu().numpy().astype(np.float32)
            logits = duty.run(frozen_logits)
            path = data / "teacher-logits" / f"{index:03d}--{arm}.npz"
            with path.open("xb") as stream: np.savez_compressed(stream, logits=logits)
            teacher_paths[row["example_id"]] = (path, sha256(path))
            if (index + 1) % 32 == 0: print(f"teacher {arm} {index + 1}/320 training only", flush=True)
        visual = [value for name, value in trainable.items() if "signhubert" in name]
        decoder = [value for name, value in trainable.items() if name.startswith("decoder.")]
        if len(visual) != 16 or len(visual) + len(decoder) != len(trainable): raise ValueError("pilot optimizer partition differs")
        optimizer = torch.optim.AdamW([{"params": visual, "lr": RECIPE["visual_learning_rate"]}, {"params": decoder, "lr": RECIPE["decoder_learning_rate"]}], weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, 320))
        updates = 0
        for epoch in range(1, 5):
            schedule = balanced_epoch(training, epoch)
            for offset in range(0, len(schedule), 4):
                batch = schedule[offset:offset + 4]
                def update():
                    optimizer.zero_grad(set_to_none=True); ce_values = []; kl_values = []
                    for row in batch:
                        if not row["training_allowed"] or row["role"] != "training": raise RuntimeError("diagnostic holdout entered a gradient update")
                        labels = tokenizer(row["reference"], return_tensors="pt").input_ids.to("mps")
                        teacher_path, digest = teacher_paths[row["example_id"]]
                        if sha256(teacher_path) != digest: raise ValueError("frozen teacher logits changed")
                        with np.load(teacher_path, allow_pickle=False) as cache: teacher = torch.from_numpy(cache["logits"]).to("mps")
                        logits = model(**inputs(row, arm), labels=labels, use_cache=False, return_dict=True).logits
                        ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), label_smoothing=.1)
                        kl = teacher_kl(logits, teacher, labels, 2.)
                        if not torch.isfinite(ce + kl) or float(kl) < -1e-5: raise RuntimeError("pilot objective is nonfinite")
                        if updates == 0 and abs(float(kl)) > 1e-5: raise RuntimeError("initial frozen parent and student distributions differ")
                        ((ce + kl) / 4).backward(); ce_values.append(float(ce)); kl_values.append(float(kl))
                    norms = visual_gradient_audit(torch, trainable, selected)
                    gradient = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.)
                    if not torch.isfinite(gradient): raise RuntimeError("pilot gradient norm is nonfinite")
                    optimizer.step(); scheduler.step()
                    if any(not torch.equal(selected[name].detach().cpu(), frozen_direct[name]) for name in DIRECT_NAMES): raise RuntimeError("pilot changed a frozen direct projection")
                    return ce_values, kl_values, float(gradient), norms
                ce_values, kl_values, gradient, norms = duty.run(update); updates += 1
                record = {"arm": arm, "epoch": epoch, "update": updates, "example_ids": [r["example_id"] for r in batch],
                          "roles": [r["role"] for r in batch], "ce": ce_values, "kl": kl_values, "gradient_norm": gradient, "visual_gradient_norms": norms,
                          "mps_active_seconds": duty.active_seconds, "mps_idle_seconds": duty.idle_seconds}
                with (output / "steps.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n"); stream.flush(); os.fsync(stream.fileno())
                if updates % 16 == 0: print(f"trained {arm} update {updates}/320 epoch={epoch}", flush=True)
            if epoch in (1, 2, 4):
                checkpoint = output / f"{arm}-epoch-{epoch}.safetensors"
                state = duty.run(lambda: complete_adapter_state(selected))
                save_file(state, str(checkpoint), metadata={"schema": "umi-studio-c5-face-learning-pilot-adapter/1", "plan_sha256": sha256(plan_path),
                    "parent_checkpoint_sha256": plan["asset_sha256"]["models/checkpoint-11625/model.safetensors"], "rank": "16", "alpha": "32.0", "visual_last_layers": "4",
                    "arm": arm, "epoch": str(epoch), "direct_parameters_frozen": "true"})
                results = evaluate(model, epoch, arm)
                decision = {"arm": arm, "epoch": epoch, "checkpoint_sha256": sha256(checkpoint), **comparison(parent_records, results)}
                evaluations.append(decision)
                (output / "diagnostic-decisions.json").write_text(json.dumps(evaluations, indent=2, sort_keys=True) + "\n")
                print(json.dumps(decision), flush=True)
        del optimizer, scheduler, model, selected, trainable, visual, decoder; gc.collect(); torch.mps.empty_cache()
    passing = [r for r in evaluations if r["passes_diagnostic_gate"]]
    matched_epochs = {epoch: {r["arm"]: r for r in evaluations if r["epoch"] == epoch} for epoch in (1, 2, 4)}
    result = {"schema": "umi-studio-c5-face-learning-pilot-decision/1", "plan_sha256": sha256(plan_path), "evaluations": evaluations,
        "selected_diagnostic_checkpoint": min(passing, key=lambda r: (r["mean_wer"], r["epoch"], r["arm"])) if passing else None,
        "matched_epoch_recovered_minus_legacy_mean_wer": {str(epoch): arms[ARMS[1]]["mean_wer"] - arms[ARMS[0]]["mean_wer"] for epoch, arms in matched_epochs.items()},
        "decision": "verify_open_cross_domain_then_scale_training" if passing else "reject_pilot_no_generalized_learning_gain",
        "mps_active_seconds": duty.active_seconds, "mps_idle_seconds": duty.idle_seconds, "maximum_allocated_bytes": duty.maximum_allocated_bytes,
        "confirmation_decoded": False, "miner_changed": False, "promoted": False, "qualification": plan["qualification"]}
    (output / "decision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", type=Path); group.add_argument("--backend-probe", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.backend_probe: probe(args.backend_probe.resolve())
    elif args.output_dir: train(args.plan.resolve(), args.output_dir.resolve())
    else: parser.error("training requires an output directory")


if __name__ == "__main__": main()
