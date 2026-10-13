"""Train the full protected RGB corpus, with serial resolution arms and fixed quality gates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import distance, normalize, sha256
from c5_full_coverage_mps import comparison
from tiny_rgb_fit import ROOT, SIDES, RgbDutyCycle, gradient_receipt, metal_working_set_bytes
from tiny_rgb_full_corpus import FULL_RECIPE, read_rows
from tiny_rgb_full_run import checked_full_plan, lr_factor, training_inventory
from tiny_rgb_full_training_tools import batched_gradient_receipt, masked_length_buckets
from tiny_rgb_pixel_cache import PixelCache


def append(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class Arm:
    def __init__(self, torch, plan, side, tokenizer, labels, cache, duty):
        from tiny_rgb_language_model import TinyRgbLanguageTranslator
        from tiny_rgb_model import TinyRgbConfig
        from tiny_rgb_mps_safe import replace_native_gelu
        from train_tiny_rgb_fit import initialization_digest
        from transformers import T5ForConditionalGeneration

        self.torch, self.plan, self.side = torch, plan, side
        self.tokenizer, self.labels, self.cache, self.duty = tokenizer, labels, cache, duty
        text = T5ForConditionalGeneration.from_pretrained(
            ROOT / plan["language_path"], local_files_only=True
        )
        torch.manual_seed(plan["seed"])
        self.model = TinyRgbLanguageTranslator(text, TinyRgbConfig(**plan["visual_config"]))
        if replace_native_gelu(self.model) != 6:
            raise RuntimeError("full RGB activation topology differs")
        self.digest = initialization_digest(self.model)
        if self.digest != plan["numerical_backend"]["visual_initialization_sha256"]:
            raise RuntimeError("full RGB initialization differs from the numerical proof")
        self.counts = self.model.parameter_counts()
        duty.run(lambda: self.model.to("mps"))
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not n.startswith("language_model.")
                    ],
                    "lr": FULL_RECIPE["visual_learning_rate"],
                },
                {
                    "params": list(self.model.language_model.parameters()),
                    "lr": FULL_RECIPE["language_learning_rate"],
                },
            ],
            weight_decay=FULL_RECIPE["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda update: lr_factor(update, plan["planned_updates_per_arm"])
        )
        self.guard = gradient_receipt

    def prepare(self, row, role, cache_allowed=True):
        pixels, _, _, hit = self.cache.get(
            row, self.side, expected_role=role, cache_allowed=cache_allowed
        )
        from tiny_rgb_data import pixel_tensor

        video, mask, label = masked_length_buckets(
            pixel_tensor(pixels)[None],
            self.labels.get((row["source"], row["example_id"])),
            self.torch,
        )
        return video, label, hit, mask

    def tensors(self, sample):
        video, label, _, mask = sample
        video = video.to("mps")
        return video, mask.to("mps"), label.to("mps") if label is not None else None

    def backward(self, sample, divisor=1):
        self.torch.mps.synchronize()
        self.torch.mps.empty_cache()
        video, mask, label = self.tensors(sample)
        result = self.model(video, mask, label)
        if not self.torch.isfinite(result.loss):
            raise RuntimeError("full RGB loss is nonfinite")
        (result.loss / divisor).backward()
        loss = float(result.loss)
        del result, video, mask, label
        self.torch.mps.synchronize()
        self.torch.mps.empty_cache()
        return loss

    def preflight(self, rows, output):
        self.model.eval()
        # Reserve the future AdamW moments without taking an optimizer step.
        reserve = self.duty.run(lambda: self.torch.empty(self.counts["total"] * 2, device="mps"))
        for row in rows:
            sample = self.prepare(row, "training")
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.duty.run(lambda sample=sample: self.backward(sample))
            original = self.duty.run(lambda: gradient_receipt(self.model, self.torch))
            batched = self.duty.run(lambda: batched_gradient_receipt(self.model, self.torch))
            if original != batched:
                raise RuntimeError("full RGB batched gradient guard changed coverage")
            append(
                output / "preflight.jsonl",
                {
                    "short_side": self.side,
                    "example_id": row["example_id"],
                    "loss": loss,
                    "real_frames": int(sample[3].sum()),
                    "padded_frames": sample[0].shape[1],
                    "pixel_shape": list(sample[0].shape[-2:]),
                    "optimizer_updates": 0,
                    "gradient_coverage": batched,
                    "optimizer_moment_memory_reserved_bytes": self.counts["total"] * 8,
                    "visual_initialization_sha256": self.digest,
                    **self.duty.receipt(),
                },
            )
        timings = {"reference": [], "batched": []}
        for repeat in range(4):
            order = ("reference", "batched") if repeat % 2 == 0 else ("batched", "reference")
            for name in order:
                started = self.duty.active_seconds
                function = gradient_receipt if name == "reference" else batched_gradient_receipt
                self.duty.run(lambda function=function: function(self.model, self.torch))
                timings[name].append(self.duty.active_seconds - started)
        before, after = (statistics.median(timings[n]) for n in ("reference", "batched"))
        self.guard = batched_gradient_receipt if after <= 0.95 * before else gradient_receipt
        receipt = {
            "short_side": self.side,
            "gradient_guard_times": timings,
            "gradient_guard_relative_time_reduction": 1 - after / before,
            "use_batched_guard": self.guard is batched_gradient_receipt,
        }
        sample = self.prepare(rows[0], "training")

        def condition():
            video, mask, label = self.tensors(sample)
            with self.torch.no_grad():
                normal_result = self.model(video, mask, label)
                normal = normal_result.logits
                frames = int(mask.sum())
                tokens = int((label != -100).sum())
                plain = self.model(video[:, :frames], mask[:, :frames], label[:, :tokens])
                padding_difference = float((plain.logits - normal[:, :tokens]).abs().max())
                padding_loss_difference = float((plain.loss - normal_result.loss).abs())
                altered = self.model(-video, mask, label).logits
                difference = float((normal - altered).abs().max())
            if (
                not math.isfinite(padding_difference)
                or not math.isfinite(padding_loss_difference)
                or padding_difference > 1e-3
                or padding_loss_difference > 5e-4
            ):
                raise RuntimeError("full RGB masked padding changed valid Metal outputs")
            if not math.isfinite(difference) or difference <= 1e-6:
                raise RuntimeError("full RGB pixels do not condition the language decoder")
            return {
                "negated_pixel_maximum_logit_change": difference,
                "masked_padding_maximum_logit_difference": padding_difference,
                "masked_padding_loss_difference": padding_loss_difference,
                "padding_probe_real_frames": frames,
                "padding_probe_real_tokens": tokens,
            }

        receipt.update(self.duty.run(condition))
        append(
            output / "preflight.jsonl", {**receipt, "optimizer_updates": 0, **self.duty.receipt()}
        )
        del reserve
        self.optimizer.zero_grad(set_to_none=True)
        return receipt

    def update(self):
        coverage = self.guard(self.model, self.torch)
        norm = float(
            self.torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), FULL_RECIPE["gradient_clip"]
            )
        )
        if not math.isfinite(norm):
            raise RuntimeError("full RGB aggregate gradient norm is nonfinite")
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {
            "gradient_norm": norm,
            "gradient_coverage": coverage,
            "learning_rates": [r["lr"] for r in self.optimizer.param_groups],
        }

    def evaluate(self, exposures, output):
        self.model.eval()
        rows = []
        for row in self.plan["evaluation"]:
            sample = self.prepare(row, "open_development")

            def predict(sample=sample):
                video, mask, _ = self.tensors(sample)
                with self.torch.no_grad():
                    tokens = self.model.generate(video, mask, **FULL_RECIPE["generation"])
                return self.tokenizer.decode(tokens[0].cpu(), skip_special_tokens=True)

            hypothesis = self.duty.run(predict)
            self.torch.mps.synchronize()
            self.torch.mps.empty_cache()
            reference = normalize(row["reference"])
            edits = distance(reference, normalize(hypothesis))
            result = {
                **{k: row[k] for k in ("example_id", "domain", "signer", "source_group")},
                "role": "open_development",
                "training_allowed": False,
                "model": "custom_rgb",
                "short_side": self.side,
                "checkpoint_examples": exposures,
                "hypothesis": hypothesis,
                "edits": edits,
                "reference_words": len(reference),
                "wer": edits / len(reference),
            }
            rows.append(result)
            append(output / "evaluation-results.jsonl", result)
        return rows


def main() -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        raise ValueError("use the early full RGB launcher")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = checked_full_plan(args.plan)
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("full RGB training needs a new checkpoints/ directory")
    import torch
    from safetensors.torch import save_model
    from transformers import AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("full RGB training requires native Metal")
    torch.mps.set_per_process_memory_fraction(
        FULL_RECIPE["maximum_process_mps_bytes"] / metal_working_set_bytes()
    )
    output.mkdir(parents=True)
    duty = RgbDutyCycle(torch, ROOT / plan["gpu_budget_policy"])
    cache = PixelCache(
        ROOT / plan["data_dir"], sha256(args.plan), read_rows(ROOT / plan["headers"])
    )
    tokenizer = AutoTokenizer.from_pretrained(ROOT / plan["language_path"], local_files_only=True)
    training = training_inventory(plan)
    by_id = {(r["source"], r["example_id"]): r for r in training}
    labels = {
        (r["source"], r["example_id"]): tokenizer(
            r["reference"], return_tensors="pt", truncation=False
        ).input_ids
        for r in training
    }
    if any(t.shape[1] > FULL_RECIPE["max_text_tokens"] for t in labels.values()):
        raise ValueError("full RGB target changed beyond the audited bound")
    headers = [r for r in read_rows(ROOT / plan["headers"]) if r["role"] == "training"]
    probes = {
        min(range(len(headers)), key=lambda i: headers[i]["sampled_frames"]),
        max(range(len(headers)), key=lambda i: headers[i]["sampled_frames"]),
        max(
            range(len(headers)),
            key=lambda i: headers[i]["sampled_frames"] * math.prod(headers[i]["source_shape"]),
        ),
    }
    probe_rows = [by_id[(headers[i]["source"], headers[i]["example_id"])] for i in sorted(probes)]
    preflights = {}
    for side in SIDES:
        arm = Arm(torch, plan, side, tokenizer, labels, cache, duty)
        preflights[side] = arm.preflight(probe_rows, output)
        del arm
        gc.collect()
        torch.mps.empty_cache()
    run = {
        "schema": "umi-studio-tiny-rgb-full-execution/1",
        "plan_sha256": sha256(args.plan),
        "preflights": preflights,
        "gpu_resource_policy": duty.budget.record,
        "miner_mode": plan["miner_mode"],
        "miner_health_claim": False,
        "training_examples_per_arm": len(training),
        "exposures_per_arm": 48800,
        "initial_diagnostic_weights_reused": False,
    }
    output.joinpath("run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    schedule = read_rows(args.plan.with_name("schedule.jsonl"))
    repeats = Counter((r["source"], r["example_id"]) for r in schedule)
    controls = plan["fixed_controls"]
    decisions = []
    for side in SIDES:
        arm = Arm(torch, plan, side, tokenizer, labels, cache, duty)
        if preflights[side]["use_batched_guard"]:
            arm.guard = batched_gradient_receipt
        # Zero-update numerical and conditioning proofs are supplied by preflight.
        # Keep all five trained checkpoint evaluations and fixed controls unchanged.
        seen = set()
        for start in range(0, len(schedule), FULL_RECIPE["gradient_accumulation"]):
            arm.model.train()
            group = schedule[start : start + FULL_RECIPE["gradient_accumulation"]]
            losses = []
            for row in group:
                identity = row["source"], row["example_id"]
                sample = arm.prepare(
                    row,
                    "training",
                    repeats[identity] >= plan["cache_policy"]["training_minimum_repeats"],
                )
                losses.append(
                    duty.run(
                        lambda sample=sample, arm=arm, group=group: arm.backward(sample, len(group))
                    )
                )
                seen.add(identity)
            gradient = duty.run(arm.update)
            exposures = start + len(group)
            append(
                output / "steps.jsonl",
                {
                    "short_side": side,
                    "exposures": exposures,
                    "update": exposures // 4,
                    "unique_training_examples_seen": len(seen),
                    "mean_loss": statistics.mean(losses),
                    **gradient,
                    **cache.receipt(),
                    **duty.receipt(),
                },
            )
            if exposures % 512 == 0:
                print(
                    json.dumps(
                        {
                            "short_side": side,
                            "exposures": exposures,
                            "unique_training_examples_seen": len(seen),
                            "mean_loss": statistics.mean(losses),
                            **cache.receipt(),
                            **duty.receipt(),
                        }
                    ),
                    flush=True,
                )
            if exposures not in FULL_RECIPE["checkpoint_after_exposures"]:
                continue
            cache.check_storage(arm.counts["total"] * 4 + 1024**2)
            checkpoint = output / f"rgb-cap{side}-after-{exposures:06d}.safetensors"
            duty.run(lambda checkpoint=checkpoint, arm=arm: save_model(arm.model, str(checkpoint)))
            predictions = arm.evaluate(exposures, output)
            decision = comparison(controls["selected_parent"], predictions)
            original = comparison(controls["original_checkpoint_common_inputs"], predictions)
            original_guard = all(
                r["mean_wer"] <= r["parent_mean_wer"]
                and r["word_weighted_wer"] <= r["parent_word_weighted_wer"]
                for r in original["domains"].values()
            )
            decision.update(
                {
                    "short_side": side,
                    "exposures": exposures,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": sha256(checkpoint),
                    "vs_original_common_inputs": original,
                    "passes_original_baseline_guard": original_guard,
                    "eligible_for_new_independent_confirmation": decision[
                        "passes_open_development_gate"
                    ]
                    and original_guard,
                    "promoted": False,
                    "unique_training_examples_seen": len(seen),
                }
            )
            decisions.append(decision)
            output.joinpath("open-decisions.json").write_text(
                json.dumps(decisions, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(decision), flush=True)
        if len(seen) != 7529:
            raise RuntimeError("full RGB arm did not cover every unique eligible clip")
        del arm
        gc.collect()
        torch.mps.empty_cache()
    checked_full_plan(args.plan)
    final = {
        "schema": "umi-studio-tiny-rgb-full-decision/1",
        "complete": True,
        "plan_sha256": sha256(args.plan),
        "evaluations": decisions,
        "candidate_for_new_confirmation": any(
            r["eligible_for_new_independent_confirmation"] for r in decisions
        ),
        "selected_model_changed": False,
        "miner_modified": False,
        "confirmation_decoded": False,
        **cache.receipt(),
        **duty.receipt(),
        "qualification": (
            "Completed open source-held-out comparison only. No automatic promotion. A newly "
            "frozen independent confirmation and final runtime verification remain required."
        ),
    }
    output.joinpath("decision.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"complete": True, "decision_sha256": sha256(output / "decision.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
