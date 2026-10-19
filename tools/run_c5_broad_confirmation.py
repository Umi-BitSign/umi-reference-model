"""One fixed eligible candidate, complete reserved inference, then blinded scoring.

The cohort is consumed before the first baseline decode. A failed inference
does not release it for a different candidate or another inference invocation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

import c5_broad_native_input as inputs
import c5_broad_semantic_scoring as semantic
from fetch_c5_broad_how2sign import ROOT, artifact, read, require, sha256
from launch_terra_visual_job import identity, now, write

MEDIA = ROOT / "runs/c5-broad-confirmation-media-qualified-v1/plan.json"
CLAIM = MEDIA.with_name("consumption.json")
BUDGET = 512 * 1024**2
SOURCES = (
    "run_c5_broad_confirmation.py",
    "c5_broad_native_input.py",
    "c5_broad_semantic_scoring.py",
    "compare_c5_native112_export.py",
    "c5_native112_parity.py",
    "c5_native112_visual_loader.py",
    "c5_native112_frame_embedder.py",
    "evaluate_c5_words.py",
    "terra_polarity_judge.py",
    "terra_quality_training.py",
    "benchmark_fleurs_dev.py",
)


def sources():
    return {"tools/" + name: sha256(ROOT / "tools" / name) for name in SOURCES}


def bound(descriptor):
    path = ROOT / descriptor["path"]
    require(
        path.resolve() == path and artifact(path) == descriptor, "confirmation bound bytes changed"
    )
    return path


def terminal(directory):
    worker, exited = read(directory / "worker.json"), read(directory / "exit.json")
    require(
        worker["uid"] == os.getuid()
        and type(exited["exit_code"]) is int
        and exited["exit_code"] == 0
        and worker["pid"] == exited["pid"],
        "confirmation needs an exited successful same-worker predecessor",
    )
    return {name: artifact(directory / (name + ".json")) for name in ("worker", "exit")}


def check_runtime(result, plan, export, parity):
    require(
        result.get("schema")
        == plan.get("schema")
        == "umi-studio-native112-full-video-compatibility/1"
        and result.get("complete") is True
        and all(
            result.get(k) is True
            for k in (
                "full_video_crop_pose_feature_token_parity",
                "fresh_linux_detection_per_request",
            )
        )
        and result.get("optimizer_updates") == result.get("new_Terra_requests") == 0
        and plan["export"] == export
        and plan["preprocessed_parity"] == parity
        and plan["case_indices"] == [0, 1]
        and plan["repeats"] == 2
        and len(result["requests"]) == 8
        and len(result["trained_checkpoint_references_independently_reproduced"]) == 2,
        "confirmation requires complete same-candidate native video compatibility",
    )


def admit(export_path, parity_path, runtime_path, preflight_path):
    # The supplemental repair preserves every frozen training/evaluation check.
    from run_c5_http502_audit_corrected import install, verify_audit

    install()
    from probe_c5_native112_full_runtime import admit as admit_parity

    export, model_plan, _, _, parity_authority = admit_parity(export_path, parity_path)
    verify_audit(bound(export["evidence"]["terminal_audit"]))
    runtime_authority = terminal(runtime_path.parent)
    runtime = read(runtime_path)
    runtime_plan_path = bound(runtime["plan"])
    require(
        runtime_path.name == "result.json"
        and runtime_plan_path == runtime_path.with_name("plan.json"),
        "unrelated native compatibility result",
    )
    runtime_plan = read(runtime_plan_path)
    check_runtime(runtime, runtime_plan, artifact(export_path), artifact(parity_path))
    for name, digest in runtime_plan["source_sha256"].items():
        require(
            sha256(ROOT / name) == digest
            and sha256(runtime_path.parent / "source-snapshot" / name) == digest,
            "native compatibility source changed",
        )
    references = [
        read(bound(d)) for d in runtime["trained_checkpoint_references_independently_reproduced"]
    ]
    expected = [
        (device, repeat, case)
        for device in ("CPU", "Metal")
        for repeat in range(2)
        for case in range(2)
    ]
    import numpy as np
    from c5_native112_parity import compare_streams, compare_tokens

    for descriptor, key in zip(runtime["requests"], expected, strict=True):
        request = read(bound(descriptor))
        require(
            (request["device"], request["repeat"], request["case_index"]) == key
            and request["case"] == runtime_plan["cases"][key[2]]
            and request["exact_token_match"] is True,
            "native compatibility request coverage changed",
        )
        reference = references[key[2]]
        compare_tokens(reference["tokens"], request["tokens"])
        require(request["text"] == reference["text"], "native compatibility text changed")
        with (
            np.load(bound(reference["features"]), allow_pickle=False) as a,
            np.load(bound(request["feature_arrays"]), allow_pickle=False) as b,
        ):
            measured = compare_streams(dict(a), dict(b), len(a["pose_features"]))
        require(
            measured == request["feature_comparison"], "native compatibility feature report changed"
        )
    preflight_authority = terminal(preflight_path.parent)
    preflight = read(preflight_path)
    require(
        preflight_path.name == "result.json"
        and preflight.get("schema") == "umi-studio-broad-confirmation-native-inputs/1"
        and preflight.get("complete") is True
        and preflight.get("clips") == 48
        and preflight.get("training_allowed") is False
        and preflight.get("model_or_scorer_execution") is False,
        "confirmation needs all48 native input preflights",
    )
    audit_path = bound(preflight["landmark_audit"])
    records = inputs.bind_records(audit_path)
    media = read(MEDIA)
    landmark_plan = read(bound(read(audit_path)["plan"]))
    require(
        landmark_plan["media"] == artifact(MEDIA)
        and len(media["examples"]) == len(records) == len(preflight["receipts"]) == 48
        and preflight["source_sha256"] == read(audit_path)["source_sha256"],
        "confirmation inputs lost their exact reserved cohort",
    )
    for record, descriptor, example in zip(
        records, preflight["receipts"], media["examples"], strict=True
    ):
        receipt = read(bound(descriptor))
        require(
            record["example"] == example
            and receipt["record"] == record
            and set(receipt["native_inputs"]) == {"224", "112"},
            "confirmation preflight reordered or replaced a clip",
        )
    return (
        {
            "export": artifact(export_path),
            "parity": artifact(parity_path),
            "runtime": artifact(runtime_path),
            "preflight": artifact(preflight_path),
            "parity_authority": parity_authority,
            "runtime_authority": runtime_authority,
            "preflight_authority": preflight_authority,
        },
        model_plan,
        records,
    )


def freeze(path, export, parity, runtime, preflight):
    inputs.owned(path, "runs")
    require(not path.exists(), "confirmation plan must be fresh")
    evidence, model_plan, records = admit(export, parity, runtime, preflight)
    from terra_polarity_judge import protocol

    value = {
        "schema": "umi-studio-broad-one-candidate-confirmation/1",
        "evidence": evidence,
        "media": artifact(MEDIA),
        "model_plan": model_plan,
        "records": records,
        "judge_protocol": protocol(),
        "scoring_recipe": semantic.RECIPE,
        "source_sha256": sources(),
        "CPU_threads": 2,
        "GPU_use": False,
        "optimizer_updates": 0,
        "training_allowed": False,
        "consume_before": "first baseline decode",
        "promotion_authorized": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, value)
    return value


def checked(path):
    inputs.owned(path, "runs")
    plan = read(path)
    require(
        plan["schema"] == "umi-studio-broad-one-candidate-confirmation/1"
        and plan["media"] == artifact(MEDIA)
        and plan["source_sha256"] == sources()
        and plan["scoring_recipe"] == semantic.RECIPE
        and plan["CPU_threads"] == 2
        and plan["GPU_use"] is False
        and plan["training_allowed"] is False,
        "confirmation plan policy, cohort or sources changed",
    )
    evidence, model_plan, records = admit(
        *(bound(plan["evidence"][k]) for k in ("export", "parity", "runtime", "preflight"))
    )
    from terra_polarity_judge import protocol

    require(
        plan["evidence"] == evidence
        and plan["model_plan"] == model_plan
        and plan["records"] == records
        and plan["judge_protocol"] == protocol(),
        "confirmation candidate or scoring protocol changed",
    )
    return plan


def consume(plan_path, output):
    # O_EXCL prevents concurrent candidates and fresh-run retries. Retain on failure.
    CLAIM.parent.mkdir(parents=True, exist_ok=True)
    write(
        CLAIM,
        {
            "schema": "umi-studio-broad-confirmation-consumption/1",
            "plan": artifact(plan_path),
            "output": str(output.relative_to(ROOT)),
            "worker": artifact(output / "worker.json"),
            "media": artifact(MEDIA),
            "consumed_at": now(),
            "replacement_or_tuning_allowed": False,
        },
    )
    directory = os.open(CLAIM.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def check_claim(plan_path, output):
    claim = read(CLAIM)
    require(
        claim["plan"] == artifact(plan_path)
        and claim["media"] == artifact(MEDIA)
        and claim["output"] == str(output.relative_to(ROOT))
        and claim["worker"] == artifact(output / "worker.json")
        and claim["replacement_or_tuning_allowed"] is False,
        "confirmation inference is not the single consumed candidate",
    )
    return artifact(CLAIM)


def guard(output, extra=0):
    import shutil

    paths = list(output.rglob("*")) if output.exists() else []
    require(all(not p.is_symlink() for p in paths), "confirmation output contains aliases")
    used = sum(p.stat().st_size for p in paths if p.is_file())
    require(
        used + extra <= BUDGET and shutil.disk_usage(ROOT).free >= 84 * 1024**3 + BUDGET,
        "confirmation storage cap or physical reserve failed",
    )


def start(output):
    inputs.owned(output, "checkpoints")
    require(not output.exists(), "confirmation stage needs a fresh output")
    guard(output)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))


def require_environment():
    require(
        all(
            os.environ.get(k) == "2"
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        )
        and all(
            os.environ.get(k) == "1"
            for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE")
        ),
        "confirmation requires the pinned offline CPU2 environment",
    )


def infer(plan_path, output):
    plan = checked(plan_path)
    require_environment()
    require(not CLAIM.exists(), "reserved cohort already consumed; no replacement inference")
    start(output)
    code = 1
    try:
        import numpy as np
        import torch
        from c5_native112_parity import check_streams, native_streams
        from c5_native112_visual_loader import load_visual_pair
        from c5_small_input_vision import SmallInputDino
        from compare_c5_native112_export import decode_native, native_translator
        from evaluate_c5_words import original_streams
        from probe_c5_small_input import legacy_models
        from probe_tiny_rgb_capacity import state_digest
        from transformers import ByT5Tokenizer

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        import cv2

        cv2.setNumThreads(1)
        model_plan = plan["model_plan"]
        tokenizer = ByT5Tokenizer.from_pretrained(
            ROOT / model_plan["reference"]["bundle_root"] / "model/models/byt5_base",
            local_files_only=True,
        )
        # Load before consumption; neither loading nor the preflight produces predictions.
        translator = native_translator(model_plan)
        hands, face = legacy_models(model_plan)
        appearance = SmallInputDino(hands.model, face.model, side=224).requires_grad_(False).eval()
        expected = model_plan["pretrained_state_sha256"]
        require(
            state_digest(appearance.state_dict()) == expected, "original224 visual weights changed"
        )
        preflight = read(bound(plan["evidence"]["preflight"]))
        preflight_rows = [read(bound(d)) for d in preflight["receipts"]]
        consume(plan_path, output)
        files, baseline, candidate = [], [], []
        with torch.inference_mode():
            for arm, side in (("baseline", 224), ("candidate", 112)):
                if arm == "candidate":
                    require(
                        state_digest(appearance.state_dict()) == expected,
                        "baseline weights mutated",
                    )
                    del appearance, hands, face
                    gc.collect()
                    embedders = load_visual_pair(
                        bound(plan["evidence"]["export"]), batch_context=nullcontext
                    )
                for index, record in enumerate(plan["records"]):
                    guard(output)
                    values = inputs.read_input(record, side=side)
                    receipt = inputs.array_receipt(values, side)
                    require(
                        receipt == preflight_rows[index]["native_inputs"][str(side)],
                        "confirmation inference pixels, pose or timeline changed after preflight",
                    )
                    if arm == "baseline":
                        streams = original_streams(appearance, values)
                        streams.pop("frame_indices")
                    else:
                        streams = native_streams(embedders, values)
                    check_streams(streams, len(values["pixels"]))
                    started = time.perf_counter()
                    text, tokens = decode_native(
                        translator, tokenizer, model_plan["reference"]["generation"], streams
                    )
                    target = output / f"{arm}-{index:04d}.npz"
                    guard(output, sum(v.nbytes for v in streams.values()) + 65536)
                    np.savez(target, **streams)
                    row = {**record["example"], "hypothesis": text}
                    path = output / f"{arm}-{index:04d}.json"
                    write(
                        path,
                        {
                            "prediction": row,
                            "tokens": tokens,
                            "inputs": receipt,
                            "features": artifact(target),
                            "decode_seconds": time.perf_counter() - started,
                        },
                    )
                    files.append(artifact(path))
                    (baseline if arm == "baseline" else candidate).append(row)
                    del values, streams
                    print(json.dumps({"arm": arm, "clips": index + 1, "of": 48}), flush=True)
        require(plan == checked(plan_path), "confirmation plan changed during inference")
        comparison = semantic.comparison_items(
            [r["example"] for r in plan["records"]], baseline, candidate
        )
        write(
            output / "result.json",
            {
                "schema": "umi-studio-broad-confirmation-inference/1",
                "complete": True,
                "plan": artifact(plan_path),
                "claim": check_claim(plan_path, output),
                "files": files,
                "baseline": baseline,
                "candidate": candidate,
                "comparison": comparison,
                "CPU_threads": 2,
                "GPU_use": False,
                "optimizer_updates": 0,
                "new_Terra_requests": 0,
                "promotion_authorized": False,
            },
        )
        code = 0
    except BaseException as error:
        write(
            output / "failure.json",
            {"error": str(error), "cohort_released": False, "automatic_retries": 0},
        )
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def checked_inference(path, plan_path, plan):
    authority = terminal(path.parent)
    receipt = read(path)
    require(
        path.name == "result.json"
        and receipt.get("complete") is True
        and receipt["schema"] == "umi-studio-broad-confirmation-inference/1"
        and receipt["plan"] == artifact(plan_path)
        and len(receipt["files"]) == 96
        and receipt["claim"] == check_claim(plan_path, path.parent),
        "incomplete or unrelated confirmation inference",
    )
    import numpy as np
    from c5_native112_parity import check_streams
    from transformers import ByT5Tokenizer

    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["model_plan"]["reference"]["bundle_root"] / "model/models/byt5_base",
        local_files_only=True,
    )
    arms = {}
    preflight = read(bound(plan["evidence"]["preflight"]))
    for offset, arm in enumerate(("baseline", "candidate")):
        arms[arm] = []
        for index, record in enumerate(plan["records"]):
            target = bound(receipt["files"][offset * 48 + index])
            require(
                target == path.with_name(f"{arm}-{index:04d}.json"),
                "inference output coverage changed",
            )
            saved = read(target)
            require(
                saved["prediction"]
                == {**record["example"], "hypothesis": saved["prediction"]["hypothesis"]}
                and saved["tokens"]
                and all(type(t) is int and t >= 0 for t in saved["tokens"])
                and tokenizer.decode(saved["tokens"], skip_special_tokens=True)
                == saved["prediction"]["hypothesis"],
                "confirmation tokens, transcript or source identity changed",
            )
            side = 224 if arm == "baseline" else 112
            require(
                saved["inputs"]
                == read(bound(preflight["receipts"][index]))["native_inputs"][str(side)],
                "confirmation inference input receipt changed",
            )
            with np.load(bound(saved["features"]), allow_pickle=False) as arrays:
                check_streams(dict(arrays), saved["inputs"]["header"]["sampled_frames"])
            arms[arm].append(saved["prediction"])
        require(arms[arm] == receipt[arm], "confirmation arm differs from actual outputs")
    comparison = semantic.comparison_items(
        [r["example"] for r in plan["records"]], arms["baseline"], arms["candidate"]
    )
    require(comparison == receipt["comparison"], "confirmation blinded comparison changed")
    return comparison, authority


def score(plan_path, inference_path, output):
    plan = checked(plan_path)
    require_environment()
    comparison, authority = checked_inference(inference_path, plan_path, plan)
    start(output)
    code = 1
    try:
        from benchmark_fleurs_dev import distance, normalize
        from terra_polarity_judge import Pacer

        write(
            CLAIM.with_name("scoring-consumption.json"),
            {
                "plan": artifact(plan_path),
                "inference": artifact(inference_path),
                "worker": artifact(output / "worker.json"),
                "started_at": now(),
                "replacement_or_tuning_allowed": False,
            },
        )
        write(output / "blinded-inputs.json", comparison)
        # Inference completion follows all development API calls. Preserve the original gap.
        exited = read(bound(authority["exit"]))
        ready = datetime.fromisoformat(exited["observed_at"]) + timedelta(seconds=60)
        while (remaining := (ready - datetime.now(timezone.utc)).total_seconds()) > 0:  # noqa: UP017 (Python3.10 runtime)
            time.sleep(min(30, remaining))
        pacer = Pacer()
        for repeat, planned in enumerate(semantic.batches(comparison), start=1):
            joined = [None] * len(comparison["items"])
            for batch, chosen in enumerate(planned):
                guard(output)
                values = pacer.score(
                    [comparison["items"][i] for i in chosen],
                    output / f"pass-{repeat}-batch-{batch:03d}",
                )["scores"]
                require(len(values) == len(chosen), "grader returned incomplete confirmation batch")
                for index, value in zip(chosen, values, strict=True):
                    joined[index] = value
                print(
                    json.dumps({"pass": repeat, "batches": batch + 1, "of": len(planned)}),
                    flush=True,
                )
            write(output / f"pass-{repeat}-scores.json", joined)
        actual = semantic.authenticated_passes(comparison, output, plan["judge_protocol"])
        inference = read(inference_path)
        diagnostic = {}
        for arm in ("baseline", "candidate"):
            pairs = [
                (normalize(r["reference"]), normalize(r["hypothesis"])) for r in inference[arm]
            ]
            edits = sum(distance(a, b) for a, b in pairs)
            words = sum(len(a) for a, _ in pairs)
            diagnostic[arm] = {
                "edits": edits,
                "reference_words": words,
                "corpus_WER": edits / words if words else None,
            }
        statistics = semantic.summarize(comparison, actual, diagnostic)
        require(plan == checked(plan_path), "confirmation admission changed during scoring")
        write(
            output / "result.json",
            {
                **statistics,
                "schema": "umi-studio-broad-confirmation-result/1",
                "plan": artifact(plan_path),
                "inference": artifact(inference_path),
                "inference_authority": authority,
                "claim": artifact(CLAIM),
                "raw_response_authentication_performed": True,
                "candidate_or_runtime_admission_verified": True,
                "complete": True,
                "optimizer_updates": 0,
                "promotion_authorized": False,
                "qualifications": read(MEDIA)["qualifications"],
            },
        )
        code = 0
    except BaseException as error:
        write(
            output / "failure.json",
            {"error": str(error), "cohort_released": False, "automatic_retries": 0},
        )
        raise
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "infer", "score"), required=True)
    for name in ("plan", "export", "parity", "runtime", "preflight", "inference", "output"):
        parser.add_argument("--" + name, type=lambda s: Path(s).absolute())
    args = parser.parse_args()
    require(args.plan is not None, "confirmation requires --plan")
    if args.mode == "freeze":
        require(
            all(getattr(args, k) is not None for k in ("export", "parity", "runtime", "preflight")),
            "freeze requires export, parity, runtime and preflight results",
        )
        freeze(args.plan, args.export, args.parity, args.runtime, args.preflight)
    else:
        require(args.output is not None, "confirmation execution requires --output")
        if args.mode == "infer":
            infer(args.plan, args.output)
        else:
            require(args.inference is not None, "scoring requires --inference")
            score(args.plan, args.inference, args.output)


if __name__ == "__main__":
    main()
