"""Compare an eligible trained export with its checkpoint on two complete training clips.

Runs CPU and Metal visual paths serially, with the published native translator
on CPU. Inputs are verified preprocessed crops, not a detector or serving test.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import export_c5_native112_visual as exporter
from c5_native112_parity import (
    check_input,
    compare_streams,
    compare_tokens,
    native_streams,
    require,
)
from c5_native112_visual_loader import authenticate_export, load_visual_pair

ROOT = Path(__file__).resolve().parents[1]
CASE_INDICES = (0, 1)
BUDGET = 64 * 1024**2


def native_translator(model_plan):
    """Load the original published feature-to-token class, in a private CPU instance."""
    import torch
    from c5_winner_mps_safe import replace_fairseq_exact_gelu, replace_native_layer_norms
    from c5_winner_reference import load_reference
    from probe_tiny_rgb_capacity import state_digest
    from tiny_rgb_mps_safe import replace_native_gelu

    reference = load_reference()
    require(reference == model_plan["reference"], "native translator reference changed")
    root = ROOT / reference["bundle_root"] / "model"
    old_path = list(sys.path)
    old_source = os.environ.get("SHUBERT_DINOV2_SOURCE")
    sys.path[:0] = [str(root / "vendor/demo"), str(root / "vendor")]
    os.environ["SHUBERT_DINOV2_SOURCE"] = str(root / "vendor/dinov2-source")
    try:
        import inference

        require(
            Path(inference.__file__).resolve() == root / "vendor/demo/inference.py",
            "native translator imported an unrelated author module",
        )
        model = inference.SignLanguageByT5ForConditionalGeneration.from_pretrained(
            root / "models/checkpoint-11625", local_files_only=True, use_safetensors=True
        ).eval()
    finally:
        sys.path[:] = old_path
        if old_source is None:
            os.environ.pop("SHUBERT_DINOV2_SOURCE", None)
        else:
            os.environ["SHUBERT_DINOV2_SOURCE"] = old_source
    require(
        (
            replace_fairseq_exact_gelu(model),
            replace_native_gelu(model),
            replace_native_layer_norms(model),
        )
        == (12, 1, 30),
        "native translator numerical topology changed",
    )
    model.requires_grad_(False).eval()
    require(
        state_digest(model.encoder.state_dict())
        == model_plan["initial_reference_hashes"]["encoder"],
        "native translator encoder values differ from the trained checkpoint path",
    )
    require(
        all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()),
        "native translator requires the original frozen CPU FP32 values",
    )
    return model


def decode_native(translator, tokenizer, generation, streams):
    import torch
    from evaluate_tiny_rgb_feature_winner import CpuBudget
    from transformers import StoppingCriteriaList

    with torch.inference_mode():
        tokens = translator.generate(
            **{name: torch.from_numpy(value.copy())[None] for name, value in streams.items()},
            **generation,
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([CpuBudget()]),
        )
    return tokenizer.decode(tokens[0], skip_special_tokens=True), tokens[0].tolist()


def reference_paths(model_plan, initialization, cases):
    import torch
    from c5_mixed_semantic_model import install_visual
    from c5_small_input_semantic import clip_input
    from c5_word_semantic_model import load_student
    from evaluate_roi_short_translation import decode

    model, tokenizer = load_student(model_plan)
    install_visual(model, initialization)
    results = []
    for values in cases:
        started = time.perf_counter()
        clip = clip_input(values)
        mask = torch.ones(1, len(clip.pixels), dtype=torch.bool)
        with torch.inference_mode():
            features, _ = model.predict_features(clip, mask)
            memory = model.semantic_memory(features, mask)
            text, tokens = decode(
                model.language, tokenizer, model_plan["reference"]["generation"], memory
            )
        results.append(
            {
                "features": {k: v[0].cpu().numpy().copy() for k, v in features.items()},
                "text": text,
                "tokens": tokens,
                "seconds": time.perf_counter() - started,
            }
        )
    del model
    gc.collect()
    return results, tokenizer


def compare_runtime(embedders, translator, tokenizer, generation, cases, references):
    results = []
    for values, expected in zip(cases, references, strict=True):
        started = time.perf_counter()
        streams = native_streams(embedders, values)
        features = compare_streams(expected["features"], streams, len(values["pixels"]))
        text, tokens = decode_native(translator, tokenizer, generation, streams)
        token_report = compare_tokens(expected["tokens"], tokens)
        require(
            text == expected["text"], "native parity decoded text differs despite token equality"
        )
        results.append(
            {
                "features": features,
                "tokens": tokens,
                "text": text,
                **token_report,
                "seconds": time.perf_counter() - started,
            }
        )
    return results


def run(export_path, output):
    output = output.absolute()
    require(
        output.resolve() == output
        and output.is_relative_to(ROOT / "checkpoints")
        and output != ROOT / "checkpoints"
        and not output.exists(),
        "native parity requires a fresh owned output",
    )
    record, model_plan, _ = authenticate_export(export_path)
    # Admission is deliberately before Torch/model imports or output creation.
    import numpy as np
    import torch
    from c5_native112_frame_embedder import Native112FrameEmbedder
    from launch_terra_visual_job import identity, now
    from terra_word_verified_inputs import load_verified_input
    from train_terra_c5_tail import gpu_lease

    require(
        all(
            os.environ.get(k) == "2"
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        ),
        "native parity requires the pinned CPU2 environment",
    )
    require(torch.backends.mps.is_available(), "native parity requires both CPU and Metal")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan_path = exporter.bound(record["evidence"]["training_plan"])
    training = exporter.read(plan_path)
    cases_bound = [training["records"][i] for i in CASE_INDICES]
    sources = dict(training["source_sha256"])
    sources.update(
        {
            str(p.relative_to(ROOT)): exporter.sha(p)
            for p in (
                Path(__file__),
                ROOT / "tools/c5_native112_parity.py",
                ROOT / "tools/c5_native112_visual_loader.py",
                ROOT / "tools/c5_native112_frame_embedder.py",
                ROOT / "tools/export_c5_native112_visual.py",
                ROOT / "tools/c5_native112_export_values.py",
            )
        }
    )
    require(
        all(exporter.sha(ROOT / name) == digest for name, digest in sources.items()),
        "native parity frozen dependency changed before execution",
    )

    def guard(extra=0):
        used = (
            sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            if output.exists()
            else 0
        )
        require(
            used + extra <= BUDGET
            and exporter.shutil.disk_usage(ROOT).free >= 80 * 1024**3 + 128 * 1024**2 + BUDGET,
            "native parity storage reserve or physical floor failed",
        )

    def write(name, value):
        data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        guard(len(data))
        with (output / name).open("xb") as f:
            f.write(data)

    guard(BUDGET)
    output.mkdir(parents=True)
    write("worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    exit_code = 1
    try:
        write(
            "plan.json",
            {
                "export": exporter.artifact(export_path),
                "case_indices": list(CASE_INDICES),
                "records": cases_bound,
                "source_sha256": sources,
                "CPU_threads": 2,
                "GPU_duty_fraction": 0.5,
                "optimizer_updates": 0,
                "new_Terra_requests": 0,
                "scope": "verified preprocessed crop-to-token parity, not detector/serving proof",
            },
        )
        for name in sources:
            target = output / "source-snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            guard((ROOT / name).stat().st_size)
            with target.open("xb") as f:
                f.write((ROOT / name).read_bytes())
        with gpu_lease():
            cases = [load_verified_input(case, purpose="training") for case in cases_bound]
            inputs = [check_input(values) for values in cases]
            references, tokenizer = reference_paths(
                model_plan, record["evidence"]["initialization"], cases
            )
            for i, row in enumerate(references):
                target = output / f"reference-features-{i:02d}.npz"
                guard(sum(v.nbytes for v in row["features"].values()) + 65536)
                np.savez(target, **row["features"])
                write(
                    f"reference-{i:02d}.json",
                    {
                        **{k: v for k, v in row.items() if k != "features"},
                        "features": exporter.artifact(target),
                    },
                )
            embedders = load_visual_pair(export_path, batch_context=nullcontext)
            translator = native_translator(model_plan)
            cpu = compare_runtime(
                embedders,
                translator,
                tokenizer,
                model_plan["reference"]["generation"],
                cases,
                references,
            )
            write("CPU.json", cpu)
            resources = {"mps_active_seconds": 0.0, "mps_idle_seconds": 0.0}

            @contextmanager
            def metal_context():
                torch.mps.synchronize()
                started = time.perf_counter()
                try:
                    yield
                finally:
                    torch.mps.synchronize()
                    elapsed = time.perf_counter() - started
                    resources["mps_active_seconds"] += elapsed
                    time.sleep(elapsed)
                    resources["mps_idle_seconds"] += elapsed

            with metal_context():
                for embedder in embedders.values():
                    embedder.model.to("mps")
            metal = {
                name: Native112FrameEmbedder(e.model, batch_context=metal_context, batch_size=2)
                for name, e in embedders.items()
            }
            mps = compare_runtime(
                metal,
                translator,
                tokenizer,
                model_plan["reference"]["generation"],
                cases,
                references,
            )
            write("Metal.json", mps)
            require(inputs == [check_input(v) for v in cases], "native parity input arrays mutated")
            require(
                all(exporter.sha(ROOT / n) == h for n, h in sources.items()),
                "native parity source changed",
            )
            require(
                exporter.artifact(export_path) == exporter.read(output / "plan.json")["export"],
                "native parity export changed",
            )
            exporter.verify_evidence(record)
            write(
                "result.json",
                {
                    "complete": True,
                    "plan": exporter.artifact(output / "plan.json"),
                    "worker": exporter.artifact(output / "worker.json"),
                    "input_arrays": inputs,
                    "CPU": cpu,
                    "Metal": mps,
                    "resources": resources,
                    "source_sha256": sources,
                    "scope": "two fixed training clips; complete stride2 crop-to-token paths",
                    "end_to_end_detector_or_serving_qualified": False,
                    "latency_scope": (
                        "one request per clip/path; no warm-up exclusion; "
                        "includes normalization/features/CPU translation; "
                        "excludes loading/detection/cropping"
                    ),
                    "independent_semantic_confirmation_completed": False,
                    "optimizer_updates": 0,
                    "new_Terra_requests": 0,
                    "promotion_authorized": False,
                },
            )
        exit_code = 0
    except BaseException as error:
        write("failure.json", {"error": str(error), "automatic_retries": 0})
        raise
    finally:
        write("exit.json", {"pid": os.getpid(), "exit_code": exit_code, "observed_at": now()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.export.resolve(), args.output)


if __name__ == "__main__":
    main()
