"""Full96-word/full34-sentence native224 comparison of a qualified temporal-memory final.

All original public controls reproduce before trained memory adapter is installed.
Both complete calibrated semantic comparisons use the unchanged owner gate.
Independent broad confirmation and deployment remain separate requirements.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from c5_timing_development_input import (
    development_row,
    expected_header,
    read_development224,
    source_visibility,
)
from c5_timing_evaluation_protocol import both_families, compare, inventory, summarize
from c5_timing_feature_data import (
    ROOT,
    WIDTHS,
    array_metadata,
    artifact,
    bound,
    check_arrays,
    owned,
    read,
    require,
    sha,
    terminal,
)
from launch_terra_visual_job import identity, now, write

POLICY = {
    "clips": {"word": 96, "sentence": 34},
    "input": "fresh complete native224 pixels and original pose; shared original DINO values",
    "control_order": "all130 public controls before trained memory adapter installation and decode",
    "generation": "original published C5 generation and existing CpuBudget; no reduced horizon",
    "selection": "audited complete final after actual trained native qualification",
    "scoring": "both complete families; two complete calibrated blinded passes each",
    "judge": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "both_original_semantic_research_gates_required": True,
    "WER_is_diagnostic_only": True,
    "CPU_threads": 2,
    "GPU_use": False,
    "optimizer_updates": 0,
    "automatic_retries": 0,
    "new_downloads": 0,
    "maximum_output_bytes": 256 * 1024**2,
    "physical_free_floor_bytes": 84 * 1024**3,
    "independent_confirmation_required": True,
    "C5_upstream_training_inventory_known": False,
    "word_labels_shared_with_training_intentionally": True,
    "promotion_authorized": False,
}


def specification(final_audit_path, native_plan, native_result, native_poll):
    from c5_temporal_memory_final import checked_final
    from qualify_c5_temporal_memory_native import checked_qualification
    from terra_polarity_judge import protocol

    final = checked_final(final_audit_path)
    qualified, _, evidence = checked_qualification(native_plan, native_result, native_poll)
    require(
        qualified["final_audit"] == artifact(final_audit_path)
        and qualified["adapter"] == final["adapter"],
        "memory evaluation native qualification belongs to another final",
    )
    inputs = inventory(final["model_plan"])
    sources = dict(qualified["source_sha256"])
    for name, digest in inputs["source_sha256"].items():
        require(
            name not in sources or sources[name] == digest,
            "memory comparison source ancestry conflicts",
        )
        sources[name] = digest
    for name in (
        "evaluate_c5_temporal_memory.py",
        "c5_timing_evaluation_protocol.py",
        "c5_timing_development_input.py",
    ):
        sources["tools/" + name] = sha(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-full-native224-temporal-memory-evaluation-plan/1",
        "final_audit": artifact(final_audit_path),
        "adapter": final["adapter"],
        "adapter_configuration": final["adapter_configuration"],
        "native_plan": artifact(native_plan),
        "native_result": artifact(native_result),
        "native_terminal_poll": artifact(native_poll),
        "native_evidence": evidence,
        "model_plan": final["model_plan"],
        "inputs": inputs,
        "policy": POLICY,
        "judge_protocol": protocol(),
        "source_sha256": sources,
        "promotion_authorized": False,
    }


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = read(path)
    require(
        plan
        == specification(
            bound(plan["final_audit"]),
            bound(plan["native_plan"]),
            bound(plan["native_result"]),
            bound(plan["native_terminal_poll"]),
        ),
        "memory evaluation changed its exact final, native qualification, inputs or protocol",
    )
    verify_snapshot(path, plan)
    return plan


def freeze(path, final_audit_path, native_plan, native_result, native_poll):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(not path.parent.exists(), "memory evaluation requires a fresh plan")
    plan = specification(final_audit_path, native_plan, native_result, native_poll)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))


def guard(output, extra=0):
    from evaluate_tiny_rgb_feature_winner import peak_bytes

    paths = list(output.rglob("*")) if output.exists() else []
    require(all(not p.is_symlink() for p in paths), "memory evaluation output has aliases")
    used = sum(p.stat().st_size for p in paths if p.is_file())
    remaining = POLICY["maximum_output_bytes"] - used
    require(
        type(extra) is int
        and 0 <= extra <= remaining
        and shutil.disk_usage(ROOT).free
        >= POLICY["physical_free_floor_bytes"] + 128 * 1024**2 + remaining
        and peak_bytes() <= 24 * 1024**3,
        "memory evaluation exceeds its original process or output/storage allowance",
    )


def cache_values(saved, case):
    require(
        saved["row"] == development_row(case) and saved["input"]["header"] == expected_header(case),
        "memory evaluation cache changed its reserved source join",
    )
    with np.load(bound(saved["streams"]), allow_pickle=False) as archive:
        values = dict(archive)
    require(
        check_arrays(values, saved["input"]) == saved["arrays"],
        "memory evaluation cache changed native streams or visibility",
    )
    header, indices, observed, valid = source_visibility(case)
    require(
        header == saved["input"]["header"]
        and values["frame_indices"].tobytes() == indices.tobytes()
        and values["source_hand_tip_visible"].tobytes() == observed.tobytes()
        and values["valid_intervals"].tobytes() == valid.tobytes(),
        "memory evaluation cache differs from full-rate source landmarks",
    )
    if case["kind"] == "word":
        require(
            saved["input"]["arrays"]["pose_features"]
            == case["word"]["verified_input"]["arrays"]["pose_features"],
            "memory evaluation word pose bits changed",
        )
    else:
        teacher = case["sentence"]["development"]["original_pair"]["teacher"]
        with np.load(
            bound({"path": teacher["cache"], "sha256": teacher["cache_sha256"]}), allow_pickle=False
        ) as original:
            require(
                original["pose_features"].tobytes() == values["pose_features"].tobytes(),
                "memory evaluation sentence pose bits changed",
            )
    return {
        "streams": {name: values[name] for name in WIDTHS},
        "valid_intervals": values["valid_intervals"],
    }


def verified_prediction(row, saved, tokenizer):
    from score_roi_short_translation import verified_prediction as original

    return original(row, saved, tokenizer)


def verify_control(row, saved, expected, tokenizer):
    value = verified_prediction(row, saved, tokenizer)
    require(
        expected["row"] == row
        and value["hypothesis"] == expected["hypothesis"]
        and (not expected["token_identity_required"] or saved["tokens"] == expected["tokens"]),
        "memory evaluation did not reproduce an original complete C5 control",
    )
    return value


def adapted_memory(model, values):
    import torch
    from c5_timing_semantic_model import clip_input

    clip = clip_input(values)
    mask = torch.ones(1, len(clip.streams["pose_features"]), dtype=torch.bool)
    details = model.encode_details(clip, mask)
    metadata = {
        "base_memory": array_metadata(details["base_memory"][0].detach().numpy()),
        "normalized_residual_penalty": float(details["normalized_residual_penalty"]),
    }
    return details["memory"], metadata, details["base_memory"][0].detach().numpy()


def infer(path, output, plan):
    import torch
    from c5_small_input_vision import SmallInputDino
    from c5_temporal_memory_candidate import install_adapter, require_candidate
    from c5_temporal_memory_final import checked_final
    from c5_temporal_memory_model import load_student, require_model
    from c5_timing_native_input import input_receipt
    from evaluate_c5_words import original_streams
    from evaluate_roi_short_translation import decode, prediction
    from probe_c5_small_input import legacy_models
    from probe_tiny_rgb_capacity import state_digest

    binding = checked_final(bound(plan["final_audit"]))
    model, tokenizer = load_student(plan["model_plan"], plan["adapter_configuration"])
    require(
        torch.count_nonzero(model.memory_adapter.output_projection.weight).item() == 0,
        "timing public controls require the untouched identity initializer",
    )
    hands, face = legacy_models(plan["model_plan"])
    appearance = SmallInputDino(hands.model, face.model, side=224).requires_grad_(False).eval()
    visual = state_digest(appearance.state_dict())
    require(
        visual == plan["model_plan"]["pretrained_state_sha256"]
        and all(
            p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad
            for p in appearance.parameters()
        ),
        "timing controls changed original public appearance values",
    )
    cases, controls = plan["inputs"]["cases"], plan["inputs"]["controls"]
    files, baseline, predictions = [], [], []
    with torch.no_grad():
        for index, (case, expected) in enumerate(zip(cases, controls, strict=True)):
            guard(output)
            row = development_row(case)
            values = read_development224(case)
            inputs = input_receipt(values)
            streams = original_streams(appearance, values)
            arrays = {
                **streams,
                "valid_intervals": values["valid_intervals"],
                "source_hand_tip_visible": values["source_hand_tip_visible"],
            }
            metadata = check_arrays(arrays, inputs)
            cache = output / f"streams-{index:04d}.npz"
            guard(output, sum(value.nbytes for value in arrays.values()) + 65536)
            with cache.open("xb") as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            feature_values = {
                "streams": {name: streams[name] for name in WIDTHS},
                "valid_intervals": values["valid_intervals"],
            }
            memory, _, base_memory = adapted_memory(model, feature_values)
            memory_cache = output / f"memory-{index:04d}.npy"
            guard(output, base_memory.nbytes + 65536)
            with memory_cache.open("xb") as stream:
                np.save(stream, base_memory, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            text, tokens = decode(
                model.language, tokenizer, plan["model_plan"]["reference"]["generation"], memory
            )
            saved = {
                **prediction(row, text),
                "row": row,
                "kind": case["kind"],
                "tokens": tokens,
                "native_crop_parity": True,
                "input": inputs,
                "streams": artifact(cache),
                "arrays": metadata,
                "base_memory": artifact(memory_cache),
                "base_memory_metadata": array_metadata(base_memory),
            }
            baseline.append(verify_control(row, saved, expected, tokenizer))
            filename = output / f"control-{index:04d}.json"
            write(filename, saved)
            files.append(artifact(filename))
            del values, streams, arrays, feature_values, memory
            print(
                json.dumps({"stage": "memory_original_controls", "clips": index + 1, "of": 130}),
                flush=True,
            )
        require(
            len(files) == 130 and state_digest(appearance.state_dict()) == visual,
            "timing controls omitted inputs or changed public appearance",
        )
        del appearance, hands, face
        gc.collect()
        require_model(model, plan["model_plan"], plan["adapter_configuration"])
        install_adapter(model, binding)
        candidate_files = []
        for index, case in enumerate(cases):
            guard(output)
            control = read(bound(files[index]))
            row = development_row(case)
            verify_control(row, control, controls[index], tokenizer)
            values = cache_values(control, case)
            memory, adapter_details, base_memory = adapted_memory(model, values)
            require(
                array_metadata(base_memory) == control["base_memory_metadata"],
                "candidate changed original semantic memory",
            )
            text, tokens = decode(
                model.language, tokenizer, plan["model_plan"]["reference"]["generation"], memory
            )
            value = {
                **prediction(row, text),
                "tokens": tokens,
                "native_crop_parity": True,
                "row": row,
                "kind": case["kind"],
                "control": files[index],
                "streams": control["streams"],
                "adapter": plan["adapter"],
                "adapter_details": adapter_details,
                "semantic_memory": array_metadata(memory[0].detach().numpy()),
            }
            predictions.append(verified_prediction(row, value, tokenizer))
            filename = output / f"candidate-{index:04d}.json"
            write(filename, value)
            candidate_files.append(artifact(filename))
            require_candidate(model, binding)
            del values, memory
            print(
                json.dumps({"stage": "memory_candidate", "clips": index + 1, "of": 130}), flush=True
            )
    require_model(model, plan["model_plan"], plan["adapter_configuration"])
    require_candidate(model, binding)
    require(
        plan == checked(path), "memory evaluation sources or candidate changed during inference"
    )
    guard(output)
    write(
        output / "inference.json",
        {
            "complete": True,
            "plan": artifact(path),
            "adapter": plan["adapter"],
            "controls": files,
            "candidates": candidate_files,
            "baseline": baseline,
            "predictions": predictions,
            "clips": 130,
            "all_original_controls_reproduced_before_candidate_installation": True,
            "all_original_and_trained_values_unchanged": True,
            "optimizer_updates": 0,
            "new_Terra_requests": 0,
        },
    )


def decoded(plan, output, plan_path):
    import torch
    from c5_temporal_memory_adapter import TemporalMemoryAdapter
    from c5_temporal_memory_final import checked_final
    from train_c5_temporal_memory import state_hashes
    from transformers import ByT5Tokenizer

    binding = checked_final(bound(plan["final_audit"]))
    receipt_path = output / "inference.json"
    receipt = read(receipt_path)
    require(
        receipt["complete"] is True
        and receipt["plan"] == artifact(plan_path)
        and receipt["adapter"] == plan["adapter"]
        and receipt["clips"] == len(receipt["controls"]) == len(receipt["candidates"]) == 130
        and receipt["all_original_controls_reproduced_before_candidate_installation"] is True
        and receipt["all_original_and_trained_values_unchanged"] is True
        and receipt["optimizer_updates"] == receipt["new_Terra_requests"] == 0,
        "memory scoring requires complete fixed-final inference and all public controls",
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["model_plan"]["reference"]["bundle_root"] / "model/models/byt5_base",
        local_files_only=True,
    )
    config = binding["adapter_configuration"]
    adapter = TemporalMemoryAdapter(
        rank=config["rank"],
        maximum_relative_residual=config["maximum_relative_residual"],
        seed=config["seed"],
    )
    require(adapter.configuration() == config, "scoring changed complete adapter configuration")
    saved_adapter = torch.load(bound(binding["adapter"]), map_location="cpu", weights_only=True)
    adapter.load_state_dict(saved_adapter["memory_adapter"], strict=True)
    adapter.requires_grad_(False).eval()
    require(
        state_hashes(adapter) == binding["state_hashes"], "scoring changed audited adapter values"
    )
    baseline, predictions, assets = [], [], [artifact(receipt_path)]
    require(
        {p.name for p in output.glob("control-*.json")}
        == {f"control-{i:04d}.json" for i in range(130)}
        and {p.name for p in output.glob("candidate-*.json")}
        == {f"candidate-{i:04d}.json" for i in range(130)}
        and {p.name for p in output.glob("streams-*.npz")}
        == {f"streams-{i:04d}.npz" for i in range(130)}
        and {p.name for p in output.glob("memory-*.npy")}
        == {f"memory-{i:04d}.npy" for i in range(130)},
        "memory scoring lost exact complete decoded file coverage",
    )
    for index, case in enumerate(plan["inputs"]["cases"]):
        control_path, candidate_path = (
            bound(receipt["controls"][index]),
            bound(receipt["candidates"][index]),
        )
        require(
            control_path == output / f"control-{index:04d}.json"
            and candidate_path == output / f"candidate-{index:04d}.json",
            "memory scoring changed decoded order",
        )
        control, candidate = read(control_path), read(candidate_path)
        require(
            bound(control["streams"]) == output / f"streams-{index:04d}.npz"
            and control["kind"] == case["kind"],
            "memory scoring changed original stream/input ordering",
        )
        row = development_row(case)
        baseline.append(verify_control(row, control, plan["inputs"]["controls"][index], tokenizer))
        values = cache_values(control, case)
        memory_path = bound(control["base_memory"])
        require(
            memory_path == output / f"memory-{index:04d}.npy",
            "scoring changed base memory coverage",
        )
        base_memory = np.load(memory_path, allow_pickle=False)
        require(
            base_memory.dtype == np.float32
            and base_memory.shape == (len(values["valid_intervals"]) + 1, 1536)
            and np.isfinite(base_memory).all()
            and array_metadata(base_memory) == control["base_memory_metadata"],
            "scoring changed complete original memory",
        )
        with torch.no_grad():
            adapted = adapter(
                torch.from_numpy(base_memory)[None],
                valid_intervals=torch.from_numpy(values["valid_intervals"]),
            )
        expected_details = {
            "base_memory": array_metadata(base_memory),
            "normalized_residual_penalty": float(adapted["normalized_residual_penalty"]),
        }
        require(
            candidate["row"] == row
            and candidate["kind"] == case["kind"]
            and candidate["control"] == receipt["controls"][index]
            and candidate["streams"] == control["streams"]
            and candidate["adapter"] == plan["adapter"]
            and candidate["adapter_details"] == expected_details
            and candidate["semantic_memory"] == array_metadata(adapted["memory"][0].numpy()),
            "memory scoring changed actual candidate/input/learned memory evidence",
        )
        predictions.append(verified_prediction(row, candidate, tokenizer))
        assets.extend(
            (
                artifact(control_path),
                artifact(candidate_path),
                control["streams"],
                control["base_memory"],
            )
        )
    require(
        baseline == receipt["baseline"] and predictions == receipt["predictions"],
        "memory inference summary differs from decoded files",
    )
    return baseline, predictions, assets


def score(path, output, plan):
    from evaluate_c5_polarity_words import authenticated_passes
    from terra_polarity_judge import Pacer
    from terra_short_curriculum_learning import score_comparison

    baseline, predictions, assets = decoded(plan, output, path)
    results, requests = {}, 0
    pacer = Pacer()
    for kind in ("word", "sentence"):
        indices = [i for i, case in enumerate(plan["inputs"]["cases"]) if case["kind"] == kind]
        rows = [development_row(plan["inputs"]["cases"][i]) for i in indices]
        original, candidate = [baseline[i] for i in indices], [predictions[i] for i in indices]
        paired = compare(kind, rows, original, candidate)
        directory = output / kind
        directory.mkdir()
        write(directory / "comparison.json", paired)
        measured = score_comparison(paired, directory / "semantic", pacer)
        actual = authenticated_passes(paired, directory / "semantic")
        require(actual == measured, "timing scores differ from complete actual raw passes")
        result = summarize(kind, paired, actual, original, candidate)
        if not result["calibration_passed"]:
            write(
                directory / "calibration-failure.json",
                {
                    "kind": kind,
                    "controls": [
                        [values[c["index"]] for c in paired["controls"]] for values in actual
                    ],
                    "model_quality_available": False,
                },
            )
            raise ValueError("memory evaluation failed original semantic calibration")
        write(directory / "decision.json", result)
        results[kind] = result
        requests += 2 * ((len(paired["items"]) + 15) // 16)
        print(
            json.dumps({"stage": "timing_semantic_decision", "kind": kind, "result": result}),
            flush=True,
        )
        guard(output)
    require(plan == checked(path), "memory evaluation protocol changed during scoring")
    require(
        (baseline, predictions, assets) == decoded(plan, output, path),
        "memory evaluation evidence changed during scoring",
    )
    write(
        output / "decision.json",
        {
            **both_families(results["word"], results["sentence"]),
            "plan": artifact(path),
            "adapter": plan["adapter"],
            "inference": artifact(output / "inference.json"),
            "inference_assets": assets,
            "optimizer_updates": 0,
            "new_Terra_requests": requests,
            "independent_confirmation_complete": False,
        },
    )


def checked_decision(plan_path, decision_path, poll_path):
    """Authenticate a complete positive or negative comparison without C5 loading."""
    from evaluate_c5_polarity_words import authenticated_passes

    owned(plan_path, "runs")
    owned(decision_path, "checkpoints")
    owned(poll_path, "runs")
    plan = checked(plan_path)
    output = decision_path.parent
    authority = terminal(output)
    worker, poll = read(bound(authority["worker.json"])), read(poll_path)
    argv = worker["argv"]
    flag = "--plan" if "--plan" in argv else "--freeze"
    require(
        decision_path.name == "decision.json"
        and poll["tool"] == "write_stdin"
        and type(poll["requested_session_id"]) is int
        and poll["actual_owned_pid"] == worker["pid"]
        and type(poll["tool_result"].get("exit_code")) is int
        and poll["tool_result"]["exit_code"] == 0
        and poll["tool_result"].get("session_id") is None
        and Path(argv[1]).absolute() == ROOT / "tools/evaluate_c5_temporal_memory.py"
        and Path(argv[argv.index(flag) + 1]).absolute() == plan_path
        and Path(argv[argv.index("--output") + 1]).absolute() == output,
        "timing quality requires actual terminal worker/program/plan evidence",
    )
    claim_path = plan_path.with_name("execution-claim.json")
    claim = read(claim_path)
    require(
        claim["plan"] == artifact(plan_path) and claim["worker"] == authority["worker.json"],
        "timing quality lost its actual single execution claim",
    )
    baseline, predictions, assets = decoded(plan, output, plan_path)
    expected_files = {
        output / name for name in ("worker.json", "exit.json", "inference.json", "decision.json")
    }
    expected_files.update(bound(item) for item in assets)
    results, requests = {}, 0
    for kind in ("word", "sentence"):
        selected = [i for i, case in enumerate(plan["inputs"]["cases"]) if case["kind"] == kind]
        rows = [development_row(plan["inputs"]["cases"][i]) for i in selected]
        original, candidate = [baseline[i] for i in selected], [predictions[i] for i in selected]
        paired = compare(kind, rows, original, candidate)
        directory = output / kind
        require(
            read(directory / "comparison.json") == paired,
            "timing quality paired comparison changed",
        )
        passes = authenticated_passes(paired, directory / "semantic")
        result = summarize(kind, paired, passes, original, candidate)
        require(
            result["calibration_passed"] and read(directory / "decision.json") == result,
            "timing quality decision differs from complete authentic calibrated scores",
        )
        results[kind] = result
        count = (len(paired["items"]) + 15) // 16
        requests += 2 * count
        expected_files.update(directory / name for name in ("comparison.json", "decision.json"))
        expected_files.add(directory / "semantic/blinded-inputs.json")
        for repeat in (1, 2):
            expected_files.add(directory / f"semantic/pass-{repeat}-scores.json")
            for batch in range(count):
                expected_files.update(
                    directory / f"semantic/pass-{repeat}-batch-{batch:03d}" / name
                    for name in ("request.json", "started.json", "response.json", "scores.json")
                )
    expected = {
        **both_families(results["word"], results["sentence"]),
        "plan": artifact(plan_path),
        "adapter": plan["adapter"],
        "inference": artifact(output / "inference.json"),
        "inference_assets": assets,
        "optimizer_updates": 0,
        "new_Terra_requests": requests,
        "independent_confirmation_complete": False,
    }
    require(
        read(decision_path) == expected,
        "timing overall decision changed actual complete family results",
    )
    actual = list(output.rglob("*"))
    require(
        all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in actual)
        and {p for p in actual if p.is_file()} == expected_files,
        "timing quality contains partial, extra or aliased output evidence",
    )
    return (
        plan,
        expected,
        [
            artifact(plan_path),
            artifact(decision_path),
            artifact(poll_path),
            artifact(claim_path),
            *authority.values(),
            *(artifact(p) for p in sorted(expected_files)),
        ],
    )


def run(path, output):
    import cv2
    import torch
    from prepare_c5_timing_features import require_environment
    from train_terra_c5_tail import gpu_lease

    require_environment()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    plan = checked(path)
    owned(output, "checkpoints")
    require(not output.exists(), "memory evaluation requires a fresh output")
    guard(output)
    output.mkdir(parents=True)
    write(output / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        write(
            path.with_name("execution-claim.json"),
            {
                "plan": artifact(path),
                "worker": artifact(output / "worker.json"),
                "claimed_at": now(),
            },
        )
        with gpu_lease():
            infer(path, output, plan)
        gc.collect()
        score(path, output, plan)
        code = 0
    finally:
        write(output / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "freeze",
        "plan",
        "output",
        "final-audit",
        "native-plan",
        "native-result",
        "native-terminal-poll",
    ):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute())
    args = parser.parse_args()
    if (
        args.freeze
        and all((args.final_audit, args.native_plan, args.native_result, args.native_terminal_poll))
        and not args.plan
    ):
        freeze(
            args.freeze,
            args.final_audit,
            args.native_plan,
            args.native_result,
            args.native_terminal_poll,
        )
        if args.output:
            run(args.freeze, args.output)
    elif args.plan and args.output and not args.freeze:
        run(args.plan, args.output)
    else:
        parser.error(
            "freeze from a complete final/native qualification, or run one full frozen comparison"
        )


if __name__ == "__main__":
    main()
