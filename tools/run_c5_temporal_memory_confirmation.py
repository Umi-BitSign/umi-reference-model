"""One locally qualified temporal-memory final on the untouched broad confirmation set.

The common cohort claim is consumed before the first original decode. Native224
appearance is shared unchanged; only the audited local temporal-memory adapter differs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import Counter
from pathlib import Path

import c5_broad_native_input as native
import c5_broad_semantic_scoring as semantic
import c5_timing_broad_input as inputs
import numpy as np
import run_c5_broad_confirmation as broad
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
from evaluate_c5_temporal_memory import adapted_memory
from launch_terra_visual_job import now, write


def specification(development_plan, development_result, development_poll, preflight_path):
    from evaluate_c5_temporal_memory import checked_decision
    from terra_polarity_judge import protocol

    # Neither reserved sources nor models are accessed before both local gates.
    development, decision, evidence = checked_decision(
        development_plan, development_result, development_poll
    )
    require(
        decision["passes_both_semantic_research_gates"] is True,
        "memory confirmation requires both complete calibrated development gains",
    )
    preflight = read(preflight_path)
    preflight_authority = terminal(preflight_path.parent)
    require(
        preflight_path.name == "result.json"
        and preflight["schema"] == "umi-studio-broad-confirmation-native-inputs/1"
        and preflight["complete"] is True
        and preflight["clips"] == len(preflight["receipts"]) == 48
        and preflight["training_allowed"] is False
        and preflight["model_or_scorer_execution"] is False
        and preflight["candidate_bound_by_preflight"] is False
        and preflight["confirmation_consumed_by_preflight"] is False,
        "memory confirmation requires the complete original native input preflight",
    )
    audit_path = bound(preflight["landmark_audit"])
    records = native.bind_records(audit_path)
    audit = read(audit_path)
    media = read(broad.MEDIA)
    landmark_plan = read(bound(audit["plan"]))
    require(
        landmark_plan["media"] == artifact(broad.MEDIA)
        and [r["example"] for r in records] == media["examples"]
        and len(records) == 48
        and Counter(r["example"]["family"] for r in records) == {"how2sign": 24, "openasl": 24}
        and len({r["example"]["source_group"] for r in records}) == 36
        and preflight["source_sha256"] == audit["source_sha256"],
        "memory confirmation changed the sealed cohort or its complete input audit",
    )
    training_plan = read(bound(read(bound(development["final_audit"]))["plan"]))
    trained_groups = {case["row"]["source_group"] for case in training_plan["records"]}
    require(
        len(training_plan["records"]) == 532
        and all(
            case["row"]["role"] == "training" and case["row"]["training_allowed"] is True
            for case in training_plan["records"]
        )
        and not trained_groups.intersection(media["reserved_source_groups"]),
        "memory training overlaps a primary or fallback confirmation source group",
    )
    preflight_assets = [
        artifact(preflight_path),
        *preflight_authority.values(),
        artifact(audit_path),
    ]
    for index, (record, descriptor) in enumerate(zip(records, preflight["receipts"], strict=True)):
        path = bound(descriptor)
        saved = read(path)
        require(
            path == preflight_path.with_name(f"input-{index:04d}.json")
            and saved["record"] == record
            and set(saved["native_inputs"]) == {"224", "112"},
            "memory confirmation preflight reordered or replaced a reserved source",
        )
        inputs.confirmation_row(record)
        preflight_assets.append(descriptor)
    sources = dict(development["source_sha256"])
    for name, digest in {**audit["source_sha256"], **broad.sources()}.items():
        require(
            name not in sources or sources[name] == digest, "confirmation source ancestry conflicts"
        )
        sources[name] = digest
    for name in ("run_c5_temporal_memory_confirmation.py", "c5_timing_broad_input.py"):
        sources["tools/" + name] = sha(ROOT / "tools" / name)
    return {
        "schema": "umi-studio-broad-native224-temporal-memory-confirmation-plan/1",
        "development_plan": artifact(development_plan),
        "development_result": artifact(development_result),
        "development_terminal_poll": artifact(development_poll),
        "development_evidence": evidence,
        "evidence": {"preflight": artifact(preflight_path)},
        "preflight_assets": preflight_assets,
        "final_audit": development["final_audit"],
        "adapter": development["adapter"],
        "adapter_configuration": development["adapter_configuration"],
        "media": artifact(broad.MEDIA),
        "model_plan": development["model_plan"],
        "records": records,
        "judge_protocol": protocol(),
        "scoring_recipe": semantic.RECIPE,
        "source_sha256": sources,
        "CPU_threads": 2,
        "GPU_use": False,
        "optimizer_updates": 0,
        "training_allowed": False,
        "consume_before": "first baseline decode",
        "promotion_authorized": False,
    }


def checked(path):
    from terra_training_recovery import verify_snapshot

    plan = read(path)
    require(
        plan
        == specification(
            *(
                bound(plan[k])
                for k in ("development_plan", "development_result", "development_terminal_poll")
            ),
            bound(plan["evidence"]["preflight"]),
        ),
        "memory confirmation changed its exact candidate, local gates, native inputs or protocol",
    )
    verify_snapshot(path, plan)
    return plan


def freeze(path, development_plan, development_result, development_poll, preflight_path):
    from c5_experiment_sources import archive_sources

    owned(path, "runs")
    require(
        not path.parent.exists() and not broad.CLAIM.exists(),
        "confirmation needs a fresh unconsumed plan",
    )
    plan = specification(development_plan, development_result, development_poll, preflight_path)
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))


def prediction(record, saved, tokenizer):
    tokens = saved["tokens"]
    require(
        isinstance(tokens, list)
        and 1 <= len(tokens) <= 2048
        and all(type(v) is int and 0 <= v < 384 for v in tokens)
        and saved["native_crop_parity"] is True
        and saved["prediction"]
        == {
            **inputs.confirmation_row(record),
            "hypothesis": tokenizer.decode(tokens, skip_special_tokens=True),
        },
        "memory confirmation lost actual native tokens, text or reserved source roles",
    )
    return saved["prediction"]


def cache_values(saved, record, preflight):
    with np.load(bound(saved["streams"]), allow_pickle=False) as archive:
        values = dict(archive)
    require(
        check_arrays(values, saved["input"]) == saved["arrays"],
        "confirmation native stream arrays changed",
    )
    header, indices, observed, valid = inputs.source_visibility(record)
    original = preflight["native_inputs"]["224"]
    require(
        preflight["record"] == record
        and saved["input"]["header"] == header == original["header"]
        and all(
            saved["input"]["arrays"][name] == original["arrays"][name]
            for name in ("pixels", "pose_features", "frame_indices")
        )
        and values["frame_indices"].tobytes() == indices.tobytes()
        and values["source_hand_tip_visible"].tobytes() == observed.tobytes()
        and values["valid_intervals"].tobytes() == valid.tobytes(),
        "confirmation changed original native pixels, pose, source grid or full-rate visibility",
    )
    return {
        "streams": {name: values[name] for name in WIDTHS},
        "valid_intervals": values["valid_intervals"],
    }


def infer(plan_path, output):
    import cv2
    import torch
    from c5_small_input_vision import SmallInputDino
    from c5_temporal_memory_candidate import install_adapter, require_candidate
    from c5_temporal_memory_final import checked_final
    from c5_temporal_memory_model import load_student, require_model
    from c5_timing_native_input import input_receipt
    from evaluate_c5_words import original_streams
    from evaluate_roi_short_translation import decode
    from probe_c5_small_input import legacy_models
    from probe_tiny_rgb_capacity import state_digest
    from train_terra_c5_tail import gpu_lease

    broad.require_environment()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    plan = checked(plan_path)
    require(not broad.CLAIM.exists(), "reserved cohort already consumed; no replacement inference")
    broad.start(output)
    code = 1
    try:
        with gpu_lease():
            binding = checked_final(bound(plan["final_audit"]))
            require(binding["adapter"] == plan["adapter"], "confirmation final adapter changed")
            model, tokenizer = load_student(plan["model_plan"], plan["adapter_configuration"])
            require(
                torch.count_nonzero(model.memory_adapter.output_projection.weight).item() == 0,
                "confirmation baseline must start from exact zero-output memory",
            )
            hands, face = legacy_models(plan["model_plan"])
            appearance = (
                SmallInputDino(hands.model, face.model, side=224).requires_grad_(False).eval()
            )
            visual = plan["model_plan"]["pretrained_state_sha256"]
            require(
                state_digest(appearance.state_dict()) == visual,
                "confirmation public appearance changed",
            )
            preflight = read(bound(plan["evidence"]["preflight"]))
            broad.consume(plan_path, output)
            controls, candidates, baseline, candidate = [], [], [], []
            with torch.no_grad():
                for index, record in enumerate(plan["records"]):
                    broad.guard(output)
                    values = inputs.read_confirmation224(record)
                    receipt = input_receipt(values)
                    streams = original_streams(appearance, values)
                    arrays = {
                        **streams,
                        "valid_intervals": values["valid_intervals"],
                        "source_hand_tip_visible": values["source_hand_tip_visible"],
                    }
                    metadata = check_arrays(arrays, receipt)
                    target = output / f"streams-{index:04d}.npz"
                    broad.guard(output, sum(v.nbytes for v in arrays.values()) + 65536)
                    with target.open("xb") as stream:
                        np.savez(stream, **arrays)
                        stream.flush()
                        os.fsync(stream.fileno())
                    saved = {"input": receipt, "streams": artifact(target), "arrays": metadata}
                    feature_values = cache_values(
                        saved, record, read(bound(preflight["receipts"][index]))
                    )
                    memory, _, base_memory = adapted_memory(model, feature_values)
                    memory_path = output / f"memory-{index:04d}.npy"
                    broad.guard(output, base_memory.nbytes + 65536)
                    with memory_path.open("xb") as stream:
                        np.save(stream, base_memory, allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    saved.update(
                        base_memory=artifact(memory_path),
                        base_memory_metadata=array_metadata(base_memory),
                    )
                    text, tokens = decode(
                        model.language,
                        tokenizer,
                        plan["model_plan"]["reference"]["generation"],
                        memory,
                    )
                    saved.update(
                        prediction={**record["example"], "hypothesis": text},
                        tokens=tokens,
                        native_crop_parity=True,
                    )
                    baseline.append(prediction(record, saved, tokenizer))
                    path = output / f"baseline-{index:04d}.json"
                    write(path, saved)
                    controls.append(artifact(path))
                    del values, streams, arrays, feature_values, memory
                    print(json.dumps({"arm": "baseline", "clips": index + 1, "of": 48}), flush=True)
                require(
                    state_digest(appearance.state_dict()) == visual, "baseline appearance mutated"
                )
                del appearance, hands, face
                gc.collect()
                require_model(model, plan["model_plan"], plan["adapter_configuration"])
                install_adapter(model, binding)
                for index, record in enumerate(plan["records"]):
                    broad.guard(output)
                    control = read(bound(controls[index]))
                    feature_values = cache_values(
                        control, record, read(bound(preflight["receipts"][index]))
                    )
                    memory, adapter_details, base_memory = adapted_memory(model, feature_values)
                    require(
                        array_metadata(base_memory) == control["base_memory_metadata"],
                        "confirmation candidate changed original semantic memory",
                    )
                    text, tokens = decode(
                        model.language,
                        tokenizer,
                        plan["model_plan"]["reference"]["generation"],
                        memory,
                    )
                    saved = {
                        "prediction": {**record["example"], "hypothesis": text},
                        "tokens": tokens,
                        "native_crop_parity": True,
                        "control": controls[index],
                        "streams": control["streams"],
                        "adapter": plan["adapter"],
                        "adapter_details": adapter_details,
                        "semantic_memory": array_metadata(memory[0].detach().numpy()),
                    }
                    candidate.append(prediction(record, saved, tokenizer))
                    path = output / f"candidate-{index:04d}.json"
                    write(path, saved)
                    candidates.append(artifact(path))
                    require_candidate(model, binding)
                    del feature_values, memory
                    print(
                        json.dumps({"arm": "candidate", "clips": index + 1, "of": 48}), flush=True
                    )
            require_model(model, plan["model_plan"], plan["adapter_configuration"])
            require_candidate(model, binding)
        require(plan == checked(plan_path), "confirmation plan changed during inference")
        write(
            output / "result.json",
            {
                "schema": "umi-studio-broad-native224-temporal-memory-inference/1",
                "complete": True,
                "plan": artifact(plan_path),
                "adapter": plan["adapter"],
                "claim": broad.check_claim(plan_path, output),
                "controls": controls,
                "candidates": candidates,
                "baseline": baseline,
                "candidate": candidate,
                "comparison": semantic.comparison_items(
                    [r["example"] for r in plan["records"]], baseline, candidate
                ),
                "all_original_and_trained_values_unchanged": True,
                "all48_baselines_before_trained_memory_adapter_installation": True,
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
    import torch
    from c5_temporal_memory_adapter import TemporalMemoryAdapter
    from c5_temporal_memory_final import checked_final
    from train_c5_temporal_memory import state_hashes
    from transformers import ByT5Tokenizer

    authority = broad.terminal(path.parent)
    receipt = read(path)
    require(
        path.name == "result.json"
        and receipt["schema"] == "umi-studio-broad-native224-temporal-memory-inference/1"
        and receipt["complete"] is True
        and receipt["plan"] == artifact(plan_path)
        and receipt["adapter"] == plan["adapter"]
        and receipt["claim"] == broad.check_claim(plan_path, path.parent)
        and len(receipt["controls"]) == len(receipt["candidates"]) == len(plan["records"]) == 48
        and receipt["all_original_and_trained_values_unchanged"] is True
        and receipt["all48_baselines_before_trained_memory_adapter_installation"] is True
        and receipt["optimizer_updates"] == receipt["new_Terra_requests"] == 0,
        "memory confirmation needs complete fixed-adapter native inference",
    )
    binding = checked_final(bound(plan["final_audit"]))
    require(binding["adapter"] == plan["adapter"], "confirmation audit belongs to another adapter")
    config = binding["adapter_configuration"]
    adapter = TemporalMemoryAdapter(
        rank=config["rank"],
        maximum_relative_residual=config["maximum_relative_residual"],
        seed=config["seed"],
    )
    require(
        adapter.configuration() == config == plan["adapter_configuration"],
        "confirmation changed exact memory adapter configuration",
    )
    saved_adapter = torch.load(bound(binding["adapter"]), map_location="cpu", weights_only=True)
    adapter.load_state_dict(saved_adapter["memory_adapter"], strict=True)
    adapter.requires_grad_(False).eval()
    require(
        state_hashes(adapter) == binding["state_hashes"],
        "confirmation replay changed audited adapter values",
    )
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / plan["model_plan"]["reference"]["bundle_root"] / "model/models/byt5_base",
        local_files_only=True,
    )
    preflight = read(bound(plan["evidence"]["preflight"]))
    expected = {path, path.with_name("worker.json"), path.with_name("exit.json")}
    baseline, candidate = [], []
    for index, record in enumerate(plan["records"]):
        control_path, candidate_path = (
            bound(receipt["controls"][index]),
            bound(receipt["candidates"][index]),
        )
        require(
            control_path == path.with_name(f"baseline-{index:04d}.json")
            and candidate_path == path.with_name(f"candidate-{index:04d}.json"),
            "memory confirmation decoded order changed",
        )
        control, saved = read(control_path), read(candidate_path)
        require(
            bound(control["streams"]) == path.with_name(f"streams-{index:04d}.npz"),
            "memory confirmation cache order changed",
        )
        values = cache_values(control, record, read(bound(preflight["receipts"][index])))
        memory_path = bound(control["base_memory"])
        require(
            memory_path == path.with_name(f"memory-{index:04d}.npy"),
            "confirmation base memory coverage changed",
        )
        base_memory = np.load(memory_path, allow_pickle=False)
        require(
            base_memory.dtype == np.float32
            and base_memory.shape == (len(values["valid_intervals"]) + 1, 1536)
            and np.isfinite(base_memory).all()
            and array_metadata(base_memory) == control["base_memory_metadata"],
            "confirmation changed complete original memory",
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
            saved["control"] == receipt["controls"][index]
            and saved["streams"] == control["streams"]
            and saved["adapter"] == plan["adapter"]
            and saved["adapter_details"] == expected_details
            and saved["semantic_memory"] == array_metadata(adapted["memory"][0].numpy()),
            "memory confirmation changed actual trained adapter or native memory evidence",
        )
        baseline.append(prediction(record, control, tokenizer))
        candidate.append(prediction(record, saved, tokenizer))
        expected.update((control_path, candidate_path, bound(control["streams"]), memory_path))
    comparison = semantic.comparison_items(
        [r["example"] for r in plan["records"]], baseline, candidate
    )
    require(
        baseline == receipt["baseline"]
        and candidate == receipt["candidate"]
        and comparison == receipt["comparison"],
        "confirmation prediction summary changed",
    )
    paths = list(path.parent.rglob("*"))
    require(
        all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in paths)
        and {p for p in paths if p.is_file()} == expected,
        "memory confirmation contains partial, extra or aliased inference",
    )
    authority["memory_inference_assets"] = [artifact(p) for p in sorted(expected)]
    return comparison, authority


def score(plan_path, inference_path, output):
    # Keep the original broad scoring, cohort claim, calibration and statistics.
    from terra_polarity_judge import bind

    bind(broad.score, checked=checked, checked_inference=checked_inference)(
        plan_path, inference_path, output
    )


def checked_tool_poll(plan_path, result_path, poll_path, *, inference_path=None):
    owned(poll_path, "runs")
    authority = broad.terminal(result_path.parent)
    worker, poll = read(bound(authority["worker"])), read(poll_path)
    argv = worker["argv"]
    require(
        poll["tool"] == "write_stdin"
        and type(poll["requested_session_id"]) is int
        and poll["actual_owned_pid"] == worker["pid"]
        and type(poll["tool_result"].get("exit_code")) is int
        and poll["tool_result"]["exit_code"] == 0
        and poll["tool_result"].get("session_id") is None
        and Path(argv[1]).absolute() == ROOT / "tools/run_c5_temporal_memory_confirmation.py"
        and Path(argv[argv.index("--plan") + 1]).absolute() == plan_path
        and Path(argv[argv.index("--output") + 1]).absolute() == result_path.parent
        and (
            (inference_path is None and "--inference" not in argv)
            or (
                inference_path is not None
                and "--inference" in argv
                and Path(argv[argv.index("--inference") + 1]).absolute() == inference_path
            )
        ),
        "memory confirmation requires actual terminal stage/worker/program/plan evidence",
    )
    return authority


def checked_confirmation(plan_path, inference_path, inference_poll, scoring_path, scoring_poll):
    """Recompute complete authenticated confirmation statistics, including failures."""
    from benchmark_fleurs_dev import distance, normalize

    owned(plan_path, "runs")
    for path in (inference_path, scoring_path):
        owned(path, "checkpoints")
        require(path.name == "result.json", "confirmation result filename changed")
    plan = checked(plan_path)
    inference_authority = checked_tool_poll(plan_path, inference_path, inference_poll)
    scoring_authority = checked_tool_poll(
        plan_path, scoring_path, scoring_poll, inference_path=inference_path
    )
    comparison, authority = checked_inference(inference_path, plan_path, plan)
    directory = scoring_path.parent
    passes = semantic.authenticated_passes(comparison, directory, plan["judge_protocol"])
    inference = read(inference_path)
    diagnostic = {}
    for arm in ("baseline", "candidate"):
        pairs = [(normalize(r["reference"]), normalize(r["hypothesis"])) for r in inference[arm]]
        edits = sum(distance(a, b) for a, b in pairs)
        words = sum(len(a) for a, _ in pairs)
        diagnostic[arm] = {
            "edits": edits,
            "reference_words": words,
            "corpus_WER": edits / words if words else None,
        }
    statistics = semantic.summarize(comparison, passes, diagnostic)
    # A failed calibration remains raw evidence and cannot establish model quality.
    require(
        statistics["interpretation_permitted"] is True,
        "memory confirmation cannot interpret failed original calibration",
    )
    expected = {
        **statistics,
        "schema": "umi-studio-broad-confirmation-result/1",
        "plan": artifact(plan_path),
        "inference": artifact(inference_path),
        "inference_authority": authority,
        "claim": artifact(broad.CLAIM),
        "raw_response_authentication_performed": True,
        "candidate_or_runtime_admission_verified": True,
        "complete": True,
        "optimizer_updates": 0,
        "promotion_authorized": False,
        "qualifications": read(broad.MEDIA)["qualifications"],
    }
    require(
        read(scoring_path) == expected,
        "memory confirmation decision changed authentic complete scores",
    )
    scoring_claim_path = broad.CLAIM.with_name("scoring-consumption.json")
    scoring_claim = read(scoring_claim_path)
    require(
        scoring_claim["plan"] == artifact(plan_path)
        and scoring_claim["inference"] == artifact(inference_path)
        and scoring_claim["worker"] == scoring_authority["worker"]
        and scoring_claim["replacement_or_tuning_allowed"] is False,
        "memory confirmation lost its single actual scoring claim",
    )
    files = {
        scoring_path,
        directory / "worker.json",
        directory / "exit.json",
        directory / "blinded-inputs.json",
    }
    for repeat, batches in enumerate(semantic.batches(comparison), start=1):
        files.add(directory / f"pass-{repeat}-scores.json")
        for batch in range(len(batches)):
            files.update(
                directory / f"pass-{repeat}-batch-{batch:03d}" / name
                for name in ("request.json", "started.json", "response.json", "scores.json")
            )
    paths = list(directory.rglob("*"))
    require(
        all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in paths)
        and {p for p in paths if p.is_file()} == files,
        "memory confirmation contains partial, extra or aliased scoring evidence",
    )
    return {
        "schema": "umi-studio-complete-temporal-memory-confirmation-audit/1",
        "complete": True,
        "plan": artifact(plan_path),
        "adapter": plan["adapter"],
        "both_complete_development_gates_passed": True,
        "actual_trained_native_qualification_verified": True,
        "independent_confirmation_complete": True,
        "substantial_model_improvement_confirmed": statistics["passes_semantic_statistics"],
        "statistics": statistics,
        "evidence": [
            artifact(plan_path),
            artifact(inference_poll),
            artifact(scoring_poll),
            artifact(broad.CLAIM),
            artifact(scoring_claim_path),
            *inference_authority.values(),
            *scoring_authority.values(),
            *authority["memory_inference_assets"],
            *(artifact(p) for p in sorted(files)),
        ],
        "promotion_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "freeze",
        "plan",
        "output",
        "development-plan",
        "development-result",
        "development-terminal-poll",
        "preflight",
        "inference",
        "audit",
        "inference-terminal-poll",
        "scoring-result",
        "scoring-terminal-poll",
    ):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute())
    args = parser.parse_args()
    if args.audit and all(
        (
            args.plan,
            args.inference,
            args.inference_terminal_poll,
            args.scoring_result,
            args.scoring_terminal_poll,
        )
    ):
        owned(args.audit, "runs")
        result = checked_confirmation(
            args.plan,
            args.inference,
            args.inference_terminal_poll,
            args.scoring_result,
            args.scoring_terminal_poll,
        )
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        write(args.audit, result)
    elif (
        args.freeze
        and all(
            (
                args.development_plan,
                args.development_result,
                args.development_terminal_poll,
                args.preflight,
            )
        )
        and not args.plan
    ):
        freeze(
            args.freeze,
            args.development_plan,
            args.development_result,
            args.development_terminal_poll,
            args.preflight,
        )
    elif args.plan and args.output and not args.freeze:
        if args.inference:
            score(args.plan, args.inference, args.output)
        else:
            infer(args.plan, args.output)
    else:
        parser.error("freeze after both local gates and native input preflight, infer, or score")


if __name__ == "__main__":
    main()
