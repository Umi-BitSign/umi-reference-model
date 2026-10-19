"""Export two research DINO files after complete audited training and semantic gates.

This does not build, deploy or qualify a serving bundle. Native input, feature,
token, latency and broader independent semantic confirmation remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from importlib import import_module
from pathlib import Path

from c5_native112_export_values import (
    merge_visual_values,
    qualified_shapes,
    require,
    verify_visual_values,
)

ROOT = Path(__file__).resolve().parents[1]
FILES = {"face": "dinov2face.safetensors", "hands": "dinov2hand.safetensors"}
BUDGET = 256 * 1024**2


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def artifact(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def bound(descriptor):
    path = ROOT / descriptor["path"]
    require(path.resolve() == path and artifact(path) == descriptor, "export source bytes changed")
    return path


def export_sources():
    paths = [Path(__file__), ROOT / "tools/c5_native112_export_values.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def verify_evidence(evidence):
    """Rehash each distinct bound artifact; conflicting hashes fail closed."""
    seen = {}

    def visit(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                path, digest = value["path"], value["sha256"]
                require(path not in seen or seen[path] == digest, "conflicting export evidence")
                if path not in seen:
                    bound(value)
                    seen[path] = digest
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(evidence)
    return len(seen)


def checked_coordinator(complete, handoff, initialization, audit, evaluations):
    """Require the exact completed serial pipeline, not isolated passing scores."""
    require(
        complete.get("complete") is True
        and complete.get("initialization") == initialization
        and complete.get("terminal_audit") == audit
        and complete.get("evaluations") == evaluations
        and set(evaluations) == {"word", "sentence"}
        and type(complete.get("optimizer_updates")) is int
        and complete["optimizer_updates"] == 0
        and type(complete.get("automatic_retries")) is int
        and complete["automatic_retries"] == 0
        and complete.get("automatic_training_continuation") is False
        and complete.get("promotion_authorized") is False,
        "visual export needs the complete same-final serial evaluation",
    )
    require(
        handoff.get("initialization") == initialization
        and handoff.get("terminal_audit") == audit
        and handoff.get("complete_evaluation_order") == ["word96", "sentence34"]
        and handoff.get("stop_on_failed_calibration") is True
        and handoff.get("cross_pipeline_minimum_start_gap_seconds") == 60
        and type(handoff.get("automatic_retries")) is int
        and handoff.get("automatic_retries") == 0
        and handoff.get("automatic_training_continuation") is False
        and handoff.get("promotion_authorized") is False,
        "visual export lost the original complete-evaluation handoff",
    )
    for kind, measured in evaluations.items():
        decision = measured["decision_values"]
        require(
            measured["kind"] == kind
            and measured.get("semantic_decline_recorded_for_evaluation_only") is False
            and decision.get("calibration_passed") is True
            and decision.get("passes_semantic_research_gate") is True
            and decision.get("stop_for_semantic_decline") is False,
            "visual export requires both authenticated calibrated semantic improvement gates",
        )


def evaluation_protocol(training_path):
    """Select the exact audited lineage's complete-evaluation implementation."""
    modules = {
        "umi-studio-C5-polarity-storage-recovery/1": "evaluate_c5_word_storage_final",
        "umi-studio-C5-polarity-HTTP502-recovery/1": "evaluate_c5_word_http502_final",
    }
    name = modules.get(read(training_path).get("schema"))
    require(name is not None, "unsupported visual export training lineage")
    return import_module(name), "tools/" + name + ".py"


def authenticate(
    training_path, result_path, audit_path, coordinator_path, word_path, sentence_path
):
    # Refuse before importing model/evaluation helpers or creating output when
    # no completed fixed-final result is available.
    require(
        training_path.is_relative_to(ROOT / "runs")
        and audit_path.is_relative_to(ROOT / "runs")
        and all(
            path.is_relative_to(ROOT / "checkpoints") and path.is_file()
            for path in (result_path, coordinator_path, word_path, sentence_path)
        ),
        "visual export requires complete owned training and evaluation receipts",
    )
    import torch
    from c5_mixed_risk_evidence import require_terminal

    torch.set_num_threads(2)

    protocol, coordinator_source = evaluation_protocol(training_path)
    initialization = protocol.prerequisites(training_path, result_path, audit_path)
    require_terminal(coordinator_path)
    complete = read(coordinator_path)
    handoff_path = bound(complete["handoff"])
    require(handoff_path == coordinator_path.with_name("handoff.json"), "unrelated export handoff")
    handoff = read(handoff_path)
    evaluations = {
        "word": protocol.previous_evaluation(word_path, "word", initialization),
        "sentence": protocol.previous_evaluation(sentence_path, "sentence", initialization),
    }
    checked_coordinator(complete, handoff, initialization, artifact(audit_path), evaluations)
    sources = protocol.control_sources()
    require(
        handoff["source_sha256"] == sources
        and handoff["coordinator_source_sha256"] == sources[coordinator_source],
        "export evaluation coordinator differs from reviewed sources",
    )
    for name, digest in sources.items():
        require(
            sha(ROOT / name) == digest
            and sha(coordinator_path.parent / "control-sources" / name) == digest,
            "export lost archived evaluation source integrity",
        )
    spacing = [
        coordinator_path.with_name(name)
        for name in (
            "training-to-word-spacing.json",
            "word-to-sentence-spacing.json",
        )
    ]
    require(
        complete["API_spacing"] == [artifact(path) for path in spacing],
        "export API spacing changed",
    )
    for path in spacing:
        times = read(path)
        start, ready = (datetime.fromisoformat(times[key]) for key in ("not_before", "ready_at"))
        require(
            start.tzinfo is not None and ready.tzinfo is not None and ready >= start,
            "export predecessor API cooldown was not respected",
        )
    return {
        "training_plan": artifact(training_path),
        "training_result": artifact(result_path),
        "terminal_audit": artifact(audit_path),
        "evaluation_complete": artifact(coordinator_path),
        "handoff": artifact(handoff_path),
        "initialization": initialization,
        "evaluations": evaluations,
        "API_spacing": complete["API_spacing"],
    }


def storage(directory, prospective=0):
    require(type(prospective) is int and prospective >= 0, "invalid export byte reservation")
    paths = list(directory.rglob("*")) if directory.exists() else []
    require(all(not path.is_symlink() for path in paths), "export cannot follow output aliases")
    used = sum(path.stat().st_size for path in paths if path.is_file())
    remaining = BUDGET - used
    require(
        prospective <= remaining
        and shutil.disk_usage(ROOT).free >= 80 * 1024**3 + 128 * 1024**2 + remaining,
        "export exceeds its256MiB reserve or original80GiB floor",
    )


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(training_path, result_path, audit_path, coordinator_path, word_path, sentence_path, output):
    inputs = [
        path.resolve()
        for path in (
            training_path,
            result_path,
            audit_path,
            coordinator_path,
            word_path,
            sentence_path,
        )
    ]
    training_path, result_path, audit_path, coordinator_path, word_path, sentence_path = inputs
    output = output.absolute()
    temporary = output.with_name("." + output.name + ".partial")
    require(
        output.resolve() == output
        and output.is_relative_to(ROOT / "checkpoints")
        and output != ROOT / "checkpoints"
        and not output.exists()
        and not temporary.exists(),
        "visual export requires a fresh owned output; partial exports are not retried",
    )
    sources = export_sources()
    evidence = authenticate(*inputs)
    verify_evidence(evidence)
    plan, result = read(training_path), read(result_path)
    require(
        result["training_examples"] == plan["planned_training_examples"] == 856,
        "visual export cannot consume a partial trained checkpoint",
    )
    shapes = qualified_shapes(read(bound(plan["model_plan"]["visual_scope"])))
    require(
        set(shapes) == set(plan["model_plan"]["expected_visual_gradient_names"]),
        "export scope differs from actual qualified training",
    )
    visual_path = bound(result["visual"])
    storage(temporary)
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file

    with safe_open(str(visual_path), framework="np") as saved:
        metadata = saved.metadata()
    reference = plan["model_plan"]["reference"]
    require(
        metadata.get("plan_sha256") == sha(training_path)
        and metadata.get("update") == "856"
        and metadata.get("optimizer_updates") == str(result["optimizer_updates"])
        and metadata.get("fixed_winner_checkpoint") == reference["translator_checkpoint_sha256"],
        "trained export tensor metadata differs from the complete audited final",
    )
    trained = load_file(str(visual_path))
    originals, original_metadata, original_files = {}, {}, {}
    for branch, filename in FILES.items():
        source = ROOT / reference["bundle_root"] / "model/models" / filename
        source_artifact = artifact(source)
        require(
            source_artifact["sha256"] == plan["model_plan"]["assets"][source_artifact["path"]],
            "export original released visual source changed",
        )
        original_files[branch] = source_artifact
        with safe_open(str(source), framework="np") as saved:
            original_metadata[branch] = saved.metadata() or {}
        originals[branch] = load_file(str(source))
    merged = merge_visual_values(originals, trained, shapes)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    files = {}
    for branch, filename in FILES.items():
        storage(temporary, sum(value.nbytes for value in merged[branch].values()) + 1024**2)
        path = temporary / filename
        provenance = {
            **original_metadata[branch],
            "schema": "umi-studio-native112-trained-visual-file/1",
            "training_plan_sha256": sha(training_path),
            "training_result_sha256": sha(result_path),
            "trained_visual_sha256": result["visual"]["sha256"],
            "original_visual_file_sha256": original_files[branch]["sha256"],
        }
        save_file(merged[branch], str(path), metadata=provenance)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        with safe_open(str(path), framework="np") as saved:
            require(saved.metadata() == provenance, "serialized visual export metadata changed")
        merged[branch] = load_file(str(path))
        files[branch] = {"path": str((output / filename).relative_to(ROOT)), "sha256": sha(path)}
    verified = verify_visual_values(originals, trained, merged, shapes)
    require(artifact(visual_path) == result["visual"], "trained final changed during export")
    require(
        all(artifact(ROOT / item["path"]) == item for item in original_files.values()),
        "original visual source changed during export",
    )
    verified_evidence = verify_evidence(evidence)
    require(export_sources() == sources, "export program changed during export")
    manifest = {
        "schema": "umi-studio-native112-trained-visual-export/1",
        "complete": True,
        "evidence": evidence,
        "files": files,
        "original_visual_files": original_files,
        "verified_values": verified,
        "source_sha256": sources,
        "bound_source_artifacts_reverified": verified_evidence,
        "native_input_side": 112,
        "temporal_stride": 2,
        "scope": "research visual files only; no serving bundle or stage-advance decision",
        "serving_feature_token_or_latency_equivalence_established": False,
        "broader_independent_semantic_confirmation_completed": False,
        "promotion_or_deployment_authorized": False,
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    storage(temporary, len(data))
    with (temporary / "EXPORT.json").open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(temporary)
    require(not output.exists(), "export output appeared before atomic publication")
    temporary.rename(output)
    sync_directory(output.parent)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "training-plan",
        "training-result",
        "terminal-audit",
        "evaluations",
        "words",
        "sentences",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        args.training_plan,
        args.training_result,
        args.terminal_audit,
        args.evaluations,
        args.words,
        args.sentences,
        args.output,
    )
    print(json.dumps({"complete": result["complete"], "files": result["files"]}))


if __name__ == "__main__":
    main()
