"""Fresh zero-update C5 risk qualification for the separately tested polarity instruction."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import probe_c5_word_semantic as original
from benchmark_fleurs_dev import sha256
from evaluate_roi_short_translation import terminal
from launch_terra_visual_job import write
from terra_polarity_judge import Pacer, authenticated_scores, bind, payload, protocol
from terra_training_recovery import artifact, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc  # noqa: UP017 - pinned runtime is Python3.10
CATALOG_REVIEW = ROOT / "runs/terra-negation-rubric-study-v1/terminal-review.json"
CONTEXT_REVIEW = ROOT / "runs/terra-negation-failed-context-v1/terminal-review.json"


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def research_support():
    evidence = []
    boundary = None
    for kind, review_path, digest in (
        (
            "catalog",
            CATALOG_REVIEW,
            "019010ee1746f4c13c1169b9c76f3d02045951a14691d002ad071561a3ce86f0",
        ),
        (
            "context",
            CONTEXT_REVIEW,
            "fef395e5aed1b1b3917ceaf8a6ddbb812a57e07df073e16c0e453507bc66b9e8",
        ),
    ):
        require(sha256(review_path) == digest, "reviewed polarity diagnostic changed")
        review = read(review_path)
        plan_path = ROOT / review["plan"]["path"]
        complete_path = ROOT / review["complete"]["path"]
        for item in (review["plan"], review["complete"], review["worker"], review["exit"]):
            require(artifact(ROOT / item["path"]) == item, "diagnostic artifact changed")
        terminal(complete_path)
        require(
            not list(complete_path.parent.rglob("failure.json")), "diagnostic has failed evidence"
        )
        plan, complete = read(plan_path), read(complete_path)
        require(
            complete.get("complete") is True
            and complete.get("optimizer_updates") == 0
            and complete.get("GPU_use") is False,
            "polarity support requires complete text-only diagnostics",
        )
        verify_snapshot(plan_path, plan)
        for name, expected in plan["source_sha256"].items():
            require(sha256(ROOT / name) == expected, "diagnostic source changed")
        if kind == "catalog":
            proposal_path = ROOT / plan["proposal"]["path"]
            require(artifact(proposal_path) == plan["proposal"], "diagnostic catalog changed")
            proposal = read(proposal_path)
            require(
                proposal["clarification_append_only"] == protocol()["clarification_append_only"],
                "polarity instruction differs from the tested catalog",
            )
            cases, requests = proposal["cases"], proposal["requests"]
        else:
            require(
                plan["clarification"] == protocol()["clarification_append_only"],
                "polarity instruction differs from the tested failed context",
            )
            cases, requests = plan["context"]["cases"], plan["requests"]
        require(
            len(complete["results"]) == len(requests) == complete["new_Terra_requests"] == 4,
            "diagnostic omitted a predeclared request",
        )
        passes = {1: {}, 2: {}}
        raw = []
        for index, (request, result) in enumerate(zip(requests, complete["results"], strict=True)):
            if kind == "catalog" and request["variant"] != "clarified":
                continue
            require(result["request"] == request, "diagnostic result request mapping changed")
            rows = [cases[i] for i in request["case_indices"]]
            directory = complete_path.parent / f"request-{index:03d}"
            require(
                artifact(ROOT / request["payload"]["path"]) == request["payload"]
                and read(ROOT / request["payload"]["path"]) == payload(rows),
                "diagnostic exact payload differs from the new protocol",
            )
            actual = authenticated_scores(rows, directory)
            recorded = result["scores"]
            require(
                (
                    actual == [row["score"] for row in recorded]
                    if kind == "catalog"
                    else actual == recorded
                ),
                "diagnostic summary differs from its raw scores",
            )
            for case_index, score in zip(request["case_indices"], actual, strict=True):
                require(
                    case_index not in passes[request["pass"]], "diagnostic double-counted a pair"
                )
                row = cases[case_index]
                require(
                    row["range"][0] <= score <= row["range"][1],
                    "clarified diagnostic did not meet its original declared bounds",
                )
                passes[request["pass"]][case_index] = score
            raw.extend(
                artifact(directory / name)
                for name in ("request.json", "started.json", "response.json", "scores.json")
            )
        require(
            all(set(values) == set(range(len(cases))) for values in passes.values()),
            "clarified diagnostic lacks two complete passes",
        )
        evidence.append(
            {
                "kind": kind,
                "review": artifact(review_path),
                "plan": artifact(plan_path),
                "complete": artifact(complete_path),
                "raw_clarified_evidence": raw,
                "pairs_per_pass": len(cases),
                "complete_passes": 2,
            }
        )
        exited = datetime.fromisoformat(read(ROOT / review["exit"]["path"])["observed_at"])
        require(exited.tzinfo is not None, "diagnostic exit needs an actual timezone")
        candidate = exited + timedelta(seconds=60)
        boundary = max(boundary, candidate) if boundary is not None else candidate
    return {
        "diagnostics": evidence,
        "API_start_not_before": boundary.isoformat(),
        "old_word_lineage_remains_exhausted": True,
        "general_judge_reliability_or_model_quality_established": False,
    }


def specification(component_path=original.COMPONENT, component_result=original.COMPONENT_RESULT):
    base = original.specification(component_path, component_result)
    support = research_support()
    sources = dict(base["source_sha256"])
    for name in ("probe_c5_polarity_semantic.py", "terra_polarity_judge.py", "c5_word_training.py"):
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        **base,
        "schema": "umi-studio-C5-polarity-semantic-gradient-proof/1",
        "judge_protocol": protocol(),
        "polarity_diagnostic_support": support,
        "source_sha256": sources,
        "fresh_original_public_C5_initialization": True,
        "failed_word_prefix_or_rewards_reused": False,
        "new_optimizer_training_authorized_by_proof": False,
    }


def checked(path):
    plan = read(path)
    if plan != specification(
        ROOT / plan["component_plan"]["path"], ROOT / plan["component_result"]["path"]
    ):
        raise ValueError(
            "polarity risk proof changed its original model/data/numerics or tested prompt"
        )
    verify_snapshot(path, plan)
    return plan


def freeze(path):
    from c5_experiment_sources import archive_sources
    from prepare_terra_word_native import guard

    path = path.resolve()
    require(
        not path.parent.exists() and path.is_relative_to(ROOT / "runs"), "fresh owned plan required"
    )
    guard()
    plan = specification()
    path.parent.mkdir(parents=True)
    write(path, plan)
    archive_sources(path, path.with_name("source-snapshot"))
    print(
        json.dumps(
            {
                "plan": artifact(path),
                "training_cases": len(plan["records"]),
                "optimizer_updates": 0,
                "initialization": "original public C5",
            }
        ),
        flush=True,
    )


def legacy_candidate_checker():
    definition = ast.parse(inspect.getsource(original.checked_candidates)).body[0]
    imports = [
        statement
        for statement in definition.body
        if isinstance(statement, ast.ImportFrom)
        and statement.module == "probe_jeff_terra_quality"
        and [(alias.name, alias.asname) for alias in statement.names] == [("terra_payload", None)]
    ]
    require(len(imports) == 1, "cannot identify original local payload import")
    definition.body.remove(imports[0])
    namespace = {**original.checked_candidates.__globals__, "terra_payload": payload}
    exec(
        compile(
            ast.Module(body=[definition], type_ignores=[]),
            "<polarity-candidate-authenticator>",
            "exec",
        ),
        namespace,
    )
    return namespace["checked_candidates"]


def checked_candidates(path, plan_path, plan, tokenizer):
    result = legacy_candidate_checker()(path, plan_path, plan, tokenizer)
    paired = result["comparison"]
    indices = original.shuffled_indices(paired)
    for offset in range(0, len(indices), 16):
        directory = path.parent / "semantic" / f"batch-{offset // 16:03d}"
        authenticated_scores([paired["items"][i] for i in indices[offset : offset + 16]], directory)
    return result


def prepare(path, output, plan):
    boundary = datetime.fromisoformat(plan["polarity_diagnostic_support"]["API_start_not_before"])
    require(boundary.tzinfo is not None, "polarity predecessor boundary lacks timezone")
    while (remaining := (boundary - datetime.now(UTC)).total_seconds()) > 0:
        time.sleep(min(30, remaining))
    write(
        output / "predecessor-spacing.json",
        {"not_before": boundary.isoformat(), "ready_at": datetime.now(UTC).isoformat()},
    )
    return bind(original.prepare, RecoveryPacer=Pacer, checked=checked)(path, output, plan)


def bound_qualification():
    return bind(original.qualify, checked_candidates=checked_candidates, checked=checked)


def qualify(path, output, plan, candidates):
    return bound_qualification()(path, output, plan, candidates)


def checked_qualification(result_path, plan_path, plan):
    return bind(original.checked_qualification, checked_candidates=checked_candidates)(
        result_path, plan_path, plan
    )


def run(path, output, mode, candidates=None):
    from train_terra_c5_tail import gpu_lease

    bound = bind(original.run, checked=checked, prepare=prepare, qualify=qualify)
    if mode == "candidates":
        with gpu_lease():
            return bound(path, output, mode, candidates)
    return bound(path, output, mode, candidates)


def pipeline(path, output):
    return bind(
        original.pipeline,
        __file__=__file__,
        checked=checked,
        checked_qualification=checked_qualification,
    )(path, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "plan", "output", "candidates"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--mode", choices=("candidates", "qualify"))
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze:
        freeze(args.freeze)
    path = args.plan or args.freeze
    if path and args.output and args.pipeline:
        pipeline(path, args.output)
    elif path and args.output and args.mode:
        run(path, args.output, args.mode, args.candidates)
    elif not args.freeze or args.output:
        parser.error(
            "freeze a fresh plan or execute its complete candidates/qualification pipeline"
        )


if __name__ == "__main__":
    main()
