from __future__ import annotations

import copy
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_mixed_semantic_model as model
import probe_c5_mixed_semantic as proof
import probe_c5_word_semantic as word
import terra_mixed_semantic_protocol as protocol
import test_c5_word_training as fixtures
import test_terra_mixed_semantic_protocol as mixed_fixtures
from terra_mixed_native_input import expected_header
from terra_training_recovery import artifact
from tiny_rgb_capacity_protocol import LIMITS

UTC = timezone.utc  # noqa: UP017 - pinned model runtime is Python3.10


@pytest.fixture
def candidates(tmp_path, monkeypatch):
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(proof, "ROOT", tmp_path)
    monkeypatch.setattr(proof, "artifact", descriptor)
    tokenizer = fixtures.Tokenizer()
    records = mixed_fixtures.cases.__wrapped__()
    pools = mixed_fixtures.pools(records, tokenizer)
    plan = {
        "records": records,
        "initialization": {"initial_visual_state_sha256": "word-final"},
        "initial_reference_hashes": {"sequence": "original", "language": "original"},
        "API_start_not_before": "2026-10-17T20:00:00+00:00",
        "expected_visual_gradient_names": [f"vision.fixture{index}" for index in range(350)],
    }
    plan_path = tmp_path / "runs/proof/plan.json"
    plan_path.parent.mkdir(parents=True)
    proof.write(plan_path, plan)
    output = tmp_path / "checkpoints/proof/candidates"
    output.mkdir(parents=True)
    proof.write(output / "worker.json", {"pid": 10})
    proof.write(output / "exit.json", {"pid": 10, "exit_code": 0})
    for index, (case, pool) in enumerate(zip(records, pools, strict=True)):
        proof.write(output / f"pool-{index:02d}.json", {"case": case, "candidates": pool})
    directory = output / "semantic"
    directory.mkdir()
    paired = protocol.paired_inputs(records, pools, tokenizer)
    proof.write(directory / "blinded-inputs.json", paired)
    proof.write(
        directory / "predecessor-spacing.json",
        {
            "not_before": plan["API_start_not_before"],
            "ready_at": "2026-10-17T20:01:00+00:00",
        },
    )
    indices = protocol.shuffled_indices(paired, proof.POLICY["training_rollout_seed"])
    pacer = fixtures.FakePacer()
    pacer.score([paired["items"][index] for index in indices], directory / "request-000")
    actual = protocol.verified_proof_rewards(
        paired, directory, proof.POLICY["training_rollout_seed"]
    )
    proof.write(output / "scores.json", actual["scores"])
    result = {
        "complete": True,
        "plan_sha256": proof.sha256(plan_path),
        "pools": pools,
        "comparison": paired,
        "qualities": actual["qualities"],
        "scores": descriptor(output / "scores.json"),
        "initialization": plan["initialization"],
        "vision_initialization_sha256": "word-final",
        "initial_reference_hashes": plan["initial_reference_hashes"],
        "optimizer_updates": 0,
        "new_Terra_requests": 1,
        "calibration_passed": True,
        "training_only": True,
        "promotion_authorized": False,
    }
    path = output / "candidates.json"
    proof.write(path, result)
    return SimpleNamespace(
        path=path,
        plan_path=plan_path,
        plan=plan,
        tokenizer=tokenizer,
        result=result,
        directory=directory,
    )


def overwrite(path, value):
    path.write_text(json.dumps(value))


def test_qualification_reuses_accepted_loop_bytecode_with_only_six_explicit_bindings():
    bound = proof.bound_qualification()
    assert bound.__code__ is word.qualify.__code__
    assert {
        key
        for key in bound.__globals__
        if bound.__globals__[key] is not word.qualify.__globals__[key]
    } == {
        "POLICY",
        "load_student",
        "load_clip",
        "require_model",
        "checked_candidates",
        "checked",
    }
    assert bound.__globals__["compare"] is word.compare
    assert bound.__globals__["verified_comparison"] is word.verified_comparison
    assert bound.__globals__["visual_gradients"] is word.visual_gradients
    assert proof.load_student is model.load_student
    assert proof.load_clip is model.load_clip
    assert proof.require_model is model.require_model
    assert proof.TENSORS == 350 and proof.PARAMETERS == 43260672
    assert proof.POLICY["risk_alpha"] == 0.005 and proof.POLICY["caption_weight"] == 0
    assert proof.require_resources is word.require_resources
    assert word.LIMITS is LIMITS


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "model",
        "initializer",
        "backbone",
        "receipt",
        "pool",
        "extra_pool",
        "raw_response",
        "parser",
        "request",
        "count",
        "scores",
        "cooldown",
        "terminal",
        "failure",
    ],
)
def test_candidates_rejoin_actual_mixed_pools_raw_rewards_and_terminal_initializer(
    candidates, mutation
):
    env = candidates
    if mutation in ("model", "initializer", "backbone", "receipt", "pool", "count", "scores"):
        result = copy.deepcopy(env.result)
        if mutation == "model":
            result["vision_initialization_sha256"] = "original-public"
        elif mutation == "initializer":
            result["initialization"]["initial_visual_state_sha256"] = "other"
        elif mutation == "backbone":
            result["initial_reference_hashes"]["language"] = "trained"
        elif mutation == "receipt":
            result["comparison"]["cases"][0]["mixed_input_receipt"] = result["comparison"]["cases"][
                1
            ]["mixed_input_receipt"]
        elif mutation == "pool":
            result["pools"][0][-1]["tokens"][0] += 1
        elif mutation == "count":
            result["new_Terra_requests"] = 2
        elif mutation == "scores":
            score_path = env.path.with_name("scores.json")
            overwrite(score_path, [10] * len(result["comparison"]["items"]))
            result["scores"] = proof.artifact(score_path)
        overwrite(env.path, result)
    elif mutation in ("raw_response", "parser", "request"):
        filename = {
            "raw_response": "response.json",
            "parser": "scores.json",
            "request": "request.json",
        }[mutation]
        path = env.directory / "request-000" / filename
        record = json.loads(path.read_text())
        if mutation == "parser":
            record["scores"][0] = 99
        elif mutation == "raw_response":
            record["model"] = "other-model"
        else:
            record["model"] = "other-model"
        overwrite(path, record)
    elif mutation == "extra_pool":
        proof.write(env.path.with_name("pool-99.json"), {})
    elif mutation == "cooldown":
        overwrite(
            env.directory / "predecessor-spacing.json",
            {
                "not_before": env.plan["API_start_not_before"],
                "ready_at": "2026-10-17T19:59:59+00:00",
            },
        )
    elif mutation == "terminal":
        overwrite(env.path.with_name("exit.json"), {"pid": 11, "exit_code": 0})
    elif mutation == "failure":
        proof.write(env.directory / "request-000/failure.json", {})
    if mutation is None:
        assert (
            proof.checked_candidates(env.path, env.plan_path, env.plan, env.tokenizer)["qualities"]
            == [[10, 3]] * 3
        )
    else:
        with pytest.raises(ValueError):
            proof.checked_candidates(env.path, env.plan_path, env.plan, env.tokenizer)


def resources():
    return {
        "current_allowed_gpu_duty_fraction": 0.5,
        "current_cpu_threads": 2,
        "memory_policy": "framework_default_no_artificial_six_GiB_cap",
        "mps_active_seconds": 2.0,
        "mps_idle_seconds": 2.0,
        "maximum_allocated_bytes": 1024,
        "maximum_driver_allocated_bytes": 2048,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "count",
        "frames",
        "receipt",
        "quality",
        "names",
        "norm",
        "logits",
        "gradient",
        "risk",
        "resources",
        "file",
        "initializer",
    ],
)
def test_mixed_full_proof_rechecks_real_case_timelines_all350_gradients_and_unchanged_limits(
    candidates, monkeypatch, mutation
):
    env = candidates
    monkeypatch.setattr(proof, "candidate_tokenizer", lambda plan: env.tokenizer)
    output = env.path.parent.parent / "qualify"
    output.mkdir()
    proof.write(output / "worker.json", {"pid": 11})
    proof.write(output / "exit.json", {"pid": 11, "exit_code": 0})
    reports = []
    for index, case in enumerate(env.plan["records"]):
        report = {
            "input_receipt": case["receipt"],
            "frames": expected_header(case["case"])["sampled_frames"],
            "actual_Terra_qualities": [10, 3],
            "CPU_risk": 0.5,
            "Metal_risk": 0.5,
            "resources": resources(),
            "comparison": {
                "loss_absolute_difference": 0,
                "logit_maximum_absolute_difference": 0,
                "total_gradient_norm_relative_error": 0,
                "gradient_differences": [
                    {
                        "parameter": name,
                        "relative_l2_error": 0,
                        "finite": True,
                        "cpu_norm": 1,
                        "actual_norm": 1,
                    }
                    for name in env.plan["expected_visual_gradient_names"]
                ],
            },
        }
        if index == 1:
            if mutation == "frames":
                report["frames"] -= 1
            elif mutation == "receipt":
                report["input_receipt"] = env.plan["records"][0]["receipt"]
            elif mutation == "quality":
                report["actual_Terra_qualities"] = [10, 10]
            elif mutation in ("names", "norm", "gradient"):
                key, value = {
                    "names": ("parameter", "vision.other"),
                    "norm": ("actual_norm", 0),
                    "gradient": ("relative_l2_error", 0.006),
                }[mutation]
                report["comparison"]["gradient_differences"][-1][key] = value
            elif mutation == "logits":
                report["comparison"]["logit_maximum_absolute_difference"] = 0.002
            elif mutation == "risk":
                report["Metal_risk"] = 1
            elif mutation == "resources":
                report["resources"]["current_allowed_gpu_duty_fraction"] = 1
        target = output / f"comparison-{index:02d}.json"
        proof.write(target, report)
        reports.append(proof.artifact(target))
    result = {
        "complete": True,
        "full_model_numerics_passed": True,
        "plan_sha256": proof.sha256(env.plan_path),
        "candidates": proof.artifact(env.path),
        "comparisons": reports,
        "vision_initialization_sha256": "word-final",
        "initial_reference_hashes": env.plan["initial_reference_hashes"],
        "trainable_tensors": 350,
        "trainable_parameters": 43260672,
        "pure_Terra_objective": True,
        "all_fixed_values_unchanged": True,
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "resources": resources(),
        "model_win_claimed": False,
        "held_out_quality_evaluated": False,
        "promotion_authorized": False,
    }
    if mutation == "count":
        result["comparisons"].pop()
    elif mutation == "initializer":
        result["vision_initialization_sha256"] = "original-public"
    elif mutation == "file":
        overwrite(output / "comparison-02.json", {})
    target = output / "qualification.json"
    proof.write(target, result)
    if mutation is None:
        assert proof.checked_qualification(target, env.plan_path, env.plan)["complete"] is True
    else:
        with pytest.raises(ValueError):
            proof.checked_qualification(target, env.plan_path, env.plan)


def test_candidate_phase_authenticates_raw_response_instead_of_pacer_return(
    candidates, monkeypatch
):
    env = candidates
    output = env.path.parent.parent / "fresh"
    output.mkdir()
    monkeypatch.setattr(proof, "prepare_pools", lambda *a: env.result["pools"])
    monkeypatch.setattr(proof, "RecoveryPacer", fixtures.FakePacer)
    monkeypatch.setattr(proof, "require_model", lambda *a: None)
    monkeypatch.setattr(proof, "checked", lambda path: env.plan)
    monkeypatch.setattr(proof, "state_digest", lambda *a: "word-final")
    monkeypatch.setattr(proof, "fixed_hashes", lambda *a: env.plan["initial_reference_hashes"])
    monkeypatch.setattr(proof, "guard", lambda *a: None)
    # Avoid a model; the phase still writes and reauthenticates all actual score files.
    monkeypatch.setattr(
        proof,
        "load_student",
        lambda plan: (
            SimpleNamespace(vision=SimpleNamespace(state_dict=lambda: {})),
            env.tokenizer,
        ),
    )
    proof.prepare(env.plan_path, output, env.plan)
    result = json.loads((output / "candidates.json").read_text())
    assert result["qualities"] == [[10, 3]] * 3
    assert 999 not in json.loads((output / "scores.json").read_text())


@pytest.mark.parametrize(
    "failure", [None, "candidates", "qualify", "missing_receipt", "bad_rewards"]
)
def test_pipeline_verifies_serial_terminal_children_and_never_retries(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(proof, "ROOT", tmp_path)
    monkeypatch.setattr(proof, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(proof, "checked", lambda path: {})
    monkeypatch.setattr(proof, "guard", lambda *a: None)
    monkeypatch.setattr(proof, "candidate_tokenizer", lambda plan: None)
    monkeypatch.setattr(proof, "identity", lambda pid, command: {"pid": pid})
    events = []
    path = tmp_path / "runs/proof/plan.json"
    path.parent.mkdir(parents=True)
    proof.write(path, {})
    output = tmp_path / "checkpoints/proof"

    def child(command, check):
        mode = command[command.index("--mode") + 1]
        events.append(mode)
        assert check is False
        if failure != "missing_receipt":
            folder = output / mode
            folder.mkdir()
            proof.write(folder / "worker.json", {"pid": 33})
            proof.write(folder / "exit.json", {"pid": 33, "exit_code": int(failure == mode)})
            proof.write(
                folder / ("candidates.json" if mode == "candidates" else "qualification.json"),
                {"complete": True},
            )
        return SimpleNamespace(returncode=int(failure == mode))

    def verify_candidates(*args):
        events.append("actual rewards")
        if failure == "bad_rewards":
            raise ValueError("invalid actual rewards")

    monkeypatch.setattr(proof.subprocess, "run", child)
    monkeypatch.setattr(proof, "checked_candidates", verify_candidates)
    monkeypatch.setattr(proof, "checked_qualification", lambda *a: events.append("actual proof"))
    if failure is None:
        proof.pipeline(path, output)
        assert events == ["candidates", "actual rewards", "qualify", "actual proof"]
        assert (output / "complete.json").exists()
    else:
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
            proof.pipeline(path, output)
        assert events.count("candidates") == 1
        assert ("qualify" in events) == (failure == "qualify")
        assert not (output / "complete.json").exists()
    assert json.loads((output / "exit.json").read_text())["exit_code"] == int(failure is not None)


@pytest.mark.parametrize("mode", ["candidates", "qualify"])
def test_both_modes_acquire_same_exclusive_C5_lease_before_model_work(tmp_path, monkeypatch, mode):
    import cv2
    import train_terra_c5_tail

    events = []
    monkeypatch.setattr(proof, "ROOT", tmp_path)
    monkeypatch.setattr(proof, "checked", lambda path: {})
    monkeypatch.setattr(proof, "guard", lambda *a: None)
    monkeypatch.setattr(proof.torch, "set_num_threads", lambda *a: None)
    monkeypatch.setattr(proof.torch, "set_num_interop_threads", lambda *a: None)
    monkeypatch.setattr(cv2, "setNumThreads", lambda *a: None)
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")

    @contextmanager
    def lease():
        events.append("lease")
        yield
        events.append("release")

    monkeypatch.setattr(train_terra_c5_tail, "gpu_lease", lease)
    monkeypatch.setattr(proof, "prepare", lambda *a: events.append("model"))
    monkeypatch.setattr(proof, "qualify", lambda *a: events.append("model"))
    proof.run(tmp_path / "plan", tmp_path / f"checkpoints/{mode}", mode, tmp_path / "candidates")
    assert events == ["lease", "model", "release"]


def test_cross_process_API_cooldown_waits_before_first_request_in_small_intervals():
    current = datetime(2026, 10, 17, 20, tzinfo=UTC)
    waits = []

    def sleep(seconds):
        nonlocal current
        waits.append(seconds)
        current += timedelta(seconds=seconds)

    boundary = current + timedelta(seconds=65)
    ready = proof.wait_for_predecessor(
        {"API_start_not_before": boundary.isoformat()}, clock=lambda: current, sleep=sleep
    )
    assert datetime.fromisoformat(ready) == boundary
    assert waits == [30, 30, 5]


@pytest.mark.parametrize("explicit_lineage", [False, True])
def test_unfinished_prerequisites_leave_no_real_frozen_plan(
    tmp_path, monkeypatch, explicit_lineage
):
    monkeypatch.setattr(proof, "ROOT", tmp_path)

    def unavailable(*args):
        raise ValueError("actual word-final prerequisite is incomplete")

    monkeypatch.setattr(proof, "evidence_specification", unavailable)
    target = tmp_path / "runs/not-ready/plan.json"
    arguments = ["--freeze", str(target)]
    if explicit_lineage:
        arguments += [
            "--training-plan",
            str(tmp_path / "runs/recovery/plan.json"),
            "--training-result",
            str(tmp_path / "checkpoints/recovery/result.json"),
        ]
    with pytest.raises(ValueError, match="incomplete"):
        proof.main(arguments)
    assert not target.parent.exists()


def test_freeze_binds_explicit_lineage_through_the_existing_specification(tmp_path, monkeypatch):
    import c5_experiment_sources

    monkeypatch.setattr(proof, "ROOT", tmp_path)
    monkeypatch.setattr(proof, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(proof, "guard", lambda: None)
    captured = []
    training = tmp_path / "runs/recovery/plan.json"
    result = tmp_path / "checkpoints/recovery/result.json"
    plan = {"records": ["complete-case"], "authenticated_initialization": "fixture"}

    def specification(training_path, result_path):
        captured.append((training_path, result_path))
        return plan

    monkeypatch.setattr(proof, "specification", specification)
    monkeypatch.setattr(c5_experiment_sources, "archive_sources", lambda *args: None)
    target = tmp_path / "runs/proof/plan.json"
    proof.main(
        [
            "--freeze",
            str(target),
            "--training-plan",
            str(training),
            "--training-result",
            str(result),
        ]
    )
    assert captured == [(training, result)]
    assert json.loads(target.read_text()) == plan


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--freeze", "plan.json", "--training-plan", "recovery.json"], "supplied together"),
        (["--freeze", "plan.json", "--training-result", "result.json"], "supplied together"),
        ([], "only with --freeze"),
        (["--plan", "plan.json", "--output", "out", "--pipeline"], "only with --freeze"),
        (["--freeze", "plan.json", "--mode", "candidates"], "only with --freeze"),
    ],
)
def test_invalid_lineage_cli_is_rejected_before_any_action(arguments, message, monkeypatch, capsys):
    if "supplied together" not in message:
        arguments = [
            *arguments,
            "--training-plan",
            "recovery.json",
            "--training-result",
            "result.json",
        ]

    def forbidden(*args):
        pytest.fail("invalid lineage arguments reached a proof action")

    for name in ("freeze", "pipeline", "run"):
        monkeypatch.setattr(proof, name, forbidden)
    with pytest.raises(SystemExit) as error:
        proof.main(arguments)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
