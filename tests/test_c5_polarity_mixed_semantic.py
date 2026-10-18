from __future__ import annotations

import json
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import c5_mixed_risk_evidence as old_evidence
import c5_mixed_semantic_model as model
import c5_polarity_mixed_evidence as evidence
import evaluate_c5_polarity_final as final
import evaluate_c5_polarity_sentences as sentences
import evaluate_c5_polarity_words as words
import probe_c5_mixed_semantic as original
import probe_c5_polarity_mixed_semantic as proof
import probe_c5_word_semantic as word
import terra_mixed_semantic_protocol as mixed
import terra_polarity_judge as judge
from terra_training_recovery import artifact
from test_c5_mixed_risk_evidence import predecessor  # noqa: F401
from test_c5_word_training import FakePacer
from test_probe_c5_mixed_semantic import candidates  # noqa: F401


def read(path):
    return json.loads(path.read_text())


def overwrite(path, value):
    path.write_text(json.dumps(value))


def clarify(directory):
    for path in directory.rglob("request.json"):
        request = read(path)
        request["instructions"] += judge.CLARIFICATION
        overwrite(path, request)
        started_path = path.with_name("started.json")
        started = read(started_path)
        started["request_sha256"] = original.sha256(path)
        overwrite(started_path, started)


def test_numerical_loop_and_all_original_limits_remain_exact():
    bound = proof.bound_qualification()
    assert bound.__code__ is original.bound_qualification().__code__ is word.qualify.__code__
    for name in ("compare", "visual_gradients", "verified_comparison", "POLICY", "LIMITS"):
        assert bound.__globals__[name] is original.bound_qualification().__globals__[name]
    assert bound.__globals__["load_student"] is evidence.load_student
    assert bound.__globals__["checked_candidates"] is proof.checked_candidates
    assert bound.__globals__["checked"] is proof.checked
    assert bound.__globals__["load_clip"] is original.load_clip
    assert bound.__globals__["require_model"] is original.require_model
    assert original.TENSORS == 350 and original.PARAMETERS == 43260672


def test_audited_initializer_is_required_before_any_model_load(tmp_path, monkeypatch):
    events = []

    def unfinished(plan, result, audit):
        events.append((plan, result, audit))
        raise ValueError("complete independent word audit required")

    monkeypatch.setattr(final, "prerequisites", unfinished)
    monkeypatch.setattr(model, "load_original_student", lambda *a: pytest.fail("model loaded"))
    binding = {
        "word_training_plan": {"path": "runs/fresh/plan.json"},
        "word_training_result": {"path": "checkpoints/fresh/result.json"},
    }
    with pytest.raises(ValueError, match="independent word audit"):
        evidence.load_student({"initialization": binding})
    assert events == [
        (
            model.ROOT / "runs/fresh/plan.json",
            model.ROOT / "checkpoints/fresh/result.json",
            model.ROOT / "runs/fresh/terminal-audit.json",
        )
    ]


def test_initializer_retains_original_model_loading_and_visual_installation(monkeypatch):
    binding = {
        "word_training_plan": {"path": "runs/fresh/plan.json"},
        "word_training_result": {"path": "checkpoints/fresh/result.json"},
        "original_model_plan": {"fixture": True},
    }
    student, tokenizer = object(), object()
    events = []
    monkeypatch.setattr(final, "prerequisites", lambda *a: binding)

    def load(plan):
        assert plan is binding["original_model_plan"]
        events.append("original public model")
        return student, tokenizer

    def install(value, actual):
        assert value is student and actual is binding
        events.append("install full visual via original checked body")

    monkeypatch.setattr(model, "load_original_student", load)
    monkeypatch.setattr(model, "install_visual", install)
    assert evidence.load_student({"initialization": binding}) == (student, tokenizer)
    assert events == ["original public model", "install full visual via original checked body"]
    with pytest.raises(ValueError, match="initializer changed"):
        evidence.checked_initialization({**binding, "extra": True})


def test_partial_training_cannot_reach_evaluations_inputs_or_create_a_plan(tmp_path, monkeypatch):
    def unfinished(*args):
        raise ValueError("full fresh training required")

    monkeypatch.setattr(final, "prerequisites", unfinished)
    monkeypatch.setattr(evidence, "previous_evaluation", lambda *a: pytest.fail("grading reached"))
    monkeypatch.setattr(old_evidence, "checked_result", lambda *a: pytest.fail("inputs reached"))
    monkeypatch.setattr(original, "guard", lambda *a: pytest.fail("output reserve reached"))
    monkeypatch.setattr(original, "ROOT", tmp_path)
    target = tmp_path / "runs/unavailable/plan.json"
    with pytest.raises(ValueError, match="full fresh training required"):
        proof.freeze(target)
    assert not target.parent.exists()


def test_specification_defaults_use_the_fresh_lineage_and_preserve_case_policy(monkeypatch):
    import tiny_rgb_full_run

    args = []

    def complete(*values):
        args.append(values)
        return {"fixture": True, "judge_protocol": judge.protocol()}

    monkeypatch.setattr(evidence, "specification", complete)
    monkeypatch.setattr(tiny_rgb_full_run, "runtime_binding", lambda: {"fixture": True})
    plan = proof.specification()
    assert args == [
        (
            proof.TRAINING_PLAN,
            proof.TRAINING_RESULT,
            original.INPUT_PLAN,
            original.INPUT_RESULT,
            proof.WORD_PARENT,
            proof.SENTENCE_PARENT,
            original.PROPOSAL,
        )
    ]
    assert "polarity" in str(proof.TRAINING_PLAN)
    assert "polarity" in str(proof.WORD_PARENT) and "polarity" in str(proof.SENTENCE_PARENT)
    assert plan["schema"] == "umi-studio-C5-polarity-mixed-semantic-gradient-proof/1"
    assert plan["policy"] is original.POLICY
    assert plan["judge_protocol"] == judge.protocol()
    assert plan["policy"]["fresh_optimizer_training_authorized"] is False


@pytest.mark.parametrize("legacy", [False, True])
def test_evidence_pins_new_sources_audit_and_the_original_word_protocol(
    tmp_path, monkeypatch, legacy
):
    protocol = {} if legacy else judge.protocol()
    base = {
        "initialization": {"original_model_plan": {"judge_protocol": protocol}},
        "source_sha256": {"previous-source": "unchanged"},
    }
    monkeypatch.setattr(old_evidence, "specification", lambda *a: base)
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(evidence, "artifact", partial(artifact, root=tmp_path))
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("c5_polarity_mixed_evidence.py", "probe_c5_polarity_mixed_semantic.py"):
        (tools / name).write_text("new source fixture")
    training = tmp_path / "runs/fresh/plan.json"
    training.parent.mkdir(parents=True)
    training.with_name("terminal-audit.json").write_text("complete audit fixture")
    if legacy:
        with pytest.raises(ValueError, match="same qualified word judge protocol"):
            evidence.specification(*([training] * 6))
    else:
        result = evidence.specification(*([training] * 6))
        assert result["judge_protocol"] == judge.protocol()
        assert result["word_terminal_audit"] == evidence.artifact(
            training.with_name("terminal-audit.json")
        )
        assert result["source_sha256"] == {
            **base["source_sha256"],
            **{str(path.relative_to(tmp_path)): original.sha256(path) for path in tools.iterdir()},
        }
        assert base["source_sha256"] == {"previous-source": "unchanged"}
        assert result["fresh_mixed_optimizer_training_authorized"] is False


@pytest.fixture
def polarity_candidate(candidates):  # noqa: F811
    clarify(candidates.directory)
    return candidates


def test_candidate_raw_authentication_requires_the_new_prompt(polarity_candidate):
    env = polarity_candidate
    assert proof.checked_candidates(env.path, env.plan_path, env.plan, env.tokenizer) == env.result
    with pytest.raises(ValueError, match="rubric or request/response"):
        original.checked_candidates(env.path, env.plan_path, env.plan, env.tokenizer)
    request = env.directory / "request-000/request.json"
    payload = read(request)
    payload["instructions"] = payload["instructions"].removesuffix(judge.CLARIFICATION)
    overwrite(request, payload)
    started = request.with_name("started.json")
    overwrite(started, {**read(started), "request_sha256": original.sha256(request)})
    with pytest.raises(ValueError, match="rubric or request/response"):
        proof.checked_candidates(env.path, env.plan_path, env.plan, env.tokenizer)


@pytest.mark.parametrize("negation", [3, 4])
def test_actual_negation_boundary_stays_three_without_score_replacement(
    polarity_candidate, negation
):
    env = polarity_candidate
    paired = env.result["comparison"]
    indices = mixed.shuffled_indices(paired, original.POLICY["training_rollout_seed"])
    slot = indices.index(paired["controls"][-1]["index"])
    request = env.directory / f"request-{slot // 16:03d}"
    response_path = request / "response.json"
    response = read(response_path)
    payload = json.loads(response["output"][0]["content"][0]["text"])
    payload["scores"][slot % 16]["score"] = negation
    response["output"][0]["content"][0]["text"] = json.dumps(payload)
    overwrite(response_path, response)
    scores_path = request / "scores.json"
    scores = read(scores_path)
    scores["scores"][slot % 16] = negation
    scores["response_sha256"] = original.sha256(response_path)
    overwrite(scores_path, scores)
    aggregate = read(env.path.with_name("scores.json"))
    aggregate[paired["controls"][-1]["index"]] = negation
    overwrite(env.path.with_name("scores.json"), aggregate)
    result = read(env.path)
    result["scores"] = original.artifact(env.path.with_name("scores.json"))
    overwrite(env.path, result)
    if negation == 3:
        assert proof.checked_candidates(env.path, env.plan_path, env.plan, env.tokenizer) == result
    else:
        with pytest.raises(ValueError, match="original calibration"):
            proof.checked_candidates(env.path, env.plan_path, env.plan, env.tokenizer)


@pytest.mark.parametrize("kind", ["word", "sentence"])
@pytest.mark.parametrize(
    "mutation", [None, "decline", "calibration", "response", "missing_pass", "legacy_prompt"]
)
def test_both_full_predecessors_reauthenticate_raw_polarity_passes_and_keep_stops(
    predecessor,  # noqa: F811
    monkeypatch,
    kind,
    mutation,
):
    env = predecessor(kind, decline=mutation == "decline", failed_control=mutation == "calibration")
    clarify(env.directory / "semantic")
    module = words if kind == "word" else sentences
    monkeypatch.setattr(module, "checked", lambda path: env.inference)
    monkeypatch.setattr(module, "checked_scores", lambda path: env.scores)
    if mutation == "response":
        path = next((env.directory / "semantic").glob("*/response.json"))
        overwrite(path, {**read(path), "model": "another-model"})
    elif mutation == "missing_pass":
        for path in (env.directory / "semantic").glob("pass-2-batch-*"):
            shutil.rmtree(path)
    elif mutation == "legacy_prompt":
        path = next((env.directory / "semantic").glob("*/request.json"))
        request = read(path)
        request["instructions"] = request["instructions"].removesuffix(judge.CLARIFICATION)
        overwrite(path, request)
        started = path.with_name("started.json")
        overwrite(started, {**read(started), "request_sha256": original.sha256(path)})
    if mutation is None:
        result = evidence.previous_evaluation(env.parent, kind, env.initialization)
        assert result["decision_values"]["calibration_passed"] is True
        assert result["decision_values"]["stop_for_semantic_decline"] is False
        assert result["decision_values"]["passes_semantic_research_gate"] is False
        assert result["stage_advance_or_promotion_authorized"] is False
    else:
        with pytest.raises(ValueError):
            evidence.previous_evaluation(env.parent, kind, env.initialization)


def test_mixed_proof_has_no_evaluation_only_decline_bypass():
    with pytest.raises(TypeError, match="evaluation_only"):
        evidence.previous_evaluation(None, "word", {}, evaluation_only=True)


def test_prepare_authenticates_actual_polarity_files_instead_of_pacer_return(
    candidates,  # noqa: F811
    monkeypatch,
):
    env = candidates
    output = env.path.parent.parent / "fresh-polarity"
    output.mkdir()

    class Pacer(FakePacer):
        def score(self, rows, directory):
            super().score(rows, directory)
            clarify(directory)
            return {"scores": [999]}

    monkeypatch.setattr(proof, "Pacer", Pacer)
    monkeypatch.setattr(original, "prepare_pools", lambda *a: env.result["pools"])
    monkeypatch.setattr(original, "require_model", lambda *a: None)
    monkeypatch.setattr(proof, "checked", lambda *a: env.plan)
    monkeypatch.setattr(original, "state_digest", lambda *a: "word-final")
    monkeypatch.setattr(original, "fixed_hashes", lambda *a: env.plan["initial_reference_hashes"])
    monkeypatch.setattr(original, "guard", lambda *a: None)
    monkeypatch.setattr(original, "wait_for_predecessor", lambda plan: plan["API_start_not_before"])
    monkeypatch.setattr(
        evidence,
        "load_student",
        lambda *a: (SimpleNamespace(vision=SimpleNamespace(state_dict=lambda: {})), env.tokenizer),
    )
    proof.prepare(env.plan_path, output, env.plan)
    result = read(output / "candidates.json")
    assert result["qualities"] == [[10, 3]] * 3
    assert 999 not in read(output / "scores.json")
    assert read(output / "semantic/request-000/request.json")["instructions"].endswith(
        judge.CLARIFICATION
    )


@pytest.mark.parametrize("failure", [None, "candidates", "qualify", "bad_rewards"])
def test_pipeline_uses_new_entrypoint_serial_children_and_never_retries(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(original, "ROOT", tmp_path)
    monkeypatch.setattr(original, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(proof, "checked", lambda *a: {})
    monkeypatch.setattr(original, "guard", lambda *a: None)
    monkeypatch.setattr(original, "candidate_tokenizer", lambda *a: None)
    monkeypatch.setattr(original, "identity", lambda pid, argv: {"pid": pid})
    events, commands = [], []
    path = tmp_path / "runs/proof/plan.json"
    path.parent.mkdir(parents=True)
    original.write(path, {})
    output = tmp_path / "checkpoints/proof"

    def child(command, check):
        assert check is False
        mode = command[command.index("--mode") + 1]
        commands.append(command)
        events.append(mode)
        folder = output / mode
        folder.mkdir()
        original.write(folder / "worker.json", {"pid": 33})
        original.write(folder / "exit.json", {"pid": 33, "exit_code": int(failure == mode)})
        filename = "candidates.json" if mode == "candidates" else "qualification.json"
        original.write(folder / filename, {"complete": True})
        return SimpleNamespace(returncode=int(failure == mode))

    def accepted(*args):
        events.append("authenticate polarity rewards")
        if failure == "bad_rewards":
            raise ValueError("invalid actual rewards")

    monkeypatch.setattr(original.subprocess, "run", child)
    monkeypatch.setattr(proof, "checked_candidates", accepted)
    monkeypatch.setattr(proof, "checked_qualification", lambda *a: events.append("full proof"))
    if failure is None:
        proof.pipeline(path, output)
        assert events == ["candidates", "authenticate polarity rewards", "qualify", "full proof"]
        assert read(output / "complete.json")["optimizer_updates"] == 0
    else:
        with pytest.raises((ValueError, RuntimeError)):
            proof.pipeline(path, output)
        assert events.count("candidates") == 1
        assert ("qualify" in events) == (failure == "qualify")
        assert not (output / "complete.json").exists()
    assert all(command[1] == str(Path(proof.__file__).resolve()) for command in commands)
    assert read(output / "exit.json")["exit_code"] == int(failure is not None)


@pytest.mark.parametrize("mode", ["candidates", "qualify", "pipeline"])
def test_real_cli_reaches_missing_plan_before_model_or_output_without_interop_error(tmp_path, mode):
    missing = tmp_path / "missing-plan.json"
    output = tmp_path / "uncreated-output"
    args = [
        sys.executable,
        str(Path(proof.__file__).resolve()),
        "--plan",
        str(missing),
        "--output",
        str(output),
    ]
    args += ["--pipeline"] if mode == "pipeline" else ["--mode", mode]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert "set_num_interop_threads" not in result.stderr
    assert not output.exists()
