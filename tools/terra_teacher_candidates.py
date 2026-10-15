"""Reference-blind C5 text candidates for the existing Terra risk objective."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from benchmark_fleurs_dev import normalize, sha256
from evaluate_tiny_rgb_feature_winner import peak_bytes
from terra_visual_learning import bounded_candidate, candidate_pool
from tiny_rgb_visual_teacher_data import ROOT, load_targets
from transformers import StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutput

DECODE = {"max_length": 513, "num_beams": 1, "do_sample": False, "use_cache": True}
CACHE_MAX_BYTES = 128 * 1024**2
TRAINING_ROLLOUT_LIMITS = {
    "maximum_process_peak_bytes": 32 * 1024**3,
    "maximum_seconds_per_clip": 1200,
}


class TrainingRolloutBudget:
    """Mixed CPU/Metal training process only; evaluator's24GiB policy is untouched."""

    def __init__(self, *, clock=time.perf_counter, memory=peak_bytes):
        self.clock, self.memory, self.calls = clock, memory, 0
        self.started = clock()

    def __call__(self, input_ids, scores, **kwargs):
        self.calls += 1
        if self.calls == 1 or self.calls % 64 == 0:
            elapsed, peak = self.clock() - self.started, self.memory()
            if (
                elapsed > TRAINING_ROLLOUT_LIMITS["maximum_seconds_per_clip"]
                or peak > TRAINING_ROLLOUT_LIMITS["maximum_process_peak_bytes"]
            ):
                error = RuntimeError(
                    f"training CPU rollout limit exceeded: peak={peak}, seconds={elapsed:.3f}"
                )
                error.terra_resource_limits = dict(TRAINING_ROLLOUT_LIMITS)
                raise error
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def generation_binding(pair, checkpoint_sha):
    raw, teacher = pair["raw"], pair["teacher"]
    if (
        pair.get("feature_pretraining_allowed") is not True
        or raw["role"] != "training"
        or raw["training_allowed"] is not True
        or teacher["role"] != "training"
        or teacher["training_allowed"] is not True
        or teacher["source"] != raw["source"]
        or teacher["id"] != raw["example_id"]
        or normalize(teacher["reference"]) != normalize(raw["reference"])
    ):
        raise ValueError("teacher candidates require aligned approved training identities")
    return {
        "source": raw["source"],
        "example_id": raw["example_id"],
        "role": "training",
        "training_allowed": True,
        "checkpoint_sha256": checkpoint_sha,
        "native_features_sha256": teacher["cache_sha256"],
        "native_sidecar_sha256": teacher.get("sidecar_sha256"),
        "frame_indices_sha256": pair["header"]["frame_indices_sha256"],
        "sampled_frames": pair["header"]["sampled_frames"],
        "decode": DECODE,
        "implementation_sha256": sha256(Path(__file__)),
    }


def binding_key(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()


def checked_candidate(value, tokenizer):
    tokens = value["tokens"]
    if (
        not tokens
        or len(tokens) > 512
        or any(type(token) is not int or not 0 <= token < 384 for token in tokens)
        or value.get("kind") != "c5_teacher"
    ):
        raise ValueError("invalid cached teacher token sequence")
    expected = bounded_candidate(torch.tensor([0, *tokens]), 512)
    if any(expected[key] != value[key] for key in ("tokens", "truncated", "termination")):
        raise ValueError("cached teacher termination differs from observed tokens")
    text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if text != value["text"]:
        raise ValueError("cached teacher text differs from its actual token sequence")
    return value


class TeacherTextCache:
    def __init__(self, root, *, workspace=ROOT, maximum_bytes=CACHE_MAX_BYTES):
        self.root = root.resolve()
        if not self.root.is_relative_to(workspace / "runs"):
            raise ValueError("teacher text cache must remain in the owned runs/ subtree")
        self.root.mkdir(parents=True, exist_ok=True)
        files = list(self.root.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in files):
            raise ValueError("teacher text cache refuses links or nested directories")
        self.used = sum(path.stat().st_size for path in files)
        self.maximum_bytes = maximum_bytes
        self.workspace = workspace
        if self.used > maximum_bytes:
            raise RuntimeError("teacher text cache exceeds its explicit byte budget")

    def get(self, binding, tokenizer):
        path = self.root / f"{binding_key(binding)}.json"
        if not path.exists():
            return None
        if path.is_symlink() or path.stat().st_size > 65536:
            raise ValueError("teacher text cache entry is unsafe or too large")
        try:
            row = json.loads(path.read_text())
            if row["generation_inputs"] != binding:
                raise ValueError("teacher cache input/model/decode bindings changed")
            return checked_candidate(row["candidate"], tokenizer)
        except (KeyError, TypeError) as error:
            raise ValueError("teacher cache evidence is incomplete") from error

    def put(self, binding, candidate):
        from terra_training_storage import POLICY

        path = self.root / f"{binding_key(binding)}.json"
        payload = (
            json.dumps({"generation_inputs": binding, "candidate": candidate}, sort_keys=True)
            + "\n"
        ).encode()
        if self.used + len(payload) > self.maximum_bytes:
            raise RuntimeError("teacher text cache reached its explicit byte budget")
        required = (
            POLICY["physical_floor_bytes"]
            + POLICY["headroom_bytes"]
            + POLICY["output_budget_bytes"]
            + self.maximum_bytes
            - self.used
        )
        if shutil.disk_usage(self.workspace).free < required:
            raise RuntimeError(
                "teacher text cache lacks its physical floor and future output reserves"
            )
        # Publish a complete entry atomically, without overwriting another writer.
        with tempfile.NamedTemporaryFile(
            dir=self.root, prefix="owned-", suffix=".partial", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            self.used += len(payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("teacher text cache has conflicting generation evidence") from None
        finally:
            temporary.unlink()


@torch.no_grad()
def teacher_candidate(student, tokenizer, pair, *, checkpoint_sha, cache):
    binding = generation_binding(pair, checkpoint_sha)
    for field in ("cache", "sidecar"):
        if pair["teacher"].get(field) and not (
            ROOT / pair["teacher"][field]
        ).resolve().is_relative_to(ROOT / "data"):
            raise ValueError("native teacher files must stay in the approved data/ subtree")
    cached = cache.get(binding, tokenizer)
    if cached is not None:
        return cached
    if sha256(ROOT / pair["teacher"]["cache"]) != pair["teacher"]["cache_sha256"]:
        raise ValueError("native teacher feature bytes changed")
    if next(student.parameters()).device.type != "cpu":
        raise ValueError("this teacher candidate protocol is CPU-only")
    student.eval()
    features = {name: torch.from_numpy(value)[None] for name, value in load_targets(pair).items()}
    native = student.semantic_encoder(**features, return_dict=True).last_hidden_state
    if (
        native.shape != (1, binding["sampled_frames"], student.language.config.d_model)
        or not torch.isfinite(native).all()
    ):
        raise ValueError("teacher candidate native timeline/width/finite values differ")
    emitted = bounded_candidate(
        student.language.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=native),
            attention_mask=torch.ones(native.shape[:2], dtype=torch.long),
            **DECODE,
            stopping_criteria=StoppingCriteriaList([TrainingRolloutBudget()]),
        )[0],
        512,
    )
    candidate = {
        **emitted,
        "text": tokenizer.decode(
            emitted["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ),
        "kind": "c5_teacher",
    }
    checked_candidate(candidate, tokenizer)
    cache.put(binding, candidate)
    return candidate


def teacher_augmented_pool(
    student, tokenizer, video, mask, reference, *, pair, seed, checkpoint_sha, cache
):
    generation_binding(pair, checkpoint_sha)  # Validate roles before any sampler runs.
    if reference != pair["raw"]["reference"]:
        raise ValueError("teacher candidate human anchor differs from its approved identity")
    pool = candidate_pool(
        student, tokenizer, video, mask, reference, seed=seed, budget_factory=TrainingRolloutBudget
    )
    teacher = teacher_candidate(
        student, tokenizer, pair, checkpoint_sha=checkpoint_sha, cache=cache
    )
    if teacher["tokens"] not in [candidate["tokens"] for candidate in pool]:
        pool.append(teacher)
    return pool


def make_visual_initialization(
    source_plan_path, checkpoint, target_plan, *, expected_sha256, root=ROOT
):
    """Transfer our Terra-trained vision, with a new optimizer and a new real proof."""
    from safetensors import safe_open
    from terra_training_recovery import artifact, verify_snapshot

    source_plan_path, checkpoint = source_plan_path.resolve(), checkpoint.resolve()
    source = json.loads(source_plan_path.read_text())
    verify_snapshot(source_plan_path, source)
    shared = (
        "reference",
        "evaluation",
        "winner_control",
        "historical_controls",
        "gates",
        "resource_policy",
        "resource_policy_sha256",
        "runtime_binding",
        "native_pairs_sha256",
        "proof_cases",
        "training_examples",
        "planned_training_examples",
    )
    if any(source[key] != target_plan[key] for key in shared):
        raise ValueError(
            "visual transfer cannot change approved data, controls, resources or quality gates"
        )
    if (
        source["recipe"]["judge_model"] != "gpt-5.6-terra"
        or source["recipe"]["feature_distillation"] is not False
        or source["recipe"]["latent_distillation"] is not False
        or any(
            source["recipe"][key] != target_plan["recipe"][key]
            for key in (
                "capacity",
                "short_side",
                "target_fps",
                "max_frames",
                "max_text_tokens",
                "seed",
            )
        )
    ):
        raise ValueError("teacher arm initialization must be the matching Terra-only visual model")
    source_artifact, visual_artifact = (
        artifact(source_plan_path, root=root),
        artifact(checkpoint, root=root),
    )
    if visual_artifact["sha256"] != expected_sha256:
        raise ValueError("teacher arm initial visual checkpoint bytes changed")
    with safe_open(str(checkpoint), framework="pt") as file:
        metadata = file.metadata()
    commit_path = checkpoint.with_suffix(".json")
    commit = json.loads(commit_path.read_text())
    if (
        metadata["plan_sha256"] != source_artifact["sha256"]
        or metadata["fixed_winner_checkpoint"]
        != target_plan["reference"]["translator_checkpoint_sha256"]
        or commit.get("complete") is not True
        or commit["visual"] != visual_artifact
        or str(commit["training_examples"]) != metadata["update"]
        or str(commit["optimizer_updates"]) != metadata["optimizer_updates"]
    ):
        raise ValueError("teacher arm initial visual checkpoint provenance/counters differ")
    return {
        "schema": "umi-studio-terra-teacher-visual-initialization/1",
        "source_plan": source_artifact,
        "visual": visual_artifact,
        "checkpoint_commit": artifact(commit_path, root=root),
        "source_training_examples": int(metadata["update"]),
        "source_optimizer_updates": int(metadata["optimizer_updates"]),
        "optimizer_state_reused": False,
        "source_is_feature_distillation": False,
        "new_teacher_objective_numerical_qualification_required": True,
    }


def checked_visual_initialization(plan, *, expected_sha256, root=ROOT):
    from terra_training_recovery import checked_artifact

    initialization = plan["visual_initialization"]
    paths = {
        key: checked_artifact(initialization[key], root=root)
        for key in ("source_plan", "visual", "checkpoint_commit")
    }
    expected = make_visual_initialization(
        paths["source_plan"], paths["visual"], plan, expected_sha256=expected_sha256, root=root
    )
    if initialization != expected:
        raise ValueError("teacher arm visual initialization descriptor changed")
    return paths


def install_visual_initialization(student, plan, *, expected_sha256, root=ROOT):
    from evaluate_terra_visual import install_weights
    from safetensors import safe_open
    from safetensors.torch import load_file

    paths = checked_visual_initialization(plan, expected_sha256=expected_sha256, root=root)
    with safe_open(str(paths["visual"]), framework="pt") as file:
        metadata = file.metadata()
    initialization = plan["visual_initialization"]
    install_weights(
        student,
        load_file(str(paths["visual"]), device="cpu"),
        metadata,
        parent_sha=initialization["source_plan"]["sha256"],
        reference_sha=plan["reference"]["translator_checkpoint_sha256"],
        update=initialization["source_training_examples"],
    )
