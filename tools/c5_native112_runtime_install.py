"""Install admitted native112 branches into a private original runtime instance.

The caller owns export admission, model loading, resource limits and actual
video/detector/feature/token/latency verification. This module executes no model.
"""

from __future__ import annotations

import json

from c5_stride2_runtime import install_post_mediapipe_stride
from resolution_policy import set_visual_crop_size

GENERATION = {"length_penalty": 0.6, "max_length": 2048,
              "no_repeat_ngram_size": 16, "num_beams": 5}


def require(value, message):
    if not value:
        raise ValueError(message)


def cpu_stopping_criteria():
    from evaluate_tiny_rgb_feature_winner import CpuBudget
    from transformers import StoppingCriteriaList

    return StoppingCriteriaList([CpuBudget()])


class FixedGeneration:
    """Keep native feature-to-token dispatch with the full frozen generation recipe."""

    def __init__(self, base, tokenizer, *, stopping_factory, observe_tokens):
        self.base, self.tokenizer = base, tokenizer
        self.stopping_factory, self.observe_tokens = stopping_factory, observe_tokens

    def __getattr__(self, name):
        return getattr(self.base, name)

    def generate(self, *args, **kwargs):
        require(not args, "native runtime generation requires named complete streams")
        require(all(name in kwargs for name in
                    ("face_features", "left_hand_features", "right_hand_features", "pose_features")),
                "native runtime generation omitted a complete stream")
        require("stopping_criteria" not in kwargs, "native runtime replaced its CPU budget")
        required = {**GENERATION, "early_stopping": True, "do_sample": False,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id}
        require(set(kwargs) <= set(required) | {"face_features", "left_hand_features",
                                                "right_hand_features", "pose_features"},
                "native runtime introduced an unreviewed generation option")
        for name, value in required.items():
            require(name not in kwargs or json.dumps(kwargs[name]) == json.dumps(value),
                    "native runtime changed frozen generation: " + name)
        result = self.base.generate(**{**kwargs, **required,
                                      "stopping_criteria": self.stopping_factory()})
        if self.observe_tokens is not None:
            self.observe_tokens(result)
        return result


def install(runtime, embedders, generation, *, translator, stopping_factory=cpu_stopping_criteria,
            observe_tokens=None):
    require(json.dumps(generation, sort_keys=True) == json.dumps(GENERATION, sort_keys=True),
            "native112 runtime requires the full original generation recipe")
    require(callable(stopping_factory) and (observe_tokens is None or callable(observe_tokens)),
            "native112 runtime needs a CPU budget factory and optional token observer")
    require(set(embedders) == {"hands", "face"} and embedders["hands"] is not embedders["face"],
            "native112 runtime needs two independent admitted embedders")
    require(getattr(runtime, "_c5_temporal_stride", None) is None
            and getattr(runtime, "_c5_native112_installed", False) is False,
            "native112 runtime cannot reinstall crops or temporal stride")
    require(runtime.generation_max_length == GENERATION["max_length"]
            and runtime.generation_num_beams == GENERATION["num_beams"]
            and runtime.model_execution_concurrency == 1 and str(runtime.device) == "cpu",
            "native112 runtime changed generation bounds or execution concurrency")
    parameters = list(translator.parameters())
    require(parameters and callable(translator.generate)
            and all(not p.requires_grad and str(p.device) == "cpu" and str(p.dtype) == "torch.float32"
                    for p in parameters), "native112 requires an admitted frozen CPU FP32 translator")
    for embedder in embedders.values():
        embedder._check_model()
    resolution = set_visual_crop_size(runtime, 112)
    install_post_mediapipe_stride(runtime)
    runtime._hand_model, runtime._face_model = embedders["hands"], embedders["face"]
    runtime._translator = FixedGeneration(translator, runtime._tokenizer,
                                          stopping_factory=stopping_factory,
                                          observe_tokens=observe_tokens)
    runtime._c5_native112_installed = True
    runtime.component_devices = {**runtime.component_devices,
                                 "dino_hand": str(embedders["hands"].device),
                                 "dino_face": str(embedders["face"].device), "shubert_byt5": "cpu"}
    return {"resolution": resolution, "temporal_stride": 2, "generation": dict(GENERATION),
            "actual_video_detector_feature_token_or_latency_proof": False,
            "export_admission_performed_by_installer": False}
