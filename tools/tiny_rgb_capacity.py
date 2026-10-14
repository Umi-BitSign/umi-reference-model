"""Visual-capacity controls with unchanged small RGB inputs and supplied language weights."""

from __future__ import annotations

from dataclasses import asdict

from tiny_rgb_language_model import ScratchRgbVision, TinyRgbLanguageTranslator
from tiny_rgb_model import TinyRgbConfig
from tiny_rgb_mps_safe import replace_native_gelu
from torch import nn


def visual_config(capacity: str) -> TinyRgbConfig:
    """Vary visual attention width/depth only; retain pixel stem and token geometry."""
    configurations = {
        "small": TinyRgbConfig(),
        "medium": TinyRgbConfig(width=384, heads=8),
        "large": TinyRgbConfig(width=512, heads=8, encoder_layers=6),
    }
    if capacity not in configurations:
        raise ValueError(f"unknown visual capacity: {capacity}")
    return configurations[capacity]


def build_capacity_translator(language_model: nn.Module, capacity: str):
    """Construct scratch vision without reinitializing the supplied language decoder."""
    model = TinyRgbLanguageTranslator(language_model, visual_config(capacity))
    replace_native_gelu(model.vision)
    return model


def visual_budget(capacity: str, language_width: int = 512) -> dict:
    """Measure instantiated visual parameters; estimate only the linear bridge bytes."""
    if language_width < 1:
        raise ValueError("language width must be positive")
    config = visual_config(capacity)
    model = ScratchRgbVision(config)
    replacements = replace_native_gelu(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    stem = sum(parameter.numel() for parameter in model.stem.parameters())
    temporal = sum(parameter.numel() for parameter in model.encoder.parameters()) + sum(
        parameter.numel() for parameter in model.encoder_norm.parameters()
    )
    # Projection weight/bias and one RMS scale, exactly as TinyRgbLanguageTranslator.
    bridge = config.width * language_width + 2 * language_width
    return {
        "capacity": capacity,
        "visual_config": asdict(config),
        "scratch_visual_parameters": total,
        "convolution_stem_parameters": stem,
        "spatial_parameters": total - stem - temporal,
        "temporal_parameters": temporal,
        "scratch_bridge_parameters_at_language_width": bridge,
        "language_width": language_width,
        "visual_and_bridge_float32_weight_bytes": 4 * (total + bridge),
        "portable_gelu_replacements": replacements,
        "spatial_tokens_per_frame": config.spatial_queries,
        "pixel_input_geometry_changed": False,
        "pretrained_visual_weights_loaded": False,
        "optimizer_updates": 0,
        "metal_training_numerics_verified": False,
        "asl_training_fit_verified": False,
        "held_out_quality_verified": False,
    }


if __name__ == "__main__":
    import json

    import torch

    torch.set_num_threads(2)
    print(json.dumps([visual_budget(name) for name in ("small", "medium", "large")], indent=2))
