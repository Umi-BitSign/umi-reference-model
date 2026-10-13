"""Equivalent float32 Fairseq exact-GELU arithmetic for Metal gradient qualification."""

from __future__ import annotations

import math

import torch


def portable_fairseq_gelu(value):
    original_dtype = value.dtype
    floats = value.float()
    return (0.5 * floats * (1.0 + torch.erf(floats / math.sqrt(2.0)))).to(original_dtype)


def replace_fairseq_exact_gelu(model):
    replaced = 0
    for module in model.modules():
        function = getattr(module, "activation_fn", None)
        if (
            getattr(function, "__module__", None) == "fairseq.modules.gelu"
            and getattr(function, "__name__", None) == "gelu"
        ):
            module.activation_fn = portable_fairseq_gelu
            replaced += 1
    return replaced
