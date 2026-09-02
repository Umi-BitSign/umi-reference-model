from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch
from torch import Tensor

from .canonical import canonical_json_sha256

S1_TENSOR_SET_DIGEST_PROFILE = "umi-s1-tensor-set-v1"
S1_TENSOR_SET_DOMAIN = b"umi-s1-tensor-set-v1\0"


class S1StateDigestError(ValueError):
    """Raised when an S1 tensor set cannot be hashed canonically."""


def tensor_payload_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise S1StateDigestError("state value is not a tensor")
    tensor = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def s1_tensor_set_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash a complete named tensor set with the FLEURS trainer's canonical profile."""

    if not state or not all(isinstance(name, str) and name for name in state):
        raise S1StateDigestError("state tensor names must be nonempty strings")
    descriptor = [
        {
            "name": name,
            "dtype": str(state[name].dtype),
            "shape": list(state[name].shape),
            "tensor_sha256": tensor_payload_sha256(state[name]),
        }
        for name in sorted(state)
    ]
    return canonical_json_sha256(descriptor, domain=S1_TENSOR_SET_DOMAIN)
