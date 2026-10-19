"""Embed native RGB crops with a caller-owned, frozen research DINO backbone.

The caller authenticates exported weights and supplies a bounded execution
context. This helper loads no model and installs no runtime. Actual trained
CPU/Metal feature, token and latency comparisons remain required.
"""

from __future__ import annotations

import numpy as np
import torch
from c5_patch_projection import LinearPatchProjection
from torch import nn


class Native112FrameEmbedder:
    """Match the native runtime's frame-embedding method without its224px resize.

    Inputs are already RGB, after the runtime's BGR conversion. Each complete
    batch, including its host transfer, runs inside the supplied context. Its
    caller owns admission, device/resource policy and exclusive model access.
    """

    def __init__(self, model, *, batch_context, batch_size=2):
        if type(batch_size) is not int or batch_size < 1 or not callable(batch_context):
            raise ValueError("native112 embedding requires a positive batch and execution context")
        if not isinstance(model, nn.Module):
            raise ValueError("native112 embedding requires a private frozen DINO module")
        self.model, self.batch_context, self.batch_size = model, batch_context, batch_size
        parameters = list(model.parameters())
        if not parameters:
            raise ValueError("native112 backbone has no parameters")
        self.device = parameters[0].device
        self._check_model()

    def _check_model(self):
        projection = getattr(getattr(self.model, "patch_embed", None), "proj", None)
        if (
            any(module.training for module in self.model.modules())
            or any(
                parameter.requires_grad
                or parameter.dtype != torch.float32
                or parameter.device != self.device
                for parameter in self.model.parameters()
            )
            or any(buffer.device != self.device for buffer in self.model.buffers())
            or not isinstance(projection, LinearPatchProjection)
            or tuple(getattr(self.model, "pos_embed", torch.empty(0)).shape) != (1, 257, 384)
            or tuple(getattr(self.model, "mask_token", torch.empty(0)).shape) != (1, 384)
        ):
            raise ValueError("native112 embedding requires frozen FP32 DINO with linear patches")
        return projection

    def extract_embeddings_from_frames(self, frames):
        """Return every input frame in order as a finite FP32 N-by-384 array."""
        if (
            not isinstance(frames, list | tuple | np.ndarray)
            or (isinstance(frames, np.ndarray) and frames.ndim != 4)
            or len(frames) < 1
            or any(
                not isinstance(frame, np.ndarray)
                or frame.dtype != np.uint8
                or frame.shape != (112, 112, 3)
                for frame in frames
            )
        ):
            raise ValueError("native112 embedding needs complete uint8 RGB112px crops")
        projection = self._check_model()
        parts = []
        for start in range(0, len(frames), self.batch_size):
            with self.batch_context(), torch.inference_mode():
                pixels = (
                    torch.from_numpy(np.stack(frames[start : start + self.batch_size]))
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .float()
                    .div_(255)
                )
                mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                # Training normalizes on CPU before its device transfer.
                pixels = ((pixels - mean) / std).to(self.device)
                calls = 0

                def check_projection(module, args, expected=pixels):
                    nonlocal calls
                    calls += 1
                    if calls != 1 or len(args) != 1 or args[0] is not expected:
                        raise ValueError("native112 backbone altered its normalized patch input")

                handle = projection.register_forward_pre_hook(check_projection)
                try:
                    features = self.model(pixels)
                finally:
                    handle.remove()
                if (
                    calls != 1
                    or not isinstance(features, torch.Tensor)
                    or features.shape != (len(pixels), 384)
                    or features.dtype != torch.float32
                    or features.device != self.device
                    or not torch.isfinite(features).all()
                ):
                    raise ValueError(
                        "native112 backbone changed complete finite FP32 feature coverage"
                    )
                parts.append(features.cpu().numpy().copy())
        return np.concatenate(parts, axis=0)
