"""Zero-initialized local corrections to complete original C5 semantic memory.

This is an untrained architecture prototype. It preserves every original time
step and blocks local temporal mixing at the supplied full-rate visibility gaps.
No C5 backbone, dataset, decoder, optimizer or grader is loaded here.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 1536


def require(value, message):
    if not value:
        raise ValueError(message)


class TemporalMemoryAdapter(nn.Module):
    def __init__(self, *, rank=8, maximum_relative_residual=0.1, seed=20261019):
        super().__init__()
        require(
            type(rank) is int and 1 <= rank <= 64, "memory adapter rank must be an integer in1..64"
        )
        require(
            type(maximum_relative_residual) in (int, float)
            and math.isfinite(maximum_relative_residual)
            and 0 < maximum_relative_residual <= 1,
            "memory adapter needs a finite relative residual bound in(0,1]",
        )
        require(type(seed) is int and 0 <= seed < 2**63, "memory adapter seed is invalid")
        self.rank = rank
        self.maximum_relative_residual = float(maximum_relative_residual)
        self.seed = seed
        self.norm = nn.LayerNorm(WIDTH, elementwise_affine=False, eps=1e-5)
        # Restore the caller's RNG state; public model loading/rollouts keep their own seed.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.input_projection = nn.Linear(
                WIDTH, rank, bias=False, device="cpu", dtype=torch.float32
            )
            self.output_projection = nn.Linear(
                rank, WIDTH, bias=False, device="cpu", dtype=torch.float32
            )
            nn.init.zeros_(self.output_projection.weight)
        self.temporal_weights = nn.Parameter(
            torch.zeros(rank, 3, device="cpu", dtype=torch.float32)
        )
        with torch.no_grad():
            self.temporal_weights[:, 1].fill_(1)

    def configuration(self):
        return {
            "schema": "umi-studio-local-temporal-memory-adapter/1",
            "width": WIDTH,
            "rank": self.rank,
            "context_frames": 3,
            "seed": self.seed,
            "maximum_relative_residual": self.maximum_relative_residual,
            "scale": "per-frame original memory RMS with floor1e-6",
            "normalization_epsilon": 1e-5,
            "output_time_steps": "all original positions",
            "unknown_interval_policy": "no added neighbor mixing across any false interval",
            "initial_inference": "bitwise identity including signed zero",
            "trainable_parameters": 2 * WIDTH * self.rank + 3 * self.rank,
        }

    def validate(self, memory, valid_intervals):
        require(
            isinstance(memory, torch.Tensor)
            and memory.device.type == "cpu"
            and memory.dtype == torch.float32
            and memory.ndim == 3
            and memory.shape[0] == 1
            and memory.shape[1] > 0
            and memory.shape[2] == WIDTH
            and not memory.requires_grad
            and torch.isfinite(memory).all(),
            "memory adapter requires complete detached original CPU FP32 memory",
        )
        require(
            isinstance(valid_intervals, torch.Tensor)
            and valid_intervals.device.type == "cpu"
            and valid_intervals.dtype == torch.bool
            and valid_intervals.shape == (memory.shape[1] - 1,),
            "memory adapter requires every original interval visibility flag",
        )
        require(
            all(
                p.device.type == "cpu" and p.dtype == torch.float32 and torch.isfinite(p).all()
                for p in self.parameters()
            ),
            "memory adapter parameters must stay finite CPU FP32",
        )

    def forward(self, memory, *, valid_intervals):
        self.validate(memory, valid_intervals)
        if (
            not torch.is_grad_enabled()
            and torch.count_nonzero(self.output_projection.weight).item() == 0
        ):
            return {"memory": memory.clone(), "normalized_residual_penalty": memory.new_zeros(())}
        hidden = F.gelu(self.input_projection(self.norm(memory)))
        zero = hidden.new_zeros((1, 1, self.rank))
        left = torch.cat((zero, hidden[:, :-1]), dim=1)
        right = torch.cat((hidden[:, 1:], zero), dim=1)
        left_mask = torch.cat((valid_intervals.new_zeros(1), valid_intervals)).to(hidden.dtype)[
            None, :, None
        ]
        right_mask = torch.cat((valid_intervals, valid_intervals.new_zeros(1))).to(hidden.dtype)[
            None, :, None
        ]
        context = (
            left * left_mask * self.temporal_weights[:, 0]
            + hidden * self.temporal_weights[:, 1]
            + right * right_mask * self.temporal_weights[:, 2]
        )
        projected = self.output_projection(context)
        require(
            torch.isfinite(hidden).all()
            and torch.isfinite(context).all()
            and torch.isfinite(projected).all(),
            "memory adapter intermediate features overflowed",
        )
        scale = memory.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        require(torch.isfinite(scale).all(), "memory adapter original-memory scale overflowed")
        delta = self.maximum_relative_residual * scale * torch.tanh(projected)
        adapted = memory + delta
        penalty = (delta / scale).square().mean()
        require(
            adapted.shape == memory.shape
            and torch.isfinite(adapted).all()
            and torch.isfinite(penalty),
            "memory adapter changed complete finite output coverage",
        )
        return {"memory": adapted, "normalized_residual_penalty": penalty}
