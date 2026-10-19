"""Four-parameter, initially identity timing of frozen complete feature streams.

Positions use sample-index units. Every stream shares the same strictly
increasing positions, with endpoints fixed independently across visibility gaps.
This module supplies neither linguistic boundaries nor a training recipe.
Linear resampling has knots at integer positions; autograd selects one adjacent
interval there. Central finite-difference checks apply away from those knots.
"""

from __future__ import annotations

import math

import torch
from torch import nn

DEFAULT_MAXIMUM_LOG_RATE = math.log(2)

WIDTHS = {
    "face_features": 384,
    "left_hand_features": 384,
    "right_hand_features": 384,
    "pose_features": 14,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(streams, valid_intervals):
    require(
        isinstance(streams, dict) and set(streams) == set(WIDTHS),
        "timing requires the complete original visual and14D pose streams",
    )
    first = streams["pose_features"]
    require(
        isinstance(first, torch.Tensor) and first.ndim == 2,
        "timing requires two-dimensional frozen feature tensors",
    )
    frames = len(first)
    require(1 <= frames <= 4096, "timing clip length exceeds its explicit4096-frame bound")
    for name, width in WIDTHS.items():
        value = streams[name]
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.float32
            and value.shape == (frames, width)
            and not value.requires_grad
            and torch.isfinite(value).all(),
            "timing requires finite frozen CPU FP32 features with original widths",
        )
    require(
        isinstance(valid_intervals, torch.Tensor)
        and valid_intervals.device.type == "cpu"
        and valid_intervals.dtype == torch.bool
        and valid_intervals.shape == (frames - 1,),
        "timing requires an explicit complete boolean interval-visibility mask",
    )
    return frames


def components(valid_intervals, frames):
    start = 0
    result = []
    for index, valid in enumerate(valid_intervals.tolist()):
        if not valid:
            result.append((start, index + 1))
            start = index + 1
    result.append((start, frames))
    return result


def motion_features(streams, start, stop):
    """Within-component RMS motion, centered/scaled separately for each stream."""
    energy = torch.stack(
        [
            (v[start + 1 : stop] - v[start : stop - 1]).square().mean(1).sqrt()
            for v in streams.values()
        ],
        dim=1,
    )
    require(torch.isfinite(energy).all(), "feature motion overflowed finite FP32")
    normalized = energy / energy.mean(0).clamp_min(1e-6)
    return torch.tanh(normalized - normalized.mean(0))


class MonotoneFeatureTiming(nn.Module):
    def __init__(self, *, maximum_log_rate=DEFAULT_MAXIMUM_LOG_RATE):
        super().__init__()
        require(
            type(maximum_log_rate) in (int, float)
            and math.isfinite(maximum_log_rate)
            and 0 < maximum_log_rate <= math.log(4),
            "timing rate bound must be finite in(0,log4]",
        )
        self.maximum_log_rate = float(maximum_log_rate)
        self.rate_weights = nn.Parameter(torch.zeros(4, dtype=torch.float32))

    def forward(self, streams, *, valid_intervals):
        frames = validate(streams, valid_intervals)
        require(
            self.rate_weights.device.type == "cpu"
            and self.rate_weights.dtype == torch.float32
            and torch.isfinite(self.rate_weights).all(),
            "timing parameters must remain finite CPU FP32",
        )
        # Fix the stream order even when a caller constructs its dictionary in a different order.
        streams = {name: streams[name] for name in WIDTHS}
        position_parts, rate_parts = [], []
        for number, (start, stop) in enumerate(components(valid_intervals, frames)):
            length = stop - start
            if number:
                rate_parts.append(self.rate_weights.new_ones(1))
            if length == 1:
                position_parts.append(self.rate_weights.new_tensor([float(start)]))
                continue
            local = motion_features(streams, start, stop)
            rates = torch.exp(self.maximum_log_rate * torch.tanh(local @ self.rate_weights))
            cumulative = torch.cumsum(rates, 0)
            inside = cumulative[:-1] * ((length - 1) / cumulative[-1])
            position_parts.append(
                torch.cat((rates.new_zeros(1), inside, rates.new_tensor([float(length - 1)])))
                + start
            )
            rate_parts.append(rates)
        positions = torch.cat(position_parts)
        rates = torch.cat(rate_parts) if rate_parts else self.rate_weights.new_empty(0)
        require(
            positions.shape == (frames,)
            and rates.shape == (frames - 1,)
            and positions[0] == 0
            and positions[-1] == frames - 1
            and torch.all(positions[1:] > positions[:-1]),
            "timing lost complete monotone coverage",
        )
        if not torch.is_grad_enabled() and torch.count_nonzero(self.rate_weights).item() == 0:
            # Exact inference identity, including signed zero; never stop training gradients here.
            warped = {name: value.clone() for name, value in streams.items()}
        elif frames == 1:
            warped = {name: value.clone() for name, value in streams.items()}
        else:
            lower = positions.floor().long().clamp(0, frames - 2)
            fraction = (positions - lower).unsqueeze(1)
            warped = {
                name: value[lower] * (1 - fraction) + value[lower + 1] * fraction
                for name, value in streams.items()
            }
        identity = torch.arange(frames, dtype=torch.float32)
        displacement = ((positions - identity) / max(frames - 1, 1)).square().mean()
        adjacent = valid_intervals[1:] & valid_intervals[:-1]
        smoothness = (
            (rates[1:].log() - rates[:-1].log())[adjacent].square().mean()
            if adjacent.any()
            else self.rate_weights.sum() * 0
        )
        return {
            "streams": warped,
            "positions": positions,
            "interval_rates": rates,
            "normalized_displacement_penalty": displacement,
            "log_rate_smoothness_penalty": smoothness,
        }
