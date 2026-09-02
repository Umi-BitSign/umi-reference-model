from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

MOTION_FEATURE_DIM = 1_184
DEFAULT_MAX_FRAMES = 120
DEFAULT_MAX_OUTPUT_TOKENS = 128
DEFAULT_VOCABULARY_SIZE = 4_096

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3


def _is_compiling() -> bool:
    return torch.compiler.is_compiling()


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _time_mask(frame_mask: Tensor, dtype: torch.dtype) -> Tensor:
    return frame_mask.unsqueeze(-1).to(dtype=dtype)


def _key_padding_bias(frame_mask: Tensor, dtype: torch.dtype) -> Tensor:
    valid = frame_mask.to(dtype=dtype).unsqueeze(1).unsqueeze(1)
    return (valid - 1.0) * 10_000.0


def _validate_motion_inputs(
    motion: Tensor,
    frame_mask: Tensor,
    *,
    max_frames: int,
) -> None:
    if motion.ndim != 3 or motion.shape[1:] != (max_frames, MOTION_FEATURE_DIM):
        raise ValueError(f"motion must have shape [batch, {max_frames}, {MOTION_FEATURE_DIM}]")
    if frame_mask.ndim != 2 or frame_mask.shape != motion.shape[:2]:
        raise ValueError(f"frame_mask must have shape [batch, {max_frames}]")
    if frame_mask.dtype != torch.int32:
        raise TypeError("frame_mask must use torch.int32")
    if not motion.is_floating_point():
        raise TypeError("motion must use a floating-point dtype")


@dataclass(frozen=True)
class PortableS0Config:
    vocabulary_size: int = 46
    max_frames: int = DEFAULT_MAX_FRAMES
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    hidden_size: int = 128
    dropout: float = 0.10

    def validate(self) -> None:
        if self.vocabulary_size < 4:
            raise ValueError("vocabulary_size must be at least 4")
        if self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if self.max_output_tokens < 2:
            raise ValueError("max_output_tokens must be at least 2")
        if self.hidden_size < 16 or self.hidden_size % 8 != 0:
            raise ValueError("hidden_size must be at least 16 and divisible by 8")
        if not 0.0 <= self.dropout < 0.5:
            raise ValueError("dropout must be in [0, 0.5)")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class _MaskedTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise_in = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.pointwise_out = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor, frame_mask: Tensor) -> Tensor:
        mask = frame_mask.unsqueeze(1).to(dtype=value.dtype)
        value = value * mask
        residual = value
        value = self.depthwise(value)
        value = self.activation(self.pointwise_in(value))
        value = self.dropout(self.pointwise_out(value))
        return self.norm(value + residual) * mask


class _OutputTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise_in = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.pointwise_out = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.depthwise(value)
        value = self.activation(self.pointwise_in(value))
        value = self.dropout(self.pointwise_out(value))
        return self.norm(value + residual)


class PortableS0(nn.Module):
    """Fixed-query character diagnostic over the flat UMI motion contract."""

    def __init__(self, config: PortableS0Config | None = None) -> None:
        super().__init__()
        self.config = config or PortableS0Config()
        self.config.validate()

        hidden_size = self.config.hidden_size
        self.stem = nn.Sequential(
            nn.Linear(MOTION_FEATURE_DIM, hidden_size),
            nn.GELU(approximate="tanh"),
        )
        self.temporal_blocks = nn.ModuleList(
            _MaskedTemporalBlock(hidden_size, 2**layer, self.config.dropout) for layer in range(4)
        )
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_queries = nn.Parameter(torch.empty(self.config.max_output_tokens, hidden_size))
        nn.init.normal_(self.output_queries, mean=0.0, std=0.02)
        self.output_mixer = nn.ModuleList(
            _OutputTemporalBlock(hidden_size, dilation, self.config.dropout) for dilation in (1, 2)
        )
        self.output_projection = nn.Linear(hidden_size, self.config.vocabulary_size)
        self.attention_scale = hidden_size**-0.5

    def validate_inputs(self, motion: Tensor, frame_mask: Tensor) -> None:
        _validate_motion_inputs(
            motion,
            frame_mask,
            max_frames=self.config.max_frames,
        )

    def forward(self, motion: Tensor, frame_mask: Tensor) -> Tensor:
        if not _is_compiling():
            self.validate_inputs(motion, frame_mask)

        mask = _time_mask(frame_mask, motion.dtype)
        encoded = self.stem(motion * mask) * mask
        encoded = encoded.transpose(1, 2)
        for block in self.temporal_blocks:
            encoded = block(encoded, frame_mask)
        encoded = encoded.transpose(1, 2)

        keys = self.key_projection(encoded)
        values = self.value_projection(encoded)
        queries = self.output_queries.unsqueeze(0).expand(motion.shape[0], -1, -1)
        scores = torch.matmul(queries, keys.transpose(1, 2)) * self.attention_scale
        scores = scores + _key_padding_bias(frame_mask, scores.dtype).squeeze(1)
        decoded = torch.matmul(torch.softmax(scores, dim=-1), values) + queries
        decoded = decoded.transpose(1, 2)
        for block in self.output_mixer:
            decoded = block(decoded)
        return self.output_projection(decoded.transpose(1, 2))

    @property
    def parameter_count(self) -> int:
        return _parameter_count(self)


@dataclass(frozen=True)
class PortableS1Config:
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE
    max_frames: int = DEFAULT_MAX_FRAMES
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    hidden_size: int = 256
    num_heads: int = 4
    feed_forward_size: int = 1_024
    encoder_layers: int = 4
    decoder_layers: int = 4
    dropout: float = 0.10

    def validate(self) -> None:
        if self.vocabulary_size <= UNK_TOKEN_ID:
            raise ValueError("vocabulary_size must contain all reserved token IDs")
        if self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if self.max_output_tokens < 2:
            raise ValueError("max_output_tokens must be at least 2")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.hidden_size < 16 or self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be at least 16 and divisible by num_heads")
        if self.feed_forward_size < self.hidden_size:
            raise ValueError("feed_forward_size must be at least hidden_size")
        if self.encoder_layers < 1 or self.decoder_layers < 1:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        if not 0.0 <= self.dropout < 0.5:
            raise ValueError("dropout must be in [0, 0.5)")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class _PortableAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.scale = self.head_size**-0.5
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, value: Tensor) -> Tensor:
        batch_size, sequence_length, _ = value.shape
        value = value.reshape(batch_size, sequence_length, self.num_heads, self.head_size)
        return value.transpose(1, 2)

    def _merge_heads(self, value: Tensor) -> Tensor:
        value = value.transpose(1, 2)
        batch_size, sequence_length, _, _ = value.shape
        return value.reshape(batch_size, sequence_length, self.num_heads * self.head_size)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        additive_mask: Tensor,
    ) -> Tensor:
        queries = self._split_heads(self.query_projection(query))
        keys = self._split_heads(self.key_projection(key_value))
        values = self._split_heads(self.value_projection(key_value))
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores + additive_mask.to(dtype=scores.dtype), dim=-1)
        context = torch.matmul(self.dropout(weights), values)
        return self.output_projection(self._merge_heads(context))


class _FeedForward(nn.Module):
    def __init__(self, hidden_size: int, feed_forward_size: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, feed_forward_size)
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(feed_forward_size, hidden_size)

    def forward(self, value: Tensor) -> Tensor:
        value = self.activation(self.input_projection(value))
        return self.output_projection(self.dropout(value))


class _EncoderLayer(nn.Module):
    def __init__(self, config: PortableS1Config) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = _PortableAttention(
            config.hidden_size,
            config.num_heads,
            config.dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward = _FeedForward(
            config.hidden_size,
            config.feed_forward_size,
            config.dropout,
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, value: Tensor, frame_mask: Tensor, additive_mask: Tensor) -> Tensor:
        mask = _time_mask(frame_mask, value.dtype)
        normalized = self.attention_norm(value)
        value = value + self.residual_dropout(self.attention(normalized, normalized, additive_mask))
        value = value * mask
        value = value + self.residual_dropout(self.feed_forward(self.feed_forward_norm(value)))
        return value * mask


class _DecoderLayer(nn.Module):
    def __init__(self, config: PortableS1Config) -> None:
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(config.hidden_size)
        self.self_attention = _PortableAttention(
            config.hidden_size,
            config.num_heads,
            config.dropout,
        )
        self.cross_attention_norm = nn.LayerNorm(config.hidden_size)
        self.cross_attention = _PortableAttention(
            config.hidden_size,
            config.num_heads,
            config.dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward = _FeedForward(
            config.hidden_size,
            config.feed_forward_size,
            config.dropout,
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        value: Tensor,
        memory: Tensor,
        decoder_mask: Tensor,
        self_attention_mask: Tensor,
        memory_attention_mask: Tensor,
    ) -> Tensor:
        mask = _time_mask(decoder_mask, value.dtype)
        normalized = self.self_attention_norm(value)
        value = value + self.residual_dropout(
            self.self_attention(normalized, normalized, self_attention_mask)
        )
        value = value * mask
        value = value + self.residual_dropout(
            self.cross_attention(
                self.cross_attention_norm(value),
                memory,
                memory_attention_mask,
            )
        )
        value = value * mask
        value = value + self.residual_dropout(self.feed_forward(self.feed_forward_norm(value)))
        return value * mask


class PortableS1(nn.Module):
    """Portable autoregressive motion-to-text Transformer reference model."""

    def __init__(self, config: PortableS1Config | None = None) -> None:
        super().__init__()
        self.config = config or PortableS1Config()
        self.config.validate()

        self.motion_stem = nn.Sequential(
            nn.Linear(MOTION_FEATURE_DIM, self.config.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )
        self.encoder_positions = nn.Parameter(
            torch.empty(self.config.max_frames, self.config.hidden_size)
        )
        self.encoder_layers = nn.ModuleList(
            _EncoderLayer(self.config) for _ in range(self.config.encoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(self.config.hidden_size)

        self.token_embedding = nn.Embedding(
            self.config.vocabulary_size,
            self.config.hidden_size,
            padding_idx=PAD_TOKEN_ID,
        )
        self.decoder_positions = nn.Parameter(
            torch.empty(self.config.max_output_tokens, self.config.hidden_size)
        )
        self.decoder_layers = nn.ModuleList(
            _DecoderLayer(self.config) for _ in range(self.config.decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(self.config.hidden_size)
        self.output_projection = nn.Linear(
            self.config.hidden_size,
            self.config.vocabulary_size,
        )
        self.output_projection.weight = self.token_embedding.weight

        causal_mask = torch.triu(
            torch.full(
                (self.config.max_output_tokens, self.config.max_output_tokens),
                -10_000.0,
            ),
            diagonal=1,
        )
        self.register_buffer(
            "causal_attention_bias",
            causal_mask.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )
        nn.init.normal_(self.encoder_positions, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_positions, mean=0.0, std=0.02)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def validate_encoder_inputs(self, motion: Tensor, frame_mask: Tensor) -> None:
        _validate_motion_inputs(
            motion,
            frame_mask,
            max_frames=self.config.max_frames,
        )

    def validate_decoder_inputs(
        self,
        decoder_input_ids: Tensor,
        memory: Tensor,
        frame_mask: Tensor,
    ) -> None:
        if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[1] != (
            self.config.max_output_tokens
        ):
            raise ValueError(
                f"decoder_input_ids must have shape [batch, {self.config.max_output_tokens}]"
            )
        if decoder_input_ids.dtype != torch.int64:
            raise TypeError("decoder_input_ids must use torch.int64")
        expected_memory_shape = (
            decoder_input_ids.shape[0],
            self.config.max_frames,
            self.config.hidden_size,
        )
        if memory.shape != expected_memory_shape:
            raise ValueError(f"memory must have shape {expected_memory_shape}")
        if frame_mask.shape != (*decoder_input_ids.shape[:1], self.config.max_frames):
            raise ValueError(f"frame_mask must have shape [batch, {self.config.max_frames}]")
        if frame_mask.dtype != torch.int32:
            raise TypeError("frame_mask must use torch.int32")

    def encode(self, motion: Tensor, frame_mask: Tensor) -> Tensor:
        if not _is_compiling():
            self.validate_encoder_inputs(motion, frame_mask)

        mask = _time_mask(frame_mask, motion.dtype)
        value = self.motion_stem(motion * mask)
        value = (value + self.encoder_positions.unsqueeze(0)) * mask
        additive_mask = _key_padding_bias(frame_mask, value.dtype)
        for layer in self.encoder_layers:
            value = layer(value, frame_mask, additive_mask)
        return self.encoder_norm(value) * mask

    def decode(
        self,
        decoder_input_ids: Tensor,
        memory: Tensor,
        frame_mask: Tensor,
    ) -> Tensor:
        if not _is_compiling():
            self.validate_decoder_inputs(decoder_input_ids, memory, frame_mask)

        decoder_mask = decoder_input_ids.ne(PAD_TOKEN_ID).to(dtype=torch.int32)
        mask = _time_mask(decoder_mask, memory.dtype)
        value = self.token_embedding(decoder_input_ids)
        value = (value + self.decoder_positions.unsqueeze(0)) * mask
        self_attention_mask = (
            _key_padding_bias(decoder_mask, value.dtype) + self.causal_attention_bias
        )
        memory_attention_mask = _key_padding_bias(frame_mask, value.dtype)
        for layer in self.decoder_layers:
            value = layer(
                value,
                memory,
                decoder_mask,
                self_attention_mask,
                memory_attention_mask,
            )
        value = self.decoder_norm(value) * mask
        return self.output_projection(value)

    def forward(
        self,
        motion: Tensor,
        frame_mask: Tensor,
        decoder_input_ids: Tensor,
    ) -> Tensor:
        memory = self.encode(motion, frame_mask)
        return self.decode(decoder_input_ids, memory, frame_mask)

    @torch.no_grad()
    def greedy_decode(
        self,
        motion: Tensor,
        frame_mask: Tensor,
        *,
        max_new_tokens: int | None = None,
    ) -> Tensor:
        token_count = self.config.max_output_tokens if max_new_tokens is None else max_new_tokens
        if not 1 <= token_count <= self.config.max_output_tokens:
            raise ValueError("max_new_tokens must be within the configured output window")

        memory = self.encode(motion, frame_mask)
        batch_size = motion.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, self.config.max_output_tokens),
            PAD_TOKEN_ID,
            dtype=torch.int64,
            device=motion.device,
        )
        decoder_input_ids[:, 0] = BOS_TOKEN_ID
        generated = torch.full(
            (batch_size, token_count),
            PAD_TOKEN_ID,
            dtype=torch.int64,
            device=motion.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=motion.device)

        for position in range(token_count):
            logits = self.decode(decoder_input_ids, memory, frame_mask)
            next_token = torch.argmax(logits[:, position], dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, PAD_TOKEN_ID),
                next_token,
            )
            generated[:, position] = next_token
            finished = torch.logical_or(finished, next_token.eq(EOS_TOKEN_ID))
            if bool(torch.all(finished).item()):
                break
            if position + 1 < self.config.max_output_tokens:
                decoder_input_ids[:, position + 1] = next_token
        return generated

    @property
    def parameter_count(self) -> int:
        return _parameter_count(self)
