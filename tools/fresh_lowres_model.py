"""Raw-frame, low-resolution translator with no sign-specific initialization.

The module deliberately accepts a language model and a frame encoder as
constructor inputs. A clean-pretrained experiment may use generic ImageNet and
T5-family weights, while the strict scratch control passes randomly initialized
versions of the same components. Neither path consumes C5/SHuBERT weights or
cached C5 encoder states.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def resize_video_short_side(video: torch.Tensor, short_side: int) -> torch.Tensor:
    """Resize TCHW RGB frames without changing their aspect ratio."""
    if video.ndim != 4 or video.shape[1] != 3 or not video.is_floating_point():
        raise ValueError("video must be a floating-point TCHW RGB tensor")
    if short_side < 64:
        raise ValueError("short side must be at least 64 pixels")
    height, width = video.shape[-2:]
    if height < 1 or width < 1:
        raise ValueError("video frames must have positive dimensions")
    scale = short_side / min(height, width)
    output_height = max(1, round(height * scale))
    output_width = max(1, round(width * scale))
    if (output_height, output_width) == (height, width):
        return video
    return F.interpolate(
        video,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


class MobileNetFrameEncoder(nn.Module):
    """Apply MobileNetV3-Small to each frame and return one vector per frame."""

    output_width = 576

    def __init__(self, features: nn.Module, frame_batch_size: int = 32) -> None:
        super().__init__()
        if frame_batch_size < 1:
            raise ValueError("frame batch size must be positive")
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.frame_batch_size = frame_batch_size

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape batch x time x 3 x height x width")
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        sequential_features = isinstance(self.features, nn.Sequential)
        trainable_indices = (
            [
                index
                for index, block in enumerate(self.features)
                if any(parameter.requires_grad for parameter in block.parameters())
            ]
            if sequential_features
            else []
        )
        if sequential_features and trainable_indices and trainable_indices != list(
            range(trainable_indices[0], len(self.features))
        ):
            raise ValueError("trainable MobileNet blocks must form a suffix")
        encoded_chunks = []
        for chunk in flat.split(self.frame_batch_size):
            if not sequential_features:
                encoded_chunks.append(self.pool(self.features(chunk)).flatten(1))
                continue
            hidden = chunk
            trainable_start = trainable_indices[0] if trainable_indices else len(self.features)
            with torch.no_grad():
                for block in self.features[:trainable_start]:
                    hidden = block(hidden)
            hidden = hidden.contiguous()
            for block in self.features[trainable_start:]:
                hidden = block(hidden)
            encoded_chunks.append(self.pool(hidden).flatten(1))
        encoded = torch.cat(encoded_chunks)
        if encoded.shape[1] != self.output_width:
            raise ValueError("MobileNet frame feature width changed")
        return encoded.reshape(batch, time, self.output_width)


def mobilenet_v3_small_encoder(weights=None, frame_batch_size: int = 32) -> MobileNetFrameEncoder:
    """Build the visual encoder; ``weights=None`` is the strict scratch arm."""
    from torchvision.models import mobilenet_v3_small

    backbone = mobilenet_v3_small(weights=weights)
    return MobileNetFrameEncoder(backbone.features, frame_batch_size)


class S3DVideoEncoder(nn.Module):
    """Return one spatially pooled Kinetics feature vector per S3D time step."""

    output_width = 1024

    def __init__(self, features: nn.Module) -> None:
        super().__init__()
        self.features = features

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape batch x time x 3 x height x width")
        encoded = self.features(frames.permute(0, 2, 1, 3, 4))
        if encoded.ndim != 5 or encoded.shape[1] != self.output_width:
            raise ValueError("S3D feature topology changed")
        return encoded.mean(dim=(-2, -1)).transpose(1, 2).contiguous()


def s3d_video_encoder(weights=None) -> S3DVideoEncoder:
    from torchvision.models.video import s3d

    backbone = s3d(weights=weights)
    return S3DVideoEncoder(backbone.features)


class DiscardedTextEncoder(nn.Module):
    main_input_name = "input_ids"

    def forward(self, *args, **kwargs):
        raise RuntimeError("the fresh model cannot run the discarded text encoder")


def discard_unused_text_encoder(language_model: T5ForConditionalGeneration) -> int:
    """Release the T5 text encoder, which is bypassed by visual encoder states."""
    encoder = language_model.encoder
    if isinstance(encoder, DiscardedTextEncoder):
        return 0
    removed = sum(parameter.numel() for parameter in encoder.parameters())
    if removed < 1:
        raise ValueError("language model has no removable text encoder")
    language_model.encoder = DiscardedTextEncoder()
    return removed


class FreshLowResolutionTranslator(nn.Module):
    """Encode raw frames temporally, then decode text with a supplied T5 model."""

    def __init__(
        self,
        language_model: T5ForConditionalGeneration,
        frame_encoder: nn.Module,
        frame_feature_width: int,
        *,
        temporal_width: int = 256,
        temporal_layers: int = 3,
        temporal_heads: int = 8,
        temporal_feedforward_width: int = 1024,
        max_frames: int = 192,
        dropout: float = 0.1,
        use_language_encoder: bool = False,
    ) -> None:
        super().__init__()
        if (
            frame_feature_width < 1
            or temporal_width < 1
            or temporal_layers < 1
            or temporal_heads < 1
            or temporal_width % temporal_heads
            or max_frames < 1
        ):
            raise ValueError("invalid fresh-model topology")
        self.frame_encoder = frame_encoder
        self.frame_projection = nn.Linear(frame_feature_width, temporal_width)
        self.temporal_positions = nn.Parameter(torch.empty(max_frames, temporal_width))
        layer = nn.TransformerEncoderLayer(
            d_model=temporal_width,
            nhead=temporal_heads,
            dim_feedforward=temporal_feedforward_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=temporal_layers, norm=nn.LayerNorm(temporal_width)
        )
        self.language_projection = nn.Linear(
            temporal_width, language_model.config.d_model, bias=False
        )
        self.language_model = language_model
        self.max_frames = max_frames
        self.use_language_encoder = use_language_encoder
        if use_language_encoder and isinstance(language_model.encoder, DiscardedTextEncoder):
            raise ValueError("language encoder was discarded before model construction")
        nn.init.normal_(self.temporal_positions, mean=0.0, std=temporal_width**-0.5)

    def _encode_with_mask(
        self, frames: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError("frames must have five dimensions")
        if frame_mask.shape != frames.shape[:2] or frame_mask.dtype != torch.bool:
            raise ValueError("frame mask must be boolean with shape batch x time")
        if frames.shape[1] > self.max_frames or not torch.all(frame_mask.any(dim=1)):
            raise ValueError("clip is empty or exceeds the configured frame limit")
        frame_parameters = list(self.frame_encoder.parameters())
        frame_device = frame_parameters[0].device if frame_parameters else frames.device
        if frame_device != frames.device and any(
            parameter.requires_grad for parameter in frame_parameters
        ):
            raise ValueError("a trainable visual encoder cannot span devices")
        hidden = self.frame_encoder(frames.to(frame_device))
        if hidden.ndim != 3 or hidden.shape[0] != frames.shape[0]:
            raise ValueError("visual encoder returned an invalid sequence")
        projection_device = self.frame_projection.weight.device
        frame_mask = frame_mask.to(projection_device)
        if hidden.shape[1] != frame_mask.shape[1]:
            frame_mask = (
                F.adaptive_max_pool1d(
                    frame_mask.to(dtype=torch.float32).unsqueeze(1), hidden.shape[1]
                )
                .squeeze(1)
                .to(dtype=torch.bool)
            )
        hidden = self.frame_projection(hidden.to(projection_device))
        scale = math.sqrt(hidden.shape[-1])
        hidden = hidden * scale + self.temporal_positions[: hidden.shape[1]]
        hidden = self.temporal_encoder(hidden, src_key_padding_mask=~frame_mask)
        hidden = self.language_projection(hidden)
        if self.use_language_encoder:
            hidden = self.language_model.encoder(
                inputs_embeds=hidden,
                attention_mask=frame_mask,
                return_dict=True,
            ).last_hidden_state
        return hidden, frame_mask

    def encode(self, frames: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        return self._encode_with_mask(frames, frame_mask)[0]

    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        memory, memory_mask = self._encode_with_mask(frames, frame_mask)
        return self.language_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=memory_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        **generation_options,
    ) -> torch.Tensor:
        memory, memory_mask = self._encode_with_mask(frames, frame_mask)
        return self.language_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=memory_mask,
            **generation_options,
        )


def initialization_manifest(
    *,
    visual_initialization: str,
    language_initialization: str,
    source_short_side: int,
    temporal_sample_fps: float,
    use_language_encoder: bool = False,
    file_sha256: Callable[[str], str] | None = None,
) -> dict:
    """Describe an experiment's initialization without implying weight provenance."""
    allowed_visual = {"random", "imagenet1k_v1", "kinetics400_s3d_v1"}
    allowed_language = {"random", "google_byt5_small", "google_t5_small"}
    if visual_initialization not in allowed_visual:
        raise ValueError("unsupported visual initialization")
    if language_initialization not in allowed_language:
        raise ValueError("unsupported language initialization")
    if source_short_side < 64 or temporal_sample_fps <= 0:
        raise ValueError("invalid spatial or temporal sampling policy")
    manifest = {
        "schema": "umi-studio-fresh-lowres-initialization/1",
        "visual_initialization": visual_initialization,
        "language_initialization": language_initialization,
        "source_short_side": source_short_side,
        "source_resize_policy": "aspect_preserving_short_side",
        "temporal_sample_fps": temporal_sample_fps,
        "language_encoder_policy": (
            f"generic_{language_initialization.removeprefix('google_')}"
            if use_language_encoder and language_initialization != "random"
            else "random" if use_language_encoder else "discarded"
        ),
        "sign_specific_checkpoint": None,
        "sign_specific_encoder_cache": False,
    }
    if file_sha256 is not None:
        manifest["model_source_sha256"] = file_sha256(__file__)
    return manifest
