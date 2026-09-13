import logging
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import random
import os
import sys
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec import Wav2Vec2Config, TransformerEncoder
from fairseq.modules import LayerNorm
logger = logging.getLogger(__name__)

@dataclass
class SignHubertConfig(Wav2Vec2Config):
    conv_pos: int = field(default=32)
    discrete: bool = field(default=False)
    codebook_size: int = field(default=256)
    channels_embed_dim: int = field(default=384)
    channels_pose_embed_dim: int = field(default=14)
    intermediate_dim: int = field(default=1024)
    mask_strategy: str = field(default='random')
    channels: str = field(default='face,left_hand,right_hand,body_posture')

@register_model('signhubert_onlyhands', dataclass=SignHubertConfig)
class SignHubertModel(BaseFairseqModel):

    def __init__(self, cfg: SignHubertConfig):
        super().__init__()
        self.cfg = cfg
        self.discrete = cfg.discrete
        self.embed = cfg.encoder_embed_dim
        self.channel_embed = cfg.channels_embed_dim
        self.channel_pose_embed = cfg.channels_pose_embed_dim
        self.intermediate_dim = cfg.intermediate_dim
        self.channels = cfg.channels.split(',')
        self.post_extract_proj = nn.Linear(cfg.intermediate_dim, cfg.encoder_embed_dim)
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_strategy = cfg.mask_strategy
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.feature_grad_mult = cfg.feature_grad_mult
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, 1, cfg.intermediate_dim // len(self.channels)).uniform_())
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.channel_embed * len(self.channels))
        if 'face' in self.channels:
            self.layer_norm_face = LayerNorm(self.channel_embed)
            self.face_proj = nn.Linear(self.channel_embed, cfg.intermediate_dim // len(self.channels))
        if 'left_hand' in self.channels:
            self.layer_norm_lhand = LayerNorm(self.channel_embed)
            self.left_hand_proj = nn.Linear(self.channel_embed, cfg.intermediate_dim // len(self.channels))
        if 'right_hand' in self.channels:
            self.layer_norm_rhand = LayerNorm(self.channel_embed)
            self.right_hand_proj = nn.Linear(self.channel_embed, cfg.intermediate_dim // len(self.channels))
        if 'body_posture' in self.channels:
            self.layer_norm_body = LayerNorm(self.channel_pose_embed)
            self.body_posture_proj = nn.Linear(self.channel_pose_embed, cfg.intermediate_dim // len(self.channels))
        self.codebook_size = cfg.codebook_size
        self.heads = []
        for i in range(len(self.channels)):
            self.heads.append(nn.Linear(cfg.encoder_embed_dim, cfg.codebook_size))
        self.heads = torch.nn.ModuleList(self.heads)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        return state

    @classmethod
    def build_model(cls, cfg: SignHubertConfig, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def apply_mask(self, x, padding_mask, mask_indices=None, mask_channel_indices=None):
        B, T, C, D = x.shape
        mask = torch.ones_like(x)
        if self.mask_prob > 0 and self.mask_strategy == 'channel':
            if mask_indices is None:
                mask_indices = torch.zeros_like(x[:, :, :, 0], dtype=bool)
                num_channels_to_mask = int(C * self.mask_prob)
                num_channels_to_mask = max(1, num_channels_to_mask)
                for i in range(B):
                    channels_to_mask = np.random.choice(C, num_channels_to_mask, replace=False)
                    mask_indices[i, :, channels_to_mask] = True
            mask[mask_indices.unsqueeze(-1).expand(-1, -1, -1, D)] = 0
        elif self.mask_prob > 0 and self.mask_strategy == 'gloss':
            if mask_indices is None:
                mask_indices_channel = compute_mask_indices((B, T), padding_mask, self.mask_prob, self.mask_length, self.mask_selection, self.mask_other, min_masks=1, no_overlap=self.no_mask_channel_overlap, min_space=self.mask_min_space, require_same_masks=self.cfg.require_same_masks, mask_dropout=self.cfg.mask_dropout)
                mask_indices_channel = torch.from_numpy(mask_indices_channel).to(x.device)
            mask_indices = mask_indices_channel.unsqueeze(2).expand(-1, -1, C)
            mask_indices = mask_indices.unsqueeze(3).expand(-1, -1, -1, D)
            mask[mask_indices] = 0
        elif self.mask_prob > 0 and self.mask_strategy == 'random':
            if mask_indices is None:
                mask_indices = compute_mask_indices((B, T * C), padding_mask, self.mask_prob, self.mask_length, self.mask_selection, self.mask_other, min_masks=1, no_overlap=self.no_mask_channel_overlap, min_space=self.mask_min_space, require_same_masks=self.cfg.require_same_masks, mask_dropout=self.cfg.mask_dropout)
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            mask_indices = mask_indices.view(B, T, C)
            mask_indices = mask_indices.unsqueeze(3).expand(-1, -1, -1, D)
            mask[mask_indices] = 0
        else:
            raise ValueError(f'unknown mask strategy {self.mask_strategy}')
        x = x * mask + self.mask_emb * (1 - mask)
        return (x, mask)

    def forward(self, source, padding_mask=None, mask=True, features_only=False, layer=None, mask_indices=None, mask_channel_indices=None, padding_count=None, kmeans_labels=None):
        channels_to_use = []
        for c in self.channels:
            if c in source[0]:
                channels_to_use.append(c)
        for c in channels_to_use:
            if c == 'face':
                face_features_list = []
                label_face_features_list = []
            elif c == 'left_hand':
                left_hand_features_list = []
                label_left_hand_features_list = []
            elif c == 'right_hand':
                right_hand_features_list = []
                label_right_hand_features_list = []
            elif c == 'body_posture':
                body_posture_features_list = []
                label_body_posture_features_list = []
        for sample in source:
            for c in channels_to_use:
                if c == 'face':
                    face_features_list.append(sample['face'])
                    label_face_features_list.append(sample['label_face'])
                elif c == 'left_hand':
                    left_hand_features_list.append(sample['left_hand'])
                    label_left_hand_features_list.append(sample['label_left_hand'])
                elif c == 'right_hand':
                    right_hand_features_list.append(sample['right_hand'])
                    label_right_hand_features_list.append(sample['label_right_hand'])
                elif c == 'body_posture':
                    body_posture_features_list.append(sample['body_posture'])
                    label_body_posture_features_list.append(sample['label_body_posture'])
        features_list = []
        labels_list = []
        for c in channels_to_use:
            if c == 'face':
                face_features = torch.stack(face_features_list)
                face_labels = torch.stack(label_face_features_list)
                face_features = self.layer_norm_face(face_features)
                face_features = self.face_proj(face_features)
                features_list.append(face_features)
                labels_list.append(face_labels)
            elif c == 'left_hand':
                left_hand_features = torch.stack(left_hand_features_list)
                left_hand_labels = torch.stack(label_left_hand_features_list)
                left_hand_features = self.layer_norm_lhand(left_hand_features)
                left_hand_features = self.left_hand_proj(left_hand_features)
                features_list.append(left_hand_features)
                labels_list.append(left_hand_labels)
            elif c == 'right_hand':
                right_hand_features = torch.stack(right_hand_features_list)
                right_hand_labels = torch.stack(label_right_hand_features_list)
                right_hand_features = self.layer_norm_rhand(right_hand_features)
                right_hand_features = self.right_hand_proj(right_hand_features)
                features_list.append(right_hand_features)
                labels_list.append(right_hand_labels)
            elif c == 'body_posture':
                body_posture_features = torch.stack(body_posture_features_list)
                body_posture_labels = torch.stack(label_body_posture_features_list)
                body_posture_features = self.layer_norm_body(body_posture_features)
                body_posture_features = self.body_posture_proj(body_posture_features)
                features_list.append(body_posture_features)
                labels_list.append(body_posture_labels)
        features = torch.stack(features_list, dim=2)
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, mask_indices=mask_indices, mask_channel_indices=mask_channel_indices)
        else:
            x = features
            mask_indices = None
        x = self.dropout_input(x)
        x = x.view(x.size(0), x.size(1), -1)
        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)
        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)
        if features_only:
            return {'x': x, 'padding_mask': padding_mask, 'layer_results': layer_results}
        result = {'losses': {}}
        predictions = []
        for i, head in enumerate(self.heads):
            channel_pred = head(x)
            predictions.append(channel_pred)
        predictions = torch.stack(predictions, dim=2)
        labels = torch.stack(labels_list, dim=2)
        predictions_flat = predictions.view(-1, self.codebook_size)
        labels_flat = labels.view(-1)
        labels_flat = labels_flat.squeeze(-1)
        mask_indices_reduced = mask_indices.any(dim=-1)
        mask_indices_flat = mask_indices_reduced.view(-1)
        masked_loss = F.cross_entropy(predictions_flat[mask_indices_flat == 0], labels_flat[mask_indices_flat == 0], reduction='none')
        result['losses']['kmeans_loss'] = masked_loss
        if 'sample_size' not in result:
            result['sample_size'] = masked_loss.numel()
        return result

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0), device=y.device)
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)
            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)
            var = zss / (zc - 1) - zs ** 2 / (zc * (zc - 1))
            return torch.sqrt(var + 1e-06).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-06).mean()

    def extract_features(self, source, padding_mask, kmeans_labels, mask=False, layer=None):
        res = self.forward(source, padding_mask, mask=mask, features_only=True, layer=layer, kmeans_labels=kmeans_labels)
        return res

    def remove_pretraining_modules(self, last_layer=None):
        self.heads = None
        self.final_proj = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList((l for i, l in enumerate(self.encoder.layers) if i <= last_layer))
