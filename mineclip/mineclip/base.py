"""
Base API for importing pretrained video models
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional

import mineclip.utils as U


__all__ = ["VideoRewardBase"]

# calculated from 21K video clips, which contains 2.8M frames
MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)


def compute_contribution_matrix(recorded_blocks, projection_matrix):
    num_layers = len(recorded_blocks)
    batch_size, num_heads, seq_len, _ = recorded_blocks[0]['attention_patterns'].shape
    embed_dim = projection_matrix.shape[1]

    # Initialize contribution tensor
    c = torch.zeros((num_layers, num_heads, seq_len, embed_dim), device=projection_matrix.device)

    for l in range(num_layers):
        attn_patterns = recorded_blocks[l]['attention_patterns']  # [batch, heads, seq, seq]
        layer_output = recorded_blocks[l]['output']  # [seq, batch, hidden]
        layer_output = layer_output.permute(1, 0, 2)  # [batch, seq, hidden]

        for h in range(num_heads):
            head_attention = attn_patterns[:, h, :, :]  # [batch, seq, seq]

            for i in range(seq_len):
                pos_attention = head_attention[:, :, i]  # [batch, seq]
                pos_attention = pos_attention.unsqueeze(-1)  # [batch, seq, 1]

                weighted_output = layer_output * pos_attention  # [batch, seq, hidden]
                avg_contribution = weighted_output.mean(dim=0)  # [seq, hidden]

                # Take just position i's contribution and project it
                c[l, h, i] = (avg_contribution[i] @ projection_matrix)  # [embed_dim]

    return c


class VideoRewardBase(nn.Module):
    def __init__(
        self,
        *,
        image_encoder: nn.Module,
        temporal_encoder: nn.Module,
        reward_head: nn.Module,
    ):
        """
        Args:
          image_encoder: [B, C, H, W] -> [B, F]
          temporal_encoder: [B, L, F] -> [B, F]
          reward_head: [B, F] -> [B, D] softmax over D classes/dims
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.temporal_encoder = temporal_encoder
        self.reward_head = reward_head

    def forward_image_features(self, frames):
        # print("A. Starting forward_image_features", flush=True)
        assert frames.ndim >= 4
        leading_dims = frames.size()[:-3]
        C, H, W = frames.size()[-3:]
        frames = frames.view(-1, C, H, W)
        # print("B. About to preprocess", flush=True)
        frames = U.basic_image_tensor_preprocess(
            frames, mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD
        )
        # print("C. About to run image encoder", flush=True)
        features = self.image_encoder(frames)
        # print("D. Got features from image encoder", flush=True)
        # print("E. Recorded blocks:", self.image_encoder.recorded_blocks, flush=True)
        # print(type(self.image_encoder.recorded_blocks[0]['input']))
        # print(self.image_encoder.recorded_blocks[0]['input'].shape)
        # print(len(self.image_encoder.recorded_blocks.keys()))
        # print(type(self.image_encoder.recorded_blocks[0]['attention_patterns']))
        # print(self.image_encoder.recorded_blocks[0]['attention_patterns'].shape)
        # print(self.image_encoder.projection.shape)

        c = compute_contribution_matrix(
            self.image_encoder.recorded_blocks,
            self.image_encoder.projection
        )

        # print(c.shape)

        # quit()
        return features.view(*leading_dims, features.size(-1))


    def forward_video_features(self, image_features):
        """
        [B, L, F] -> [B, F]
        """
        B, L, F = image_features.size()
        video_feats = self.temporal_encoder(image_features)
        assert video_feats.shape[0] == B
        return video_feats

    def forward_reward_head(self, video_features, text_tokens=None, softmax=False):
        """
        [B, F] -> [B, D]
        """
        B, F = video_features.size()
        if text_tokens is not None:
            rewards = self.reward_head(video_features, text_tokens)
        else:
            rewards = self.reward_head(video_features)
        if torch.is_tensor(rewards):
            assert rewards.shape[0] == B
            if softmax:
                rewards = torch.nn.functional.softmax(rewards, dim=1)
        return rewards

    def forward(self, videos, text_tokens=None, is_video_features=False):
        """
        Args:
            videos: [B, F] if is_video_features else [B, L, C, H, W]
            is_video_features: pass in [B, F] of already-computed video features
            text_tokens: [B, L, D]
        """
        if is_video_features:
            assert videos.ndim == 2
            return self.forward_reward_head(videos, text_tokens=text_tokens)
        else:
            assert videos.ndim == 5, "video must be 5D (raw pixels)"
            return self.forward_reward_head(
                self.forward_video_features(self.forward_image_features(videos)),
                text_tokens=text_tokens,
            )

    def load_ckpt(self, ckpt_or_path, strip_prefix="model.", strict=False):
        if isinstance(ckpt_or_path, dict):
            ckpt = ckpt_or_path
        else:
            ckpt_path = U.f_expand(ckpt_or_path)
            assert U.f_exists(ckpt_path), f"ckpt not found: {ckpt_path}"
            ckpt = U.torch_load(ckpt_path)
        # `ret` might contain key matching info if strict=False
        ret = U.load_state_dict(
            self, ckpt["state_dict"], strip_prefix=strip_prefix, strict=strict
        )
        return ret
