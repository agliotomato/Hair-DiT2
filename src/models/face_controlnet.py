"""
FaceControlNet: SD3.5 ControlNet for face/boundary blending in hair inpainting.

Initialized from SD3.5 transformer blocks 18~23 (last 6 double blocks).
Role: provide residuals for the last 6 SD3.5 blocks to blend the hair-face boundary.

Design rationale (from InstantX Qwen-Image ControlNet Inpainting):
  - Copy the last 6 pretrained transformer blocks for the face/boundary ControlNet
  - These late blocks specialize in fine detail and boundary coherence
  - FaceControlNet residuals are injected only into SD3.5 blocks 18~23,
    blended with HairControlNet residuals using the matte mask

Forward:
  inputs:
    noisy_latent:            (B, 16, 64, 64)  noisy latent (same as HairControlNet)
    face_latent:             (B, 16, 64, 64)  VAE-encoded masked_face = face * (1-matte)
    encoder_hidden_states:   (B, 333, 4096)   null text embeddings from HairControlNet
    pooled_projections:      (B, 2048)        null pooled from HairControlNet
    sigmas:                  (B,)             flow matching sigma values

  output:
    block_samples: list of 6 residuals, each (B, seq_len, inner_dim)
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from diffusers import SD3ControlNetModel, SD3Transformer2DModel
from diffusers.models.normalization import AdaLayerNormZero


# Index range of source SD3.5 transformer blocks to copy into FaceControlNet
FACE_CONTROLNET_START_BLOCK = 18
FACE_CONTROLNET_END_BLOCK = 24   # exclusive


class FaceControlNet(nn.Module):
    """
    SD3.5 ControlNet for face boundary blending.

    Trainable:
      - controlnet (SD3ControlNetModel, 6 blocks initialized from SD3.5 blocks 18~23)

    No null embeddings — these are provided by HairControlNet and passed in.
    No MatteCNN — face_latent (16ch) is the direct conditioning input.
    """

    def __init__(
        self,
        model_id: str,
        num_layers: int = 6,
        local_files_only: bool = False,
    ):
        super().__init__()

        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )

        # Build 6-block ControlNet (initially copies blocks 0~5)
        # num_extra_conditioning_channels=0: face_latent is pure 16ch, no extra channel
        self.controlnet = SD3ControlNetModel.from_transformer(
            transformer,
            num_layers=num_layers,
            num_extra_conditioning_channels=0,
            load_weights_from_transformer=True,
        )

        # Replace transformer_blocks with deep copies of blocks 18~23
        source_blocks = transformer.transformer_blocks[
            FACE_CONTROLNET_START_BLOCK:FACE_CONTROLNET_END_BLOCK
        ]
        self.controlnet.transformer_blocks = nn.ModuleList([
            copy.deepcopy(blk) for blk in source_blocks
        ])

        # SD3.5 block 23 has context_pre_only=True which breaks the 2-tuple return
        # in SD3ControlNetModel.forward. Setting it to False requires replacing
        # norm1_context (AdaLayerNormContinuous) with AdaLayerNormZero to match
        # the expected forward call signature.
        for blk in self.controlnet.transformer_blocks:
            if blk.context_pre_only:
                # inner_dim: norm1 is AdaLayerNormZero, linear maps time_emb → 6*inner_dim
                inner_dim = blk.norm1.linear.weight.shape[0] // 6
                blk.norm1_context = AdaLayerNormZero(inner_dim, bias=True).to(
                    dtype=blk.norm1.linear.weight.dtype,
                    device=blk.norm1.linear.weight.device,
                )
                blk.context_pre_only = False

        del transformer
        torch.cuda.empty_cache()

    def forward(
        self,
        noisy_latent: torch.Tensor,
        face_latent: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Args:
            noisy_latent:          (B, 16, 64, 64) noisy latent
            face_latent:           (B, 16, 64, 64) VAE-encoded masked_face
            encoder_hidden_states: (B, 333, 4096) null text embeddings (from HairControlNet)
            pooled_projections:    (B, 2048) null pooled (from HairControlNet)
            sigmas:                (B,) flow matching sigmas

        Returns:
            block_samples: list of 6 tensors, each (B, seq_len, inner_dim)
                           — residuals for SD3.5 blocks 18~23
        """
        device = noisy_latent.device
        dtype = noisy_latent.dtype

        block_samples = self.controlnet(
            hidden_states=noisy_latent,
            controlnet_cond=face_latent.to(device=device, dtype=dtype),
            encoder_hidden_states=encoder_hidden_states.to(device=device, dtype=dtype),
            pooled_projections=pooled_projections.to(device=device, dtype=dtype),
            timestep=sigmas.to(device=device, dtype=dtype),
            return_dict=False,
        )[0]

        return block_samples  # list of 6 residuals
