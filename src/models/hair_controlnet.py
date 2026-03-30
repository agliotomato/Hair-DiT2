"""
HairControlNet: SD3.5 ControlNet for hair region generation.

Components:
- SD3ControlNetModel (trainable, initialized from SD3.5-medium transformer)
- MatteCNN: 1ch matte → 16ch features at latent resolution (512→64)
- null_encoder_hidden_states: nn.Parameter (1, 333, 4096), learned null text embedding
- null_pooled_projections:    nn.Parameter (1, 2048), learned null pooled embedding

Forward:
  inputs: noisy_latent (B,16,64,64), sketch (B,3,512,512), matte (B,1,512,512), sigmas (B,)
  1. sketch → frozen VAE encode → sketch_latent (B,16,64,64)
  2. matte → MatteCNN → matte_feat (B,16,64,64)
  3. ctrl_cond = sketch_latent + matte_feat
  4. SD3ControlNetModel forward → block_samples
  5. return block_samples, null_encoder_hs (expanded to B), null_pooled (expanded to B)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import SD3ControlNetModel, SD3Transformer2DModel

import torch.nn.functional as F

from src.models.vae_wrapper import VAEWrapper


class MatteCNN(nn.Module):
    """
    Lightweight CNN to embed 1-channel matte into 16-channel latent-resolution features.

    Spatial: 512 → 256 → 128 → 64  (three stride-2 convolutions)
    Channels:  1  →  16 →  32 →  16
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 1 → 16, 512 → 256
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            # Block 2: 16 → 32, 256 → 128
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            # Block 3: 32 → 16, 128 → 64
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )

    def forward(self, matte: torch.Tensor) -> torch.Tensor:
        """
        Args:
            matte: (B, 1, 512, 512) in [0, 1]
        Returns:
            feat: (B, 16, 64, 64)
        """
        return self.net(matte)


class HairControlNet(nn.Module):
    """
    SD3.5 ControlNet model for hair region synthesis.

    Trainable components:
      - controlnet (SD3ControlNetModel)
      - matte_cnn (MatteCNN)
      - null_encoder_hidden_states (nn.Parameter)
      - null_pooled_projections (nn.Parameter)

    Frozen components (passed in, not stored as submodules):
      - vae (VAEWrapper) — used only for sketch encoding in forward()
    """

    # Null embedding shapes for SD3.5-medium
    NULL_ENC_SHAPE = (1, 333, 4096)
    NULL_POOLED_SHAPE = (1, 2048)

    def __init__(
        self,
        model_id: str,
        vae: VAEWrapper,
        num_layers: int = 12,
        local_files_only: bool = False,
    ):
        super().__init__()

        # Load transformer temporarily just to initialize ControlNet
        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )

        self.controlnet = SD3ControlNetModel.from_transformer(
            transformer,
            num_layers=num_layers,
            load_weights_from_transformer=True,
            # SD3ControlNetModel defaults to extra_conditioning_channels=1,
            # so pos_embed_input expects 17ch. We provide 17ch ctrl_cond:
            #   16ch: sketch_latent + matte_feat
            #    1ch: raw matte_latent (explicit spatial mask)
        )
        # Free transformer memory — it's held separately in Trainer
        del transformer
        torch.cuda.empty_cache()

        self.matte_cnn = MatteCNN()

        # Learned null text conditioning (trained alongside ControlNet)
        self.null_encoder_hidden_states = nn.Parameter(
            torch.zeros(*self.NULL_ENC_SHAPE)
        )
        self.null_pooled_projections = nn.Parameter(
            torch.zeros(*self.NULL_POOLED_SHAPE)
        )

        # Keep VAE reference (frozen, not a submodule to avoid double registration)
        self._vae = vae

    def forward(
        self,
        noisy_latent: torch.Tensor,
        sketch: torch.Tensor,
        matte: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_latent: (B, 16, 64, 64) noisy target latent
            sketch:       (B, 3, 512, 512) colored sketch in [0, 1]
            matte:        (B, 1, 512, 512) hair matte in [0, 1]
            sigmas:       (B,) flow matching sigma values

        Returns:
            block_samples:      list of 12 ControlNet residuals, each (B, seq_len, inner_dim)
            null_encoder_hs:    (B, 333, 4096) expanded null text embeddings
            null_pooled:        (B, 2048) expanded null pooled embeddings
        """
        B = noisy_latent.shape[0]
        device = noisy_latent.device
        dtype = noisy_latent.dtype

        # 1. Encode sketch through frozen VAE → latent conditioning
        sketch_latent = self._vae.encode(sketch.to(dtype=dtype))   # (B, 16, 64, 64)
        sketch_latent = sketch_latent.to(device=device, dtype=dtype)

        # 2. Encode matte through trainable CNN → latent-resolution features
        matte_feat = self.matte_cnn(matte.to(device=device, dtype=dtype))  # (B, 16, 64, 64)

        # 3. Combine into control conditioning (17ch for SD3ControlNetModel API)
        #    16ch: sketch_latent + matte_feat  (structural + matte learned features)
        #     1ch: raw matte downsampled       (explicit spatial mask)
        matte_latent = F.interpolate(
            matte.to(device=device, dtype=dtype), size=(64, 64), mode="bilinear", align_corners=False
        )  # (B, 1, 64, 64)
        ctrl_cond = torch.cat([sketch_latent + matte_feat, matte_latent], dim=1)  # (B, 17, 64, 64)

        # 4. Expand null embeddings to batch size
        null_enc_hs = self.null_encoder_hidden_states.expand(B, -1, -1).to(device=device, dtype=dtype)
        null_pooled = self.null_pooled_projections.expand(B, -1).to(device=device, dtype=dtype)

        # 5. Sigmas → timestep format expected by SD3 (1D, float)
        timestep = sigmas.to(device=device, dtype=dtype)

        # 6. ControlNet forward
        block_samples = self.controlnet(
            hidden_states=noisy_latent,
            controlnet_cond=ctrl_cond,
            encoder_hidden_states=null_enc_hs,
            pooled_projections=null_pooled,
            timestep=timestep,
            return_dict=False,
        )[0]

        return block_samples, null_enc_hs, null_pooled
