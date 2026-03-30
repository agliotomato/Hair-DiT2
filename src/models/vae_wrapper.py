"""SD3.5 VAE wrapper - 16ch latents, scaling=1.5305, shift=0.0609.

Provides encode/decode with proper scaling/shift handling for SD3.5-medium.
All parameters are frozen — VAE is used only for perceptual compression.

SD3.5 VAE conventions:
  encode: images [0,1] → normalize [-1,1] → VAE encode → latent = (raw - shift) * scale
  decode: raw = latent / scale + shift → VAE decode → images [-1,1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


SD35_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
SD35_SUBFOLDER = "vae"

SCALING_FACTOR = 1.5305
SHIFT_FACTOR = 0.0609


class VAEWrapper(nn.Module):
    """
    Thin wrapper around SD3.5-medium AutoencoderKL.

    Usage:
        vae = VAEWrapper.from_pretrained()
        latent = vae.encode(images)    # (B, 16, 64, 64)
        recon  = vae.decode(latent)    # (B, 3, 512, 512) in [-1, 1]
    """

    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae
        self.scaling_factor: float = getattr(vae.config, "scaling_factor", SCALING_FACTOR)
        self.shift_factor: float = getattr(vae.config, "shift_factor", SHIFT_FACTOR)
        # Freeze all parameters
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = SD35_MODEL_ID,
        subfolder: str = SD35_SUBFOLDER,
        torch_dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ) -> "VAEWrapper":
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        return cls(vae)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.

        Args:
            images: (B, 3, H, W) float, values in [0, 1]
                    Internally normalized to [-1, 1] before encoding.

        Returns:
            latent: (B, 16, H/8, W/8) scaled by (raw - shift) * scale
        """
        images_11 = self.normalize(images)
        # Cast to VAE dtype
        images_11 = images_11.to(dtype=next(self.vae.parameters()).dtype)
        posterior = self.vae.encode(images_11).latent_dist
        raw_latent = posterior.sample()
        latent = (raw_latent - self.shift_factor) * self.scaling_factor
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image space.

        NOTE: @torch.no_grad() intentionally omitted.
        VAE parameters are frozen (requires_grad=False), so they are not updated.
        But the computation graph must be maintained so that LPIPS gradients can
        flow back: L_lpips → pred_rgb → VAE decode → x0_pred → v_pred → HairControlNet.

        Args:
            latent: (B, 16, H/8, W/8) in scaled space

        Returns:
            images: (B, 3, H, W) in [-1, 1]
        """
        latent = latent.to(dtype=next(self.vae.parameters()).dtype)
        raw = latent / self.scaling_factor + self.shift_factor
        images = self.vae.decode(raw).sample
        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode only (convenience for training)."""
        return self.encode(images)

    @staticmethod
    def normalize(images_01: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] images to [-1,1] for VAE input."""
        return images_01 * 2.0 - 1.0

    @staticmethod
    def denormalize(images_11: torch.Tensor) -> torch.Tensor:
        """Convert [-1,1] images to [0,1]."""
        return (images_11 + 1.0) * 0.5
