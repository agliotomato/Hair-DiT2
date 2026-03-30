"""
Flow matching losses for SD3.5 Dual ControlNet hair inpainting.

L_total = w_flow * L_flow + w_lpips * L_lpips + w_edge * L_edge

L_flow: matte-only MSE(v_pred, v_target) where v_target = noise - latents
  - Only computed inside the matte region (no background gradient)
  - Normalized by matte area to avoid scale sensitivity with matte size
L_lpips: perceptual loss on decoded x0_pred in matte region
  - x0_pred = x_t - sigma * v_pred  (flow matching x0 recovery)
  - decoded through SD3.5 VAE
L_edge: sketch edge alignment (braid Phase 2 only)

Key change vs hair-dit losses.py:
  FlowMatchingLoss: background weighting (outside_weight=0.1) removed.
  Now pure matte-region loss, normalized by matte area sum.
"""

from __future__ import annotations

from typing import Optional

import kornia.filters as KF
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.utils import resize_matte_to_latent
from src.models.vae_wrapper import VAEWrapper


class FlowMatchingLoss(nn.Module):
    """
    Matte-region MSE between predicted velocity and target velocity.

    v_target = noise - latents  (flow matching velocity)
    Loss computed ONLY inside the matte (hair) region.
    Normalized by matte area to handle varying matte sizes.
    """

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        matte_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            v_pred:       (B, 16, 64, 64) predicted velocity
            v_target:     (B, 16, 64, 64) target velocity (noise - latents)
            matte_latent: (B, 1, 64, 64) [0, 1] matte at latent resolution

        Returns:
            loss: scalar
        """
        diff_sq = (v_pred - v_target) ** 2  # (B, 16, 64, 64)

        # Matte-only: zero outside the hair region
        # matte_latent (B,1,64,64) broadcasts over 16 channels
        masked = matte_latent * diff_sq

        # Normalize by matte area × channels to be independent of matte size
        area = matte_latent.sum() * v_pred.shape[1] + 1e-8
        return masked.sum() / area


class PerceptualLoss(nn.Module):
    """
    LPIPS perceptual loss computed on the decoded hair region (within matte).
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()
        self.lpips_fn = lpips.LPIPS(net=net)
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        matte: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_rgb:   (B, 3, 512, 512) in [-1, 1]
            target_rgb: (B, 3, 512, 512) in [-1, 1]
            matte:      (B, 1, 512, 512) in [0, 1]

        Returns:
            loss: scalar
        """
        pred_masked   = pred_rgb   * matte
        target_masked = target_rgb * matte
        # LPIPS operates in float32
        return self.lpips_fn(
            pred_masked.float(),
            target_masked.float(),
        ).mean()


class SketchEdgeAlignmentLoss(nn.Module):
    """
    Structural fidelity loss for braid fine-tuning.

    Penalizes regions where the sketch has strokes but the generated hair
    has no corresponding edges (strand boundaries).
    """

    def __init__(self, stroke_threshold: float = 0.1):
        super().__init__()
        self.stroke_threshold = stroke_threshold

    def forward(
        self,
        pred_rgb: torch.Tensor,
        sketch: torch.Tensor,
        matte: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_rgb: (B, 3, 512, 512) in [-1, 1]
            sketch:   (B, 3, 512, 512) in [0, 1]
            matte:    (B, 1, 512, 512) in [0, 1]

        Returns:
            loss: scalar
        """
        # Sketch stroke mask: any channel > threshold, intersected with matte
        sketch_mask = (sketch.max(dim=1, keepdim=True).values > self.stroke_threshold).float()
        sketch_mask = sketch_mask * matte

        # Sobel edge magnitude on predicted hair (grayscale)
        pred_gray = pred_rgb.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        pred_gray = (pred_gray + 1.0) * 0.5  # [-1,1] → [0,1]

        # spatial_gradient returns (B, C, 2, H, W)
        grad = KF.spatial_gradient(pred_gray.float())   # (B, 1, 2, H, W)
        edge_mag = grad.norm(dim=2).clamp(0, 1)        # (B, 1, H, W)

        # Penalize: stroke present AND edge absent
        loss = (sketch_mask * (1.0 - edge_mag)).mean()
        return loss


class HairLoss(nn.Module):
    """
    Combined flow-matching loss for SD3.5 Dual ControlNet hair inpainting.

    Args:
        phase:             "pretrain" (unbraid) or "finetune" (braid)
        w_flow:            weight for flow matching loss (default 1.0)
        w_lpips:           weight for perceptual loss (default 0.1)
        w_edge:            weight for edge alignment loss (default 0.05)
        lpips_warmup_frac: fraction of total steps before LPIPS activates (pretrain only)
    """

    def __init__(
        self,
        phase: str = "pretrain",
        w_flow: float = 1.0,
        w_lpips: float = 0.1,
        w_edge: float = 0.05,
        lpips_warmup_frac: float = 0.3,
    ):
        super().__init__()
        assert phase in ("pretrain", "finetune"), f"Unknown phase: {phase}"
        self.phase = phase
        self.w_flow = w_flow
        self.w_lpips = w_lpips
        self.w_edge = w_edge
        self.lpips_warmup_frac = lpips_warmup_frac

        self.flow_loss = FlowMatchingLoss()
        self.perc_loss = PerceptualLoss()
        self.edge_loss = SketchEdgeAlignmentLoss()

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        matte_latent: torch.Tensor,
        # Optional for LPIPS / edge loss
        x_t: Optional[torch.Tensor] = None,
        sigmas: Optional[torch.Tensor] = None,   # (B, 1, 1, 1) for x0 recovery
        vae: Optional[VAEWrapper] = None,
        target_rgb: Optional[torch.Tensor] = None,  # (B,3,H,W) in [0,1]
        sketch: Optional[torch.Tensor] = None,
        matte: Optional[torch.Tensor] = None,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            v_pred:        (B, 16, 64, 64) predicted velocity
            v_target:      (B, 16, 64, 64) ground-truth velocity (noise - latents)
            matte_latent:  (B, 1, 64, 64) matte at latent scale
            x_t:           (B, 16, 64, 64) noisy latent at time sigma
            sigmas:        (B, 1, 1, 1) flow matching sigma values
            vae:           VAEWrapper for decoding x0 predictions
            target_rgb:    (B, 3, H, W) ground truth in [0, 1]
            sketch:        (B, 3, H, W) colored sketch in [0, 1]
            matte:         (B, 1, H, W) full-res matte in [0, 1]
            current_step:  current training step
            total_steps:   total training steps

        Returns:
            total_loss: scalar tensor
            loss_dict:  dict with individual loss values (for logging)
        """
        # Primary flow matching loss — always active, matte-region only
        l_flow = self.flow_loss(v_pred, v_target, matte_latent)
        total_loss = self.w_flow * l_flow
        loss_dict = {"loss_flow": l_flow.item()}

        # Determine if perceptual loss should activate
        lpips_active = (
            self.phase == "finetune"
            or current_step >= int(self.lpips_warmup_frac * total_steps)
        )

        if lpips_active and vae is not None and x_t is not None and sigmas is not None:
            # Flow matching x0 recovery: x0 = x_t - sigma * v_pred
            # x0_pred must stay in the computation graph so LPIPS gradient
            # flows back through v_pred → Dual ControlNet.
            # VAE params are frozen (requires_grad=False) so no VAE updates occur.
            x0_pred = x_t - sigmas * v_pred            # (B, 16, 64, 64)
            pred_rgb_11 = vae.decode(x0_pred)          # (B, 3, H, W) in [-1, 1]

            if target_rgb is not None and matte is not None:
                target_rgb_11 = VAEWrapper.normalize(target_rgb)
                l_lpips = self.perc_loss(pred_rgb_11, target_rgb_11, matte)
                total_loss = total_loss + self.w_lpips * l_lpips
                loss_dict["loss_lpips"] = l_lpips.item()

            # Edge loss: braid fine-tune only
            if self.phase == "finetune" and sketch is not None and matte is not None:
                l_edge = self.edge_loss(pred_rgb_11, sketch, matte)
                total_loss = total_loss + self.w_edge * l_edge
                loss_dict["loss_edge"] = l_edge.item()

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict
