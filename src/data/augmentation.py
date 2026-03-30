"""
Augmentation pipeline for hair-inpaint (hair-dit2) training.

All augmentations operate on the sample dict:
  {sketch, matte, masked_face, target, img, filename}

Phase 1 (pretrain / unbraid):
  - SketchColorJitter (p=0.8):  uniform color jitter on sketch only → color robustness
  - ThicknessJitter (p=0.5):    stroke dilation → line-thickness robustness
  - MatteBoundaryPerturbation (p=0.3): elastic matte warp → boundary robustness
  - AppearanceJitter (p=0.5):   color jitter on GT target → structure-appearance disentanglement

Phase 2 (finetune / braid):
  - StrokeColorSampler (p=1.0): per-stroke color sampling from actual hair pixels
  - ThicknessJitter (p=0.3)
  - MatteBoundaryPerturbation (p=0.2)

Key changes vs hair-dit augmentation.py:
  - MatteBoundaryPerturbation now also recomputes masked_face after warp
  - SketchColorJitter and AppearanceJitter are new (Phase 1 only)
  - StrokeColorSampler reads from full GT target (not img*matte)
"""

from __future__ import annotations

import random

import kornia.filters as KF
import kornia.morphology as KM
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# 1. Stroke Color Sampler  (Phase 2: SketchHairSalon 방식)
# ---------------------------------------------------------------------------

class StrokeColorSampler:
    """
    Per-stroke color sampling from actual target hair pixels.

    SketchHairSalon 논문 방식:
      각 stroke 영역에 대해, 매 iteration마다 target 이미지의 해당 위치 픽셀 중
      무작위로 1개를 샘플링하여 stroke 색으로 할당.

    Phase 2 only. Maintains stroke↔target color correspondence.

    Args:
        p:              적용 확률
        min_pixels:     stroke 영역 픽셀 수가 이 값 미만이면 해당 stroke 색 유지
        quantize_bits:  unique stroke 검출을 위한 양자화 비트 수
    """

    def __init__(self, p: float = 1.0, min_pixels: int = 10, quantize_bits: int = 5):
        self.p = p
        self.min_pixels = min_pixels
        self.shift = 8 - quantize_bits

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        sketch = sample["sketch"]  # (3, H, W) float32 [0,1]
        target = sample["target"]  # (3, H, W) float32 [0,1]  full GT image
        matte  = sample["matte"]   # (1, H, W) float32 [0,1]

        sketch_aug = self._resample_colors(sketch, target, matte)
        return {**sample, "sketch": sketch_aug}

    def _resample_colors(
        self,
        sketch: torch.Tensor,
        target: torch.Tensor,
        matte: torch.Tensor,
    ) -> torch.Tensor:
        # uint8 양자화로 unique stroke 레이블 검출
        sketch_u8 = (sketch * 255).byte()  # (3, H, W)
        sketch_q  = (sketch_u8 >> self.shift) << self.shift  # quantized

        flat_q = sketch_q.view(3, -1).T  # (N, 3)
        unique_colors = torch.unique(flat_q, dim=0)  # (K, 3)

        out = sketch.clone()

        # matte mask for filtering valid hair pixels
        matte_flat = matte.squeeze(0)  # (H, W)

        for color in unique_colors:
            r, g, b = color.tolist()

            # 검은 배경(non-hair stroke) 은 건드리지 않음
            if r == 0 and g == 0 and b == 0:
                continue

            # 이 stroke 레이블에 해당하는 픽셀 마스크
            mask = (
                (sketch_q[0] == r) &
                (sketch_q[1] == g) &
                (sketch_q[2] == b)
            )  # (H, W) bool

            # target에서 이 stroke 위치 AND matte 내 픽셀 추출
            hair_mask = mask & (matte_flat > 0.1)
            hair_pixels = target[:, hair_mask]  # (3, N)

            # 유효한 픽셀만 (target이 0이 아닌 것)
            valid = hair_pixels.sum(dim=0) > 0.05  # (N,)
            if valid.sum() < self.min_pixels:
                continue

            hair_pixels_valid = hair_pixels[:, valid]  # (3, M)

            # GAN 방식: 67% 평균 색상, 33% 랜덤 픽셀
            if random.randint(0, 5) < 2:
                idx = random.randint(0, hair_pixels_valid.shape[1] - 1)
                sampled_color = hair_pixels_valid[:, idx]  # (3,)
            else:
                sampled_color = hair_pixels_valid.float().mean(dim=1)  # (3,)

            # 해당 stroke 모든 픽셀을 sampled_color로 교체
            out[:, mask] = sampled_color.unsqueeze(1)

        return out


# ---------------------------------------------------------------------------
# 2. Sketch Color Jitter  (Phase 1 only)
# ---------------------------------------------------------------------------

class SketchColorJitter:
    """
    Apply random color jitter uniformly to the sketch image (target unchanged).

    Phase 1 only — breaks stroke↔target color correspondence intentionally,
    forcing the model to rely on stroke structure rather than color for guidance.
    Not used in Phase 2 where StrokeColorSampler maintains correspondence.

    Args:
        p:          probability of applying augmentation
        brightness: ColorJitter brightness range
        contrast:   ColorJitter contrast range
        saturation: ColorJitter saturation range
        hue:        ColorJitter hue range
    """

    def __init__(
        self,
        p: float = 0.8,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1,
    ):
        self.p = p
        self.jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        sketch = sample["sketch"]  # (3, H, W) float32 [0,1]

        # Apply jitter via PIL round-trip
        sketch_pil = TF.to_pil_image(sketch.clamp(0, 1))
        sketch_jittered = TF.to_tensor(self.jitter(sketch_pil))

        return {**sample, "sketch": sketch_jittered}


# ---------------------------------------------------------------------------
# 3. Appearance Jitter  (Phase 1 only)
# ---------------------------------------------------------------------------

class AppearanceJitter:
    """
    Apply random color jitter to the target GT image (sketch unchanged).

    Forces structure-appearance disentanglement:
      - model must use sketch structure (not target colors) to generate hair
      - prevents color shortcut learning in Phase 1

    After jittering target, recomputes masked_face = target_jittered * (1 - matte).
    Not used in Phase 2 where StrokeColorSampler provides correct color correspondence.

    Args:
        p:          probability of applying augmentation
        brightness: ColorJitter brightness range
        contrast:   ColorJitter contrast range
        saturation: ColorJitter saturation range
        hue:        ColorJitter hue range
    """

    def __init__(
        self,
        p: float = 0.5,
        brightness: float = 0.3,
        contrast: float = 0.2,
        saturation: float = 0.3,
        hue: float = 0.05,
    ):
        self.p = p
        self.jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        target = sample["target"]  # (3, H, W) float32 [0,1]
        matte  = sample["matte"]   # (1, H, W) float32 [0,1]

        target_pil = TF.to_pil_image(target.clamp(0, 1))
        target_jittered = TF.to_tensor(self.jitter(target_pil))

        # Recompute masked_face with jittered target
        masked_face_new = target_jittered * (1.0 - matte)

        return {**sample, "target": target_jittered, "img": target_jittered, "masked_face": masked_face_new}


# ---------------------------------------------------------------------------
# 4. Sketch Thickness Jitter
# ---------------------------------------------------------------------------

class ThicknessJitter:
    """
    Randomly dilate sketch strokes by 0-2 pixels.

    Args:
        p:          probability of applying augmentation
        max_kernel: maximum dilation kernel size (default 3 → ±1px dilation)
    """

    def __init__(self, p: float = 0.5, max_kernel: int = 3):
        self.p = p
        self.max_kernel = max_kernel  # must be odd

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        sketch = sample["sketch"].unsqueeze(0)  # (1, 3, H, W)
        k = random.choice([k for k in range(3, self.max_kernel + 1, 2)])
        kernel = torch.ones(k, k, device=sketch.device)
        sketch_dilated = KM.dilation(sketch, kernel).squeeze(0)
        return {**sample, "sketch": sketch_dilated.clamp(0, 1)}


# ---------------------------------------------------------------------------
# 5. Matte Boundary Perturbation
# ---------------------------------------------------------------------------

class MatteBoundaryPerturbation:
    """
    Apply small elastic deformation to the matte, then recompute target and masked_face.

    Args:
        p:         probability of applying augmentation
        amplitude: max pixel displacement (default 4)
        sigma:     smoothness of displacement field (default 10)
    """

    def __init__(self, p: float = 0.3, amplitude: float = 4.0, sigma: float = 10.0):
        self.p = p
        self.amplitude = amplitude
        self.sigma = sigma

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        matte = sample["matte"].unsqueeze(0)   # (1, 1, H, W)
        img   = sample["img"].unsqueeze(0)     # (1, 3, H, W)

        H, W = matte.shape[-2], matte.shape[-1]

        noise = torch.randn(1, 2, H, W, device=matte.device) * self.amplitude
        kernel_size = int(6 * self.sigma / H * H) | 1
        kernel_size = max(kernel_size, 3)
        noise = KF.gaussian_blur2d(noise, (kernel_size, kernel_size), (self.sigma, self.sigma))

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=matte.device),
            torch.linspace(-1, 1, W, device=matte.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        disp = noise.permute(0, 2, 3, 1)
        disp[..., 0] /= (W / 2)
        disp[..., 1] /= (H / 2)
        grid_warped = (grid + disp).clamp(-1, 1)

        matte_warped = F.grid_sample(matte.float(), grid_warped, mode="bilinear", align_corners=True)
        matte_warped = matte_warped.squeeze(0).clamp(0, 1)

        img_sq = img.squeeze(0)
        # target stays as the full GT image (unchanged)
        # masked_face must be recomputed with warped matte
        masked_face_new = img_sq * (1.0 - matte_warped)

        return {**sample, "matte": matte_warped, "masked_face": masked_face_new}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

class ComposeAug:
    """Sequential composition of augmentations."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_augmentation_pipeline(phase: str = "pretrain") -> ComposeAug:
    """
    Build the augmentation pipeline for a given training phase.

    Phase 1 (pretrain):
      SketchColorJitter (p=0.8): sketch color robustness (no correspondence)
      ThicknessJitter (p=0.5):   stroke width variation
      MatteBoundaryPerturbation (p=0.3): matte edge robustness
      AppearanceJitter (p=0.5):  structure-appearance disentanglement

    Phase 2 (finetune):
      StrokeColorSampler (p=1.0): per-stroke color from actual hair pixels
      ThicknessJitter (p=0.3):    stroke width variation (reduced)
      MatteBoundaryPerturbation (p=0.2): matte edge robustness (reduced)

    Args:
        phase: "pretrain" (unbraid) or "finetune" (braid)
    """
    if phase == "pretrain":
        return ComposeAug([
            SketchColorJitter(p=0.8),
            ThicknessJitter(p=0.5),
            MatteBoundaryPerturbation(p=0.3),
            AppearanceJitter(p=0.5),
        ])
    elif phase == "finetune":
        return ComposeAug([
            StrokeColorSampler(p=1.0),
            ThicknessJitter(p=0.3),
            MatteBoundaryPerturbation(p=0.2),
        ])
    else:
        raise ValueError(f"Unknown phase: {phase}. Must be 'pretrain' or 'finetune'.")
