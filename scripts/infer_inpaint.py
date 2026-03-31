"""
Inference: face photo + sketch + matte → full image (hair inpainted).

Key design vs hair-dit infer.py:
  - No composite.py: model outputs full 512×512 image directly
  - Face re-injection at each denoising step → pixel-perfect face preservation
  - FaceControlNet receives masked_face_latent (face with hair zeroed)

Usage:
  python scripts/infer_inpaint.py \
    --config  configs/phase2_braid.yaml \
    --checkpoint checkpoints/phase2_braid/best.pth \
    --split   braid_test \
    --num_samples 16 \
    --num_steps   20 \
    --output_dir  outputs/infer_inpaint
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

from src.data.dataset import HairInpaintDataset
from src.data.utils import resize_matte_to_latent
from src.models.face_controlnet import FaceControlNet
from src.models.hair_controlnet import HairControlNet
from src.models.vae_wrapper import VAEWrapper


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.pop("base", None)
    if base_path:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = deep_merge(base_cfg, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Sampling with face re-injection
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inpaint_sampling(
    hair_controlnet: HairControlNet,
    face_controlnet: FaceControlNet,
    transformer: SD3Transformer2DModel,
    vae: VAEWrapper,
    scheduler: FlowMatchEulerDiscreteScheduler,
    sketch: torch.Tensor,         # (1, 3, 512, 512) [0,1]
    matte: torch.Tensor,          # (1, 1, 512, 512) [0,1]
    face_image: torch.Tensor,     # (1, 3, 512, 512) [0,1] original face photo
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Full inpainting sampling with face identity preservation.

    Face region is re-injected at every denoising step:
      latents = latents * matte_down + face_latent * (1 - matte_down)

    Returns:
      (1, 3, 512, 512) in [0, 1]
    """
    scheduler.set_timesteps(num_steps, device=device)
    bf16 = torch.bfloat16

    # Encode full face image (for re-injection — preserves identity exactly)
    face_latent = vae.encode(face_image.to(device=device, dtype=bf16))  # (1,16,64,64)

    # Encode masked_face (face with hair zeroed — FaceControlNet conditioning)
    masked_face = face_image * (1.0 - matte)
    masked_face_latent = vae.encode(masked_face.to(device=device, dtype=bf16))  # (1,16,64,64)

    # Matte at latent resolution (64×64)
    matte_down = resize_matte_to_latent(matte.to(device))  # (1, 1, 64, 64)

    # matte_tokens for blending residuals in token space
    # SD3.5 patch_size=2 → 32×32 tokens → seq_len=1024
    matte_tokens = F.interpolate(
        matte.to(device=device, dtype=bf16), size=(32, 32), mode="bilinear", align_corners=False
    ).flatten(2).permute(0, 2, 1)  # (1, 1024, 1)

    # Initialize: face region from GT latent, hair region from noise
    noise = torch.randn(1, 16, 64, 64, device=device, dtype=bf16)
    latents = face_latent * (1.0 - matte_down) + noise * matte_down

    sketch_bf16 = sketch.to(device=device, dtype=bf16)
    matte_bf16  = matte.to(device=device, dtype=bf16)

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="steps", leave=False)):
        sigma = scheduler.sigmas[i].to(device)
        sigmas_1d = sigma.view(1).to(dtype=bf16)

        # HairControlNet → 12 residuals + null embeddings
        hair_res_list, null_enc_hs, null_pooled = hair_controlnet(
            noisy_latent=latents,
            sketch=sketch_bf16,
            matte=matte_bf16,
            sigmas=sigmas_1d,
        )

        # FaceControlNet → 6 residuals (reuses null embeddings)
        face_res_list = face_controlnet(
            noisy_latent=latents,
            face_latent=masked_face_latent,
            encoder_hidden_states=null_enc_hs,
            pooled_projections=null_pooled,
            sigmas=sigmas_1d,
        )

        # Cast to bf16
        hair_res_list = [r.to(dtype=bf16) for r in hair_res_list]
        face_res_list = [r.to(dtype=bf16) for r in face_res_list]
        null_enc_hs   = null_enc_hs.to(dtype=bf16)
        null_pooled   = null_pooled.to(dtype=bf16)

        # Build 24 mixed residuals
        # blocks  0~17: hair_res[i//2]
        # blocks 18~23: hair_res[i//2]*matte + face_res[i-18]*(1-matte)
        mixed = []
        for j in range(18):
            mixed.append(hair_res_list[j // 2])
        for j in range(18, 24):
            r_h = hair_res_list[j // 2]
            r_f = face_res_list[j - 18]
            mixed.append(r_h * matte_tokens + r_f * (1.0 - matte_tokens))

        v_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=null_enc_hs,
            pooled_projections=null_pooled,
            timestep=sigmas_1d,
            block_controlnet_hidden_states=mixed,
            return_dict=False,
        )[0]

        latents = scheduler.step(v_pred, t, latents, return_dict=False)[0]

        # Face re-injection: restore face region to GT latent at every step
        # This guarantees pixel-perfect face identity preservation
        latents = latents * matte_down + face_latent * (1.0 - matte_down)

    image = vae.decode(latents)                     # (1, 3, 512, 512) in [-1, 1]
    image = (image.float().clamp(-1, 1) + 1) / 2   # [0, 1]
    return image


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) [0,1] tensor → (H, W, 3) uint8."""
    t = t.squeeze(0).float().cpu()
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def make_panel(sketch, matte, masked_face, gen, target) -> np.ndarray:
    """5-panel: sketch | matte | masked_face | generated | GT"""
    return np.concatenate([
        to_uint8(sketch),
        to_uint8(matte),
        to_uint8(masked_face),
        to_uint8(gen),
        to_uint8(target),
    ], axis=1)  # (512, 2560, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--split",       default="braid_test")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_steps",   type=int, default=20)
    parser.add_argument("--output_dir",  default="outputs/infer_inpaint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = cfg["model"]["model_id"]
    local_files_only = cfg.get("local_files_only", False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VAE...")
    vae = VAEWrapper.from_pretrained(
        model_id=model_id, torch_dtype=torch.bfloat16, local_files_only=local_files_only,
    ).to(device).eval()

    print("Loading Transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=local_files_only,
    ).to(device).eval()

    print("Loading HairControlNet...")
    hair_controlnet = HairControlNet(
        model_id=model_id,
        vae=vae,
        num_layers=cfg["model"].get("num_hair_controlnet_layers", 12),
        local_files_only=local_files_only,
    )

    print("Loading FaceControlNet...")
    face_controlnet = FaceControlNet(
        model_id=model_id,
        num_layers=cfg["model"].get("num_face_controlnet_layers", 6),
        local_files_only=local_files_only,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    hair_controlnet.load_state_dict(ckpt["hair_controlnet"])
    face_controlnet.load_state_dict(ckpt["face_controlnet"])

    hair_controlnet = hair_controlnet.to(device=device, dtype=torch.bfloat16).eval()
    face_controlnet = face_controlnet.to(device=device, dtype=torch.bfloat16).eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", local_files_only=local_files_only,
    )

    print(f"Dataset: {args.split}")
    dataset = HairInpaintDataset(split=args.split)
    n = min(args.num_samples, len(dataset))

    rows = []
    for idx in tqdm(range(n), desc="Generating"):
        data        = dataset[idx]
        sketch      = data["sketch"].unsqueeze(0)       # (1,3,512,512)
        matte       = data["matte"].unsqueeze(0)        # (1,1,512,512)
        masked_face = data["masked_face"].unsqueeze(0)  # (1,3,512,512)
        target      = data["target"].unsqueeze(0)       # (1,3,512,512) full GT
        face_image  = data["img"].unsqueeze(0)          # (1,3,512,512) = target

        gen = run_inpaint_sampling(
            hair_controlnet=hair_controlnet,
            face_controlnet=face_controlnet,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            sketch=sketch,
            matte=matte,
            face_image=face_image,
            num_steps=args.num_steps,
            device=device,
        )

        # Save individual generated image
        Image.fromarray(to_uint8(gen.cpu())).save(output_dir / f"{idx:04d}_gen.png")

        rows.append(make_panel(sketch, matte, masked_face, gen.cpu(), target))

    grid = np.concatenate(rows, axis=0)
    grid_path = output_dir / "grid.png"
    Image.fromarray(grid).save(grid_path)
    print(f"\nGrid saved: {grid_path}")
    print("Columns: sketch | matte | masked_face | generated | GT")


if __name__ == "__main__":
    main()
