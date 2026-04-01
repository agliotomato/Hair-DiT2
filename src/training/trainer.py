"""
Trainer for SD3.5 Dual ControlNet hair region generation (hair-inpaint).

Holds:
  - HairControlNet (trainable, blocks 0~11)
  - FaceControlNet (trainable, blocks 18~23)
  - SD3Transformer2DModel (frozen)
  - VAEWrapper SD3.5 (frozen)
  - FlowMatchEulerDiscreteScheduler

Mixed Residuals Logic:
  - Block 0~17: residuals_hair만 주입
  - Block 18~23: (residuals_hair * matte) + (residuals_face * (1 - matte)) 혼합 주입
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
import bitsandbytes as bnb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentation import build_augmentation_pipeline
from src.data.dataset import HairInpaintDataset
from src.data.utils import resize_matte_to_latent
from src.models.face_controlnet import FaceControlNet
from src.models.hair_controlnet import HairControlNet
from src.models.vae_wrapper import VAEWrapper
from src.training.ema import EMAModel
from src.training.losses import HairLoss


class Trainer:
    """Trainer for Phase 1 (pretrain) and Phase 2 (finetune)."""

    def __init__(self, config: dict):
        self.cfg = config
        self.phase = config["training"]["phase"]
        assert self.phase in ("pretrain", "finetune"), f"Unknown phase: {self.phase}"

        self.accelerator = Accelerator(
            mixed_precision=config["training"].get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
            log_with="tensorboard",
            project_dir=config["checkpointing"]["output_dir"],
        )

        self._setup_models()
        self._setup_data()
        self._setup_optimizer()
        self._prepare_accelerator()

        self.loss_fn = HairLoss(
            phase=self.phase,
            w_flow=config["training"]["loss_weights"].get("flow", 1.0),
            w_lpips=config["training"]["loss_weights"].get("lpips", 0.1),
            w_edge=config["training"]["loss_weights"].get("edge", 0.0),
            lpips_warmup_frac=config["training"]["loss_weights"].get("lpips_warmup_frac", 0.3),
        ).to(self.accelerator.device)

        # EMA tracks both controlnets
        self.ema_hair = EMAModel(
            self.accelerator.unwrap_model(self.hair_controlnet),
            decay=config["training"].get("ema_decay", 0.9999),
        )
        self.ema_face = EMAModel(
            self.accelerator.unwrap_model(self.face_controlnet),
            decay=config["training"].get("ema_decay", 0.9999),
        )

        self.output_dir = Path(config["checkpointing"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logit_mean = config["training"].get("logit_mean", 0.0)
        self.logit_std  = config["training"].get("logit_std",  1.0)
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        self._load_checkpoint()

    def _setup_models(self):
        cfg = self.cfg
        model_id = cfg["model"]["model_id"]
        local_files_only = cfg.get("local_files_only", False)

        self.vae = VAEWrapper.from_pretrained(
            model_id=model_id, torch_dtype=torch.bfloat16, local_files_only=local_files_only
        )
        self.vae.eval()

        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=local_files_only
        )
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        self.transformer.eval()

        # HairControlNet (12 blocks → 12 residuals)
        self.hair_controlnet = HairControlNet(
            model_id=model_id,
            vae=self.vae,
            num_layers=cfg["model"].get("num_hair_controlnet_layers", 12),
            local_files_only=local_files_only,
        )

        # FaceControlNet (6 blocks from SD3.5 blocks 18~23 → 6 residuals)
        self.face_controlnet = FaceControlNet(
            model_id=model_id,
            num_layers=cfg["model"].get("num_face_controlnet_layers", 6),
            local_files_only=local_files_only,
        )

        if cfg["training"].get("gradient_checkpointing", True):
            self.transformer.enable_gradient_checkpointing()
            self.hair_controlnet.controlnet.enable_gradient_checkpointing()
            self.face_controlnet.controlnet.enable_gradient_checkpointing()
            self.vae.vae.enable_gradient_checkpointing()

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler", local_files_only=local_files_only
        )


    def _setup_data(self):
        cfg = self.cfg["training"]
        aug = build_augmentation_pipeline(self.phase)
        split_train = f"{cfg['dataset']}_train"
        split_val   = f"{cfg['dataset']}_test"

        self.train_loader = DataLoader(
            HairInpaintDataset(split=split_train, augmentation=aug),
            batch_size=cfg.get("batch_size", 4), shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            HairInpaintDataset(split=split_val),
            batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=2,
        )

    def _setup_optimizer(self):
        cfg = self.cfg["training"]
        lr = cfg.get("learning_rate", 1e-4)

        params = list(self.hair_controlnet.parameters()) + list(self.face_controlnet.parameters())
        self.optimizer = bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)

        epochs = cfg.get("epochs", 200)
        total_steps = epochs * len(self.train_loader)
        warmup = cfg.get("warmup_steps", 500)

        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps - warmup, 1), eta_min=1e-6
        )
        self.warmup_steps = warmup
        self.total_steps = total_steps

    def _prepare_accelerator(self):
        (
            self.hair_controlnet,
            self.face_controlnet,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.hair_controlnet, self.face_controlnet, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler
        )
        device = self.accelerator.device
        self.vae = self.vae.to(device)
        self.transformer = self.transformer.to(device)

    def _sample_sigmas(self, bsz: int, device: torch.device) -> torch.Tensor:
        n_train = self.scheduler.config.num_train_timesteps
        u = torch.sigmoid(torch.normal(self.logit_mean, self.logit_std, size=(bsz,), device=device))
        indices = (u * n_train).long().clamp(0, n_train - 1)
        return self.scheduler.sigmas[indices.cpu()].to(device=device).view(bsz, 1, 1, 1)

    def _load_checkpoint(self):
        resume_from = self.cfg["training"].get("resume_from")
        if not resume_from:
            return
        path = Path(resume_from)
        if not path.exists():
            self.accelerator.print(f"[Resume] Checkpoint not found: {path}, starting from scratch")
            return
        ckpt = torch.load(path, map_location="cpu")
        self.accelerator.unwrap_model(self.hair_controlnet).load_state_dict(ckpt["hair_controlnet"])
        self.accelerator.unwrap_model(self.face_controlnet).load_state_dict(ckpt["face_controlnet"])
        if "ema_hair" in ckpt:
            self.ema_hair.load_state_dict(ckpt["ema_hair"])
        if "ema_face" in ckpt:
            self.ema_face.load_state_dict(ckpt["ema_face"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.global_step = ckpt.get("global_step", 0)
        self.start_epoch = ckpt.get("epoch", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.accelerator.print(
            f"[Resume] Loaded {path} — epoch {self.start_epoch}, step {self.global_step}, best_val {self.best_val_loss:.4f}"
        )

    def train(self):
        cfg = self.cfg["training"]
        epochs = cfg.get("epochs", 200)
        grad_clip = cfg.get("gradient_clip", 1.0)
        eval_every = self.cfg["checkpointing"].get("eval_every", 10)
        save_every = self.cfg["checkpointing"].get("save_every", 20)

        self.accelerator.print(f"Starting {self.phase} training for {epochs} epochs")

        for epoch in range(self.start_epoch, epochs):
            self.hair_controlnet.train()
            self.face_controlnet.train()
            epoch_losses = []
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not self.accelerator.is_local_main_process)

            for batch in progress:
                loss, log_dict = self._train_step(batch, grad_clip=grad_clip)
                epoch_losses.append(log_dict["loss_total"])

                if self.global_step < self.warmup_steps:
                    lr_scale = min(1.0, (self.global_step + 1) / max(self.warmup_steps, 1))
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.cfg["training"]["learning_rate"] * lr_scale
                else:
                    self.lr_scheduler.step()

                self.ema_hair.update(self.accelerator.unwrap_model(self.hair_controlnet))
                self.ema_face.update(self.accelerator.unwrap_model(self.face_controlnet))
                self.global_step += 1

                progress.set_postfix({k: f"{v:.4f}" for k, v in log_dict.items()})
                self.accelerator.log(log_dict, step=self.global_step)

            if (epoch + 1) % eval_every == 0:
                val_loss = self._validate()
                self.accelerator.print(f"Val loss: {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pth", epoch=epoch + 1)
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch+1}.pth", epoch=epoch + 1)

        self._save_checkpoint("final.pth", epoch=epochs)
        self.accelerator.end_training()

    def _train_step(self, batch: dict, grad_clip: float = 1.0) -> tuple[torch.Tensor, dict]:
        with self.accelerator.accumulate(self.hair_controlnet, self.face_controlnet):
            device = self.accelerator.device
            target = batch["target"]             # (B, 3, 512, 512)
            masked_face = batch["masked_face"]   # (B, 3, 512, 512)
            sketch = batch["sketch"]
            matte = batch["matte"]
            B = target.shape[0]

            with torch.no_grad():
                latents = self.vae.encode(target)               # GT
                face_latent = self.vae.encode(masked_face)      # Face conditioning

            sigmas = self._sample_sigmas(B, device)
            noise = torch.randn_like(latents)

            # Partial noise: face region keeps GT latent, hair region gets σ*noise
            # pipeline.md: noisy = gt_latent*(1-matte) + σ*noise*matte
            matte_latent = resize_matte_to_latent(matte)  # (B, 1, 64, 64)
            noisy_latents = latents * (1.0 - matte_latent) + sigmas * noise * matte_latent

            noisy_latents = noisy_latents.to(dtype=torch.bfloat16)
            sigmas_1d = sigmas.view(B).to(dtype=torch.bfloat16)

            # HairControlNet (12 blocks → 12 residuals)
            hair_res_list, null_enc_hs, null_pooled = self.hair_controlnet(
                noisy_latent=noisy_latents, sketch=sketch, matte=matte, sigmas=sigmas_1d
            )

            # FaceControlNet (outputs 6 for blocks 18~23)
            face_res_list = self.face_controlnet(
                noisy_latent=noisy_latents, face_latent=face_latent.to(dtype=torch.bfloat16),
                encoder_hidden_states=null_enc_hs, pooled_projections=null_pooled, sigmas=sigmas_1d
            )

            # Cast to bf16
            hair_res_list = [r.to(dtype=torch.bfloat16) for r in hair_res_list]
            face_res_list = [r.to(dtype=torch.bfloat16) for r in face_res_list]
            null_enc_hs = null_enc_hs.to(dtype=torch.bfloat16)
            null_pooled = null_pooled.to(dtype=torch.bfloat16)

            # matte_tokens: SD3.5 patch_size=2 → 32×32 tokens → seq_len=1024
            # (B,1,32,32) → flatten → (B,1024,1) for broadcasting over (B,1024,inner_dim)
            matte_tokens = F.interpolate(
                matte.to(device, dtype=torch.bfloat16), size=(32, 32), mode="bilinear", align_corners=False
            ).flatten(2).permute(0, 2, 1)  # (B, 1024, 1)

            # Mix 24 residuals: HairControlNet has 12 residuals → 2 blocks each (i//2)
            # blocks  0~17: hair_res[i//2]
            # blocks 18~23: hair_res[i//2]*matte + face_res[i-18]*(1-matte)
            mixed_residuals = []
            for i in range(18):
                mixed_residuals.append(hair_res_list[i // 2])
            for i in range(18, 24):
                r_h = hair_res_list[i // 2]       # indices 9,9,10,10,11,11
                r_f = face_res_list[i - 18]        # indices 0,1,2,3,4,5
                mixed_residuals.append(r_h * matte_tokens + r_f * (1.0 - matte_tokens))

            # Transformer v_pred
            v_pred = self.transformer(
                hidden_states=noisy_latents, encoder_hidden_states=null_enc_hs, pooled_projections=null_pooled,
                timestep=sigmas_1d, block_controlnet_hidden_states=mixed_residuals, return_dict=False
            )[0]
            v_target = (noise - latents).to(dtype=torch.bfloat16)

            loss, log_dict = self.loss_fn(
                v_pred=v_pred, v_target=v_target, matte_latent=matte_latent.to(dtype=torch.bfloat16),
                x_t=noisy_latents, sigmas=sigmas.to(dtype=torch.bfloat16), vae=self.vae,
                target_rgb=target, sketch=sketch, matte=matte, current_step=self.global_step, total_steps=self.total_steps
            )

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(list(self.hair_controlnet.parameters()) + list(self.face_controlnet.parameters()), grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss, log_dict

    @torch.no_grad()
    def _validate(self) -> float:
        self.hair_controlnet.eval()
        self.face_controlnet.eval()
        total_loss = 0.0
        n_batches = 0
        device = self.accelerator.device

        for batch in self.val_loader:
            target, masked_face, sketch, matte = batch["target"], batch["masked_face"], batch["sketch"], batch["matte"]
            B = target.shape[0]

            latents = self.vae.encode(target)
            face_latent = self.vae.encode(masked_face)
            sigmas = self._sample_sigmas(B, device)
            noise = torch.randn_like(latents)

            matte_latent = resize_matte_to_latent(matte)
            noisy_latents = latents * (1.0 - matte_latent) + sigmas * noise * matte_latent

            noisy_latents = noisy_latents.to(dtype=torch.bfloat16)
            sigmas_1d = sigmas.view(B).to(dtype=torch.bfloat16)

            hair_res_list, null_enc_hs, null_pooled = self.hair_controlnet(
                noisy_latents, sketch, matte, sigmas_1d
            )
            face_res_list = self.face_controlnet(
                noisy_latents, face_latent.to(dtype=torch.bfloat16), null_enc_hs, null_pooled, sigmas_1d
            )

            hair_res_list = [r.to(dtype=torch.bfloat16) for r in hair_res_list]
            face_res_list = [r.to(dtype=torch.bfloat16) for r in face_res_list]
            matte_tokens = F.interpolate(
                matte.to(dtype=torch.bfloat16), size=(32, 32), mode="bilinear", align_corners=False
            ).flatten(2).permute(0, 2, 1)  # (B, 1024, 1)

            mixed_residuals = []
            for i in range(18):
                mixed_residuals.append(hair_res_list[i // 2])
            for i in range(18, 24):
                r_h = hair_res_list[i // 2]
                r_f = face_res_list[i - 18]
                mixed_residuals.append(r_h * matte_tokens + r_f * (1.0 - matte_tokens))

            v_pred = self.transformer(
                hidden_states=noisy_latents, encoder_hidden_states=null_enc_hs.to(dtype=torch.bfloat16),
                pooled_projections=null_pooled.to(dtype=torch.bfloat16),
                timestep=sigmas_1d, block_controlnet_hidden_states=mixed_residuals, return_dict=False
            )[0]
            v_target = (noise - latents).to(dtype=torch.bfloat16)

            _, log_dict = self.loss_fn(v_pred=v_pred, v_target=v_target, matte_latent=matte_latent.to(dtype=torch.bfloat16))
            total_loss += log_dict["loss_total"]
            n_batches += 1

        self.hair_controlnet.train()
        self.face_controlnet.train()
        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, filename: str, epoch: int = 0):
        if not self.accelerator.is_main_process:
            return
        ckpt = {
            "hair_controlnet": self.accelerator.unwrap_model(self.hair_controlnet).state_dict(),
            "face_controlnet": self.accelerator.unwrap_model(self.face_controlnet).state_dict(),
            "ema_hair": self.ema_hair.state_dict(),
            "ema_face": self.ema_face.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
        }
        save_path = self.output_dir / filename
        torch.save(ckpt, save_path)
        self.accelerator.print(f"Saved checkpoint: {save_path}")
