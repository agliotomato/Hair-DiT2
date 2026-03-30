"""
EMA (Exponential Moving Average) model wrapper.

Maintains a shadow copy of model weights updated as:
  ema_weights = decay * ema_weights + (1 - decay) * model_weights

Used for stable evaluation — EMA model typically produces better samples
than the training checkpoint.
"""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn


class EMAModel:
    """
    EMA wrapper for a PyTorch model.

    Args:
        model:  the model to track
        decay:  EMA decay rate (default 0.9999)
        device: device for EMA weights (defaults to model's device)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.device = device

        # Initialize shadow weights from model
        for name, param in model.named_parameters():
            if param.requires_grad:
                shadow = param.data.clone()
                if device is not None:
                    shadow = shadow.to(device)
                self.shadow[name] = shadow

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights from current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_to(self, model: nn.Module):
        """Temporarily copy EMA weights into model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore_to(self, model: nn.Module, original_weights: dict[str, torch.Tensor]):
        """Restore original training weights after evaluation."""
        for name, param in model.named_parameters():
            if name in original_weights:
                param.data.copy_(original_weights[name])

    def state_dict(self) -> dict:
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()}, "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.decay = state["decay"]
        self.shadow = state["shadow"]
