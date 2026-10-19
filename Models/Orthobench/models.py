from __future__ import annotations

from typing import Tuple
import torch
import timm
from timm.data import resolve_model_data_config


def create_timm_model(model_name: str, num_classes: int, pretrained: bool = True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def model_normalization(model) -> tuple[tuple[float, ...], tuple[float, ...]]:
    config = resolve_model_data_config(model)
    return tuple(config["mean"]), tuple(config["std"])


def extract_features(model, x: torch.Tensor) -> torch.Tensor:
    """Return pooled feature vector for most timm image classifiers."""
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.ndim == 4:
            if hasattr(model, "global_pool"):
                pooled = model.global_pool(feats)
                return pooled.flatten(1)
            return feats.mean(dim=(2, 3))
        if feats.ndim == 3:  # ViT tokens, usually [B, tokens, C]
            return feats[:, 0] if feats.shape[1] > 1 else feats.mean(dim=1)
        return feats.flatten(1)
    # fallback: remove classifier not reliable, but keeps script from crashing silently
    return model(x)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_classifier_and_last_block(model):
    # Works for many timm CNN/ViT families. It is intentionally permissive.
    for name, p in model.named_parameters():
        if any(key in name for key in ["classifier", "fc", "head"]):
            p.requires_grad = True
    if hasattr(model, "blocks"):
        for p in model.blocks[-1].parameters():
            p.requires_grad = True
    elif hasattr(model, "stages"):
        for p in model.stages[-1].parameters():
            p.requires_grad = True
    elif hasattr(model, "layer4"):
        for p in model.layer4.parameters():
            p.requires_grad = True
