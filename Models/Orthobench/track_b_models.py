from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TrackBSpec:
    key: str
    backend: str
    source: str
    description: str


TRACK_B_SPECS: Dict[str, TrackBSpec] = {
    "vit_b16": TrackBSpec(
        "vit_b16", "timm", "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "Supervised ViT-B/16 (ImageNet-21k to ImageNet-1k)",
    ),
    "swin_t": TrackBSpec(
        "swin_t", "timm", "swin_tiny_patch4_window7_224.ms_in1k",
        "Supervised Swin-T (ImageNet-1k)",
    ),
    "dinov2_vitb14": TrackBSpec(
        "dinov2_vitb14", "timm", "vit_base_patch14_dinov2.lvd142m",
        "DINOv2 ViT-B/14 self-supervised foundation encoder",
    ),
    "biomedclip_vitb16": TrackBSpec(
        "biomedclip_vitb16", "open_clip",
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "BiomedCLIP ViT-B/16 biomedical vision-language image encoder",
    ),
}


class TrackBClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int, model_key: str):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_classes)
        self.feature_dim = int(feature_dim)
        self.model_key = model_key

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(features, dict):
            for key in ("x_norm_clstoken", "pooled", "features", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            features = features[:, 0]
        return features.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def _open_clip_visual(pretrained: bool):
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "BiomedCLIP requires open_clip_torch. Install requirements_track_b.txt."
        ) from exc
    if pretrained:
        clip_model, _ = open_clip.create_model_from_pretrained(
            TRACK_B_SPECS["biomedclip_vitb16"].source
        )
    else:
        # The BiomedCLIP visual tower is a standard OpenCLIP ViT-B/16. The saved
        # Track-B state dict restores its biomedical weights after construction.
        clip_model = open_clip.create_model("ViT-B-16", pretrained=None)
    visual = clip_model.visual
    feature_dim = int(getattr(visual, "output_dim", 512))
    return visual, feature_dim


def create_track_b_model(
    model_key: str,
    num_classes: int,
    pretrained: bool = True,
    img_size: int = 224,
) -> TrackBClassifier:
    if model_key not in TRACK_B_SPECS:
        raise KeyError(f"Unknown Track-B model {model_key!r}; choose from {list(TRACK_B_SPECS)}")
    spec = TRACK_B_SPECS[model_key]
    if spec.backend == "timm":
        import timm

        # DINOv2 ViT-B/14 is published with a 518-pixel default configuration.
        # Passing img_size makes timm interpolate its pretrained positional
        # embeddings to the benchmark's common 224-pixel input resolution.
        encoder = timm.create_model(
            spec.source, pretrained=pretrained, num_classes=0, img_size=img_size
        )
        feature_dim = int(encoder.num_features)
    elif spec.backend == "open_clip":
        encoder, feature_dim = _open_clip_visual(pretrained)
    else:
        raise ValueError(spec.backend)
    return TrackBClassifier(encoder, feature_dim, num_classes, model_key)


def track_b_normalization(model_key: str, model: TrackBClassifier):
    if TRACK_B_SPECS[model_key].backend == "timm":
        from timm.data import resolve_model_data_config

        config = resolve_model_data_config(model.encoder)
        return tuple(config["mean"]), tuple(config["std"])
    # OpenCLIP normalization used by BiomedCLIP's published preprocessing.
    return (
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )


def configure_regime(model: TrackBClassifier, regime: str) -> None:
    if regime not in {"linear_probe", "full_ft"}:
        raise ValueError("regime must be linear_probe or full_ft")
    for parameter in model.parameters():
        parameter.requires_grad = regime == "full_ft"
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def set_training_state(model: TrackBClassifier, regime: str) -> None:
    model.train()
    if regime == "linear_probe":
        # Prevent dropout/stochastic depth and running-state changes in a frozen encoder.
        model.encoder.eval()


def load_track_b_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("track") != "B":
        raise ValueError(f"Not a Track-B checkpoint: {path}")
    # BiomedCLIP's HF-Hub definition uses OpenCLIP's timm-backed visual trunk,
    # which is architecturally different from the generic OpenCLIP ViT-B/16.
    # Reconstruct it through the same published configuration used in training,
    # then overwrite those initial weights with the saved Track-B state dict.
    # The pretrained files are already cached after training/preflight.
    reconstruct_with_published_config = (
        checkpoint["model_key"] == "biomedclip_vitb16"
    )
    model = create_track_b_model(
        checkpoint["model_key"], len(checkpoint["classes"]),
        pretrained=reconstruct_with_published_config,
        img_size=int(checkpoint.get("img_size", 224)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
