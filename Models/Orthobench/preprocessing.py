from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessingConfig:
    img_size: int = 224
    resize_mode: str = "pad"
    intensity_mode: str = "percentile"
    percentile_low: float = 0.5
    percentile_high: float = 99.5
    mask_mode: str = "none"
    border_fraction: float = 0.04
    corner_fraction: float = 0.18
    pad_value: int = 0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    version: str = "3.0"

    def __post_init__(self):
        if self.img_size <= 0:
            raise ValueError("img_size must be positive")
        if self.resize_mode not in {"pad", "stretch"}:
            raise ValueError("resize_mode must be pad or stretch")
        if self.intensity_mode not in {"percentile", "minmax", "none"}:
            raise ValueError("intensity_mode must be percentile, minmax, or none")
        if self.mask_mode not in {"none", "border", "corners", "border_corners"}:
            raise ValueError("mask_mode must be none, border, corners, or border_corners")
        if not 0 <= self.percentile_low < self.percentile_high <= 100:
            raise ValueError("invalid percentile interval")
        if not 0 <= self.border_fraction < 0.5 or not 0 <= self.corner_fraction < 0.5:
            raise ValueError("mask fractions must lie in [0, 0.5)")
        if not 0 <= self.pad_value <= 255:
            raise ValueError("pad_value must lie in [0, 255]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None, img_size: int = 224):
        if not value:
            # Legacy v2 checkpoints used square stretching plus percentile scaling.
            return cls(img_size=img_size, resize_mode="stretch", version="2-legacy")
        data = dict(value)
        for key in ("mean", "std"):
            if key in data:
                data[key] = tuple(float(x) for x in data[key])
        return cls(**data)


def normalize_pixels(image: Image.Image, cfg: PreprocessingConfig) -> Image.Image:
    high_bit_modes = {"I", "I;16", "I;16B", "I;16L", "F"}
    raw = np.asarray(image) if image.mode in high_bit_modes else np.asarray(image.convert("L"))
    arr = raw.astype(np.float32)
    if cfg.intensity_mode == "percentile":
        lo, hi = np.percentile(arr, [cfg.percentile_low, cfg.percentile_high])
    elif cfg.intensity_mode == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
    else:
        if raw.dtype == np.uint8 or arr.max(initial=0) <= 255:
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
        lo = float(arr.min())
        hi = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else float(arr.max())
    if hi <= lo:
        out = np.zeros_like(arr, dtype=np.uint8)
    else:
        out = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def mask_artifacts(image: Image.Image, cfg: PreprocessingConfig) -> Image.Image:
    if cfg.mask_mode == "none":
        return image
    arr = np.asarray(image).copy()
    h, w = arr.shape[:2]
    if cfg.mask_mode in {"border", "border_corners"}:
        bh = int(round(h * cfg.border_fraction)); bw = int(round(w * cfg.border_fraction))
        if bh:
            arr[:bh, ...] = cfg.pad_value; arr[h - bh:, ...] = cfg.pad_value
        if bw:
            arr[:, :bw, ...] = cfg.pad_value; arr[:, w - bw:, ...] = cfg.pad_value
    if cfg.mask_mode in {"corners", "border_corners"}:
        ch = int(round(h * cfg.corner_fraction)); cw = int(round(w * cfg.corner_fraction))
        arr[:ch, :cw, ...] = cfg.pad_value; arr[:ch, w - cw:, ...] = cfg.pad_value
        arr[h - ch:, :cw, ...] = cfg.pad_value; arr[h - ch:, w - cw:, ...] = cfg.pad_value
    return Image.fromarray(arr)


def resize_image(image: Image.Image, cfg: PreprocessingConfig) -> Image.Image:
    size = (cfg.img_size, cfg.img_size)
    if cfg.resize_mode == "stretch":
        return image.resize(size, Image.Resampling.BILINEAR)
    w, h = image.size
    scale = min(cfg.img_size / w, cfg.img_size / h)
    new_w = max(1, int(round(w * scale))); new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new(image.mode, size, color=cfg.pad_value)
    canvas.paste(resized, ((cfg.img_size - new_w) // 2, (cfg.img_size - new_h) // 2))
    return canvas


def load_preprocessed_rgb(path: str, cfg: PreprocessingConfig) -> Image.Image:
    with Image.open(path) as raw:
        image = normalize_pixels(raw, cfg)
    image = mask_artifacts(image, cfg)
    image = resize_image(image, cfg)
    return image.convert("RGB")
