from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from orthobench.preprocessing import PreprocessingConfig, load_preprocessed_rgb

VIEW_CLASSES = ["AP", "PA", "LATERAL", "SUNRISE"]
VIEW_TO_IDX = {c: i for i, c in enumerate(VIEW_CLASSES)}


def _find_path_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    candidates = [preferred] if preferred else []
    candidates += ["png_path", "image_path", "path", "filepath", "file_path", "dicom_png_path"]
    for c in candidates:
        if c and c in df.columns:
            return c
    raise ValueError(f"No image path column found. Tried: {candidates}")


def normalize_view_label(x: Any) -> str:
    s = str(x).strip().upper()
    synonyms = {
        "LAT": "LATERAL",
        "LATERAL VIEW": "LATERAL",
        "SUNRISE VIEW": "SUNRISE",
        "SKYLINE": "SUNRISE",
        "MERCHANT": "SUNRISE",
        "AP VIEW": "AP",
        "PA VIEW": "PA",
    }
    return synonyms.get(s, s)


def build_transform(train: bool, cfg: PreprocessingConfig) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomRotation(degrees=5, fill=cfg.pad_value),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=cfg.pad_value),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])


class ViewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, img_size: int = 224, path_col: Optional[str] = None, preprocessing=None):
        self.df = df.reset_index(drop=True).copy()
        self.path_col = _find_path_column(self.df, path_col)
        self.preprocessing = PreprocessingConfig.from_dict(preprocessing, img_size)
        self.transform = build_transform(train=train, cfg=self.preprocessing)
        if "view_label" not in self.df.columns:
            raise ValueError("Manifest must contain view_label for ViewDataset")
        self.df["view_label_norm"] = self.df["view_label"].map(normalize_view_label)
        self.df = self.df[self.df["view_label_norm"].isin(VIEW_TO_IDX)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        img = load_preprocessed_rgb(path, self.preprocessing)
        x = self.transform(img)
        y = VIEW_TO_IDX[row["view_label_norm"]]
        uid = str(row.get("image_id", row.get("uid", idx)))
        return x, torch.tensor(y, dtype=torch.long), uid


class DomainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, img_size: int = 224, path_col: Optional[str] = None, preprocessing=None):
        self.df = df.reset_index(drop=True).copy()
        self.path_col = _find_path_column(self.df, path_col)
        self.preprocessing = PreprocessingConfig.from_dict(preprocessing, img_size)
        self.transform = build_transform(train=train, cfg=self.preprocessing)
        if "domain_label" not in self.df.columns:
            raise ValueError("Manifest must contain domain_label for DomainDataset")
        self.df["domain_label"] = pd.to_numeric(self.df["domain_label"], errors="coerce")
        self.df = self.df[self.df["domain_label"].isin([0, 1])].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, str]]:
        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        try:
            img = load_preprocessed_rgb(path, self.preprocessing)
        except Exception:
            return None
        x = self.transform(img)
        y = int(row["domain_label"])
        uid = str(row.get("image_id", row.get("uid", idx)))
        return x, torch.tensor(y, dtype=torch.long), uid


def collate_drop_none(batch: List[Optional[Tuple[torch.Tensor, torch.Tensor, str]]]):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys, uids = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(uids)
