from __future__ import annotations

"""Track C zero-shot evaluation for OrthoRegistryBench.

This script deliberately keeps zero-shot VLM evaluation separate from the
supervised Track A/B experiments while reusing their locked manifests,
preprocessing contract, labels, and primary metrics.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from orthobench.datasets import VIEW_CLASSES, normalize_view_label
from orthobench.preprocessing import PreprocessingConfig, load_preprocessed_rgb

SCRIPT_VERSION = "1.1"


MODEL_SPECS = {
    "clip_vitb32": {
        "kind": "contrastive",
        # OpenAI's ViT-B/32 weights were trained with QuickGELU.
        "source": "ViT-B-32-quickgelu",
        "pretrained": "openai",
    },
    "biomedclip_vitb16": {
        "kind": "contrastive",
        "source": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "pretrained": None,
    },
    "qwen2_5_vl_3b": {
        "kind": "generative",
        "source": "Qwen/Qwen2.5-VL-3B-Instruct",
    },
}


PROMPTS = {
    "view": {
        "AP": [
            "an AP knee radiograph",
            "an anteroposterior radiographic view of the knee",
            "a frontal AP x-ray of a knee",
        ],
        "PA": [
            "a PA knee radiograph",
            "a posteroanterior radiographic view of the knee",
            "a weight-bearing PA x-ray of a knee",
        ],
        "LATERAL": [
            "a lateral knee radiograph",
            "a side-view x-ray of a knee",
            "a lateral radiographic view of the knee joint",
        ],
        "SUNRISE": [
            "a sunrise knee radiograph",
            "a skyline radiographic view of the patellofemoral joint",
            "an axial patellar x-ray view",
        ],
    },
    "domain": {
        "NON_KNEE": [
            "a radiograph that is not an image of the knee joint",
            "a non-knee musculoskeletal x-ray",
            "a radiograph of an anatomical region other than the knee",
        ],
        "KNEE": [
            "a radiograph of the knee joint",
            "a musculoskeletal x-ray centered on a knee",
            "a knee x-ray",
        ],
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_path_column(frame: pd.DataFrame, preferred: str | None) -> str:
    candidates = [preferred] if preferred else []
    candidates += [
        "png_path", "image_path", "path", "filepath", "file_path",
        "dicom_png_path",
    ]
    for name in candidates:
        if name and name in frame.columns:
            return name
    raise ValueError(f"No image path column found. Tried {candidates}")


def uid_column(frame: pd.DataFrame) -> str | None:
    for name in ("uid", "image_id"):
        if name in frame.columns:
            return name
    return None


def prepare_split(
    frame: pd.DataFrame,
    task: str,
    split_col: str,
    split_value: str,
    group_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    if split_col not in frame.columns:
        raise ValueError(f"Manifest is missing split column {split_col!r}")
    work = frame[frame[split_col].astype(str) == split_value].copy()
    if task == "view":
        if "has_view_label" in work.columns:
            work = work[pd.to_numeric(work["has_view_label"], errors="coerce") == 1]
        work["label_name"] = work["view_label"].map(normalize_view_label)
        class_names = list(VIEW_CLASSES)
    else:
        work["domain_label"] = pd.to_numeric(work["domain_label"], errors="coerce")
        work = work[work["domain_label"].isin([0, 1])]
        work["label_name"] = work["domain_label"].astype(int).map(
            {0: "NON_KNEE", 1: "KNEE"}
        )
        class_names = ["NON_KNEE", "KNEE"]
    work = work[work["label_name"].isin(class_names)].copy()
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    work["y_true"] = work["label_name"].map(class_to_idx).astype(int)
    uid_col = uid_column(work)
    if uid_col is None:
        work["track_c_uid"] = work.index.astype(str)
    else:
        work["track_c_uid"] = work[uid_col].astype(str)
    if group_col not in work.columns:
        work[group_col] = work["track_c_uid"]
    work[group_col] = work[group_col].fillna(work["track_c_uid"]).astype(str)
    if work.empty:
        raise ValueError(f"No labeled rows for {task}: {split_col}={split_value}")
    return work.reset_index(drop=True), class_names


def balanced_limit(
    frame: pd.DataFrame, n_per_class: int | None, seed: int
) -> pd.DataFrame:
    if not n_per_class:
        return frame
    pieces = []
    for _, subset in frame.groupby("y_true", sort=True):
        pieces.append(subset.sample(min(n_per_class, len(subset)), random_state=seed))
    return pd.concat(pieces).sort_index().reset_index(drop=True)


def preprocessing_config(img_size: int) -> PreprocessingConfig:
    # Intensity scaling, padding, and masking match the primary Track A/B run.
    # Model-native normalization is subsequently applied by each VLM processor.
    return PreprocessingConfig(
        img_size=img_size,
        resize_mode="pad",
        intensity_mode="percentile",
        percentile_low=0.5,
        percentile_high=99.5,
        mask_mode="none",
        pad_value=0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        version="3.0-track-c-native-normalization",
    )


def load_image(path: str, cfg: PreprocessingConfig) -> Image.Image:
    return load_preprocessed_rgb(path, cfg)


def mean_prompt_features(model, tokenizer, prompts, device):
    class_features = []
    with torch.inference_mode():
        for prompt_group in prompts:
            tokens = tokenizer(prompt_group).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.mean(dim=0)
            features = features / features.norm()
            class_features.append(features)
    return torch.stack(class_features, dim=0)


def evaluate_contrastive(
    model_key: str,
    frame: pd.DataFrame,
    class_names: list[str],
    path_col: str,
    cfg: PreprocessingConfig,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    try:
        import open_clip
    except ImportError as error:
        raise SystemExit(
            "Track C contrastive models require open_clip_torch. "
            "Install it with: python3 -m pip install open_clip_torch"
        ) from error

    spec = MODEL_SPECS[model_key]
    kwargs: dict[str, Any] = {}
    if spec.get("pretrained") is not None:
        kwargs["pretrained"] = spec["pretrained"]
    model, _, native_preprocess = open_clip.create_model_and_transforms(
        spec["source"], **kwargs
    )
    tokenizer = open_clip.get_tokenizer(spec["source"])
    model = model.eval().to(device)
    prompt_groups = [PROMPTS["view" if len(class_names) == 4 else "domain"][c]
                     for c in class_names]
    text_features = mean_prompt_features(model, tokenizer, prompt_groups, device)
    logit_scale = 1.0
    if hasattr(model, "logit_scale"):
        logit_scale = float(model.logit_scale.detach().exp().clamp(max=100).cpu())

    records = []
    for start in range(0, len(frame), batch_size):
        batch_frame = frame.iloc[start:start + batch_size]
        tensors, valid_rows = [], []
        for row_index, row in batch_frame.iterrows():
            try:
                image = load_image(str(row[path_col]), cfg)
                tensors.append(native_preprocess(image))
                valid_rows.append((row_index, row, ""))
            except Exception as error:
                records.append({
                    "row_index": row_index,
                    "uid": row["track_c_uid"],
                    "group_id": row["track_c_group"],
                    "y_true": int(row["y_true"]),
                    "y_pred": -1,
                    "valid_output": 0,
                    "error": f"{type(error).__name__}: {error}",
                })
        if not tensors:
            continue
        images = torch.stack(tensors).to(device)
        with torch.inference_mode():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ text_features.T
            probabilities = logits.softmax(dim=1).float().cpu().numpy()
        for (_, row, _), probability in zip(valid_rows, probabilities):
            record = {
                "row_index": int(row.name),
                "uid": row["track_c_uid"],
                "group_id": row["track_c_group"],
                "y_true": int(row["y_true"]),
                "y_pred": int(probability.argmax()),
                "valid_output": 1,
                "error": "",
            }
            record.update({f"prob_{name}": float(probability[i])
                           for i, name in enumerate(class_names)})
            records.append(record)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(records).sort_values("row_index").reset_index(drop=True)


def load_qwen(model_id: str):
    try:
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError as error:
        raise SystemExit(
            "Qwen evaluation requires a recent transformers installation."
        ) from error
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = ModelClass.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    # Some released generation configs contain sampling-only values even
    # though this benchmark uses deterministic greedy decoding.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def qwen_prompt(task: str, class_names: list[str]) -> str:
    choices = ", ".join(class_names)
    if task == "view":
        question = "Classify the projection/view of this knee radiograph."
    else:
        question = "Decide whether this radiograph depicts the knee joint."
    return (
        f"{question} Return exactly one label from this list: {choices}. "
        "Do not provide an explanation, punctuation, or any additional text."
    )


def parse_generated_label(text: str, class_names: list[str]) -> tuple[int, bool]:
    # Main metrics use strict protocol compliance: explanations, JSON, multiple
    # labels, punctuation, and out-of-vocabulary text are counted as failures.
    cleaned = text.strip().upper()
    if cleaned in class_names:
        return class_names.index(cleaned), True
    return -1, False


def evaluate_qwen(
    model_key: str,
    task: str,
    frame: pd.DataFrame,
    class_names: list[str],
    path_col: str,
    cfg: PreprocessingConfig,
    max_new_tokens: int,
) -> pd.DataFrame:
    spec = MODEL_SPECS[model_key]
    model, processor = load_qwen(spec["source"])
    device = model.device
    prompt = qwen_prompt(task, class_names)
    records = []
    for row_index, row in frame.iterrows():
        started = time.perf_counter()
        raw_output, error_message = "", ""
        try:
            image = load_image(str(row[path_col]), cfg)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[chat_text], images=[image], padding=True, return_tensors="pt"
            ).to(device)
            input_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            raw_output = processor.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            prediction, valid = parse_generated_label(raw_output, class_names)
        except Exception as error:
            prediction, valid = -1, False
            error_message = f"{type(error).__name__}: {error}"
        records.append({
            "row_index": row_index,
            "uid": row["track_c_uid"],
            "group_id": row["track_c_group"],
            "y_true": int(row["y_true"]),
            "y_pred": int(prediction),
            "valid_output": int(valid),
            "raw_output": raw_output,
            "error": error_message,
            "latency_seconds": time.perf_counter() - started,
        })
        if (row_index + 1) % 25 == 0:
            print(f"{model_key} {task}: {row_index + 1}/{len(frame)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(records)


def metric_values(y_true, y_pred, labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    observed_recalls = []
    for label in np.unique(y_true):
        mask = y_true == label
        observed_recalls.append(float(np.mean(y_pred[mask] == label)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean(observed_recalls)),
        # Primary definition exactly matches Tracks A/B. On OAI this evaluates
        # PA/LATERAL plus any additional class predicted by a model.
        "macro_f1": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        # Secondary sensitivity metric always averages the four official view
        # classes (or both official domain classes), even when absent in truth.
        "macro_f1_fixed_classes": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )),
    }


def clustered_bootstrap(
    predictions: pd.DataFrame,
    labels: list[int],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if replicates <= 0:
        return {}
    rng = np.random.default_rng(seed)
    group_to_indices = {
        group: indices.to_numpy()
        for group, indices in predictions.groupby("group_id").groups.items()
    }
    groups = np.asarray(list(group_to_indices), dtype=object)
    metric_names = (
        "accuracy", "balanced_accuracy", "macro_f1",
        "macro_f1_fixed_classes", "weighted_f1",
    )
    estimates = {name: [] for name in metric_names}
    for _ in range(replicates):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate([group_to_indices[group] for group in sampled_groups])
        sample = predictions.loc[sampled_indices]
        values = metric_values(sample["y_true"], sample["y_pred"], labels)
        for name, value in values.items():
            estimates[name].append(value)
    output = {}
    for name, values in estimates.items():
        output[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
        output[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
    return output


def summarize_predictions(
    predictions: pd.DataFrame,
    class_names: list[str],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    labels = list(range(len(class_names)))
    y_true = predictions["y_true"].to_numpy(int)
    y_pred = predictions["y_pred"].to_numpy(int)
    metrics = metric_values(y_true, y_pred, labels)
    metrics.update(clustered_bootstrap(predictions, labels, bootstrap_replicates, seed))
    metrics["n"] = len(predictions)
    metrics["n_groups"] = predictions["group_id"].nunique()
    metrics["valid_output_rate"] = float(predictions["valid_output"].mean())
    metrics["invalid_output_count"] = int((predictions["valid_output"] == 0).sum())
    if "latency_seconds" in predictions:
        metrics["mean_latency_seconds"] = float(predictions["latency_seconds"].mean())

    report = pd.DataFrame(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )).T.reset_index(names="class")
    matrix_labels = labels + [-1]
    matrix_names = class_names + ["INVALID"]
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=matrix_labels),
        index=[f"true_{name}" for name in matrix_names],
        columns=[f"pred_{name}" for name in matrix_names],
    )
    return metrics, report, matrix


def evaluation_splits(task: str, args) -> list[tuple[str, str, str]]:
    if task == "view":
        candidates = [
            (args.view_internal_split_col, args.view_internal_split, "internal_test"),
            (args.view_external_split_col, args.view_external_split, "OAI_test"),
        ]
    else:
        candidates = [
            (args.domain_internal_split_col, args.domain_internal_split, "internal_test"),
            (args.domain_external_split_col, args.domain_external_split, "fracatlas_external"),
        ]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_SPECS),
        default=["clip_vitb32", "biomedclip_vitb16", "qwen2_5_vl_3b"],
    )
    parser.add_argument("--tasks", nargs="+", choices=["view", "domain"],
                        default=["view", "domain"])
    parser.add_argument("--path_col", default="png_path")
    parser.add_argument("--group_col", default="group_id")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument(
        "--preflight_per_class", type=int, default=None,
        help="Balanced per-class sample limit. Omit for full evaluation.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--view_internal_split_col", default="split_view")
    parser.add_argument("--view_internal_split", default="test")
    parser.add_argument("--view_external_split_col", default="split_view_external_oai")
    parser.add_argument("--view_external_split", default="OAI_test")
    parser.add_argument("--domain_internal_split_col", default="split_domain")
    parser.add_argument("--domain_internal_split", default="test")
    parser.add_argument("--domain_external_split_col", default="split_domain_external")
    parser.add_argument("--domain_external_split", default="FracAtlas_test")
    args = parser.parse_args()

    seed_everything(args.seed)
    output = Path(args.out_dir)
    prediction_dir = output / "predictions"
    report_dir = output / "reports"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    path_col = find_path_column(manifest, args.path_col)
    cfg = preprocessing_config(args.img_size)
    prompt_path = output / "track_c_prompt_registry.json"
    prompt_path.write_text(json.dumps({
        "script_version": SCRIPT_VERSION,
        "prompts": PROMPTS,
        "qwen_output_rule": "exactly one predefined class label",
        "preprocessing": cfg.to_dict(),
        "external_test_prompt_tuning": False,
    }, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows = []
    for task in args.tasks:
        for split_col, split_value, split_alias in evaluation_splits(task, args):
            subset, class_names = prepare_split(
                manifest, task, split_col, split_value, args.group_col
            )
            subset = balanced_limit(subset, args.preflight_per_class, args.seed)
            subset["track_c_group"] = subset[args.group_col].astype(str)
            mode = (
                f"preflight{args.preflight_per_class}pc"
                if args.preflight_per_class else "full"
            )
            print(f"\n{task} {split_alias}: n={len(subset)}, mode={mode}", flush=True)
            for model_key in args.models:
                prediction_path = prediction_dir / (
                    f"pred_track_c_{model_key}_{task}_{split_alias}_{mode}.csv"
                )
                if args.resume and prediction_path.exists():
                    print(f"Loading existing {prediction_path}", flush=True)
                    predictions = pd.read_csv(prediction_path)
                elif MODEL_SPECS[model_key]["kind"] == "contrastive":
                    predictions = evaluate_contrastive(
                        model_key, subset, class_names, path_col, cfg,
                        args.batch, device,
                    )
                    predictions.to_csv(prediction_path, index=False)
                else:
                    predictions = evaluate_qwen(
                        model_key, task, subset, class_names, path_col, cfg,
                        args.max_new_tokens,
                    )
                    predictions.to_csv(prediction_path, index=False)

                name_lookup = {i: name for i, name in enumerate(class_names)}
                name_lookup[-1] = "INVALID"
                predictions["y_true_name"] = predictions["y_true"].map(name_lookup)
                predictions["y_pred_name"] = predictions["y_pred"].map(name_lookup)
                # Rewrite once so resumed and newly generated files have the
                # same analysis-ready schema.
                predictions.to_csv(prediction_path, index=False)

                metrics, report, matrix = summarize_predictions(
                    predictions, class_names, args.bootstrap_replicates, args.seed
                )
                report.to_csv(
                    report_dir / f"per_class_{model_key}_{task}_{split_alias}_{mode}.csv",
                    index=False,
                )
                matrix.to_csv(
                    report_dir / f"confusion_{model_key}_{task}_{split_alias}_{mode}.csv"
                )
                row = {
                    "track": "C",
                    "script_version": SCRIPT_VERSION,
                    "regime": "zero_shot_prompt_ensemble" if
                        MODEL_SPECS[model_key]["kind"] == "contrastive" else
                        "zero_shot_generative",
                    "model": model_key,
                    "model_source": MODEL_SPECS[model_key]["source"],
                    "task": task,
                    "split": split_alias,
                    "split_column": split_col,
                    "split_value": split_value,
                    "mode": mode,
                    **metrics,
                    "prediction_file": str(prediction_path),
                }
                summary_rows.append(row)
                print(json.dumps(row, default=str), flush=True)

    summary_path = output / (
        f"track_c_zeroshot_preflight{args.preflight_per_class}pc_summary.csv"
        if args.preflight_per_class else "track_c_zeroshot_summary.csv"
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
