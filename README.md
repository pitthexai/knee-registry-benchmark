# OrthoRegistryBench: Knee Radiograph Registry Benchmark

Code for benchmarking automated knee radiograph registry labeling across convolutional networks, pretrained vision encoders, and zero-shot vision-language models. The benchmark addresses four-class knee view classification and binary knee/non-knee domain gating, with external evaluation, uncertainty estimation, and maintenance under dataset shift.

## Benchmark tracks

| Track | Models | Evaluation approach |
| --- | --- | --- |
| **Track A** | EfficientNet-B0, ResNet-50, DenseNet-121, ConvNeXt-Tiny | Supervised full fine-tuning |
| **Track B** | ViT-B/16, Swin-Tiny, DINOv2 ViT-B/14, BiomedCLIP ViT-B/16 | Linear probing and full fine-tuning |
| **Track C** | CLIP ViT-B/32, BiomedCLIP ViT-B/16, Qwen2.5-VL-3B-Instruct | Zero-shot classification using class descriptions or instructions |

View classes are `AP`, `PA`, `LATERAL`, and `SUNRISE`. Domain labels are `0 = NON_KNEE` and `1 = KNEE`. External evaluation code uses OAI for view classification and FracAtlas for domain gating.

## Repository contents

| Location | Contents |
| --- | --- |
| `Code/Orthobench/` | Shared datasets, preprocessing, models, metrics, conformal prediction, statistics, sensitivity analysis, and triage utilities |
| `Code/Scripts/` | Training, evaluation, maintenance, and analysis entry points |
| `Code/Tests/` | Tests for core utilities, preprocessing, and peripheral masking |
| `Dataset/` | Placeholder; image data and manifests are not included |
| `Figures/` | Placeholder; figures are not included in this snapshot |

### Script guide

| Script | Purpose |
| --- | --- |
| `01_train_view_multimodel.py` | Supervised view classifier training |
| `02_eval_view_external.py` | External view evaluation |
| `05_conformal_view.py` | Temperature scaling and conformal view prediction sets |
| `06_fewshot_view_maintenance.py` | Few-shot view classifier maintenance |
| `07_domain_fracatlas_maintenance.py` | Domain classifier maintenance using FracAtlas embeddings |
| `08_summarize_results.py` | Result aggregation |
| `09_eval_peripheral_sensitivity.py` | Evaluation under peripheral masking |
| `09_final_benchmark_analysis.py` | Final benchmark analysis |
| `10_analyze_peripheral_sensitivity.py` | Analysis of masking sensitivity outputs |
| `11_track_b_preflight.py` | Track B model loading and forward-pass checks |
| `12_train_track_b.py` | Track B linear probing and full fine-tuning |
| `13_eval_track_b_external.py` | Track B external view evaluation |
| `14_conformal_track_b.py` | Track B conformal evaluation |
| `15_extract_track_b_embeddings.py` | Track B embedding extraction |
| `16_fewshot_view_maintenance_track_b.py` | Track B few-shot view maintenance |
| `17_track_c_zeroshot.py` | Track C zero-shot view and domain evaluation |
| `18_harmonized_final_analysis.py` | Harmonized analysis of prediction outputs |

Files ending in `_before_fix.py` are alternate snapshots; use the unsuffixed entry points in the examples below.

## Setup

The examples below use Linux/bash, including a VS Code Remote SSH terminal. Python 3.10 or newer is recommended based on the syntax used in the source. A CUDA-capable GPU is recommended for training and VLM inference.

```bash
git clone https://github.com/pitthexai/knee-registry-benchmark.git
cd knee-registry-benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision timm numpy pandas pillow scikit-learn scipy joblib pytest
python -m pip install open_clip_torch transformers accelerate
```

Install a PyTorch build appropriate for your CUDA environment. The package list above is a starting environment inferred from the source, not a pinned or validated environment specification. Model loading may download pretrained weights. Qwen requires a Transformers version exposing `Qwen2_5_VLForConditionalGeneration` or a compatible automatic image-to-text model class.

### Resolve the package name before running

Scripts import `orthobench`, but this snapshot stores the shared package as `Code/Orthobench`. On a case-sensitive filesystem, rename that directory and add `Code` to the Python path:

```bash
mv Code/Orthobench Code/orthobench
export PYTHONPATH="$PWD/Code${PYTHONPATH:+:$PYTHONPATH}"
```

Run the rename once. Set `PYTHONPATH` in each new shell, and run the commands below from the repository root. This avoids `ModuleNotFoundError: No module named 'orthobench'`.

## Data and manifests

Obtain source images under each dataset's access conditions and prepare a CSV manifest pointing to local images. Raw images, trained checkpoints, embeddings, and experiment results are not included in this repository snapshot. The scripts consume prepared manifests; they do not download datasets or generate the complete benchmark split manifest.

| Column | Use |
| --- | --- |
| `png_path` | Local image path; alternative columns may be selected with `--path_col` |
| `uid` or `image_id` | Image identifier |
| `group_id` | Patient/study grouping for analyses that use clustered resampling |
| `view_label` | `AP`, `PA`, `LATERAL`, or `SUNRISE` |
| `has_view_label` | Numeric `1` for images with a valid view label |
| `domain_label` | Numeric `0` for non-knee and `1` for knee |
| `split_view` | Internal view splits: `train`, `val`, `test` |
| `split_view_external_oai` | External view splits such as `OAI_adapt`, `OAI_calib`, `OAI_test` |
| `split_domain` | Internal domain split; Track C evaluates `test` by default |
| `split_domain_external` | External domain split; Track C evaluates `FracAtlas_test` by default |

Required columns depend on the script. Preserve the intended patient/study separation across splits. Provide valid grouping identifiers for clustered confidence intervals: Track C falls back to image identifiers when group identifiers are missing, which does not preserve patient-level clustering.

The primary preprocessing defaults resize to a padded 224 × 224 image, apply intensity scaling using the 0.5th–99.5th percentiles, and leave peripheral masking disabled. Shared training augmentation uses rotations up to 5 degrees and small affine translations/scaling. Model-specific normalization is applied; Track C uses each model's native processor after shared image preprocessing. Border and corner masking are available for sensitivity experiments.

## Example workflows

Replace `/path/to/manifest.csv` with your prepared manifest. These examples describe the included command-line interfaces; full experiments require the corresponding data and dependencies.

### Track A: CNN training

Explicitly select the four CNNs, because script 01 also includes transformer models in its default model list.

```bash
CUDA_VISIBLE_DEVICES=2 python Code/Scripts/01_train_view_multimodel.py \
  --manifest /path/to/manifest.csv \
  --out_dir outputs/track_a/seed0 \
  --models tf_efficientnet_b0 resnet50 densenet121 convnext_tiny \
  --seed 0 --epochs 25 --img_size 224 \
  --resize_mode pad --intensity_mode percentile --mask_mode none
```

External view evaluation:

```bash
CUDA_VISIBLE_DEVICES=2 python Code/Scripts/02_eval_view_external.py \
  --manifest /path/to/manifest.csv \
  --ckpt_dir outputs/track_a/seed0/view_ckpts \
  --out_dir outputs/track_a/seed0/external
```

### Track B: pretrained encoders

Check model availability first:

```bash
CUDA_VISIBLE_DEVICES=2 python Code/Scripts/11_track_b_preflight.py
```

Run linear probing and full fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=2 python Code/Scripts/12_train_track_b.py \
  --manifest /path/to/manifest.csv \
  --out_dir outputs/track_b/seed0 \
  --models vit_b16 swin_t dinov2_vitb14 biomedclip_vitb16 \
  --regimes linear_probe full_ft --seed 0
```

Linear probing trains a classification head with the encoder frozen; full fine-tuning updates the encoder and head. For multi-seed experiments, repeat training with seeds `0`, `1`, `2`, `3`, and `4`, using a separate output directory per seed. Four models × two regimes × five seeds produce 40 Track B training runs.

### Track C: zero-shot VLM evaluation

```bash
CUDA_VISIBLE_DEVICES=2 python Code/Scripts/17_track_c_zeroshot.py \
  --manifest /path/to/manifest.csv \
  --out_dir outputs/track_c \
  --models clip_vitb32 biomedclip_vitb16 qwen2_5_vl_3b \
  --tasks view domain --seed 0
```

Use `--preflight_per_class 5` with a separate output directory for a small preliminary run. Omit this flag for full evaluation. `CUDA_VISIBLE_DEVICES=2` selects physical GPU 2; change or omit it for your environment.

## Zero-shot prompts and decoding

Exact class descriptions are defined in `PROMPTS` in `Code/Scripts/17_track_c_zeroshot.py`.

- CLIP and BiomedCLIP use three descriptions per class. Each text embedding is normalized, embeddings are averaged within class and normalized again, and image–text similarity determines the predicted class.
- Qwen view instruction: `Classify the projection/view of this knee radiograph. Return exactly one label from this list: AP, PA, LATERAL, SUNRISE. Do not provide an explanation, punctuation, or any additional text.`
- Qwen domain instruction: `Decide whether this radiograph depicts the knee joint. Return exactly one label from this list: NON_KNEE, KNEE. Do not provide an explanation, punctuation, or any additional text.`

Qwen uses deterministic generation with sampling disabled and a default maximum of 12 new tokens. Parsing strips surrounding whitespace and converts text to uppercase, then requires an exact permitted label. Additional explanations, punctuation, or unrecognized outputs are marked invalid and counted as failures in primary metrics.

## Uncertainty, maintenance, and analysis

The included scripts support temperature scaling, adaptive prediction sets, class-conditional conformal evaluation, few-shot view adaptation, domain classifier maintenance, and peripheral masking analysis. Domain maintenance uses logistic regression, histogram gradient boosting, or random forests on precomputed embeddings.

Use each script's `--help` for its required inputs, for example:

```bash
python Code/Scripts/05_conformal_view.py --help
python Code/Scripts/16_fewshot_view_maintenance_track_b.py --help
python Code/Scripts/18_harmonized_final_analysis.py --help
```

Output locations vary by script. Training scripts create `view_ckpts/` and `results/`; Track C creates `predictions/` and `reports/`. The harmonized analysis exports run-level and model-level metrics, maintenance deltas, and analysis coverage tables. Track C supports grouped bootstrap confidence intervals, with 2,000 replicates by default.

## Tests

After setting up the lowercase package name and `PYTHONPATH`:

```bash
python -m pytest Code/Tests
```

## Current release scope

This snapshot contains experiment source code rather than a complete reproduction bundle. It does not include a pinned dependency file, dataset manifests, pretrained experiment checkpoints, or final result tables. Numbered baseline scripts 03 and 04 are absent; in particular, the baseline domain training/evaluation workflow and Track A embedding extraction are not supplied as standalone entry points here. Domain maintenance therefore requires separately prepared embedding inputs. Some source messages reference requirements files that are also absent from this snapshot.

## Citation and license

Manuscript citation details and a repository license are not supplied in this snapshot. Add the finalized paper citation when available and an appropriate `LICENSE` file specifying reuse terms. Source datasets and pretrained models retain their own access conditions and licenses.

Repository: [pitthexai/knee-registry-benchmark](https://github.com/pitthexai/knee-registry-benchmark).
