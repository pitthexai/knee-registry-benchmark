import numpy as np
from PIL import Image

from orthobench.preprocessing import PreprocessingConfig, mask_artifacts, normalize_pixels, resize_image


def test_pad_resize_preserves_aspect_ratio():
    image = Image.new("L", (200, 100), color=255)
    cfg = PreprocessingConfig(img_size=100, resize_mode="pad", pad_value=0)
    out = np.asarray(resize_image(image, cfg))
    assert out.shape == (100, 100)
    assert np.all(out[:25] == 0) and np.all(out[25:75] == 255)


def test_border_mask_changes_only_requested_border():
    image = Image.new("L", (100, 100), color=255)
    cfg = PreprocessingConfig(mask_mode="border", border_fraction=0.1, pad_value=0)
    out = np.asarray(mask_artifacts(image, cfg))
    assert np.all(out[:10] == 0) and np.all(out[:, :10] == 0)
    assert out[50, 50] == 255


def test_checkpoint_round_trip():
    cfg = PreprocessingConfig(resize_mode="pad", mask_mode="corners")
    assert PreprocessingConfig.from_dict(cfg.to_dict()) == cfg


def test_uint16_percentile_processing_preserves_dynamic_range():
    arr = np.linspace(0, 4095, 400, dtype=np.uint16).reshape(20, 20)
    image = Image.fromarray(arr, mode="I;16")
    out = np.asarray(normalize_pixels(image, PreprocessingConfig()))
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
