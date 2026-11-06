"""Image degradations applied at inference only, to a saved checkpoint. No retraining.

ORDER OF OPERATIONS (fixed; asserted):
    load RGB -> Resize(256) -> CenterCrop(224) -> uint8 HxWx3 array
    -> degrade(...)            <- here, on uint8 pixels
    -> ToTensor -> Normalize
Degrading after normalisation would corrupt a different quantity, so `degrade` accepts
and returns uint8 only. This is the same place ImageNet-C applies its corruptions.

SEVERITIES (DECISIONS.md D7). Severity 0 is the identity. Levels 1-5 are:
  gaussian_blur : sigma = 1, 2, 3, 4, 6 px          ImageNet-C `gaussian_blur`, verbatim
  brightness    : HSV V += 0.1, 0.2, 0.3, 0.4, 0.5  ImageNet-C `brightness`, verbatim
  darkness      : HSV V -= 0.1, 0.2, 0.3, 0.4, 0.5  our mirror of `brightness` (ImageNet-C
                                                    only brightens)
  occlusion     : one black square covering 5, 10, 20, 30, 40 % of the image area at a
                  seeded random position             our design (ImageNet-C has no
                                                    occlusion corruption)
Source: hendrycks/robustness, ImageNet-C/create_c/make_imagenet_c.py (commit e0377d4),
Hendrycks & Dietterich, ICLR 2019. As there, the float result is cast with np.uint8
(truncation). Unlike there, it is not re-encoded as JPEG (quality 85): corruption happens
in memory, so no second lossy step is added.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch
from PIL import Image
from skimage.color import hsv2rgb, rgb2hsv
from skimage.filters import gaussian
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import Config
from src.data import MEAN, STD, Record

CORRUPTIONS = ("gaussian_blur", "brightness", "darkness", "occlusion")
SEVERITIES = (0, 1, 2, 3, 4, 5)
BLUR_SIGMA = (1, 2, 3, 4, 6)
BRIGHTNESS_DELTA = (0.1, 0.2, 0.3, 0.4, 0.5)
OCCLUSION_AREA = (0.05, 0.10, 0.20, 0.30, 0.40)
OCCLUSION_VALUE = 0  # black


def severity_parameter(corruption: str, severity: int) -> float:
    """The physical parameter for a (corruption, severity), 0.0 at severity 0."""
    assert corruption in CORRUPTIONS and severity in SEVERITIES
    if severity == 0:
        return 0.0
    table = {"gaussian_blur": BLUR_SIGMA, "brightness": BRIGHTNESS_DELTA,
             "darkness": BRIGHTNESS_DELTA, "occlusion": OCCLUSION_AREA}[corruption]
    return float(table[severity - 1])


def image_seed(base_seed: int, key: str, corruption: str, severity: int) -> int:
    """Stable per-(image, corruption, severity) seed; independent of process and order."""
    return zlib.crc32(f"{base_seed}|{key}|{corruption}|{severity}".encode())


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)  # truncation, as ImageNet-C's np.uint8


def degrade(img: np.ndarray, corruption: str, severity: int, seed: int = 0) -> np.ndarray:
    """Apply one corruption at one severity to a uint8 HxWx3 image. Returns uint8 HxWx3."""
    assert img.dtype == np.uint8, f"degrade expects uint8 pixels, got {img.dtype}"
    assert img.ndim == 3 and img.shape[2] == 3, f"expected HxWx3, got {img.shape}"
    p = severity_parameter(corruption, severity)
    if severity == 0:
        return img.copy()

    if corruption == "gaussian_blur":
        out = _to_uint8(gaussian(img / 255.0, sigma=p, channel_axis=-1))
    elif corruption in ("brightness", "darkness"):
        hsv = rgb2hsv(img / 255.0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + (p if corruption == "brightness" else -p), 0, 1)
        out = _to_uint8(hsv2rgb(hsv))
    else:  # occlusion
        h, w, _ = img.shape
        side = int(round(np.sqrt(p * h * w)))
        assert 0 < side <= min(h, w)
        rng = np.random.default_rng(seed)
        top = int(rng.integers(0, h - side + 1))
        left = int(rng.integers(0, w - side + 1))
        out = img.copy()
        out[top:top + side, left:left + side, :] = OCCLUSION_VALUE

    assert out.dtype == np.uint8 and out.shape == img.shape
    return out


class DegradedDataset(Dataset):
    """Evaluation dataset with one corruption at one severity. Severity 0 produces tensors
    identical to the clean eval transform in src.data.build_transforms."""

    def __init__(self, rows: list[Record], cls2idx: dict[str, int], cfg: Config,
                 corruption: str, severity: int):
        assert corruption in CORRUPTIONS and severity in SEVERITIES
        self.rows, self.cls2idx, self.cfg = rows, cls2idx, cfg
        self.corruption, self.severity = corruption, severity
        self.geometry = transforms.Compose([transforms.Resize(cfg.resize),
                                            transforms.CenterCrop(cfg.img_size)])
        self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(MEAN, STD)])

    def __len__(self) -> int:
        return len(self.rows)

    def pixels(self, i: int) -> np.ndarray:
        """The degraded uint8 image, before normalisation."""
        r = self.rows[i]
        with Image.open(r.path) as im:
            img = np.asarray(self.geometry(im.convert("RGB")), dtype=np.uint8)
        key = r.path.rsplit("/", 1)[-1]
        return degrade(img, self.corruption, self.severity,
                       image_seed(self.cfg.seed, key, self.corruption, self.severity))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.to_tensor(self.pixels(i)), self.cls2idx[self.rows[i].cls]
