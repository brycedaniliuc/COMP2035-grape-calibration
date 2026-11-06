"""Degradations, on real fixture images."""

import numpy as np
import pytest
import torch

from src.config import Config
from src.data import build_transforms, index_dataset
from src.degrade import CORRUPTIONS, SEVERITIES, DegradedDataset, degrade


@pytest.fixture(scope="module")
def leaf(fixture_root) -> np.ndarray:
    ds = DegradedDataset(index_dataset(fixture_root)[:1], {"Black_rot": 0}, Config(),
                         "gaussian_blur", 0)
    img = ds.pixels(0)
    assert img.shape == (224, 224, 3) and img.dtype == np.uint8
    return img


@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_severity_zero_is_pixel_identical(leaf, corruption):
    out = degrade(leaf, corruption, 0, seed=123)
    assert np.array_equal(out, leaf) and out is not leaf


@pytest.mark.parametrize("corruption", CORRUPTIONS)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_dtype_and_shape_unchanged(leaf, corruption, severity):
    out = degrade(leaf, corruption, severity, seed=1)
    assert out.dtype == np.uint8 and out.shape == leaf.shape


def test_rejects_non_uint8(leaf):
    with pytest.raises(AssertionError, match="uint8"):
        degrade(leaf.astype(np.float32) / 255, "gaussian_blur", 1)


def test_blur_variance_decreases(leaf):
    v = [degrade(leaf, "gaussian_blur", s).astype(float).var() for s in SEVERITIES]
    assert all(a > b for a, b in zip(v, v[1:], strict=False)), v


def test_brightness_mean_increases_and_darkness_decreases(leaf):
    up = [degrade(leaf, "brightness", s).mean() for s in SEVERITIES]
    down = [degrade(leaf, "darkness", s).mean() for s in SEVERITIES]
    assert all(a < b for a, b in zip(up, up[1:], strict=False)), up
    assert all(a > b for a, b in zip(down, down[1:], strict=False)), down


def test_occlusion_patch_pixels_increase(leaf):
    black = [int((degrade(leaf, "occlusion", s, seed=7) == 0).all(axis=2).sum())
             for s in SEVERITIES]
    assert all(a < b for a, b in zip(black, black[1:], strict=False)), black
    # severity 5 covers 40% of the area: at least that many black pixels
    assert black[5] >= int(0.40 * 224 * 224 * 0.99)


def test_occlusion_seeded(leaf):
    a = degrade(leaf, "occlusion", 3, seed=11)
    b = degrade(leaf, "occlusion", 3, seed=11)
    c = degrade(leaf, "occlusion", 3, seed=12)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_deterministic_corruptions_ignore_seed(leaf):
    for corruption in ("gaussian_blur", "brightness", "darkness"):
        assert np.array_equal(degrade(leaf, corruption, 4, seed=1),
                              degrade(leaf, corruption, 4, seed=2))


def test_severity_zero_dataset_matches_clean_eval_tensors(fixture_root):
    """The identity assertion at tensor level: severity 0 through DegradedDataset gives
    exactly the tensors the clean evaluation pipeline gives."""
    recs = index_dataset(fixture_root)
    cfg = Config()
    cls2idx = {c: i for i, c in enumerate(sorted({r.cls for r in recs}))}
    _, eval_tf = build_transforms(cfg)
    from src.data import LeafDataset
    clean = LeafDataset(recs, eval_tf, cls2idx)
    for corruption in CORRUPTIONS:
        deg = DegradedDataset(recs, cls2idx, cfg, corruption, 0)
        for i in range(0, len(recs), 9):
            x0, y0 = clean[i]
            x1, y1 = deg[i]
            assert y0 == y1 and torch.equal(x0, x1), (corruption, recs[i].path)


def test_dataset_occlusion_reproducible_across_instances(fixture_root):
    recs = index_dataset(fixture_root)[:5]
    a = DegradedDataset(recs, {r.cls: 0 for r in recs}, Config(), "occlusion", 4)
    b = DegradedDataset(recs, {r.cls: 0 for r in recs}, Config(), "occlusion", 4)
    for i in range(len(recs)):
        assert np.array_equal(a.pixels(i), b.pixels(i))
