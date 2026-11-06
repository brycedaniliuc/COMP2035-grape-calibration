"""ECE, reliability bins and temperature scaling.

These use hand-built arrays on purpose: a known-answer test needs inputs whose ECE can be
worked out on paper.
"""

import math

import numpy as np
import pytest

from src.calibrate import calibration_summary, ece_and_bins, lbfgs_temperature


def test_ece_known_answer():
    """Worked by hand, 15 bins of width 1/15 (edges ..., 0.5333, 0.6000, ..., 0.9333, 1.0).

    confs   0.95 0.95 0.95 0.95 | 0.55 0.55
    correct    1    1    1    0 |    1    0
      bin (0.9333, 1.0]  : n=4, acc=3/4=0.75, conf=0.95, |gap|=0.20, weight 4/6
      bin (0.5333, 0.6]  : n=2, acc=1/2=0.50, conf=0.55, |gap|=0.05, weight 2/6
    ECE = (4/6)(0.20) + (2/6)(0.05) = 0.13333 + 0.01667 = 0.15
    """
    confs = np.array([0.95, 0.95, 0.95, 0.95, 0.55, 0.55])
    correct = np.array([1, 1, 1, 0, 1, 0])
    ece, bins = ece_and_bins(confs, correct, 15)
    assert ece == pytest.approx(0.15, abs=1e-12)
    filled = [b for b in bins if b["n"]]
    assert [(b["n"], b["acc"]) for b in filled] == [(2, 0.5), (4, 0.75)]


def test_ece_is_zero_when_calibrated():
    # 5 predictions at 0.8, 4 correct: accuracy equals confidence in the only bin
    ece, _ = ece_and_bins(np.full(5, 0.8), np.array([1, 1, 1, 1, 0]), 15)
    assert ece == pytest.approx(0.0, abs=1e-12)


def test_ece_underconfident_saturated_case():
    # The September shape: every prediction right, confidence ~0.92 -> ECE = 1 - mean conf
    confs = np.array([0.90, 0.92, 0.94, 0.96])
    ece, _ = ece_and_bins(confs, np.ones(4), 15)
    assert ece == pytest.approx(1 - confs.mean(), abs=1e-12)


def test_bin_edges_first_closed_others_right_closed():
    _, bins = ece_and_bins(np.array([0.0, 0.6, 1.0]), np.array([1, 1, 1]), 15)
    assert bins[0]["n"] == 1  # 0.0 lands in the first bin, closed at both ends
    assert bins[8]["n"] == 1 and bins[8]["hi"] == pytest.approx(0.6)  # 0.6 -> (0.5333, 0.6]
    assert bins[9]["n"] == 0
    assert bins[14]["n"] == 1  # 1.0 -> (0.9333, 1.0]
    assert math.isnan(bins[1]["acc"])  # empty bins report NaN, not 0


def test_bin_counts_and_weights_sum():
    rng = np.random.default_rng(0)
    confs = rng.uniform(0.25, 1.0, 1000)
    _, bins = ece_and_bins(confs, rng.integers(0, 2, 1000), 15)
    assert sum(b["n"] for b in bins) == 1000


def test_out_of_range_confidence_raises():
    with pytest.raises(AssertionError):
        ece_and_bins(np.array([1.2]), np.array([1]), 15)


def _overconfident_logits(n=400, seed=0):
    """Labels are right 70% of the time but logits are sharp: an overconfident model."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    pred = np.where(rng.uniform(size=n) < 0.7, y, (y + 1) % 4)
    logits = rng.normal(0, 0.1, (n, 4)).astype(np.float32)
    logits[np.arange(n), pred] += 6.0
    return logits, y


def test_temperature_softens_overconfident_logits():
    logits, y = _overconfident_logits()
    fit = lbfgs_temperature(logits, y)
    assert fit["t_lbfgs_endpoint"] > 1.0
    assert fit["val_nll_after"] < fit["val_nll_before"]
    assert not fit["degenerate"]


def test_temperature_preserves_predictions_and_reduces_ece():
    logits, y = _overconfident_logits(seed=1)
    val_logits, val_y = _overconfident_logits(seed=2)
    t = lbfgs_temperature(val_logits, val_y)["t_lbfgs_endpoint"]
    s = calibration_summary(logits, y, t, 15)  # asserts accuracy and predictions unchanged
    assert s["ece_at_t"] < s["ece"]


def test_temperature_with_zero_errors_is_flagged_degenerate():
    y = np.arange(40) % 4
    logits = np.zeros((40, 4), dtype=np.float32)
    logits[np.arange(40), y] = 2.0
    fit = lbfgs_temperature(logits, y)
    assert fit["degenerate"] and fit["n_val_errors"] == 0
    assert 0 < fit["t_lbfgs_endpoint"] < 1  # sharpens: NLL keeps falling as T -> 0
