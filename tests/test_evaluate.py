"""Statistics, scores, coverage and threshold derivation: known answers by hand."""

import math

import numpy as np
import pytest

from src.evaluate import (
    Z95,
    apply_threshold,
    bootstrap_ece_ci,
    derive_threshold,
    risk_coverage,
    scores,
    threshold_table,
    wilson_ci,
)


def test_wilson_all_correct_known_answer():
    # k = n: lower bound = n / (n + z^2). For n = 610: 610 / 613.8415 = 0.99374
    lo, hi = wilson_ci(610, 610)
    assert lo == pytest.approx(610 / (610 + Z95**2), abs=1e-12)
    assert hi == 1.0


def test_wilson_none_correct_known_answer():
    lo, hi = wilson_ci(0, 50)
    assert lo == 0.0 and hi == pytest.approx(Z95**2 / (50 + Z95**2), abs=1e-12)


def test_wilson_half():
    # p = 0.5, n = 100: centre 0.5, half = z*sqrt(.25/100 + z^2/40000) / (1 + z^2/100)
    lo, hi = wilson_ci(50, 100)
    half = Z95 * math.sqrt(0.25 / 100 + Z95**2 / 40000) / (1 + Z95**2 / 100)
    assert (lo, hi) == pytest.approx((0.5 - half, 0.5 + half), abs=1e-12)


def test_risk_coverage_known_answer():
    """scores 0.9 0.8 0.7 0.6, correct 1 0 1 1 (already in descending order)
    selective risk at k = 1..4: 0, 1/2, 1/3, 1/4  ->  AURC = (0 + .5 + .3333 + .25) / 4"""
    rc = risk_coverage(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1.0, 0.0, 1.0, 1.0]))
    assert rc["aurc"] == pytest.approx((0 + 1 / 2 + 1 / 3 + 1 / 4) / 4, abs=1e-12)
    assert rc["accuracy_at_coverage"]["1.0"] == 0.75
    assert rc["accuracy_at_coverage"]["0.5"] == 0.5  # top 2: 1 of 2 correct
    assert rc["accuracy_at_coverage"]["0.1"] == 1.0  # top 1
    assert rc["n_distinct_scores"] == 4


def test_report_aurc_rounds_to_the_stable_precision():
    """D20: reported AURC has two decimals; the JSON keeps full precision."""
    from src.evaluate import AURC_DECIMALS, report_aurc
    assert AURC_DECIMALS == 2
    assert report_aurc(0.186396) == 0.19 and report_aurc(0.179281) == 0.18
    assert report_aurc(0.0) == 0.0


def test_saturation_known_answer():
    """D20: three predictions tied at the top (one wrong), two more tied lower down."""
    from src.evaluate import saturation
    from src.verify import saturation_v2
    s = np.array([1.0, 1.0, 1.0, 0.9, 0.5, 0.5, 0.2], dtype=np.float32)
    c = np.array([1, 0, 1, 1, 0, 1, 1], dtype=float)
    want = {"top_score": 1.0, "top_tie": 3, "top_tie_wrong": 1, "tied_fraction": 5 / 7}
    assert saturation(s, c) == want and saturation_v2(s, c) == want
    distinct = saturation(np.array([0.3, 0.2, 0.1]), np.ones(3))
    assert distinct["top_tie"] == 0 and distinct["tied_fraction"] == 0.0


def test_no_aurc_is_reported_from_a_saturated_curve_or_at_t_lbfgs():
    """D20: reported_aurc is the only way an AURC reaches the report."""
    from src.evaluate import SATURATION_LIMIT, reported_aurc, risk_coverage
    y = np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=float)
    fine = risk_coverage(np.linspace(1, 0, 10), y)
    tied = risk_coverage(np.r_[np.ones(5), np.linspace(0.9, 0, 5)], y)
    assert tied["saturation"]["tied_fraction"] > SATURATION_LIMIT
    row: dict = {"corruption": "darkness", "severity": 5, "scores": {"max_softmax": fine},
                 "temperature": {"T_lbfgs": {"coverage": {"max_softmax": fine}}}}
    assert reported_aurc(row, "scores", "max_softmax") == round(fine["aurc"], 2)
    with pytest.raises(ValueError, match="not reported"):
        reported_aurc(row, "T_lbfgs", "max_softmax")  # never, even when it does not saturate
    row["scores"]["max_softmax"] = tied  # saturated: nothing is reported (sat. / a gap)
    assert reported_aurc(row, "scores", "max_softmax") is None


def test_scores_known_answers():
    logits = np.array([[2.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    s = scores(logits)
    assert s["margin"].tolist() == [1.5, 0.0]
    assert s["neg_entropy"][1] == pytest.approx(-math.log(4), abs=1e-6)  # uniform
    assert s["max_softmax"][1] == pytest.approx(0.25)


def test_threshold_table():
    conf = np.array([0.95, 0.85, 0.75, 0.65])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    row = {r["threshold"]: r for r in threshold_table(conf, correct, (0.8, 0.99))}
    assert row[0.8]["n_accepted"] == 2 and row[0.8]["acc_accepted"] == 1.0
    assert row[0.8]["acc_referred"] == 0.0
    assert row[0.99]["n_accepted"] == 0 and row[0.99]["acc_accepted"] is None


def test_bootstrap_ci_brackets_and_is_seeded():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 300)
    correct = (rng.uniform(size=300) < conf - 0.1).astype(float)
    a = bootstrap_ece_ci(conf, correct, 15, seed=1, n_boot=200)
    b = bootstrap_ece_ci(conf, correct, 15, seed=1, n_boot=200)
    assert a == b and a[0] < a[1]


def test_derive_threshold_known_answer():
    """Two conditions. Clean: all right. Shifted: scores 0.9 0.8 0.7 0.6, correct 1 1 0 1.
    For 95%: t = 0.7 accepts 0.9 0.8 0.7 in 'shifted' -> 2/3, fails; t = 0.8 -> 2/2, passes;
    clean passes at every t. Lowest candidate passing everywhere = 0.8."""
    conds = {"clean": (np.array([0.99, 0.95, 0.9]), np.ones(3)),
             "shifted": (np.array([0.9, 0.8, 0.7, 0.6]), np.array([1.0, 1.0, 0.0, 1.0]))}
    r = derive_threshold(conds, 0.95)
    assert r["found"] and r["threshold"] == 0.8
    assert r["per_condition"]["shifted"]["coverage"] == 0.5
    assert r["per_condition"]["clean"]["coverage"] == 1.0
    assert r["vacuous_conditions"] == []


def test_apply_threshold_reports_held_out_failures():
    """t = 0.8 chosen elsewhere. Here: 'a' scores 0.9 0.85 0.7 correct 1 0 1 -> accepts 2,
    1/2 right, below 0.95; 'b' scores 0.5 0.4 -> accepts nothing (vacuous)."""
    conds = {"a": (np.array([0.9, 0.85, 0.7]), np.array([1.0, 0.0, 1.0])),
             "b": (np.array([0.5, 0.4]), np.array([1.0, 1.0]))}
    r = apply_threshold(conds, 0.8, 0.95)
    assert r["below_target"] == ["a"] and r["vacuous_conditions"] == ["b"]
    assert r["per_condition"]["a"]["n_accepted"] == 2
    assert r["per_condition"]["a"]["acc_accepted"] == 0.5
    assert r["coverage_mean"] == (2 / 3 + 0) / 2


def test_derive_threshold_vacuous_when_only_rejecting_everything_works():
    # the highest-scoring prediction is wrong: only a threshold above it "works"
    conds = {"bad": (np.array([0.9, 0.8]), np.array([0.0, 1.0]))}
    r = derive_threshold(conds, 0.95)
    # 0.9 accepts only the wrong one; 0.8 accepts both (50%): no candidate satisfies it
    assert r["found"] is False and r["threshold"] is None
