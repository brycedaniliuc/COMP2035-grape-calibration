"""verify's threshold check (D20): images placed by float64 score, near-ties inside the band
settled by the counts the evaluation reported, the test side compared on the stored scores.
It must pass an honest threshold and fail every kind of wrong one."""

import copy

import numpy as np

from src.evaluate import apply_threshold, scores, threshold_analysis
from src.verify import THRESHOLD_BAND, threshold_checks


def store(conf: dict[str, list[float]], wrong: dict[str, list[int]]) -> dict:
    """Two classes; max softmax exactly conf (to float32), class 0 predicted; the images in
    `wrong` have label 1. Every condition has the same number of images."""
    st: dict = {}
    n = len(next(iter(conf.values())))
    for k, ps in conf.items():
        p = np.asarray(ps, np.float64)
        st[k] = np.stack([np.log(p), np.log1p(-p)], 1).astype(np.float32)
    st["labels"] = np.zeros(n, np.int64)
    for k in conf:  # one label vector: wrong images are wrong in every condition they name
        st["labels"][wrong.get(k, [])] = 1
    return st


def with_stored(st: dict) -> dict:
    return {**st, **{f"{k}__scores__max_softmax": scores(v)["max_softmax"]
                     for k, v in st.items() if k != "labels"}}


VAL = {"a_s1": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90],
       "b_s1": [0.995, 0.985, 0.975, 0.965, 0.955, 0.945, 0.935, 0.925, 0.915, 0.905]}
TEST = {"a_s1": [0.991, 0.981, 0.971, 0.961, 0.951, 0.941, 0.931, 0.921, 0.911, 0.901],
        "b_s1": [0.996, 0.986, 0.976, 0.966, 0.956, 0.946, 0.936, 0.926, 0.916, 0.906]}


def sweep(val: dict, test: dict) -> dict:
    return {"threshold": threshold_analysis({k: v for k, v in test.items() if "__" not in k},
                                            val)}


def result(sw: dict, val: dict, test: dict) -> dict[str, tuple[str, str]]:
    return {c[0]: (c[2], c[3]) for c in threshold_checks(sw, test, val)}


LOWEST = "threshold: chosen on validation, lowest that meets target"
HELD_OUT = "threshold: held-out test numbers re-derived"


def test_an_honest_threshold_passes_on_the_stored_scores():
    val, test = store(VAL, {"a_s1": [6]}), with_stored(store(TEST, {"a_s1": [6]}))
    sw = sweep(val, test)
    assert sw["threshold"]["by_score"]["max_softmax"]["0.90"]["validation"]["found"]
    got = result(sw, val, test)
    assert all(s == "PASS" for s, _ in got.values()), got
    assert "on the stored scores (exact)" in got[HELD_OUT][1]


def test_a_threshold_that_is_not_the_lowest_fails():
    val, test = store(VAL, {"a_s1": [6]}), with_stored(store(TEST, {"a_s1": [6]}))
    sw = sweep(val, test)
    r = sw["threshold"]["by_score"]["max_softmax"]["0.90"]
    t = r["validation"]["threshold"]
    higher = min(float(x) for k in VAL for x in scores(val[k])["max_softmax"] if x > t)
    conds = {k: (scores(val[k])["max_softmax"], (val[k].argmax(1) == val["labels"]).astype(
        float)) for k in VAL}
    r["validation"] = {"found": True, **apply_threshold(conds, higher, 0.90)}  # consistent
    tconds = {k: (scores(test[k])["max_softmax"], (test[k].argmax(1) == test["labels"])
                  .astype(float)) for k in TEST}
    r["test"] = apply_threshold(tconds, higher, 0.90)
    got = result(sw, val, test)
    assert got[LOWEST][0] == "FAIL" and "0.90" in got[LOWEST][1]
    assert got[HELD_OUT][0] == "PASS"  # the test numbers at that threshold are right


def test_reported_counts_that_no_placement_gives_fail():
    val, test = store(VAL, {"a_s1": [6]}), with_stored(store(TEST, {"a_s1": [6]}))
    sw = sweep(val, test)
    bad = copy.deepcopy(sw)
    bad["threshold"]["by_score"]["max_softmax"]["0.90"]["validation"]["per_condition"][
        "a_s1"]["n_accepted"] += 1
    got = result(bad, val, test)
    assert got[LOWEST][0] == "FAIL" and "no placement" in got[LOWEST][1]


def test_wrong_held_out_numbers_fail():
    val, test = store(VAL, {"a_s1": [6]}), with_stored(store(TEST, {"a_s1": [6]}))
    sw = sweep(val, test)
    bad = copy.deepcopy(sw)
    bad["threshold"]["by_score"]["max_softmax"]["0.90"]["test"]["per_condition"]["b_s1"][
        "n_accepted"] -= 1
    assert result(bad, val, test)[HELD_OUT][0] == "FAIL"


def test_a_near_tie_inside_the_band_is_settled_by_the_reported_counts():
    """An eleventh validation image, wrong, sits 3e-7 below the image the threshold came
    from: closer than the band, so another machine's float32 might order the two either
    way. The counts and accuracies the evaluation reported say which side it fell on."""
    test = with_stored(store(TEST, {"a_s1": [6]}))
    t = sweep(store(VAL, {"a_s1": [6]}), test)["threshold"]["by_score"]["max_softmax"][
        "0.90"]["validation"]["threshold"]
    conf = {"a_s1": [*VAL["a_s1"], t - 3e-7], "b_s1": [*VAL["b_s1"], 0.5]}
    val = store(conf, {"a_s1": [6, 10]})
    sw = sweep(val, test)
    r = sw["threshold"]["by_score"]["max_softmax"]["0.90"]
    assert r["validation"]["threshold"] == t and 3e-7 < THRESHOLD_BAND  # still that image
    got = result(sw, val, test)
    assert got[LOWEST][0] == "PASS" and "1 target(s)" in got[LOWEST][1], got[LOWEST]
    # and if the evaluation had reported the wrong image as accepted, nothing would fit
    bad = copy.deepcopy(sw)
    pc = bad["threshold"]["by_score"]["max_softmax"]["0.90"]["validation"]["per_condition"]
    pc["a_s1"]["n_accepted"], pc["a_s1"]["acc_accepted"] = 11, 9 / 11
    assert result(bad, val, test)[LOWEST][0] == "FAIL"


def test_without_stored_scores_an_image_at_the_threshold_is_undecidable():
    val = store(VAL, {"a_s1": [6]})
    test_conf = {k: list(v) for k, v in TEST.items()}
    sw0 = sweep(val, store(test_conf, {"a_s1": [6]}))
    t = sw0["threshold"]["by_score"]["max_softmax"]["0.90"]["validation"]["threshold"]
    test_conf["b_s1"][9] = t + 1e-7  # a test image on the boundary, no stored arrays
    test = store(test_conf, {"a_s1": [6]})
    got = result(sweep(val, test), val, test)
    assert got[HELD_OUT][0] == "FAIL" and "undecidable" in got[HELD_OUT][1]
