"""Calibration under shift: the degradation sweep (Stage 4), coverage curves and the
threshold (Stage 5).

    python -m src.evaluate --results outputs/<run>/results_<run_id>.json

Reads the run's config, checkpoint and split from its results JSON. Applies every
corruption at every severity to the `source` test set (originals only), at inference only,
and to the validation set: the threshold is chosen on degraded validation images and checked
on the degraded test images, which it never saw.

Temperature protocol (BRIEF.md Stage 4, AMENDED; DECISIONS.md D1):
  1. degeneracy: validation NLL over a T grid, and the LBFGS endpoint for several max_iter
  2. T = 1 (no temperature) is the primary condition
  3. a fixed grid of T, each chosen in advance and held fixed across every severity
  4. the LBFGS endpoint on clean validation, held fixed; labelled "as-fitted (degenerate)"
     when validation had zero errors. It is never called a fitted temperature then.
No temperature is ever refitted on degraded data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.calibrate import ece_and_bins, lbfgs_temperature, softmax_outputs
from src.config import Config
from src.data import class_list, index_dataset, make_splits
from src.degrade import CORRUPTIONS, SEVERITIES, DegradedDataset, severity_parameter
from src.train import (
    _json_safe,
    build_model,
    collect_logits,
    configure_determinism,
    select_device,
    set_seed,
)

T_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
NLL_T_RANGE = (0.01, 5.0)
LBFGS_MAX_ITERS = (5, 10, 20, 50, 100, 200, 500)
N_BOOT = 1000
Z95 = 1.959963984540054


# ---- statistics ---------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    assert n > 0 and 0 <= k <= n
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # the bounds are exactly 0 at k = 0 and exactly 1 at k = n; avoid rounding residue
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def bootstrap_ece_ci(confs: np.ndarray, correct: np.ndarray, n_bins: int, seed: int,
                     n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on ECE, resampling predictions with replacement."""
    rng = np.random.default_rng(seed)
    n = confs.size
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = ece_and_bins(confs[idx], correct[idx], n_bins)[0]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---- confidence scores ----------------------------------------------------------------

def scores(logits: np.ndarray, temperature: float = 1.0) -> dict[str, np.ndarray]:
    """Three confidence scores, higher = more confident.
    max_softmax: max class probability at T. neg_entropy: minus predictive entropy (nats)
    at T. margin: top logit minus second logit (raw logits, as BRIEF.md specifies)."""
    z = torch.as_tensor(logits, dtype=torch.float32)
    p = F.softmax(z / temperature, 1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1)
    top2 = z.topk(2, dim=1).values
    return {"max_softmax": p.max(1).values.numpy(), "neg_entropy": (-ent).numpy(),
            "margin": (top2[:, 0] - top2[:, 1]).numpy()}


# ---- coverage (Stage 5) ----------------------------------------------------------------

def risk_coverage(score: np.ndarray, correct: np.ndarray) -> dict:
    """Selective classification (Geifman & El-Yaniv 2017): accept the most confident
    predictions first. Returns the full risk-coverage curve, accuracy at coverage 1.0 ... 0.1,
    and AURC (mean selective risk over coverage k/n, k = 1..n). Ties are broken by input
    order (stable sort), which is deterministic."""
    n = score.size
    order = np.argsort(-score, kind="stable")
    cum_correct = np.cumsum(correct[order])
    k = np.arange(1, n + 1)
    risk = 1 - cum_correct / k
    grid = {}
    for c in np.round(np.arange(1.0, 0.05, -0.1), 1):
        m = max(1, int(math.ceil(c * n)))
        grid[f"{c:.1f}"] = float(cum_correct[m - 1] / m)
    return {"aurc": float(risk.mean()), "accuracy_at_coverage": grid,
            "n_distinct_scores": int(np.unique(score).size)}


def threshold_table(conf: np.ndarray, correct: np.ndarray,
                    thresholds: tuple[float, ...] = THRESHOLDS) -> list[dict]:
    """Accept when conf >= threshold. Accuracy above (accepted) and below (referred)."""
    rows = []
    for t in thresholds:
        acc = conf >= t
        rows.append({"threshold": t, "n_accepted": int(acc.sum()), "coverage": float(acc.mean()),
                     "acc_accepted": float(correct[acc].mean()) if acc.any() else None,
                     "acc_referred": float(correct[~acc].mean()) if (~acc).any() else None})
    return rows


# ---- one condition ----------------------------------------------------------------------

def evaluate_logits(logits: np.ndarray, labels: np.ndarray, classes: list[str],
                    temps: dict[str, float], n_bins: int, seed: int) -> dict:
    _, preds, _, correct = softmax_outputs(logits, labels)
    n, k = labels.size, int(correct.sum())
    lo, hi = wilson_ci(k, n)
    out: dict = {"n": n, "n_correct": k, "accuracy": k / n, "accuracy_wilson95": [lo, hi],
                 "per_class": {}, "temperature": {}, "scores": {}}
    for i, c in enumerate(classes):
        m = labels == i
        out["per_class"][c] = {"n": int(m.sum()), "accuracy": float(correct[m].mean())}
    for name, t in temps.items():
        _, preds_t, conf_t, correct_t = softmax_outputs(logits, labels, t)
        assert np.array_equal(preds, preds_t), f"T={t} changed a prediction"
        ece, bins = ece_and_bins(conf_t, correct_t, n_bins)
        out["temperature"][name] = {
            "T": t, "mean_confidence": float(conf_t.mean()), "ece": ece,
            "ece_bootstrap95": list(bootstrap_ece_ci(conf_t, correct_t, n_bins, seed)),
            "reliability_bins": bins,
            "per_class_mean_confidence": {c: float(conf_t[labels == i].mean())
                                          for i, c in enumerate(classes)},
            # T can reorder images by max softmax or entropy, so each T has its own coverage
            # curve. Margin / T ranks exactly as margin, so its T = 1 curve (in "scores") holds.
            "coverage": {k: risk_coverage(s, correct_t)
                         for k, s in scores(logits, t).items() if k != "margin"},
        }
    for name, s in scores(logits).items():
        out["scores"][name] = risk_coverage(s, correct)
    out["thresholds_max_softmax_T1"] = threshold_table(scores(logits)["max_softmax"], correct)
    return out


# ---- degeneracy demonstration (condition 1) ----------------------------------------------

def degeneracy(val_logits: np.ndarray, val_labels: np.ndarray, cfg: Config) -> dict:
    """Validation NLL over T, and where LBFGS stops for several max_iter. The T grid starts
    at NLL_T_RANGE[0] or below the smallest endpoint, whichever is lower, so every endpoint
    lies on the plotted curve."""
    z = torch.as_tensor(val_logits, dtype=torch.float32)
    y = torch.as_tensor(val_labels, dtype=torch.long)

    def nll_at(t: float) -> float:
        return float(F.cross_entropy(z / t, y))

    endpoints = {str(m): lbfgs_temperature(val_logits, val_labels, cfg.temperature_lr, m)
                 ["t_lbfgs_endpoint"] for m in LBFGS_MAX_ITERS}
    lower = min(NLL_T_RANGE[0], 0.8 * min(endpoints.values()))
    ts = np.geomspace(lower, NLL_T_RANGE[1], 240)
    nll = [nll_at(t) for t in ts]
    i_min = int(np.argmin(nll))
    return {"T_grid": ts.tolist(), "val_nll": nll,
            "argmin_T": float(ts[i_min]), "minimum_at_lower_edge": i_min == 0,
            "nll_monotone_increasing": bool(np.all(np.diff(nll) > 0)),
            "n_val_errors": int((z.argmax(1) != y).sum()),
            "lbfgs_endpoint_by_max_iter": endpoints,
            "lbfgs_endpoint_nll_by_max_iter": {m: nll_at(t) for m, t in endpoints.items()}}


# ---- the sweep ----------------------------------------------------------------------------

def sweep(results_path: str | Path, data_root: str | None = None,
          batch_size: int | None = None, num_workers: int | None = None) -> dict:
    results_path = Path(results_path)
    res = json.loads(results_path.read_text())
    overrides = {k: v for k, v in {"data_root": data_root, "batch_size": batch_size,
                                    "num_workers": num_workers}.items() if v is not None}
    cfg = Config(**{**res["config"], **overrides})
    assert cfg.split_mode == "source", "the sweep runs on the source split only (BRIEF.md)"
    out_dir = results_path.parent
    run_id = res["run_id"]

    device = select_device()
    set_seed(cfg.seed)
    configure_determinism(cfg.deterministic)

    records = index_dataset(cfg.data_root)
    classes = class_list(records)
    assert classes == res["classes"]
    cls2idx = {c: i for i, c in enumerate(classes)}
    splits = make_splits(records, cfg)
    assert splits.summary()["split_sizes"] == res["split_sizes"], "split differs from the run"

    model, _ = build_model(cfg.model, len(classes), pretrained=False)
    state = torch.load(out_dir / res["artefacts"]["checkpoint"], map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    def logits_for(rows: list, corruption: str, severity: int) -> tuple[np.ndarray, np.ndarray]:
        ds = DegradedDataset(rows, cls2idx, cfg, corruption, severity)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
        return collect_logits(model, dl, device)

    # Clean validation logits: the only data any temperature is chosen on.
    val_logits, val_y = logits_for(splits.val, CORRUPTIONS[0], 0)
    saved = np.load(out_dir / res["artefacts"]["logits"])
    val_diff = float(np.abs(val_logits - saved["val_logits"]).max())
    t_fit = lbfgs_temperature(val_logits, val_y, cfg.temperature_lr, cfg.temperature_max_iter)
    t_lbfgs = t_fit["t_lbfgs_endpoint"]
    lbfgs_label = ("T_lbfgs (as-fitted, degenerate: zero validation errors)"
                   if t_fit["degenerate"] else "T_lbfgs (fitted on clean validation)")
    temps = {f"T={t:g}": t for t in T_GRID}
    temps["T_lbfgs"] = t_lbfgs
    print(f"temperatures held fixed for every severity: {temps}")
    print(f"T_lbfgs is {lbfgs_label}")

    test_y = np.array([cls2idx[r.cls] for r in splits.test])
    rows: list[dict] = []
    logits_store: dict[str, np.ndarray] = {"labels": test_y}
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            lg, y = logits_for(splits.test, corruption, severity)
            assert np.array_equal(y, test_y), "test label order drifted"
            logits_store[f"{corruption}_s{severity}"] = lg
            row = {"corruption": corruption, "severity": severity,
                   "parameter": severity_parameter(corruption, severity),
                   **evaluate_logits(lg, y, classes, temps, cfg.n_bins, cfg.seed)}
            rows.append(row)
            t1 = row["temperature"]["T=1"]
            print(f"  {corruption:<14} s{severity}  acc {row['accuracy']:.4f} "
                  f"[{row['accuracy_wilson95'][0]:.3f},{row['accuracy_wilson95'][1]:.3f}]  "
                  f"conf {t1['mean_confidence']:.4f}  ECE(T=1) {t1['ece']:.4f}  "
                  f"ECE(T_lbfgs) {row['temperature']['T_lbfgs']['ece']:.4f}")

    # The same 24 conditions on the validation images: the threshold is chosen there, so the
    # test images stay held out for checking it.
    val_store: dict[str, np.ndarray] = {"labels": val_y}
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            lg, y = logits_for(splits.val, corruption, severity)
            assert np.array_equal(y, val_y), "validation label order drifted"
            val_store[f"{corruption}_s{severity}"] = lg
        print(f"  validation images, {corruption}: severities 0-5 done")
    assert all(np.array_equal(val_store[f"{c}_s0"], val_logits) for c in CORRUPTIONS), \
        "validation severity 0 differs from the clean validation logits"

    # Held fixed: every row used exactly the same temperatures.
    for r in rows:
        assert {k: v["T"] for k, v in r["temperature"].items()} == temps
    # Severity 0 is the identity: every corruption's s0 gives the same logits, and those
    # reproduce the training run's clean test logits.
    s0 = [logits_store[f"{c}_s0"] for c in CORRUPTIONS]
    assert all(np.array_equal(s0[0], x) for x in s0[1:]), "severity 0 differs by corruption"
    clean_diff = float(np.abs(s0[0] - saved["test_logits"]).max())
    clean = rows[0]
    assert clean["accuracy"] == res["test_accuracy"], "clean accuracy differs from the run"
    t1 = clean["temperature"]["T=1"]
    for got, want, what in ((t1["mean_confidence"], res["mean_confidence"], "mean confidence"),
                            (t1["ece"], res["ece"], "ECE"),
                            (t_lbfgs, res["t_lbfgs"]["t_lbfgs_endpoint"], "T_lbfgs"),
                            (clean["temperature"]["T_lbfgs"]["ece"], res["ece_at_t_lbfgs"],
                             "ECE at T_lbfgs")):
        assert abs(got - want) <= 1e-6, f"clean {what} {got} differs from the run's {want}"
    print(f"T identical across all {len(rows)} rows: {temps}")
    print(f"clean row vs training run: accuracy {clean['accuracy']} == {res['test_accuracy']}"
          f", max |logit diff| test {clean_diff:.2e}, val {val_diff:.2e}")

    sweep_res = {
        "run_id": run_id, "config": cfg.to_dict(), "classes": classes,
        "n_test": int(logits_store["labels"].size), "n_val": int(val_y.size),
        "temperatures": temps, "t_lbfgs": t_fit, "t_lbfgs_label": lbfgs_label,
        "degeneracy": degeneracy(val_logits, val_y, cfg),
        "clean_check": {"max_abs_logit_diff_test": clean_diff,
                        "max_abs_logit_diff_val": val_diff,
                        "accuracy_matches_run": True},
        "protocol": {"corruptions": list(CORRUPTIONS), "severities": list(SEVERITIES),
                     "t_grid": list(T_GRID), "thresholds": list(THRESHOLDS),
                     "n_bootstrap": N_BOOT, "ci": "Wilson 95% (accuracy), "
                     "percentile bootstrap 95% (ECE)"},
        "rows": rows,
        "threshold": threshold_analysis(logits_store, val_store),
    }
    sweep_path = out_dir / f"sweep_{run_id}.json"
    sweep_path.write_text(json.dumps(_json_safe(sweep_res), indent=1, allow_nan=False))
    logits_path = out_dir / f"sweep_logits_{run_id}.npz"
    # numpy stubs type **kwds against the allow_pickle: bool keyword; the call is correct
    np.savez(logits_path, **logits_store)  # type: ignore[arg-type]
    val_path = out_dir / f"sweep_val_logits_{run_id}.npz"
    np.savez(val_path, **val_store)  # type: ignore[arg-type]
    print(f"written -> {sweep_path}\nwritten -> {logits_path}\nwritten -> {val_path}")
    return sweep_res


# ---- the threshold (Stage 5) --------------------------------------------------------------

TARGETS = (0.90, 0.95, 0.99)
PRIMARY_TARGET = 0.95


def derive_threshold(conditions: dict[str, tuple[np.ndarray, np.ndarray]],
                     target: float) -> dict:
    """Lowest threshold t on a score such that, in EVERY condition, the predictions with
    score >= t have accuracy >= target. A condition where t accepts nothing is satisfied
    vacuously (everything is referred) and is listed, so a threshold that only "works" by
    referring everything is visible as such.

    conditions: name -> (score, correct). Candidates are every observed score value."""
    cand = np.unique(np.concatenate([s for s, _ in conditions.values()]))
    ok = np.ones(cand.size, dtype=bool)
    for s, c in conditions.values():
        order = np.argsort(-s, kind="stable")
        s_desc, cum = s[order], np.cumsum(c[order])
        # number accepted at each candidate: count of s >= t
        k = s.size - np.searchsorted(np.sort(s), cand, side="left")
        assert k[0] == np.sum(s_desc >= cand[0]) and k[-1] == np.sum(s_desc >= cand[-1])
        acc = np.where(k > 0, cum[np.maximum(k, 1) - 1] / np.maximum(k, 1), 1.0)
        ok &= acc >= target
    if not ok.any():
        return {"target": target, "threshold": None, "found": False}
    return {"found": True, **apply_threshold(conditions, float(cand[np.argmax(ok)]), target)}


def apply_threshold(conditions: dict[str, tuple[np.ndarray, np.ndarray]], t: float,
                    target: float) -> dict:
    """Accept score >= t in every condition: coverage, accuracy on the accepted predictions
    (Wilson 95%), the conditions below the target, and those where t accepts nothing."""
    per: dict[str, dict] = {}
    for name, (s, c) in conditions.items():
        a = s >= t
        n_acc = int(a.sum())
        per[name] = {"coverage": float(a.mean()), "n_accepted": n_acc,
                     "acc_accepted": float(c[a].mean()) if n_acc else None,
                     "acc_accepted_wilson95": list(wilson_ci(int(c[a].sum()), n_acc))
                     if n_acc else None}
    covs = [float(v["coverage"]) for v in per.values()]
    vacuous = [n for n, v in per.items() if v["n_accepted"] == 0]
    below = [n for n, v in per.items() if v["n_accepted"] and v["acc_accepted"] < target]
    return {"target": target, "threshold": t,
            "coverage_min": min(covs), "coverage_mean": float(np.mean(covs)),
            "referred_mean": 1 - float(np.mean(covs)), "below_target": below,
            "vacuous_conditions": vacuous, "accepts_nothing_anywhere": len(vacuous) == len(per),
            "per_condition": per}


# Stage 5: state what the threshold does not cover. Printed with every threshold.
THRESHOLD_LIMITS = (
    "PlantVillage laboratory images: one detached leaf on a plain background, not field "
    "photographs",
    "four classes (black rot, esca, leaf blight, healthy); any other disease or crop is out "
    "of scope",
    "only the four synthetic corruptions (Gaussian blur, brightening, darkening, square "
    "occlusion) at severities 1-5; background clutter, other cameras, motion blur and "
    "several leaves in one image are untested",
    "one trained model (one seed); another training run can move the threshold",
    "a condition where the threshold accepts nothing meets the target only vacuously: "
    "everything there is referred to a human",
)


def _conditions(store: dict[str, np.ndarray], score: str) -> dict:
    y = store["labels"]
    return {k: (scores(lg)[score], (lg.argmax(1) == y).astype(float))
            for k, lg in store.items() if k != "labels"}


def threshold_analysis(test_store: dict[str, np.ndarray],
                       val_store: dict[str, np.ndarray]) -> dict:
    """Thresholds for each score at each target, across every (corruption, severity).

    The threshold is CHOSEN on the degraded validation images and then APPLIED, unchanged,
    to the degraded test images, so the test numbers are held out. The threshold chosen on
    the test images themselves is kept for comparison only ("in_sample_test"): there every
    condition meets the target by construction, which overstates what new images get."""
    assert set(test_store) == set(val_store), "validation and test conditions differ"
    out: dict = {"primary_target": PRIMARY_TARGET,
                 "chosen_on": "degraded validation images (every corruption x severity)",
                 "evaluated_on": "degraded test images (held out)",
                 "score_units": {"max_softmax": "probability, T = 1",
                                 "neg_entropy": "minus entropy in nats, T = 1",
                                 "margin": "top logit minus second logit"},
                 "limits": list(THRESHOLD_LIMITS), "by_score": {}}
    for score in ("max_softmax", "neg_entropy", "margin"):
        cv, ct = _conditions(val_store, score), _conditions(test_store, score)
        by: dict[str, dict] = {}
        for tgt in TARGETS:
            chosen = derive_threshold(cv, tgt)
            by[f"{tgt:.2f}"] = {
                "validation": chosen,
                "test": apply_threshold(ct, chosen["threshold"], tgt) if chosen["found"]
                else None,
                "in_sample_test": derive_threshold(ct, tgt)}
        out["by_score"][score] = by
    return out


def _referral_note(r: dict) -> str:
    if r["accepts_nothing_anywhere"]:
        return " -- accepts NOTHING anywhere: not usable"
    n = len(r["vacuous_conditions"])
    return f", {n} condition(s) fully referred" if n else ""


def print_threshold(th: dict) -> None:
    """Plain-language summary of the threshold analysis."""
    print(f"threshold chosen on {th['chosen_on']}; checked on {th['evaluated_on']}")
    for score, by_t in th["by_score"].items():
        print(f"threshold on {score} ({th['score_units'][score]}):")
        for tgt, r in by_t.items():
            v, te, ins = r["validation"], r["test"], r["in_sample_test"]
            if not v["found"]:
                print(f"  target {tgt}: NO threshold reaches it in every validation condition")
                continue
            print(f"  target {tgt}: t = {v['threshold']:.6g}")
            print(f"    held-out test: {len(te['below_target'])} of {len(te['per_condition'])}"
                  f" conditions below {tgt}; coverage min {te['coverage_min']:.3f}, mean "
                  f"{te['coverage_mean']:.3f} (referred to a human: {te['referred_mean']:.1%}"
                  f" on average){_referral_note(te)}")
            if ins["found"]:
                print(f"    for comparison, chosen on the test images themselves (optimistic):"
                      f" t = {ins['threshold']:.6g}")
    print("what the threshold does not cover:")
    for s in th["limits"]:
        print("  -", s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results_<run_id>.json from src.train")
    ap.add_argument("--data-root", help="override the data root recorded in the run")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--num-workers", type=int)
    a = ap.parse_args()
    print_threshold(sweep(a.results, a.data_root, a.batch_size, a.num_workers)["threshold"])


if __name__ == "__main__":
    main()
