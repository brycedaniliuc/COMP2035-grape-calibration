"""Independent re-derivation of every published number, from saved logits and filenames.

    python -m src.verify --run <run_dir>/results_<run_id>.json [--determinism]
    python -m src.verify --generated report/generated   # report numbers not stale
    python -m src.verify --ensemble <run_dir>/ensemble_<ens_id>.json
    python -m src.verify --plantseg <run_dir>/plantseg_<run_id>.json
    python -m src.verify --mcdropout <run_dir>/mcdropout_<mc_id>.json

Deliberately does NOT import src.calibrate, src.evaluate's statistics or src.data's filename
parsing: softmax, accuracy, ECE binning, coverage and leaf identity are written again here,
differently. If both implementations agree the number is trustworthy; if not, there is a bug.

Tier 1 runs anywhere (fixture included) and must pass: exit status 1 on any Tier 1 FAIL.
Tier 2 compares a full ResNet-50 `source` run with the September 2026 results. It is
PENDING COLAB until such a run exists, and is never "made to pass" by adjusting anything.
Tolerances (DECISIONS.md D2): split sizes and audit exact; accuracy, mean confidence and ECE
within 0.01 absolute; T is information only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SEPTEMBER = REPO / "results" / "source" / "results_resnet50_source.json"
SEPTEMBER_AUDIT = REPO / "results" / "source" / "audit.json"
LEAF = re.compile(r"([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})___")
ECE_TOL, PROB_TOL = 1e-5, 1e-5
AURC_TOL = 1e-9  # ranking runs on the same float32 scores (see scores32)
SCORE_TOL = 1e-5  # float32 scores vs this module's float64 softmax

Check = tuple[str, int, str, str]  # (name, tier, PASS / FAIL / PENDING COLAB, detail)


# ---- second implementations ------------------------------------------------------------

def softmax64(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece_v2(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> tuple[float, np.ndarray]:
    """Bin index by ceil(conf * n_bins) - 1 (bins (lo, hi], first bin also takes 0), then a
    weighted sum via bincount. Returns ECE and per-bin counts."""
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    s_acc = np.bincount(idx, weights=correct, minlength=n_bins)
    s_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    nz = cnt > 0
    gap = np.abs(s_acc[nz] - s_conf[nz])  # = n_b * |acc_b - conf_b|
    return float(gap.sum() / conf.size), cnt


def leaf_of(name: str) -> str:
    m = LEAF.search(name.rsplit("/", 1)[-1])
    assert m, f"no UUID in {name}"
    return m.group(1).lower()


def selective_acc(score: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Accuracy of the k most confident, k = 1..n, ties kept in input order."""
    order = sorted(range(score.size), key=lambda i: (-score[i], i))
    return np.cumsum(correct[order]) / np.arange(1, score.size + 1)


# ---- checks ------------------------------------------------------------------------------

def tier1(run_path: Path, require_sweep: bool = False) -> list[Check]:
    out: list[Check] = []
    res = json.loads(run_path.read_text())
    d = run_path.parent
    nb = res["config"]["n_bins"]
    z = np.load(d / res["artefacts"]["logits"])
    tl, ty = z["test_logits"], z["test_labels"]
    p = softmax64(tl)
    correct = (p.argmax(1) == ty).astype(float)

    def add(name: str, ok: bool, detail: str) -> None:
        out.append((name, 1, "PASS" if ok else "FAIL", detail))

    acc = correct.mean()
    add("accuracy from logits = reported", acc == res["test_accuracy"],
        f"{acc:.6f} vs {res['test_accuracy']:.6f}")
    mc = p.max(1).mean()
    add("mean confidence from logits = reported", abs(mc - res["mean_confidence"]) < PROB_TOL,
        f"{mc:.6f} vs {res['mean_confidence']:.6f}")
    e2, cnt = ece_v2(p.max(1), correct, nb)
    add("ECE re-derived independently = reported", abs(e2 - res["ece"]) < ECE_TOL,
        f"{e2:.6f} vs {res['ece']:.6f}")
    bins = res["reliability_bins"]
    add("bin counts sum to test size", sum(b["n"] for b in bins) == ty.size,
        f"{sum(b['n'] for b in bins)} vs {ty.size}")
    add("bin weights sum to 1", abs(sum(b["n"] for b in bins) / ty.size - 1) < 1e-12, "")
    add("bin counts = independent bincount", [b["n"] for b in bins] == cnt.tolist(), "")
    t = res["t_lbfgs"]["t_lbfgs_endpoint"]
    pt = softmax64(tl, t)
    add("T_lbfgs leaves every prediction unchanged",
        bool(np.array_equal(pt.argmax(1), p.argmax(1))), f"T = {t:.4g}")
    e2t, _ = ece_v2(pt.max(1), correct, nb)
    add("ECE at T_lbfgs re-derived = reported", abs(e2t - res["ece_at_t_lbfgs"]) < ECE_TOL,
        f"{e2t:.6f} vs {res['ece_at_t_lbfgs']:.6f}")

    # leakage: leaves re-derived from filenames with a separate regex
    sp = json.loads((d / res["artefacts"]["splits"]).read_text())
    leaves = {k: {leaf_of(f) for f in v} for k, v in sp.items()}
    if res["config"]["split_mode"] == "source":
        add("no test leaf in train (from filenames)", not (leaves["test"] & leaves["train"]),
            f"{len(leaves['test'] & leaves['train'])} shared")
        add("no val leaf in train (from filenames)", not (leaves["val"] & leaves["train"]), "")
        add("split sizes = reported", {k: len(v) for k, v in sp.items()} == res["split_sizes"],
            str({k: len(v) for k, v in sp.items()}))

    sw_path = d / f"sweep_{res['run_id']}.json"
    val_path = d / f"sweep_val_logits_{res['run_id']}.npz"
    if sw_path.exists() and not val_path.exists():
        out.append(("sweep validation logits present", 1, "FAIL", f"missing {val_path.name}"))
    elif sw_path.exists():
        out += tier1_sweep(res, json.loads(sw_path.read_text()),
                           dict(np.load(d / f"sweep_logits_{res['run_id']}.npz")), tl,
                           dict(np.load(val_path)))
    elif require_sweep:
        out.append(("sweep present", 1, "FAIL", f"missing {sw_path.name}"))
    else:  # notebook 01 verifies before notebook 02 has produced the sweep
        out.append(("sweep and figure checks", 1, "SKIPPED",
                    "no sweep yet (run src.evaluate; use --require-sweep to demand it)"))
    return out


def scores32(logits: np.ndarray, t: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(max softmax, minus entropy) in float32, the numbers the evaluation code ranks by.
    Confidences can sit closer together than float32 rounding, so ranking on a float64
    recomputation can order near-ties differently and change a curve without any bug.
    Every check therefore ranks on these, after checking they agree with this module's own
    float64 softmax (score_agreement); the ranking and counting code stays separate."""
    import torch
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    p = torch.softmax(z / t, 1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1)
    return p.max(1).values.numpy(), (-ent).numpy()


def score_agreement(logits: np.ndarray, t: float = 1.0) -> float:
    """Largest difference between scores32 and the float64 softmax and entropy."""
    m32, e32 = scores32(logits, t)
    p = softmax64(logits, t)
    e64 = (p * np.log(np.clip(p, 1e-12, None))).sum(1)
    return float(max(np.abs(m32 - p.max(1)).max(), np.abs(e32 - e64).max()))


def aurc_v2(score: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean(1 - selective_acc(score, correct)))


def tier1_sweep(res: dict, sw: dict, lg: dict, test_logits: np.ndarray,
                val_lg: dict) -> list[Check]:
    out: list[Check] = []
    nb = sw["config"]["n_bins"]
    y = lg["labels"]
    bad_acc, bad_ece, bad_pred, bad_t, cov_diff, agree = [], [], [], [], 0.0, 0.0
    for r in sw["rows"]:
        key = f"{r['corruption']}_s{r['severity']}"
        logits = lg[key]
        correct = (softmax64(logits).argmax(1) == y).astype(float)
        if correct.mean() != r["accuracy"]:
            bad_acc.append(key)
        for name, cond in r["temperature"].items():
            pt = softmax64(logits, cond["T"])
            if not np.array_equal(pt.argmax(1), logits.argmax(1)):
                bad_pred.append(f"{key}/{name}")
            if abs(ece_v2(pt.max(1), correct, nb)[0] - cond["ece"]) >= ECE_TOL:
                bad_ece.append(f"{key}/{name}")
            if cond["T"] != sw["temperatures"][name]:
                bad_t.append(f"{key}/{name}")
            agree = max(agree, score_agreement(logits, cond["T"]))
            for k, s in zip(("max_softmax", "neg_entropy"), scores32(logits, cond["T"]),
                            strict=True):
                cov_diff = max(cov_diff, abs(aurc_v2(s, correct)
                                             - cond["coverage"][k]["aurc"]))
    n = len(sw["rows"])
    out.append(("sweep: accuracy re-derived, every row", 1, "PASS" if not bad_acc else "FAIL",
                f"{n} rows" + (f"; mismatched {bad_acc}" if bad_acc else "")))
    out.append(("sweep: ECE re-derived, every row x T", 1, "PASS" if not bad_ece else "FAIL",
                f"{n * len(sw['temperatures'])} values" + (f"; {bad_ece[:5]}" if bad_ece else "")))
    out.append(("sweep: no T changes a prediction", 1, "PASS" if not bad_pred else "FAIL", ""))
    out.append(("sweep: T identical across severities", 1, "PASS" if not bad_t else "FAIL",
                str(sw["temperatures"])))
    s0 = [k for k in lg if k.endswith("_s0")]
    same = all(np.array_equal(lg[k], test_logits) for k in s0)
    out.append(("severity 0 = clean logits, bit-identical", 1, "PASS" if same else "FAIL",
                f"{len(s0)} corruptions"))
    out.append(("sweep: float32 scores = float64 softmax and entropy", 1,
                "PASS" if agree < SCORE_TOL else "FAIL", f"max |diff| {agree:.1e}"))
    out.append(("sweep: AURC at every T re-derived", 1, "PASS" if cov_diff < AURC_TOL
                else "FAIL", f"max |diff| {cov_diff:.1e}, {n * len(sw['temperatures']) * 2} "
                "curves (max softmax, entropy)"))
    out += threshold_checks(sw, lg, val_lg)
    out += figure_checks(sw, lg)
    return out


def threshold_checks(sw: dict, lg: dict, val_lg: dict) -> list[Check]:
    """The max-softmax threshold, with separate code: on validation it meets every target
    and the next lower score does not; on test, the accepted counts, accuracies and the
    conditions below target are as reported. Scores are first checked against this module's
    own float64 softmax; the counting is then exact (no tolerance band, which near-tied
    confidences would make meaningless)."""
    keys = [k for k in lg if k != "labels"]
    stores = (("v", val_lg), ("t", lg))
    conf = {s: {k: scores32(st[k])[0] for k in keys} for s, st in stores}
    agree = max(score_agreement(st[k]) for s, st in stores for k in keys)
    ok = {s: {k: (st[k].argmax(1) == st["labels"]) for k in keys} for s, st in stores}

    def meets(split: str, t: float, target: float) -> bool:
        return all(ok[split][k][conf[split][k] >= t].mean() >= target
                   for k in keys if (conf[split][k] >= t).any())

    all_val = np.concatenate(list(conf["v"].values()))
    bad_val, bad_test, n = [], [], 0
    for tgt, r in sw["threshold"]["by_score"]["max_softmax"].items():
        if not r["validation"]["found"]:
            continue
        n += 1
        t, target = r["validation"]["threshold"], float(tgt)
        lower = all_val[all_val < t]
        if not meets("v", t, target) or (lower.size and meets("v", float(lower.max()), target)):
            bad_val.append(tgt)
        below = []
        for k in keys:
            acc_mask = conf["t"][k] >= t
            got = r["test"]["per_condition"][k]
            want_acc = float(ok["t"][k][acc_mask].mean()) if acc_mask.any() else None
            if want_acc is not None and want_acc < target:
                below.append(k)
            if got["n_accepted"] != int(acc_mask.sum()) or (
                    want_acc is not None and abs(got["acc_accepted"] - want_acc) > 1e-12):
                bad_test.append(f"{tgt}/{k}")
        if sorted(below) != sorted(r["test"]["below_target"]):
            bad_test.append(f"{tgt}/below_target")
    return [("threshold: scores agree with float64 softmax", 1,
             "PASS" if agree < SCORE_TOL else "FAIL", f"max |diff| {agree:.1e}"),
            ("threshold: chosen on validation, lowest that meets target", 1,
             "PASS" if not bad_val else "FAIL", f"{n} targets" + (f"; {bad_val}" if bad_val
                                                                 else "")),
            ("threshold: held-out test numbers re-derived", 1,
             "PASS" if not bad_test else "FAIL", f"{n} targets x {len(keys)} conditions"
             + (f"; {bad_test[:5]}" if bad_test else ""))]


def figure_checks(sw: dict, lg: dict) -> list[Check]:
    """Build each figure and read the plotted numbers back out of matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    from src import figures as F
    from src.degrade import CORRUPTIONS
    out: list[Check] = []
    F.style("paper")
    rows = {(r["corruption"], r["severity"]): r for r in sw["rows"]}
    y = lg["labels"]

    def plotted(fig) -> list[np.ndarray]:
        return [np.asarray(ln.get_ydata(), float) for ax in fig.axes for ln in ax.get_lines()]

    def has(series: list[np.ndarray], want: np.ndarray) -> bool:
        return any(s.shape == want.shape and np.allclose(s, want, atol=1e-12) for s in series)

    def check(name: str, fig, wants: list[np.ndarray]) -> None:
        got = plotted(fig)
        miss = sum(not has(got, w) for w in wants)
        out.append((f"figure {name}: plotted = data", 1, "PASS" if miss == 0 else "FAIL",
                    f"{len(wants) - miss}/{len(wants)} series found"))
        plt.close(fig)

    sev = range(6)
    check("accuracy_vs_severity", F.accuracy_vs_severity(sw)("paper"),
          [np.array([rows[(c, s)]["accuracy"] for s in sev]) for c in CORRUPTIONS])
    check("ece_vs_severity", F.ece_vs_severity(sw)("paper"),
          [np.array([rows[(c, s)]["temperature"][k]["ece"] for s in sev])
           for c in CORRUPTIONS for k in ("T=1", "T_lbfgs")])
    check("ece_by_fixed_t", F.ece_by_fixed_t(sw)("paper"),
          [np.array([rows[(c, s)]["temperature"][k]["ece"] for s in sev])
           for c in CORRUPTIONS for k in sw["temperatures"] if k.startswith("T=")])
    check("aurc_by_score", F.aurc_by_score(sw)("paper"),
          [np.array([rows[(c, s)]["scores"][k]["aurc"] for s in sev])
           for c in CORRUPTIONS for k in ("max_softmax", "neg_entropy", "margin")]
          + [np.array([rows[(c, s)]["temperature"]["T_lbfgs"]["coverage"]["max_softmax"]
                       ["aurc"] for s in sev]) for c in CORRUPTIONS])
    check("coverage_curves (vs independent recomputation)", F.coverage_curves(sw, lg)("paper"),
          [selective_acc(scores32(lg[f"{c}_s{s}"])[0],
                         (lg[f"{c}_s{s}"].argmax(1) == y).astype(float))
           for c in CORRUPTIONS for s in sev])
    check("temperature_degeneracy", F.degeneracy(sw)("paper"),
          [np.array(sw["degeneracy"]["val_nll"])])

    fig = F.reliability(sw)("paper")
    clean = rows[(CORRUPTIONS[0], 0)]
    bars = [sorted(round(p.get_height(), 12) for p in ax.patches if isinstance(p, Rectangle))
            for ax in fig.axes]
    want = [sorted(round(b["acc"], 12) for b in clean["temperature"][k]["reliability_bins"]
                   if b["n"]) for k in ("T=1", "T_lbfgs")]
    out.append(("figure reliability_clean: bar heights = bins", 1,
                "PASS" if bars == want else "FAIL", f"{sum(map(len, want))} bars"))
    plt.close(fig)

    fig = F.confidence_hist(sw, lg)("paper")
    ok = True
    for ax, c in zip(fig.axes, CORRUPTIONS, strict=False):
        tops = {round(v, 9) for p in ax.patches if isinstance(p, Polygon)
                for v in p.get_xy()[:, 1]}
        for s in sev:
            conf = softmax64(lg[f"{c}_s{s}"]).max(1)
            h, _ = np.histogram(conf, bins=np.linspace(0, 1, sw["config"]["n_bins"] + 1))
            ok &= {round(v, 9) for v in h[h > 0] / conf.size} <= tops
    out.append(("figure confidence_hist: step heights = histogram", 1,
                "PASS" if ok else "FAIL", "24 histograms"))
    plt.close(fig)
    return out


def tier2(run_path: Path) -> list[Check]:
    res = json.loads(run_path.read_text())
    sep = json.loads(SEPTEMBER.read_text())
    full = (res["config"]["model"] == "resnet50" and res["config"]["split_mode"] == "source"
            and res["audit"]["n_images"] == 9027)
    names = ["split sizes = September (6315/609/610), exact",
             "audit = September audit.json, exact",
             "clean accuracy within 0.01 of 1.0000",
             "clean mean confidence within 0.01 of 0.9222",
             "clean ECE within 0.01 of 0.0778",
             "T_lbfgs vs September LBFGS endpoint 0.237 (information only)"]
    if res["config"].get("train_seed") is not None:
        return [(n, 2, "N/A", "ensemble member: only the seed 42 run is compared with "
                 "September") for n in names]
    if not full:
        return [(n, 2, "PENDING COLAB", "needs a full ResNet-50 source run") for n in names]
    audit = json.loads(SEPTEMBER_AUDIT.read_text())

    def tol(a: float, b: float) -> str:
        return "PASS" if abs(a - b) <= 0.01 else "FAIL"

    return [
        (names[0], 2, "PASS" if res["split_sizes"] == sep["split_sizes"] else "FAIL",
         str(res["split_sizes"])),
        (names[1], 2, "PASS" if res["audit"] == audit else "FAIL", ""),
        (names[2], 2, tol(res["test_accuracy"], sep["test_accuracy"]),
         f"{res['test_accuracy']:.4f}"),
        (names[3], 2, tol(res["mean_confidence"], sep["mean_confidence"]),
         f"{res['mean_confidence']:.4f}"),
        (names[4], 2, tol(res["ece"], sep["ece"]), f"{res['ece']:.4f}"),
        (names[5], 2, "INFO", f"{res['t_lbfgs']['t_lbfgs_endpoint']:.4g} "
         f"(degenerate={res['t_lbfgs']['degenerate']})"),
    ]


def determinism(data_root: str) -> list[Check]:
    """The same config, twice, from scratch: training + sweep must be bit-identical."""
    from src.config import Config
    from src.evaluate import sweep
    from src.train import run
    outs = []
    with tempfile.TemporaryDirectory() as tmp:
        for tag in ("a", "b"):
            cfg = Config(data_root=data_root, model="efficientnet_b0", pretrained=False,
                         batch_size=8, head_epochs=1, finetune_epochs=1, num_workers=0,
                         out_dir=f"{tmp}/{tag}")
            res = run(cfg, stamp="det")
            sweep(Path(f"{tmp}/{tag}/results_{res['run_id']}.json"))
            outs.append({**dict(np.load(f"{tmp}/{tag}/logits_{res['run_id']}.npz")),
                         **{f"sweep/{k}": v for k, v in
                            np.load(f"{tmp}/{tag}/sweep_logits_{res['run_id']}.npz").items()},
                         **{f"sweep_val/{k}": v for k, v in np.load(
                             f"{tmp}/{tag}/sweep_val_logits_{res['run_id']}.npz").items()}})
        same = outs[0].keys() == outs[1].keys() and all(
            np.array_equal(outs[0][k], outs[1][k]) for k in outs[0])
    return [("same config twice: bit-identical logits (train + sweep)", 1,
             "PASS" if same else "FAIL", f"{len(outs[0])} arrays")]


def _entropy64(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-300, None))).sum(-1)


def sample_method_checks(label: str, method: dict, probs: dict, val_probs: dict,
                         test_samples: dict, val_samples: dict, n_bins: int) -> list[Check]:
    """A method that averages S samples (ensemble members, MC dropout passes), re-derived
    from the samples' own logits: the mean softmax, the uncertainty split (D13), every row's
    accuracy, ECE and AURC (max softmax and mutual information), and the threshold."""
    out: list[Check] = []
    mean_diff, unc_diff = 0.0, 0.0
    for store, samples in ((probs, test_samples), (val_probs, val_samples)):
        for key, zs in samples.items():
            p = np.stack([softmax64(z) for z in zs])
            m = p.mean(0)
            mean_diff = max(mean_diff, float(np.abs(m - store[key]).max()))
            if store is probs:
                total, expected = _entropy64(m), _entropy64(p).mean(0)
                own = {"total_entropy": total, "expected_entropy": expected,
                       "mutual_info": total - expected,
                       "variance": ((p - m) ** 2).mean(0).mean(1)}
                for u, v in own.items():
                    unc_diff = max(unc_diff, float(np.abs(v - store[f"{key}__{u}"]).max()))
    n_s = {len(zs) for zs in test_samples.values()}
    out.append((f"{label}: mean softmax re-derived from the samples' logits", 1,
                "PASS" if mean_diff < PROB_TOL else "FAIL",
                f"{n_s.pop() if len(n_s) == 1 else n_s} samples; max |diff| {mean_diff:.1e}"))
    out.append((f"{label}: total entropy, aleatoric part, mutual information, variance", 1,
                "PASS" if unc_diff < PROB_TOL else "FAIL", f"max |diff| {unc_diff:.1e}"))
    y, bad = probs["labels"], []
    for r in method["rows"]:
        key = f"{r['corruption']}_s{r['severity']}"
        p, mi = probs[key], probs[f"{key}__mutual_info"]
        correct = (p.argmax(1) == y).astype(float)
        if correct.mean() != r["accuracy"]:
            bad.append(f"{key}/accuracy")
        if abs(ece_v2(p.max(1), correct, n_bins)[0] - r["temperature"]["T=1"]["ece"]) >= ECE_TOL:
            bad.append(f"{key}/ece")
        if abs(aurc_v2(-mi, correct) - r["scores"]["neg_mutual_info"]["aurc"]) >= AURC_TOL:
            bad.append(f"{key}/aurc_mutual_info")
        if abs(aurc_v2(scores32(np.log(np.clip(p, 1e-300, None)))[0], correct)
               - r["scores"]["max_softmax"]["aurc"]) >= AURC_TOL:
            bad.append(f"{key}/aurc_max_softmax")
    out.append((f"{label}: accuracy, ECE, AURC re-derived, every row", 1,
                "PASS" if not bad else "FAIL", f"{len(method['rows'])} rows"
                + (f"; {bad[:5]}" if bad else "")))

    def logs(st: dict) -> dict:
        return {k: (np.log(np.clip(v, 1e-300, None)) if k != "labels" else v)
                for k, v in st.items() if "__" not in k}
    val_logs = logs(val_probs)
    val_logs["labels"] = val_probs["labels"]
    out += [(f"{label} {name}", t, st, det)
            for name, t, st, det in threshold_checks(method, logs(probs), val_logs)]
    return out


def ensemble_checks(ens_path: Path) -> list[Check]:
    """The deep ensemble, re-derived from each member's saved sweep logits with this
    module's own softmax, ECE, variance and coverage code."""
    from src import figures as F
    ens = json.loads(ens_path.read_text())
    d = ens_path.parent
    res = [json.loads((d / m["results"]).read_text()) for m in ens["members"]]
    out: list[Check] = []
    splits = [(d / r["artefacts"]["splits"]).read_text() for r in res]
    out.append(("ensemble: members share one split (files identical)", 1,
                "PASS" if all(s == splits[0] for s in splits) else "FAIL",
                f"{len(res)} members"))
    seeds = [r["config"]["seed"] if r["config"].get("train_seed") is None
             else r["config"]["train_seed"] for r in res]
    skip = {"train_seed", "out_dir", "data_root", "num_workers"}
    same = all({k: v for k, v in r["config"].items() if k not in skip}
               == {k: v for k, v in res[0]["config"].items() if k not in skip} for r in res)
    out.append(("ensemble: one protocol, distinct training seeds", 1,
                "PASS" if same and len(set(seeds)) == len(seeds) else "FAIL", f"seeds {seeds}"))

    probs = dict(np.load(d / f"ensemble_probs_{ens['ens_id']}.npz"))
    val_probs = dict(np.load(d / f"ensemble_val_probs_{ens['ens_id']}.npz"))
    member = {k: [dict(np.load(d / f"{prefix}_{r['run_id']}.npz")) for r in res]
              for k, prefix in (("test", "sweep_logits"), ("val", "sweep_val_logits"))}
    keys = [k for k in probs if "__" not in k and k != "labels"]
    out += sample_method_checks(
        "ensemble", ens, probs, val_probs,
        {k: [m[k] for m in member["test"]] for k in keys},
        {k: [m[k] for m in member["val"]] for k in keys}, res[0]["config"]["n_bins"])

    import matplotlib.pyplot as plt
    F.style("paper")
    single = json.loads((d / ens["single_model"]["sweep"]).read_text())
    fig = F.ensemble_comparison(single, ens)("paper")
    got = [np.asarray(ln.get_ydata(), float) for ax in fig.axes for ln in ax.get_lines()]
    plt.close(fig)
    from src.degrade import CORRUPTIONS
    wants = []
    for sw in (single, ens):
        for c in CORRUPTIONS:
            rows = sorted((r for r in sw["rows"] if r["corruption"] == c),
                          key=lambda r: r["severity"])
            wants += [np.array([f(r) for r in rows]) for _, f, _ in F.COMPARISON_METRICS]
    miss = sum(not any(g.shape == w.shape and np.allclose(g, w, atol=1e-12) for g in got)
               for w in wants)
    out.append(("figure ensemble_vs_single: plotted = data", 1, "PASS" if not miss else "FAIL",
                f"{len(wants) - miss}/{len(wants)} series found"))
    return out


def mcdropout_checks(mc_path: Path) -> list[Check]:
    """MC dropout, re-derived from the saved logits of every stochastic pass."""
    import matplotlib.pyplot as plt
    import torch

    from src import figures as F
    mc = json.loads(mc_path.read_text())
    d, mid = mc_path.parent, mc["mc_id"]
    probs = dict(np.load(d / f"mcdropout_probs_{mid}.npz"))
    val_probs = dict(np.load(d / f"mcdropout_val_probs_{mid}.npz"))
    tp = dict(np.load(d / f"mcdropout_passes_{mid}.npz"))
    vp = dict(np.load(d / f"mcdropout_val_passes_{mid}.npz"))
    keys = [k for k in probs if "__" not in k and k != "labels"]
    base = json.loads((d / f"results_{mc['base_run_id']}.json").read_text())
    out = sample_method_checks("mc dropout", mc, probs, val_probs,
                               {k: list(tp[k]) for k in keys}, {k: list(vp[k]) for k in keys},
                               base["config"]["n_bins"])
    clean = tp[keys[0]]
    differ = clean.shape[0] == mc["n_passes"] and not all(
        np.array_equal(clean[0], clean[s]) for s in range(1, clean.shape[0]))
    out.append(("mc dropout: passes stored and different (dropout active)", 1,
                "PASS" if differ else "FAIL", f"{clean.shape[0]} passes of {mc['n_passes']}"))
    head = d / mc["training"]["head"]
    if head.exists():
        state = torch.load(head, map_location="cpu", weights_only=True)
        only_head = set(state) == {"1.weight", "1.bias"}
        out.append(("mc dropout: only the head retrained (backbone = seed 42 run)", 1,
                    "PASS" if only_head else "FAIL", f"saved: {sorted(state)}"))
    else:
        out.append(("mc dropout: only the head retrained (backbone = seed 42 run)", 1,
                    "SKIPPED", f"{head.name} not here (checkpoints stay in Drive)"))

    F.style("paper")
    single = json.loads((d / mc["single_model"]["sweep"]).read_text())
    ens = F.ensemble_for(mc["base_run_id"], d)
    fig = F.three_methods(single, ens, mc)("paper")
    got = [np.asarray(ln.get_ydata(), float) for ax in fig.axes for ln in ax.get_lines()]
    plt.close(fig)
    from src.degrade import CORRUPTIONS
    wants = []
    for sw in [single] + ([ens] if ens else []) + [mc]:
        for c in CORRUPTIONS:
            rows = sorted((r for r in sw["rows"] if r["corruption"] == c),
                          key=lambda r: r["severity"])
            wants += [np.array([f(r) for r in rows]) for _, f, _ in F.COMPARISON_METRICS]
    miss = sum(not any(g.shape == w.shape and np.allclose(g, w, atol=1e-12) for g in got)
               for w in wants)
    out.append(("figure methods_comparison: plotted = data", 1, "PASS" if not miss else "FAIL",
                f"{len(wants) - miss}/{len(wants)} series"
                f"{' (with the ensemble)' if ens else ' (no ensemble run)'}"))
    return out


def _has_accuracy(obj: object) -> bool:
    if isinstance(obj, dict):
        return any(bool(re.search("accura|correct", k)) or _has_accuracy(v)
                   for k, v in obj.items())
    if isinstance(obj, list):
        return any(_has_accuracy(v) for v in obj)
    return False


def _wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (0.0 if k == 0 else centre - half), (1.0 if k == n else centre + half)


def plantseg_checks(ps_path: Path, filter_csv: Path | None = None) -> list[Check]:
    """PlantSeg phase B2, from the saved logits and the committed leaf filter, with this
    module's own selection, softmax, Wilson interval and ECE. The out-of-distribution
    sections must carry no accuracy of any kind (D16)."""
    import csv

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from src import figures as F
    ps = json.loads(ps_path.read_text())
    d = ps_path.parent
    lg = dict(np.load(d / f"plantseg_logits_{ps['run_id']}.npz"))
    sw_lg = dict(np.load(d / f"sweep_logits_{ps['run_id']}.npz"))
    sw = json.loads((d / f"sweep_{ps['run_id']}.json").read_text())
    filter_csv = filter_csv or REPO / "results" / "plantseg" / "leaf_filter.csv"
    rows = list(csv.DictReader(open(filter_csv)))
    names = {"black_rot": "grape black rot", "downy_mildew": "grape downy mildew",
             "leaf_spot": "grape leaf spot", "leafroll": "grapevine leafroll disease"}
    want = {g: [r["id"] for r in rows if r["disease"] == n and r["tag"] == "leaf"
                and r["duplicate_of"] == "" and r["label_conflict_with"] == ""]
            for g, n in names.items()}
    got = {"black_rot": ps["black_rot"]["ids"],
           **{g: ps["out_of_distribution"][g]["ids"] for g in names if g != "black_rot"}}
    npz_ids = {g: [str(x) for x in lg[f"{g}__ids"]] for g in names}
    out: list[Check] = [("plantseg: images = leaf filter, each photograph once", 1,
                         "PASS" if want == got == npz_ids else "FAIL",
                         ", ".join(f"{g} {len(v)}" for g, v in want.items()))]
    out.append(("plantseg: out-of-distribution sections carry no accuracy", 1,
                "FAIL" if _has_accuracy(ps["out_of_distribution"]) else "PASS",
                "downy mildew, leaf spot, leafroll"))

    target = ps["classes"].index("Black_rot")
    unusual = re.compile("close-up|wide view|shoot", re.I)
    reason = {r["id"]: r["reason"] for r in rows}
    bad = []
    usual = [j for j, i in enumerate(want["black_rot"]) if not unusual.search(reason[i])]
    for key, sel in (("black_rot", list(range(len(want["black_rot"])))),
                     ("black_rot_usual_framing", usual)):
        a = ps[key]
        p = softmax64(lg["black_rot"][sel])
        correct = (p.argmax(1) == target).astype(float)
        n, k = correct.size, int(correct.sum())
        lo, hi = _wilson(k, n)
        if a["ids"] != [want["black_rot"][j] for j in sel]:
            bad.append(f"{key}/ids")
        if (a["n"], a["n_correct"]) != (n, k) or abs(a["accuracy"] - k / n) > 1e-12:
            bad.append(f"{key}/accuracy")
        if max(abs(lo - a["accuracy_wilson95"][0]), abs(hi - a["accuracy_wilson95"][1])) > 1e-9:
            bad.append(f"{key}/wilson")
        if abs(p.max(1).mean() - a["mean_confidence"]) > PROB_TOL:
            bad.append(f"{key}/mean_confidence")
        if "ece" in a:  # D17: not estimable at this n
            bad.append(f"{key}/ece reported")
        own = ece_v2(scores32(lg["black_rot"][sel])[0].astype(np.float64), correct, 15)[1]
        if a["bin_occupancy"]["counts"] != [int(c) for c in own]:
            bad.append(f"{key}/bin_occupancy")
    out.append(("plantseg: black rot accuracy, Wilson, confidence, bin occupancy (no ECE)", 1,
                "PASS" if not bad else "FAIL", f"n = {ps['black_rot']['n']}, sensitivity "
                f"n = {ps['black_rot_usual_framing']['n']}" + (f"; {bad}" if bad else "")))

    t = ps["threshold"]
    groups = {"black_rot": (lg["black_rot"], ps["black_rot"]),
              **{g: (lg[g], ps["out_of_distribution"][g]) for g in names if g != "black_rot"}}
    worst = min((r for r in sw["rows"] if r["severity"] == 5),
                key=lambda r: float((sw_lg[f"{r['corruption']}_s5"].argmax(1)
                                     == sw_lg["labels"]).mean()))
    refs = {"plantvillage_clean": "gaussian_blur_s0",
            "plantvillage_most_severe": f"{worst['corruption']}_s5"}
    bad = [ref for ref, cond in refs.items() if ps["references"][ref]["condition"] != cond]
    for ref, cond in refs.items():
        groups[ref] = (sw_lg[cond], ps["references"][ref])
    for g, (z, a) in groups.items():
        conf = scores32(z)[0]
        pred = softmax64(z).argmax(1)
        c = a["confidence"]
        if c["n"] != conf.size or abs(c["mean"] - float(conf.mean())) > PROB_TOL \
                or abs(c["median"] - float(np.median(conf))) > PROB_TOL:
            bad.append(f"{g}/confidence")
        if t is not None and abs(c["share_without_warning"] - float((conf >= t).mean())) > 1e-12:
            bad.append(f"{g}/share_without_warning")
        if "predicted_class" in a and a["predicted_class"] != {
                cl: int((pred == i).sum()) for i, cl in enumerate(ps["classes"])}:
            bad.append(f"{g}/predicted_class")
    out.append(("plantseg: confidence distributions, classes assigned, references", 1,
                "PASS" if not bad else "FAIL",
                f"{len(groups)} groups; most severe = {refs['plantvillage_most_severe']}"
                + (f"; {bad}" if bad else "")))

    F.style("paper")
    confs = {g: scores32(z)[0] for g, (z, _) in groups.items()}
    fig = F.plantseg_confidence(ps, confs)("paper")
    lines = [np.asarray(ln.get_ydata(), float) for ax in fig.axes for ln in ax.get_lines()]
    wants = [F.survival(confs[g])[1] for _, series in F.PLANTSEG_PANELS for g, *_ in series]
    miss = sum(not any(x.shape == w.shape and np.allclose(x, w) for x in lines) for w in wants)
    plt.close(fig)
    fig = F.plantseg_predicted_class(ps)("paper")
    widths = sorted(round(r.get_width(), 9) for ax in fig.axes for r in ax.patches
                    if isinstance(r, Rectangle) and r.get_width() > 0)
    shares = sorted(round(a["predicted_class"][cl] / a["n"], 9)
                    for g, (_, a) in groups.items() if g in names
                    for cl in ps["classes"] if a["predicted_class"][cl] > 0)
    plt.close(fig)
    out.append(("figures plantseg: plotted = data", 1,
                "PASS" if miss == 0 and widths == shares else "FAIL",
                f"{len(wants) - miss}/{len(wants)} curves, {len(widths)} bar segments"))
    return out


GENERATOR = REPO / "tools" / "make_tables.py"
HEADER = re.compile(r"% (generator-sha256|source|pending|excluded|body-sha256): (.*)")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generated_checks(gen_dir: Path) -> list[Check]:
    """Each file tools/make_tables.py wrote must still match what it was built from: every
    source JSON unchanged, the generator unchanged, the body not edited by hand, and no
    pending source (the Colab sweep) that now exists."""
    files = sorted(Path(gen_dir).glob("*.tex"))
    if not files:
        return [("generated report files present", 1, "FAIL", f"none in {gen_dir}")]
    out: list[Check] = []
    for f in files:
        lines = f.read_text().splitlines(keepends=True)
        head = {}
        sources, pending, n_head = [], [], 0
        for line in lines:
            m = HEADER.match(line)
            if not (line.startswith("% ") and (m or n_head == 0)):
                break
            n_head += 1
            if m and m.group(1) == "source":
                path, _, h = m.group(2).rpartition(" sha256=")
                sources.append((path, h.strip()))
            elif m and m.group(1) == "pending":
                pending.append(m.group(2).partition(" glob=")[2].strip())
            elif m:
                head[m.group(1)] = m.group(2).strip()
        problems = []
        if "body-sha256" not in head or "generator-sha256" not in head:
            problems.append("no generator header: not written by tools/make_tables.py")
        else:
            if _sha("".join(lines[n_head:]).encode()) != head["body-sha256"]:
                problems.append("edited by hand")
            if _sha(GENERATOR.read_bytes()) != head["generator-sha256"]:
                problems.append("tools/make_tables.py changed since")
        for path, h in sources:
            p = Path(path) if Path(path).is_absolute() else REPO / path
            if not p.exists():
                problems.append(f"source missing: {path}")
            elif _sha(p.read_bytes()) != h:
                problems.append(f"source changed: {path}")
        for g in pending:
            if sorted(REPO.glob(g)):
                problems.append(f"pending source now exists: {g}")
        out.append((f"generated {f.name} is current", 1, "FAIL" if problems else "PASS",
                    "; ".join(problems) if problems else
                    f"{len(sources)} source(s)" + (f", {len(pending)} pending" if pending
                                                    else "")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="results_<run_id>.json")
    ap.add_argument("--ensemble", metavar="JSON", help="ensemble_<ens_id>.json to verify")
    ap.add_argument("--plantseg", metavar="JSON", help="plantseg_<run_id>.json to verify")
    ap.add_argument("--mcdropout", metavar="JSON", help="mcdropout_<mc_id>.json to verify")
    ap.add_argument("--generated", metavar="DIR",
                    help="check that the report's generated numbers and tables are current "
                         "(re-run tools/make_tables.py if not)")
    ap.add_argument("--require-sweep", action="store_true",
                    help="fail if the run has no sweep yet (notebook 02 sets this)")
    ap.add_argument("--determinism", metavar="DATA_ROOT",
                    help="also train + sweep twice on DATA_ROOT (the fixture) and compare")
    a = ap.parse_args()
    if not (a.run or a.generated or a.ensemble or a.plantseg or a.mcdropout):
        ap.error("give at least one of --run, --ensemble, --plantseg, --mcdropout, "
                 "--generated")
    checks: list[Check] = []
    if a.run:
        run_path = Path(a.run)
        checks += tier1(run_path, a.require_sweep)
        if a.determinism:
            checks += determinism(a.determinism)
        checks += tier2(run_path)
    if a.ensemble:
        checks += ensemble_checks(Path(a.ensemble))
    if a.plantseg:
        checks += plantseg_checks(Path(a.plantseg))
    if a.mcdropout:
        checks += mcdropout_checks(Path(a.mcdropout))
    if a.generated:
        checks += generated_checks(Path(a.generated))
    w = max(len(c[0]) for c in checks)
    print(f"\n{'check':<{w}}  tier  status         detail")
    for name, tier, status, detail in checks:
        print(f"{name:<{w}}  {tier:^4}  {status:<13}  {detail}")
    fails = [c for c in checks if c[1] == 1 and c[2] == "FAIL"]
    passed = sum(c[1] == 1 and c[2] == "PASS" for c in checks)
    skipped = sum(c[1] == 1 and c[2] == "SKIPPED" for c in checks)
    print(f"\nTier 1: {passed} passed, {len(fails)} failed, {skipped} skipped. "
          f"Tier 2: {sum(c[1] == 2 for c in checks)} checks, "
          f"{sum(c[2] == 'PENDING COLAB' for c in checks)} pending Colab.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
