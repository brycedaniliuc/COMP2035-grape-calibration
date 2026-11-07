"""Independent re-derivation of every published number, from saved logits and filenames.

    python -m src.verify --run <run_dir>/results_<run_id>.json [--determinism]

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
    if sw_path.exists():
        out += tier1_sweep(res, json.loads(sw_path.read_text()),
                           dict(np.load(d / f"sweep_logits_{res['run_id']}.npz")), tl)
    elif require_sweep:
        out.append(("sweep present", 1, "FAIL", f"missing {sw_path.name}"))
    else:  # notebook 01 verifies before notebook 02 has produced the sweep
        out.append(("sweep and figure checks", 1, "SKIPPED",
                    "no sweep yet (run src.evaluate; use --require-sweep to demand it)"))
    return out


def tier1_sweep(res: dict, sw: dict, lg: dict, test_logits: np.ndarray) -> list[Check]:
    out: list[Check] = []
    nb = sw["config"]["n_bins"]
    y = lg["labels"]
    bad_acc, bad_ece, bad_pred, bad_t = [], [], [], []
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
    out += figure_checks(sw, lg)
    return out


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
           for c in CORRUPTIONS for k in ("max_softmax", "neg_entropy", "margin")])
    check("coverage_curves (vs independent recomputation)", F.coverage_curves(sw, lg)("paper"),
          [selective_acc(softmax64(lg[f"{c}_s{s}"]).max(1).astype(np.float32),
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
                            np.load(f"{tmp}/{tag}/sweep_logits_{res['run_id']}.npz").items()}})
        same = outs[0].keys() == outs[1].keys() and all(
            np.array_equal(outs[0][k], outs[1][k]) for k in outs[0])
    return [("same config twice: bit-identical logits (train + sweep)", 1,
             "PASS" if same else "FAIL", f"{len(outs[0])} arrays")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results_<run_id>.json")
    ap.add_argument("--require-sweep", action="store_true",
                    help="fail if the run has no sweep yet (notebook 02 sets this)")
    ap.add_argument("--determinism", metavar="DATA_ROOT",
                    help="also train + sweep twice on DATA_ROOT (the fixture) and compare")
    a = ap.parse_args()
    run_path = Path(a.run)
    checks = tier1(run_path, a.require_sweep)
    if a.determinism:
        checks += determinism(a.determinism)
    checks += tier2(run_path)
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
