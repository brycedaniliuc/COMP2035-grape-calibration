"""All plotting. Every figure is rendered twice: ACM single-column (3.3 in wide, vector
PDF) for the paper and 16:9 (300 dpi PNG) for slides.

    python -m src.figures grid --data-root tests/fixture --out figures
    python -m src.figures sweep --sweep <run_dir>/sweep_<run_id>.json --out figures
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

BLUE, RED, INK, MUTED, GRID = "#2a78d6", "#e34948", "#0b0b0b", "#52514e", "#d8d7d2"

SIZES = {"paper": (3.3, None), "slide": (13.333, 7.5)}
FONT = {"paper": 7.0, "slide": 16.0}


def style(size: str) -> None:
    f = FONT[size]
    matplotlib.rcParams.update({
        "font.size": f, "axes.labelsize": f, "axes.titlesize": f + 0.5,
        "xtick.labelsize": f - 1, "ytick.labelsize": f - 1, "legend.fontsize": f - 1,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
        "lines.linewidth": 1.2 if size == "paper" else 2.5, "pdf.fonttype": 42,
    })


def render(build: Callable[[str], Figure], name: str, out: str | Path, tag: str = ""
           ) -> list[Path]:
    """Build the figure at both sizes and save paper PDF + slide PNG. Returns the paths.
    `tag` (e.g. the run id) is appended to every filename so runs never overwrite."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for size, ext, dpi in (("paper", "pdf", None), ("slide", "png", 300)):
        style(size)
        fig = build(size)
        p = out / f"{name}_{size}{'_' + tag if tag else ''}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=dpi if dpi is not None else "figure")
        plt.close(fig)
        paths.append(p)
    return paths


def _figsize(size: str, aspect: float) -> tuple[float, float]:
    """aspect = height / width for the paper size; slides are always 16:9."""
    w, h = SIZES[size]
    return (w, h) if h is not None else (w, w * aspect)


# ---- corruption grid ---------------------------------------------------------------

def corruption_grid(data_root: str, filename: str | None = None) -> Callable[[str], Figure]:
    """One real leaf, every corruption at every severity."""
    from src.config import Config
    from src.data import index_dataset
    from src.degrade import CORRUPTIONS, SEVERITIES, DegradedDataset, severity_parameter

    recs = [r for r in index_dataset(data_root) if not r.aug]
    rec = next(r for r in recs if filename is None or r.path.endswith(filename))
    labels = {"gaussian_blur": "Blur", "brightness": "Brighten",
              "darkness": "Darken", "occlusion": "Occlude"}
    units = {"gaussian_blur": lambda p: f"σ {p:g} px",
             "brightness": lambda p: f"V +{p:g}", "darkness": lambda p: f"V −{p:g}",
             "occlusion": lambda p: f"{p:.0%}"}
    imgs = {c: [DegradedDataset([rec], {rec.cls: 0}, Config(), c, s).pixels(0)
                for s in SEVERITIES] for c in CORRUPTIONS}

    def build(size: str) -> Figure:
        fig, axes = plt.subplots(len(CORRUPTIONS), len(SEVERITIES),
                                 figsize=_figsize(size, 0.80),
                                 gridspec_kw={"wspace": 0.04, "hspace": 0.32})
        for i, c in enumerate(CORRUPTIONS):
            for j, s in enumerate(SEVERITIES):
                ax = axes[i, j]
                ax.imshow(imgs[c][j])
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                p = severity_parameter(c, s)
                ax.set_title("clean" if s == 0 else units[c](p),
                             fontsize=FONT[size] - (2 if size == "paper" else 3), pad=1.5)
                if j == 0:
                    ax.set_ylabel(labels[c], fontsize=FONT[size] - 0.5)
                if i == len(CORRUPTIONS) - 1:
                    ax.set_xlabel(str(s), fontsize=FONT[size] - 0.5)
        fig.supxlabel("Severity level", fontsize=FONT[size],
                      y=0.02 if size == "paper" else 0.03)
        fig.suptitle(f"Corruption protocol (n = 1 leaf, {rec.cls.replace('_', ' ')})",
                     y=0.98 if size == "paper" else 0.97)
        return fig

    return build


# ---- sweep figures ---------------------------------------------------------------------
# Inputs: the sweep JSON and sweep logits written by src.evaluate. Every number plotted
# comes from those two files; nothing is recomputed from images here.

LABEL = {"gaussian_blur": "Gaussian blur", "brightness": "Brighten",
         "darkness": "Darken", "occlusion": "Occlusion"}
STYLE = {  # identity never by colour alone: colour + line style + marker
    "gaussian_blur": (BLUE, "-", "o"), "brightness": (RED, "--", "s"),
    "darkness": (INK, ":", "^"), "occlusion": (MUTED, "-.", "D")}


def _ramp(frac: float) -> str:
    """One-hue ramp for ordered series: frac 0 = light grey, 1 = INK."""
    lo, hi = np.array([0xb0, 0xaf, 0xaa]), np.array([0x0b, 0x0b, 0x0b])
    return "#" + "".join(f"{int(round(v)):02x}" for v in lo + (hi - lo) * frac)


def _sev_colour(s: int) -> str:
    """Severity colour: clean is BLUE; 1..5 run light grey -> INK."""
    return BLUE if s == 0 else _ramp((s - 1) / 4)


def _rows(sweep: dict, corruption: str) -> list[dict]:
    return sorted((r for r in sweep["rows"] if r["corruption"] == corruption),
                  key=lambda r: r["severity"])


def _panels(size: str, n: int = 4, aspect: float = 0.85, sharey: bool = True
            ) -> tuple[Figure, list]:
    shape = (2, 2) if size == "paper" else (1, n)
    fig, axes = plt.subplots(*shape, figsize=_figsize(size, aspect), sharex=True,
                             sharey=sharey, constrained_layout=True)
    return fig, list(np.ravel(axes))


def _t_lbfgs_name(sweep: dict) -> str:
    t = sweep["temperatures"]["T_lbfgs"]
    return (f"T = {t:.3g}, LBFGS endpoint (as-fitted, degenerate)" if sweep["t_lbfgs"]["degenerate"]
            else f"T = {t:.3g}, fitted on clean validation")


def _finish(fig: Figure, axes: list, size: str, xlabel: str, ylabel: str, ncols: int) -> None:
    """Shared axis labels at body size, and one legend anchored just below the figure
    (savefig bbox_inches="tight" grows the canvas to hold it), so it can collide with
    neither the suptitle nor the shared x label."""
    fig.supxlabel(xlabel, fontsize=FONT[size])
    fig.supylabel(ylabel, fontsize=FONT[size])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=ncols, frameon=False)


def accuracy_vs_severity(sweep: dict) -> Callable[[str], Figure]:
    from src.degrade import CORRUPTIONS

    def build(size: str) -> Figure:
        fig, ax = plt.subplots(figsize=_figsize(size, 0.72), constrained_layout=True)
        for c in CORRUPTIONS:
            rs = _rows(sweep, c)
            x = np.array([r["severity"] for r in rs], float)
            y = np.array([r["accuracy"] for r in rs])
            lo = y - np.array([r["accuracy_wilson95"][0] for r in rs])
            hi = np.array([r["accuracy_wilson95"][1] for r in rs]) - y
            col, ls, mk = STYLE[c]
            off = (CORRUPTIONS.index(c) - 1.5) * 0.06  # dodge so CIs do not overlap
            ax.errorbar(x + off, y, yerr=[lo, hi], color=col, ls=ls, marker=mk,
                        ms=3 if size == "paper" else 7, capsize=2, label=LABEL[c])
        ax.set(xlabel="Severity level (0 = clean)", ylabel="Test accuracy (fraction)",
               xticks=range(6))
        ax.grid(axis="y")
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(0.5, 0.0), ncols=4, frameon=False)
        fig.suptitle("Accuracy under corruption, Wilson 95% CI "
                     f"(n = {sweep['n_test']} test images)")
        return fig

    return build


def ece_vs_severity(sweep: dict) -> Callable[[str], Figure]:
    from src.degrade import CORRUPTIONS
    lb = _t_lbfgs_name(sweep)

    def build(size: str) -> Figure:
        fig, axes = _panels(size)
        for ax, c in zip(axes, CORRUPTIONS, strict=True):
            rs = _rows(sweep, c)
            x = [r["severity"] for r in rs]
            for key, col, ls, mk, lab in (("T=1", BLUE, "-", "o", "T = 1 (no temperature)"),
                                          ("T_lbfgs", RED, "--", "s", lb)):
                y = [r["temperature"][key]["ece"] for r in rs]
                lo = [r["temperature"][key]["ece_bootstrap95"][0] for r in rs]
                hi = [r["temperature"][key]["ece_bootstrap95"][1] for r in rs]
                ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
                ax.plot(x, y, color=col, ls=ls, marker=mk, ms=3 if size == "paper" else 7,
                        label=lab)
            ax.set_title(LABEL[c])
            ax.set_xticks(range(6))
            ax.grid(axis="y")
        _finish(fig, axes, size, "Severity level (0 = clean)",
                "ECE, 15 bins (bands: bootstrap 95% CI)", 1 if size == "paper" else 2)
        fig.suptitle(f"Calibration error under corruption (n = {sweep['n_test']} test images)")
        return fig

    return build


def _selective_accuracy(score: np.ndarray, correct: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="stable")
    k = np.arange(1, score.size + 1)
    return k / score.size, np.cumsum(correct[order]) / k


def coverage_curves(sweep: dict, logits: dict) -> Callable[[str], Figure]:
    from src.degrade import CORRUPTIONS, SEVERITIES
    from src.evaluate import scores
    y = logits["labels"]

    def build(size: str) -> Figure:
        fig, axes = _panels(size)
        for ax, c in zip(axes, CORRUPTIONS, strict=True):
            for s in SEVERITIES:
                lg = logits[f"{c}_s{s}"]
                cov, acc = _selective_accuracy(scores(lg)["max_softmax"],
                                               (lg.argmax(1) == y).astype(float))
                ax.plot(cov, acc, color=_sev_colour(s), lw=1.0 if size == "paper" else 2.2,
                        label="clean" if s == 0 else f"severity {s}")
            ax.set_title(LABEL[c])
            ax.set_xlim(0, 1)
            ax.grid(axis="y")
        _finish(fig, axes, size, "Coverage (fraction accepted, most confident first)",
                "Accuracy on accepted (fraction)", 3 if size == "paper" else 6)
        fig.suptitle(f"Coverage curves, max softmax at T = 1 (n = {sweep['n_test']} test images)")
        return fig

    return build


def confidence_hist(sweep: dict, logits: dict) -> Callable[[str], Figure]:
    from src.degrade import CORRUPTIONS, SEVERITIES
    from src.evaluate import scores
    bins = np.linspace(0, 1, sweep["config"]["n_bins"] + 1)

    def build(size: str) -> Figure:
        fig, axes = _panels(size)
        for ax, c in zip(axes, CORRUPTIONS, strict=True):
            for s in SEVERITIES:
                conf = scores(logits[f"{c}_s{s}"])["max_softmax"]
                w = np.full(conf.size, 1 / conf.size)
                ax.hist(conf, bins=bins, weights=w, histtype="step", color=_sev_colour(s),
                        lw=1.0 if size == "paper" else 2.2,
                        label="clean" if s == 0 else f"severity {s}")
            ax.set_title(LABEL[c])
            ax.set_xlim(0, 1)
            ax.grid(axis="y")
        _finish(fig, axes, size, "Confidence (max softmax, T = 1; 15 equal-width bins)",
                "Fraction of test images", 3 if size == "paper" else 6)
        fig.suptitle(f"Confidence distribution by severity (n = {sweep['n_test']} test images)")
        return fig

    return build


def reliability(sweep: dict) -> Callable[[str], Figure]:
    """Clean test set: T = 1 against the LBFGS endpoint T."""
    clean = next(r for r in sweep["rows"] if r["severity"] == 0)
    panels = (("T=1", "T = 1 (no temperature)"), ("T_lbfgs", _t_lbfgs_name(sweep)))

    def build(size: str) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=_figsize(size, 0.58), sharey=True,
                                 constrained_layout=True)
        for ax, (key, title) in zip(axes, panels, strict=True):
            cond = clean["temperature"][key]
            b = [r for r in cond["reliability_bins"] if r["n"]]
            centre = [(r["lo"] + r["hi"]) / 2 for r in b]
            width = (b[0]["hi"] - b[0]["lo"]) * 0.9 if b else 0.06
            ax.bar(centre, [r["acc"] for r in b], width=width, color=BLUE, label="Accuracy")
            ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="Perfect calibration")
            for r, xc in zip(b, centre, strict=True):
                ax.annotate(f"{r['n']}", (xc, r["acc"]), ha="center", va="bottom",
                            fontsize=FONT[size] - 2, color=MUTED, xytext=(0, 1),
                            textcoords="offset points")
            ax.set(xlim=(0, 1), ylim=(0, 1.08), xlabel="Confidence (max softmax)")
            ax.set_title(f"{title}\nECE = {cond['ece']:.3f}", fontsize=FONT[size] - 1)
            ax.grid(axis="y")
        axes[0].set_ylabel("Accuracy in bin (fraction)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(0.5, 0.0), ncols=2, frameon=False)
        fig.suptitle(f"Reliability on clean test images (n = {sweep['n_test']}; "
                     "numbers = images per bin)")
        return fig

    return build


def degeneracy(sweep: dict) -> Callable[[str], Figure]:
    d = sweep["degeneracy"]
    n_val = sweep["t_lbfgs"]["n_val"]

    def build(size: str) -> Figure:
        fig, ax = plt.subplots(figsize=_figsize(size, 0.68), constrained_layout=True)
        ax.plot(d["T_grid"], d["val_nll"], color=BLUE, label="Validation NLL")
        ends, end_nll = d["lbfgs_endpoint_by_max_iter"], d["lbfgs_endpoint_nll_by_max_iter"]
        groups: dict[float, list[str]] = {}
        for m, t in ends.items():  # max_iter values that stopped at the same T share a label
            groups.setdefault(round(float(np.log10(t)), 2), []).append(m)  # ~2% apart
        step = 9 if size == "paper" else 18  # endpoints can sit close together: stagger the
        for i, ms in enumerate(sorted(groups.values(), key=lambda g: ends[g[0]])):  # labels
            t, y = ends[ms[0]], end_nll[ms[0]]
            ax.plot(t, y, marker="s", color=RED, ms=3 if size == "paper" else 7, ls="none")
            ax.annotate(", ".join(ms), (t, y), xytext=(0, step * (i + 1)),
                        textcoords="offset points", ha="center", fontsize=FONT[size] - 2,
                        color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6},
                        bbox={"fc": "white", "ec": "none", "pad": 0.3})
        ax.plot([], [], marker="s", color=RED, ls="none",
                label="LBFGS endpoint (label = max_iter)")
        ax.set_xscale("log")
        ax.set(xlabel="Temperature T (log scale)", ylabel="Validation NLL (nats per image)")
        verdict = ("no minimum: NLL falls as T → 0" if d["minimum_at_lower_edge"]
                   else f"minimum at T ≈ {d['argmin_T']:.2g}")
        ax.set_title(f"Temperature fit on clean validation (n = {n_val}, "
                     f"{d['n_val_errors']} errors): {verdict}", fontsize=FONT[size] - 0.5)
        ax.grid(axis="y")
        ax.legend(frameon=False)
        return fig

    return build


def aurc_by_score(sweep: dict) -> Callable[[str], Figure]:
    from src.degrade import CORRUPTIONS
    sty = {"max_softmax": (BLUE, "-", "o", "Max softmax, T = 1"),
           "neg_entropy": (RED, "--", "s", "Entropy, T = 1"),
           "margin": (INK, ":", "^", "Logit margin")}

    def build(size: str) -> Figure:
        fig, axes = _panels(size)
        for ax, c in zip(axes, CORRUPTIONS, strict=True):
            rs = _rows(sweep, c)
            for k, (col, ls, mk, lab) in sty.items():
                ax.plot([r["severity"] for r in rs], [r["scores"][k]["aurc"] for r in rs],
                        color=col, ls=ls, marker=mk, ms=3 if size == "paper" else 7, label=lab)
            ax.plot([r["severity"] for r in rs],  # the temperature-scaled method (paper §3.2)
                    [r["temperature"]["T_lbfgs"]["coverage"]["max_softmax"]["aurc"] for r in rs],
                    color=BLUE, ls="-.", marker="o", mfc="white",
                    ms=3 if size == "paper" else 7, label=f"Max softmax, {_t_lbfgs_name(sweep)}")
            ax.set_title(LABEL[c])
            ax.set_xticks(range(6))
            ax.grid(axis="y")
        _finish(fig, axes, size, "Severity level (0 = clean)", "AURC (lower is better)", 2)
        fig.suptitle(f"Ranking quality of the confidence scores (n = {sweep['n_test']})")
        return fig

    return build


def ece_by_fixed_t(sweep: dict) -> Callable[[str], Figure]:
    """Amended protocol, condition 3: every T in the fixed grid, chosen in advance and held
    fixed across all severities. Does any single clean-data temperature help under shift?"""
    from src.degrade import CORRUPTIONS
    grid = [k for k in sweep["temperatures"] if k.startswith("T=")]
    ts = [sweep["temperatures"][k] for k in grid]
    lo, hi = np.log(min(ts)), np.log(max(ts))

    def colour(t: float) -> str:  # one ramp, light (small T) -> dark (large T); T = 1 in BLUE
        if t == 1.0:
            return BLUE
        return _ramp(float((np.log(t) - lo) / (hi - lo)))

    def build(size: str) -> Figure:
        fig, axes = _panels(size)
        for ax, c in zip(axes, CORRUPTIONS, strict=True):
            rs = _rows(sweep, c)
            for k, t in zip(grid, ts, strict=True):
                ax.plot([r["severity"] for r in rs], [r["temperature"][k]["ece"] for r in rs],
                        color=colour(t), ls="-" if t == 1.0 else "--",
                        marker="o" if t == 1.0 else ".", ms=3 if size == "paper" else 7,
                        lw=1.4 if t == 1.0 else 1.0, label=f"T = {t:g}")
            ax.set_title(LABEL[c])
            ax.set_xticks(range(6))
            ax.grid(axis="y")
        _finish(fig, axes, size, "Severity level (0 = clean)", "ECE, 15 bins",
                3 if size == "paper" else 5)
        fig.suptitle(f"ECE for each fixed temperature, held across severities "
                     f"(n = {sweep['n_test']})")
        return fig

    return build


SWEEP_FIGURES = {
    "accuracy_vs_severity": lambda s, lg: accuracy_vs_severity(s),
    "ece_vs_severity": lambda s, lg: ece_vs_severity(s),
    "ece_by_fixed_t": lambda s, lg: ece_by_fixed_t(s),
    "coverage_curves": coverage_curves,
    "confidence_hist": confidence_hist,
    "reliability_clean": lambda s, lg: reliability(s),
    "temperature_degeneracy": lambda s, lg: degeneracy(s),
    "aurc_by_score": lambda s, lg: aurc_by_score(s),
}


def render_sweep(sweep_path: str | Path, out: str | Path, tag: str = "") -> list[Path]:
    sweep_path = Path(sweep_path)
    sweep = json.loads(sweep_path.read_text())
    lg = dict(np.load(sweep_path.parent / sweep_path.name.replace("sweep_", "sweep_logits_")
                      .replace(".json", ".npz")))
    paths = []
    for name, make in SWEEP_FIGURES.items():
        paths += render(make(sweep, lg), name, out, tag)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("figure", choices=["grid", "sweep"])
    ap.add_argument("--data-root", help="grid: dataset root")
    ap.add_argument("--leaf", help="grid: filename of the leaf to show")
    ap.add_argument("--sweep", help="sweep: path to sweep_<run_id>.json")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--tag", default="", help="appended to every filename, e.g. the run id")
    a = ap.parse_args()
    if a.figure == "grid":
        assert a.data_root, "--data-root is required for the grid"
        paths = render(corruption_grid(a.data_root, a.leaf), "corruption_grid", a.out, a.tag)
    else:
        assert a.sweep, "--sweep is required"
        paths = render_sweep(a.sweep, a.out, a.tag)
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
