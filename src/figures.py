"""All plotting. Every figure is rendered twice: ACM single-column (3.3 in wide, vector
PDF) for the paper and 16:9 (300 dpi PNG) for slides.

    python -m src.figures grid --data-root tests/fixture --out figures
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib

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


def render(build: Callable[[str], Figure], name: str, out: str | Path) -> list[Path]:
    """Build the figure at both sizes and save paper PDF + slide PNG. Returns the paths."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for size, ext, dpi in (("paper", "pdf", None), ("slide", "png", 300)):
        style(size)
        fig = build(size)
        p = out / f"{name}_{size}.{ext}"
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("figure", choices=["grid"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--leaf", help="filename of the leaf to show (default: first original)")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    if a.figure == "grid":
        for p in render(corruption_grid(a.data_root, a.leaf), "corruption_grid", a.out):
            print("wrote", p)


if __name__ == "__main__":
    main()
