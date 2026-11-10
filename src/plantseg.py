"""PlantSeg field photographs (upgrade 2, phase B2; DECISIONS.md D14, D15, D16).

    python -m src.plantseg --results <run>/results_<run_id>.json --zip Plantseg.zip

The images are the leaves selected in `results/plantseg/leaf_filter.csv` (D15), each
photograph counted once: tag "leaf", no `duplicate_of`, no `label_conflict_with`. Two
analyses, never merged:

A. **Black rot leaves.** The one PlantSeg class that is also one of the model's four
   (Black_rot). Accuracy with a Wilson interval, mean confidence and the confidence
   distribution. n is small and goes with every number. No ECE: at n = 25 over 15 bins it is
   not estimable, and the bin occupancy is reported to show why (D17). A sensitivity check
   repeats it without the close-ups, wide views and shoots (D15).
B. **Downy mildew, leaf spot and leafroll leaves.** The true disease is none of the model's
   four classes, so a prediction can never be correct and **no accuracy is computed**. Only
   the confidence distribution, and which of the four classes the model assigns:
   out-of-distribution confidence.

Both are set against the same model's confidence on the clean PlantVillage test images and on
the most severe simulated corruption (the severity-5 condition with the lowest accuracy), all
at T = 1, and against the threshold chosen on validation: the share of each group that would
pass without a warning. Images get the evaluation transform (resize 256, centre crop 224),
exactly as the PlantVillage test images do.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.calibrate import ece_and_bins
from src.evaluate import scores, wilson_ci
from src.predict import Predictor
from src.provenance import not_a_result
from src.train import _json_safe

REPO = Path(__file__).resolve().parent.parent
FILTER = REPO / "results" / "plantseg" / "leaf_filter.csv"
SOURCE = {"zenodo_record": "16665770", "doi": "10.5281/zenodo.16665770", "version": "6 of 7",
          "file": "Plantseg.zip", "bytes": 1691076634,
          "md5": "23385d877bc84bab51b54630055c75b8", "licence": "CC BY 4.0",
          "url": "https://zenodo.org/api/records/16665770/files/Plantseg.zip/content"}
GROUPS = {"black_rot": "grape black rot", "downy_mildew": "grape downy mildew",
          "leaf_spot": "grape leaf spot", "leafroll": "grapevine leafroll disease"}
OOD = ("downy_mildew", "leaf_spot", "leafroll")
MAPS_TO = "Black_rot"  # the model's class for PlantSeg's "grape black rot"
# Sensitivity check for analysis A: leave out framings unlike PlantVillage's single leaves,
# as recorded in the filter's reason column (D15).
UNUSUAL_FRAMING = re.compile(r"close-up|wide view|shoot", re.IGNORECASE)
SPLIT_DIR = {"Training": "train", "Validation": "val", "Test": "test"}
N_BOOT = 1000
# D17 (owner, 2026-11-10): no ECE for the black rot subset. At n = 25 over 15 bins it is not
# estimable: the bin occupancy below is the justification, reported with every run. (The
# first real-image run showed it: the bootstrap interval missed its own estimate.)


def selection(filter_csv: str | Path = FILTER) -> dict[str, list[dict]]:
    """The analysable leaves per group: each photograph once (D15)."""
    rows = list(csv.DictReader(open(filter_csv)))
    out = {}
    for g, disease in GROUPS.items():
        out[g] = [r for r in rows if r["disease"] == disease and r["tag"] == "leaf"
                  and not r["duplicate_of"] and not r["label_conflict_with"]]
    return out


def md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def infer(pred: Predictor, z: zipfile.ZipFile, rows: list[dict], batch: int = 32) -> np.ndarray:
    """Logits for the listed PlantSeg images, read straight from the zip (nothing extracted)."""
    out = []
    for i in range(0, len(rows), batch):
        xs = []
        for r in rows[i:i + batch]:
            data = z.read(f"Plantseg/images/{SPLIT_DIR[r['plantseg_split']]}/{r['image']}")
            with Image.open(io.BytesIO(data)) as im:
                xs.append(pred.tf(im.convert("RGB")))
        out.append(pred.model(torch.stack(xs).to(pred.device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(pred.classes)), np.float32)


def confidence_summary(conf: np.ndarray, threshold: float | None, seed: int) -> dict:
    """The distribution of max-softmax confidence (T = 1): no claim about correctness."""
    rng = np.random.default_rng(seed)
    boot = [float(conf[rng.integers(0, conf.size, conf.size)].mean()) for _ in range(N_BOOT)]
    edges = np.linspace(0, 1, 11)
    out = {"n": int(conf.size), "mean": float(conf.mean()),
           "mean_bootstrap95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
           "median": float(np.median(conf)),
           "quartiles": [float(np.percentile(conf, 25)), float(np.percentile(conf, 75))],
           "share_at_least": {f"{t:.2f}": float((conf >= t).mean()) for t in (0.5, 0.9, 0.95)},
           "histogram": {"edges": edges.tolist(),
                         "counts": np.histogram(conf, edges)[0].astype(int).tolist()}}
    if threshold is not None:  # accepted without a warning by the warning tool (Stage 10)
        out["share_without_warning"] = float((conf >= threshold).mean())
    return out


def predicted(logits: np.ndarray, classes: list[str]) -> dict[str, int]:
    k = logits.argmax(1)
    return {c: int((k == i).sum()) for i, c in enumerate(classes)}


def black_rot_analysis(logits: np.ndarray, ids: list[str], classes: list[str], n_bins: int,
                       threshold: float | None, seed: int) -> dict:
    """Analysis A: labels inside the model's classes, so accuracy is defined."""
    target = classes.index(MAPS_TO)
    conf = scores(logits)["max_softmax"]
    correct = (logits.argmax(1) == target).astype(float)
    n, k = int(correct.size), int(correct.sum())
    bins = ece_and_bins(conf, correct, n_bins)[1]
    counts = [b["n"] for b in bins]
    occupied = [c for c in counts if c]
    occupancy = {"n_bins": n_bins, "counts": counts, "occupied": len(occupied),
                 "largest": max(occupied), "occupied_with_fewer_than_5":
                 sum(c < 5 for c in occupied),
                 "why_no_ece": f"{n} images over {n_bins} bins: {len(occupied)} occupied, "
                               f"{sum(c < 5 for c in occupied)} of them by fewer than 5 "
                               "images. A bin's accuracy from so few images is not an "
                               "estimate, so ECE is not reported (D17)."}
    return {"label": f"PlantSeg grape black rot leaves -> model class {MAPS_TO}",
            "ids": ids, "n": n, "n_correct": k, "accuracy": k / n,
            "accuracy_wilson95": list(wilson_ci(k, n)), "mean_confidence": float(conf.mean()),
            "bin_occupancy": occupancy,
            "mean_confidence_when_correct": float(conf[correct == 1].mean()) if k else None,
            "mean_confidence_when_wrong": float(conf[correct == 0].mean()) if k < n else None,
            "confidence": confidence_summary(conf, threshold, seed),
            "predicted_class": predicted(logits, classes)}


def ood_analysis(group: str, logits: np.ndarray, ids: list[str], classes: list[str],
                 threshold: float | None, seed: int) -> dict:
    """Analysis B: the true disease is none of the model's classes. Confidence only."""
    conf = scores(logits)["max_softmax"]
    return {"label": f"PlantSeg {GROUPS[group]} leaves: out-of-distribution confidence "
                     "(no correct answer exists, so no accuracy)",
            "ids": ids, "n": int(conf.size),
            "confidence": confidence_summary(conf, threshold, seed),
            "predicted_class": predicted(logits, classes)}


def references(sweep: dict, sweep_logits: dict, threshold: float | None, seed: int) -> dict:
    """The same model on PlantVillage test images at T = 1: clean, and the most severe
    simulated corruption (the severity-5 condition with the lowest accuracy)."""
    worst = min((r for r in sweep["rows"] if r["severity"] == 5), key=lambda r: r["accuracy"])
    out = {}
    for name, key, what in (("plantvillage_clean", "gaussian_blur_s0", "clean test images"),
                            ("plantvillage_most_severe",
                             f"{worst['corruption']}_s5",
                             f"{worst['corruption']} severity 5, the severity-5 condition "
                             f"with the lowest accuracy ({worst['accuracy']:.4f})")):
        conf = scores(sweep_logits[key])["max_softmax"]
        out[name] = {"label": f"PlantVillage {what}", "condition": key,
                     "confidence": confidence_summary(conf, threshold, seed)}
    return out


def sweep_threshold(sweep: dict) -> tuple[float | None, str]:
    """The max-softmax threshold chosen on validation at the primary target (D10)."""
    th = sweep.get("threshold", {})
    if "chosen_on" not in th:
        return None, "sweep predates the held-out threshold (D10): no threshold used"
    t = th["by_score"]["max_softmax"][f"{th['primary_target']:.2f}"]["validation"]
    if not t["found"]:
        return None, "no threshold met the target on validation"
    return float(t["threshold"]), (f"max softmax {t['threshold']:.4g}, chosen on validation for "
                                   f"{th['primary_target']:.2f} accuracy")


def plantseg_eval(results_path: str | Path, zip_path: str | Path,
                  filter_csv: str | Path = FILTER, check_md5: bool = True) -> dict:
    results_path = Path(results_path)
    res = json.loads(results_path.read_text())
    d, run_id = results_path.parent, res["run_id"]
    sweep = json.loads((d / f"sweep_{run_id}.json").read_text())
    sweep_logits = dict(np.load(d / f"sweep_logits_{run_id}.npz"))
    if check_md5:
        got = md5(zip_path)
        assert got == SOURCE["md5"], f"{zip_path} is not PlantSeg v6 (md5 {got})"
    threshold, threshold_note = sweep_threshold(sweep)
    pred = Predictor(results_path, threshold=None)
    classes, seed, n_bins = pred.classes, res["config"]["seed"], res["config"]["n_bins"]
    assert MAPS_TO in classes
    sel = selection(filter_csv)
    logits = {}
    with zipfile.ZipFile(zip_path) as z:
        for g, rows in sel.items():
            logits[g] = infer(pred, z, rows)
    ids = {g: [r["id"] for r in rows] for g, rows in sel.items()}
    usual = [i for i, r in enumerate(sel["black_rot"]) if not UNUSUAL_FRAMING.search(r["reason"])]
    out = {
        "run_id": run_id, "not_a_result": not_a_result(res), "source": SOURCE,
        "classes": classes, "temperature": 1.0,
        "transform": "resize 256, centre crop 224 (the evaluation transform)",
        "selection": "results/plantseg/leaf_filter.csv: tag leaf, no duplicate_of, no "
                     "label_conflict_with (DECISIONS.md D15)",
        "threshold": threshold, "threshold_note": threshold_note,
        "black_rot": black_rot_analysis(logits["black_rot"], ids["black_rot"], classes, n_bins,
                                        threshold, seed),
        "black_rot_usual_framing": black_rot_analysis(
            logits["black_rot"][usual], [ids["black_rot"][i] for i in usual], classes, n_bins,
            threshold, seed),
        "out_of_distribution": {g: ood_analysis(g, logits[g], ids[g], classes, threshold, seed)
                                for g in OOD},
        "references": references(sweep, sweep_logits, threshold, seed),
    }
    out["black_rot_usual_framing"]["label"] += (" (sensitivity: without close-ups, wide views "
                                                "and shoots)")
    (d / f"plantseg_{run_id}.json").write_text(
        json.dumps(_json_safe(out), indent=1, allow_nan=False))
    store = {**logits, **{f"{g}__ids": np.array(ids[g]) for g in logits}}
    # numpy stubs type **kwds against the allow_pickle: bool keyword; the call is correct
    np.savez(d / f"plantseg_logits_{run_id}.npz", **store)  # type: ignore[arg-type]
    print_report(out)
    print(f"written -> {d / f'plantseg_{run_id}.json'}")
    return out


def print_report(ps: dict) -> None:
    print(f"PlantSeg v6 (Zenodo {ps['source']['zenodo_record']}), model {ps['run_id']}, T = 1")
    print(f"threshold: {ps['threshold_note']}")
    for key in ("black_rot", "black_rot_usual_framing"):
        a = ps[key]
        lo, hi = a["accuracy_wilson95"]
        print(f"\n{a['label']}\n  n = {a['n']}  <- small: read the interval, not the point")
        print(f"  accuracy {a['accuracy']:.3f}  (Wilson 95% [{lo:.3f}, {hi:.3f}]), "
              f"{a['n_correct']} of {a['n']}")
        print(f"  mean confidence {a['mean_confidence']:.3f}; no ECE: "
              f"{a['bin_occupancy']['why_no_ece']}")
        print(f"  predicted: {a['predicted_class']}")
    print("\nOUT-OF-DISTRIBUTION CONFIDENCE (no accuracy: the true disease is not one of the "
          "model's classes)")
    groups = {**{g: ps["out_of_distribution"][g] for g in OOD},
              **{k: v for k, v in ps["references"].items()},
              "black_rot (for comparison)": ps["black_rot"]}
    for g, a in groups.items():
        c = a["confidence"]
        extra = (f", without warning {c['share_without_warning']:.1%}"
                 if "share_without_warning" in c else "")
        print(f"  {g:<28} n {c['n']:4d}  mean {c['mean']:.3f}  median {c['median']:.3f}  "
              f">=0.9 {c['share_at_least']['0.90']:.1%}{extra}")
    for g in OOD:
        print(f"  {g} assigned to: {ps['out_of_distribution'][g]['predicted_class']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results_<run_id>.json (its sweep too)")
    ap.add_argument("--zip", required=True, help="Plantseg.zip, Zenodo record 16665770")
    ap.add_argument("--filter", default=str(FILTER))
    a = ap.parse_args()
    plantseg_eval(a.results, a.zip, a.filter)


if __name__ == "__main__":
    main()
