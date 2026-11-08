"""The prototype output promised in the submitted paper (abstract, §1): a disease prediction,
a confidence indicator, and an explicit warning for uncertain cases (Stage 10).

    python -m src.predict --results <run>/results_<run_id>.json \\
        --sweep <run>/sweep_<run_id>.json --images <folder> --out predictions.csv

or, with a threshold not derived from this repo's data, `--threshold 0.9` instead of
`--sweep`. The confidence is the max softmax probability at T = 1, the score the threshold
is chosen on. warning = confidence < threshold. Every run prints where the threshold came
from, whether the data support it, and what the warning does not cover.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.config import Config
from src.data import IMG_EXT, build_transforms
from src.evaluate import PRIMARY_TARGET, THRESHOLD_LIMITS, scores
from src.train import build_model, select_device


@dataclass(frozen=True)
class Prediction:
    image: str
    predicted_class: str
    confidence: float  # max softmax probability at T = 1
    warning: bool      # confidence < threshold


@dataclass(frozen=True)
class Threshold:
    value: float | None
    source: str
    supported: bool    # chosen on validation AND met the target on every held-out condition
    statement: str


def warning_meaning(th: Threshold) -> list[str]:
    """What the flag says, and does not say, for this particular threshold."""
    if th.value is None:
        no = "there is no supported threshold, so every image is flagged."
    elif th.supported:
        no = ("warning = no does NOT mean the prediction is right. It means predictions this "
              "confident met the target accuracy on the held-out test images in every tested "
              "condition. Individual predictions can still be wrong.")
    elif th.source.startswith("sweep"):
        no = ("warning = no means only that the confidence reached the threshold. This "
              "threshold failed the target on held-out data, so it carries no accuracy claim.")
    else:
        no = ("warning = no means only that the confidence reached the hand-given threshold; "
              "nothing in this repo says how often such predictions are right.")
    return ["warning = yes: the confidence is below the threshold. Refer the image to a "
            "person instead of acting on the prediction.", no,
            "the warning cannot recognise an image outside the model's scope (not a grape "
            "leaf, or a disease outside the four classes): the model always answers with one "
            "of its four classes, sometimes confidently."]


def threshold_from_sweep(sweep: dict, target: float = PRIMARY_TARGET) -> Threshold:
    """The max-softmax threshold chosen on degraded validation images, with its held-out test
    result. A threshold that is missing, or that fails on held-out data, is reported as such."""
    th = sweep["threshold"]
    assert "chosen_on" in th, ("this sweep chose its threshold on the test images (code before "
                               "Stage 9): re-run src.evaluate on the run")
    assert th["score_units"]["max_softmax"] == "probability, T = 1"
    r = th["by_score"]["max_softmax"][f"{target:.2f}"]
    src = f"sweep {sweep['run_id']}, max softmax, target {target:.2f}, chosen on {th['chosen_on']}"
    if not r["validation"]["found"]:
        return Threshold(None, src, False, f"no threshold reaches {target:.2f} accuracy in "
                         "every validation condition: no value is supported by this data")
    te = r["test"]
    n = len(te["per_condition"])
    supported = not te["below_target"] and not te["accepts_nothing_anywhere"]
    statement = (f"held-out test images: {n - len(te['below_target'])} of {n} conditions met "
                 f"{target:.2f} on accepted predictions; {len(te['vacuous_conditions'])} were "
                 f"fully referred; {te['referred_mean']:.1%} referred on average")
    if not supported:
        statement += ". NOT SUPPORTED: it fails the target on held-out data"
    return Threshold(float(r["validation"]["threshold"]), src, supported, statement)


def manual_threshold(value: float) -> Threshold:
    return Threshold(value, "given on the command line", False,
                     "not derived from data in this repo: no accuracy claim attaches to it")


class Predictor:
    """Load a trained run once, then predict any number of images."""

    def __init__(self, results_path: str | Path, threshold: float | None):
        results_path = Path(results_path)
        res = json.loads(results_path.read_text())
        cfg = Config(**res["config"])
        self.classes: list[str] = res["classes"]
        self.threshold = threshold
        self.device = select_device()
        self.model, _ = build_model(cfg.model, len(self.classes), pretrained=False)
        self.model.load_state_dict(torch.load(results_path.parent / res["artefacts"]["checkpoint"],
                                              map_location=self.device, weights_only=True))
        self.model = self.model.to(self.device).eval()
        self.tf = build_transforms(cfg)[1]  # the evaluation transform: resize, centre crop
        self.run_id: str = res["run_id"]

    @torch.no_grad()
    def predict(self, paths: list[Path], names: list[str] | None = None,
                batch_size: int = 32) -> list[Prediction]:
        names = names or [str(p) for p in paths]
        out: list[Prediction] = []
        for i in range(0, len(paths), batch_size):
            batch = []
            for p in paths[i:i + batch_size]:
                with Image.open(p) as im:
                    batch.append(self.tf(im.convert("RGB")))
            logits = self.model(torch.stack(batch).to(self.device)).float().cpu().numpy()
            conf = scores(logits)["max_softmax"]
            for j, (k, c) in enumerate(zip(logits.argmax(1), conf, strict=True)):
                warn = self.threshold is None or bool(c < self.threshold)
                out.append(Prediction(names[i + j], self.classes[int(k)], float(c), warn))
        return out


def predict_image(predictor: Predictor, path: str | Path) -> Prediction:
    """One image -> (predicted class, confidence, warning)."""
    return predictor.predict([Path(path)])[0]


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)


def write_csv(preds: list[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "predicted_class", "confidence", "warning"])
        for p in preds:
            w.writerow([p.image, p.predicted_class, f"{p.confidence:.6f}",
                        "yes" if p.warning else "no"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results_<run_id>.json of the model")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sweep", help="sweep_<run_id>.json: use the threshold chosen there")
    g.add_argument("--threshold", type=float, help="a threshold given by hand (0..1)")
    ap.add_argument("--target", type=float, default=PRIMARY_TARGET,
                    help="accuracy target of the sweep threshold (0.90, 0.95 or 0.99)")
    ap.add_argument("--images", required=True, help="folder of images (searched recursively)")
    ap.add_argument("--out", required=True, help="CSV to write")
    a = ap.parse_args()

    if a.sweep:
        sw = json.loads(Path(a.sweep).read_text())
        th = threshold_from_sweep(sw, a.target)
    else:
        assert 0.0 <= a.threshold <= 1.0, "the threshold is a probability"
        th = manual_threshold(a.threshold)
    pred = Predictor(a.results, th.value)
    if a.sweep:
        assert sw["run_id"] == pred.run_id, "the sweep belongs to a different model run"

    folder = Path(a.images)
    paths = find_images(folder)
    assert paths, f"no images in {folder}"
    preds = pred.predict(paths, [str(p.relative_to(folder)) for p in paths])
    write_csv(preds, Path(a.out))

    n_warn = sum(p.warning for p in preds)
    print(f"model: {pred.run_id}")
    print(f"threshold: {th.value if th.value is not None else 'NONE'} ({th.source})")
    print(f"  {th.statement}")
    if th.value is None:
        print("  every image is flagged: with no supported threshold, no prediction is "
              "trusted without a person checking it")
    print(f"{len(preds)} images, {n_warn} flagged with a warning ({n_warn / len(preds):.1%})"
          f" -> {a.out}")
    print("what the warning means:")
    for s in warning_meaning(th):
        print("  -", s)
    print("what the threshold does not cover:")
    for s in THRESHOLD_LIMITS:
        print("  -", s)


if __name__ == "__main__":
    main()
