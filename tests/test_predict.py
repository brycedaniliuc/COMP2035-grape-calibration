"""The prediction + confidence + warning tool (Stage 10), on the real fixture images."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data import LeafDataset, build_transforms, class_list, index_dataset
from src.evaluate import scores
from src.predict import (
    Predictor,
    find_images,
    main,
    manual_threshold,
    predict_image,
    threshold_from_sweep,
    warning_meaning,
)
from src.train import build_model, collect_logits


@pytest.fixture(scope="module")
def run_dir(fixture_root, tmp_path_factory):
    """An untrained (seeded) model saved the way src.train saves a run."""
    d = tmp_path_factory.mktemp("run")
    cfg = Config(data_root=str(fixture_root), model="efficientnet_b0", pretrained=False)
    classes = class_list(index_dataset(fixture_root))
    torch.manual_seed(0)
    model, _ = build_model(cfg.model, len(classes), pretrained=False)
    torch.save(model.state_dict(), d / "best_x.pt")
    (d / "results_x.json").write_text(json.dumps(
        {"run_id": "x", "config": cfg.to_dict(), "classes": classes,
         "artefacts": {"checkpoint": "best_x.pt"}}))
    return d


def test_predictions_match_the_evaluation_pipeline(run_dir, fixture_root):
    """Same transform, same model, same score as the sweep: confidence = max softmax of the
    logits src.train would collect for these images."""
    p = Predictor(run_dir / "results_x.json", threshold=0.5)
    recs = index_dataset(fixture_root)
    paths = [Path(r.path) for r in recs]
    preds = p.predict(paths, batch_size=7)
    assert len(preds) == len(recs) == 80
    cfg = Config(model="efficientnet_b0", pretrained=False)
    ds = LeafDataset(recs, build_transforms(cfg)[1], {c: i for i, c in enumerate(p.classes)})
    logits, _ = collect_logits(p.model, DataLoader(ds, batch_size=16), p.device)
    want = scores(logits)["max_softmax"]
    assert np.allclose([q.confidence for q in preds], want, atol=1e-6)
    assert [q.predicted_class for q in preds] == [p.classes[k] for k in logits.argmax(1)]
    assert all(q.warning == (q.confidence < 0.5) for q in preds)


def test_threshold_edges(run_dir, fixture_root):
    img = find_images(fixture_root)[0]
    p = Predictor(run_dir / "results_x.json", threshold=None)
    assert predict_image(p, img).warning  # no supported threshold: everything is flagged
    c = predict_image(p, img).confidence
    p.threshold = c
    assert not predict_image(p, img).warning  # accepted at exactly t, as in derive_threshold
    p.threshold = float(np.nextafter(np.float32(c), np.float32(2)))
    assert predict_image(p, img).warning


def _sweep(found: bool, below: list, vacuous: list) -> dict:
    per: dict[str, dict] = {f"c{i}": {} for i in range(24)}
    return {"run_id": "x", "threshold": {
        "chosen_on": "degraded validation images",
        "score_units": {"max_softmax": "probability, T = 1"},
        "by_score": {"max_softmax": {"0.95": {
            "validation": {"found": found, "threshold": 0.8 if found else None},
            "test": {"per_condition": per, "below_target": below, "vacuous_conditions": vacuous,
                     "accepts_nothing_anywhere": False, "referred_mean": 0.25}
            if found else None}}}}}


def test_threshold_from_sweep_reports_support_honestly():
    ok = threshold_from_sweep(_sweep(True, [], ["c1"]))
    assert ok.value == 0.8 and ok.supported and "24 of 24" in ok.statement
    bad = threshold_from_sweep(_sweep(True, ["c3"], []))
    assert bad.value == 0.8 and not bad.supported and "NOT SUPPORTED" in bad.statement
    none = threshold_from_sweep(_sweep(False, [], []))
    assert none.value is None and not none.supported
    hand = manual_threshold(0.9)
    assert not hand.supported
    # the explanation of "no warning" only claims accuracy for a supported threshold
    assert "met the target" in warning_meaning(ok)[1]
    for th in (bad, none, hand):
        assert "met the target" not in warning_meaning(th)[1]


def test_cli_writes_csv(run_dir, fixture_root, tmp_path, monkeypatch, capsys):
    out = tmp_path / "pred.csv"
    monkeypatch.setattr(sys, "argv", ["predict", "--results", str(run_dir / "results_x.json"),
                                      "--threshold", "0.6", "--images", str(fixture_root),
                                      "--out", str(out)])
    main()
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 80
    assert set(rows[0]) == {"image", "predicted_class", "confidence", "warning"}
    assert all(r["warning"] == ("yes" if float(r["confidence"]) < 0.6 else "no") for r in rows)
    printed = capsys.readouterr().out
    assert "not derived from data" in printed and "does not cover" in printed
