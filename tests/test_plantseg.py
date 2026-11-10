"""PlantSeg phase B2 (D16) on a stand-in: fixture images packed like Plantseg.zip, a filter
that exercises every exclusion, and a fixture-trained run with its sweep."""

import csv
import json
import zipfile

import pytest

from src.config import Config
from src.data import index_dataset
from src.evaluate import sweep
from src.plantseg import plantseg_eval, selection, sweep_threshold
from src.train import run
from src.verify import _has_accuracy, plantseg_checks

COLS = ["id", "image", "disease", "tag", "reason", "duplicate_of", "label_conflict_with",
        "plantseg_split", "image_licence", "sheet"]
ROWS = [  # id, disease, tag, reason, duplicate_of, label_conflict_with
    ("BR-001", "grape black rot", "leaf", "", "", ""),
    ("BR-002", "grape black rot", "leaf", "close-up of lesions on a leaf blade", "", ""),
    ("BR-003", "grape black rot", "leaf", "", "", ""),
    ("BR-004", "grape black rot", "leaf", "", "BR-001", ""),          # duplicate: out
    ("BR-005", "grape black rot", "berry", "", "", ""),               # berry: out
    ("DM-001", "grape downy mildew", "leaf", "", "", ""),
    ("DM-002", "grape downy mildew", "leaf", "", "", ""),
    ("DM-003", "grape downy mildew", "leaf", "", "", "LS-003"),       # label conflict: out
    ("LS-001", "grape leaf spot", "leaf", "", "", ""),
    ("LS-002", "grape leaf spot", "other", "a composite", "", ""),    # other: out
    ("LS-003", "grape leaf spot", "leaf", "", "", "DM-003"),          # label conflict: out
    ("LR-001", "grapevine leafroll disease", "leaf", "", "", ""),
    ("LR-002", "grapevine leafroll disease", "leaf", "", "", ""),
]


@pytest.fixture(scope="module")
def setup(fixture_root, tmp_path_factory):
    d = tmp_path_factory.mktemp("ps")
    cfg = Config(data_root=str(fixture_root), model="efficientnet_b0", pretrained=False,
                 batch_size=8, head_epochs=1, finetune_epochs=0, num_workers=0, out_dir=str(d))
    res = run(cfg, stamp="ps")
    results = d / f"results_{res['run_id']}.json"
    sweep(results)
    # The untrained fixture model reaches no threshold on validation; set one in this copy of
    # the sweep so the "accepted without a warning" path runs end to end, verification included.
    sw_path = d / f"sweep_{res['run_id']}.json"
    sw = json.loads(sw_path.read_text())
    sw["threshold"]["by_score"]["max_softmax"]["0.95"]["validation"] = {"found": True,
                                                                        "threshold": 0.3}
    sw_path.write_text(json.dumps(sw))
    images = [r.path for r in index_dataset(fixture_root)][:len(ROWS)]
    zpath, fpath = d / "Plantseg.zip", d / "filter.csv"
    with zipfile.ZipFile(zpath, "w") as z, fpath.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for (i, disease, tag, reason, dup, conflict), img in zip(ROWS, images, strict=True):
            name = f"{i}.jpg"
            z.write(img, f"Plantseg/images/train/{name}")
            w.writerow({"id": i, "image": name, "disease": disease, "tag": tag,
                        "reason": reason, "duplicate_of": dup, "label_conflict_with": conflict,
                        "plantseg_split": "Training", "image_licence": "CC0", "sheet": "x"})
    ps = plantseg_eval(results, zpath, fpath, check_md5=False)
    return d, ps, fpath


def test_selection_counts_each_photograph_once(setup):
    _, _, fpath = setup
    ids = {g: [r["id"] for r in rows] for g, rows in selection(fpath).items()}
    assert ids == {"black_rot": ["BR-001", "BR-002", "BR-003"],
                   "downy_mildew": ["DM-001", "DM-002"], "leaf_spot": ["LS-001"],
                   "leafroll": ["LR-001", "LR-002"]}


def test_two_analyses_never_merged(setup):
    _, ps, _ = setup
    br = ps["black_rot"]
    assert br["n"] == 3 and 0 <= br["accuracy"] <= 1 and len(br["accuracy_wilson95"]) == 2
    assert "ece" not in br  # D17: not estimable at this n; the occupancy says why
    occ = br["bin_occupancy"]
    assert sum(occ["counts"]) == 3 and occ["occupied"] == sum(c > 0 for c in occ["counts"])
    assert ps["black_rot_usual_framing"]["ids"] == ["BR-001", "BR-003"]  # close-up left out
    assert set(ps["out_of_distribution"]) == {"downy_mildew", "leaf_spot", "leafroll"}
    assert not _has_accuracy(ps["out_of_distribution"])  # MUST NOT report accuracy
    for g in ps["out_of_distribution"].values():
        assert "out-of-distribution" in g["label"] and sum(g["predicted_class"].values()) == g["n"]


def test_references_threshold_and_verification(setup):
    d, ps, fpath = setup
    assert ps["threshold"] == 0.3
    assert "share_without_warning" in ps["black_rot"]["confidence"]
    assert ps["references"]["plantvillage_most_severe"]["condition"].endswith("_s5")
    saved = json.loads((d / f"plantseg_{ps['run_id']}.json").read_text())
    assert saved["source"]["md5"] == "23385d877bc84bab51b54630055c75b8"
    checks = plantseg_checks(d / f"plantseg_{ps['run_id']}.json", fpath)
    assert len(checks) == 5 and all(c[2] == "PASS" for c in checks), checks


def test_sweep_threshold_cases():
    found = {"chosen_on": "v", "primary_target": 0.95, "by_score": {"max_softmax": {
        "0.95": {"validation": {"found": True, "threshold": 0.8}}}}}
    assert sweep_threshold({"threshold": found})[0] == 0.8
    none = {"chosen_on": "v", "primary_target": 0.95, "by_score": {"max_softmax": {
        "0.95": {"validation": {"found": False, "threshold": None}}}}}
    assert sweep_threshold({"threshold": none}) == (None, "no threshold met the target on "
                                                          "validation")
    assert sweep_threshold({"threshold": {"by_score": {}}})[0] is None  # pre-D10 sweep
