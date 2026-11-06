"""Indexing, filename parsing, leaf grouping, splits and audit, on the real-image fixture."""

import json
import os
import shutil
from pathlib import Path

import pytest

from src.audit import leaking_leaves, supplied_split_audit
from src.config import Config
from src.data import PARTS, index_dataset, make_splits, parse_filename

REPO = Path(__file__).resolve().parent.parent
FULL_DATA = Path(os.environ.get(
    "GRAPE_DATA_ROOT",
    "/Users/bryce/Downloads/Literature Review paper/Grape/grape_dataset"))


def cfg(mode: str, root: Path) -> Config:
    return Config(data_root=str(root), split_mode=mode)


# ---- filename parsing ---------------------------------------------------------

def test_parse_real_original_and_augmented_names():
    leaf, aug, suf = parse_filename("004175d8-dc74-4285-8401-3cc9565730bb___FAM_B.Rot 0626.JPG")
    assert (leaf, aug, suf) == ("004175d8-dc74-4285-8401-3cc9565730bb", False, "")
    leaf, aug, suf = parse_filename(
        "004175d8-dc74-4285-8401-3cc9565730bb___FAM_B.Rot 0626_flipLR.JPG")
    assert (leaf, aug, suf) == ("004175d8-dc74-4285-8401-3cc9565730bb", True, "_flipLR")


def test_uuid_is_lowercased():
    leaf, _, _ = parse_filename("004175D8-DC74-4285-8401-3CC9565730BB___x_90deg.jpg")
    assert leaf == "004175d8-dc74-4285-8401-3cc9565730bb"


def test_missing_uuid_raises():
    with pytest.raises(ValueError, match="no source-leaf UUID"):
        parse_filename("IMG_0001.JPG")


# ---- indexing on the fixture --------------------------------------------------

def test_index_matches_manifest_exactly(fixture_root, manifest):
    recs = index_dataset(fixture_root)
    got = sorted((r.partition, Path(r.path).parent.name, Path(r.path).name, r.leaf, r.aug,
                  r.suffix) for r in recs)
    want = sorted((f["partition"], f["class_dir"], f["filename"], f["leaf"], f["augmented"],
                   f["suffix"]) for f in manifest["files"])
    assert got == want


def test_both_sides_have_originals_and_augmented(fixture_root):
    recs = index_dataset(fixture_root)
    for p in ("train", "test"):
        assert sum(not r.aug for r in recs if r.partition == p) > 0
        assert sum(r.aug for r in recs if r.partition == p) > 0


def test_leaf_in_two_classes_raises(fixture_root, tmp_path):
    shutil.copytree(fixture_root / "train", tmp_path / "train")
    src = next((tmp_path / "train" / "Grape___healthy").iterdir())
    shutil.copy(src, tmp_path / "train" / "Grape___Black_rot" / src.name)
    with pytest.raises(ValueError, match="two classes"):
        index_dataset(tmp_path)


# ---- audit ---------------------------------------------------------------------

def test_audit_reproduces_expected_fixture_audit(fixture_root, manifest):
    a = supplied_split_audit(index_dataset(fixture_root))
    e = manifest["expected_fixture_audit"]
    assert a["n_images"] == e["n_images"] == 80
    assert a["n_source_leaves"] == e["n_source_leaves"]
    assert (a["n_originals"], a["n_augmented"]) == (e["n_originals"], e["n_augmented"])
    assert a["supplied_train_leaves"] == e["train_leaves"]
    assert a["supplied_test_leaves"] == e["test_leaves"]
    assert a["test_leaves_also_in_train"] == e["test_leaves_also_in_train"] == 8
    assert a["pct_test_leaves_also_in_train"] == e["pct_test_leaves_also_in_train"]
    assert a["n_test_originals"] == e["n_test_originals"]
    assert a["n_test_originals_leaf_in_train"] == e["n_test_originals_leaf_in_train"] == 4
    assert a["pct_test_originals_leaf_in_train"] == e["pct_test_originals_leaf_in_train"]
    for cls_dir, v in e["per_class"].items():
        assert a["per_class_pct_test_leaves_also_in_train"][cls_dir.split("___")[-1]] == v["pct"]


def test_audit_finds_exactly_the_planted_leaks(fixture_root, manifest):
    assert leaking_leaves(index_dataset(fixture_root)) == manifest["leak_leaves"]


@pytest.mark.skipif(not FULL_DATA.is_dir(), reason="full dataset not on this machine")
def test_full_dataset_audit_matches_september_audit_json():
    a = supplied_split_audit(index_dataset(FULL_DATA))
    want = json.loads((REPO / "results" / "source" / "audit.json").read_text())
    assert a == want


# ---- splits ----------------------------------------------------------------------

def test_source_split_groups_leaves_and_discards_eval_augmented(fixture_root):
    recs = index_dataset(fixture_root)
    sp = make_splits(recs, cfg("source", fixture_root))
    leaves = {s: {r.leaf for r in sp[s]} for s in PARTS}
    assert not (leaves["train"] & leaves["val"])
    assert not (leaves["train"] & leaves["test"])
    assert not (leaves["val"] & leaves["test"])
    assert all(not r.aug for r in sp.val + sp.test)
    # every augmented copy of a val/test leaf is dropped, none is moved to train
    eval_leaves = leaves["val"] | leaves["test"]
    assert sp.dropped_augmented == sum(r.aug and r.leaf in eval_leaves for r in recs)
    assert {r.path for r in sp.train} == {r.path for r in recs if r.leaf in leaves["train"]}
    # leaves are whole: every image of a train leaf is in train
    assert len(sp.train) + len(sp.val) + len(sp.test) + sp.dropped_augmented == len(recs)


@pytest.mark.parametrize("mode", ["source", "naive", "supplied"])
def test_splits_are_deterministic(fixture_root, mode):
    recs = index_dataset(fixture_root)
    a, b = make_splits(recs, cfg(mode, fixture_root)), make_splits(recs, cfg(mode, fixture_root))
    for s in PARTS:
        assert [r.path for r in a[s]] == [r.path for r in b[s]]


def test_supplied_split_keeps_supplied_test_folder(fixture_root):
    recs = index_dataset(fixture_root)
    sp = make_splits(recs, cfg("supplied", fixture_root))
    assert {r.path for r in sp.test} == {r.path for r in recs if r.partition == "test"}
    assert {r.path for r in sp.train + sp.val} == {r.path for r in recs if r.partition == "train"}


def test_naive_split_uses_every_image_once(fixture_root):
    recs = index_dataset(fixture_root)
    sp = make_splits(recs, cfg("naive", fixture_root))
    paths = [r.path for s in PARTS for r in sp[s]]
    assert sorted(paths) == sorted(r.path for r in recs)
