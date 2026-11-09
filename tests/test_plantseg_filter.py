"""The committed PlantSeg leaf filter (DECISIONS.md D15) is complete and consistent."""

import csv
from collections import Counter
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "results" / "plantseg" / "leaf_filter.csv"


def rows():
    return list(csv.DictReader(CSV.open()))


def test_every_grape_image_tagged_once_with_a_valid_tag():
    rs = rows()
    assert len(rs) == 564 and len({r["id"] for r in rs}) == 564
    assert {r["tag"] for r in rs} <= {"leaf", "berry", "other"}
    assert all(r["reason"] for r in rs if r["tag"] == "other")
    counts = Counter((r["disease"], r["tag"]) for r in rs)
    assert counts[("grape black rot", "leaf")] == 25
    assert sum(counts.values()) == 564


def test_duplicates_and_label_conflicts_are_consistent():
    by = {r["id"]: r for r in rows()}
    for r in by.values():
        if r["duplicate_of"]:
            first = by[r["duplicate_of"]]
            assert first["disease"] == r["disease"] and first["tag"] == r["tag"]
            assert not first["duplicate_of"]  # points at the kept copy
        for other in filter(None, r["label_conflict_with"].split(";")):
            assert r["id"] in by[other]["label_conflict_with"].split(";")
            assert by[other]["disease"] != r["disease"] and by[other]["tag"] == r["tag"]


def test_the_analysable_leaves():
    """Phase B2 uses each photograph once: leaf, not a duplicate, no label conflict."""
    use = Counter(r["disease"] for r in rows() if r["tag"] == "leaf"
                  and not r["duplicate_of"] and not r["label_conflict_with"])
    assert use == {"grape black rot": 25, "grape downy mildew": 174, "grape leaf spot": 74,
                   "grapevine leafroll disease": 61}
