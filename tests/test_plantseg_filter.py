"""The committed PlantSeg leaf filter (DECISIONS.md D15) is complete and well formed."""

import csv
from collections import Counter
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "results" / "plantseg" / "leaf_filter.csv"


def test_every_grape_image_tagged_once_with_a_valid_tag():
    rows = list(csv.DictReader(CSV.open()))
    assert len(rows) == 564 and len({r["id"] for r in rows}) == 564
    assert {r["tag"] for r in rows} <= {"leaf", "berry", "other"}
    assert all(r["reason"] for r in rows if r["tag"] == "other")
    counts = Counter((r["disease"], r["tag"]) for r in rows)
    assert counts[("grape black rot", "leaf")] == 25  # the B2 analysis rests on this number
    assert sum(counts.values()) == 564
