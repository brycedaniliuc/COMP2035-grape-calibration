"""Contact sheets of the PlantSeg grape images, for the leaf filter (DECISIONS.md D15).

    .venv/bin/python tools/plantseg_contact_sheets.py <Plantseg.zip> <out_dir>
    .venv/bin/python tools/plantseg_contact_sheets.py <Plantseg.zip> <out_dir> \\
        --tags results/plantseg/leaf_filter.csv          # review sheets, grouped by tag

Reads the images straight from the zip (nothing is extracted), takes `Plant == "Grape"` from
Plantseg/Metadata.csv, orders each class by filename, and gives every image a stable id:
BR (black rot), DM (downy mildew), LS (leaf spot) or LR (leafroll), plus a number. Writes
sheets of 25 (5 x 5) and index.csv. With --tags, also writes review sheets of each class's
images grouped by tag, for spot-checking. The sheets are PlantSeg images: keep out_dir
OUTSIDE the repository.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASSES = {"grape black rot": "BR", "grape downy mildew": "DM", "grape leaf spot": "LS",
           "grapevine leafroll disease": "LR"}
SPLIT_DIR = {"Training": "train", "Validation": "val", "Test": "test"}
COLS, CELL, PER_SHEET = 5, 280, 25


def natural(name: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def grape_index(z: zipfile.ZipFile) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(z.read("Plantseg/Metadata.csv").decode("utf-8-sig"))))
    out = []
    for disease, prefix in CLASSES.items():
        mine = sorted((r for r in rows if r["Plant"] == "Grape" and r["Disease"] == disease),
                      key=lambda r: natural(r["Name"]))
        for i, r in enumerate(mine, 1):
            out.append({"id": f"{prefix}-{i:03d}", "image": r["Name"], "disease": disease,
                        "plantseg_split": r["Split"], "image_licence": r["License"],
                        "zip_path": f"Plantseg/images/{SPLIT_DIR[r['Split']]}/{r['Name']}",
                        "sheet": f"{prefix}_{(i - 1) // PER_SHEET + 1:02d}"})
    assert len(out) == 564, f"expected the 564 grape images of D14, found {len(out)}"
    return out


def sheet(z: zipfile.ZipFile, items: list[dict], title: str, path: Path) -> None:
    font = ImageFont.load_default(size=20)
    rows = (len(items) + COLS - 1) // COLS
    img = Image.new("RGB", (COLS * CELL, 40 + rows * CELL), "white")
    d = ImageDraw.Draw(img)
    d.text((8, 8), title, fill="black", font=font)
    for k, it in enumerate(items):
        x, y = (k % COLS) * CELL, 40 + (k // COLS) * CELL
        with z.open(it["zip_path"]) as fh:
            im = Image.open(fh).convert("RGB")
        im.thumbnail((CELL - 10, CELL - 36))
        img.paste(im, (x + 5, y + 30))
        d.text((x + 5, y + 4), it["id"], fill="black", font=font)
    img.save(path, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip")
    ap.add_argument("out_dir")
    ap.add_argument("--tags", help="leaf_filter.csv: also write review sheets grouped by tag")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.zip) as z:
        index = grape_index(z)
        if not a.tags:
            with (out / "index.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(index[0]))
                w.writeheader()
                w.writerows(index)
            for name in dict.fromkeys(it["sheet"] for it in index):
                items = [it for it in index if it["sheet"] == name]
                disease = items[0]["disease"]
                sheet(z, items, f"{name}: {disease}, {items[0]['id']} to {items[-1]['id']}",
                      out / f"sheet_{name}.jpg")
            print(f"{len(index)} images, {len(set(it['sheet'] for it in index))} sheets -> {out}")
            return
        tags = {r["id"]: r["tag"] for r in csv.DictReader(open(a.tags))}
        assert set(tags) == {it["id"] for it in index}, "tags do not cover the 564 images"
        for disease, prefix in CLASSES.items():
            for tag in ("leaf", "berry", "other"):
                items = [it for it in index if it["disease"] == disease and tags[it["id"]] == tag]
                for s in range(0, len(items), PER_SHEET):
                    part = items[s:s + PER_SHEET]
                    sheet(z, part, f"REVIEW {prefix} tagged {tag.upper()} "
                          f"({s + 1}-{s + len(part)} of {len(items)})",
                          out / f"review_{prefix}_{tag}_{s // PER_SHEET + 1:02d}.jpg")
        print(f"review sheets -> {out}")


if __name__ == "__main__":
    main()
