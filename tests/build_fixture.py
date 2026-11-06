"""Build tests/fixture/ by copying real images out of the supplied grape dataset.

Usage:
    .venv/bin/python tests/build_fixture.py "/path/to/grape_dataset"

Reads filenames only; never opens an image. The source folder is read, never written.
Selection is deterministic (sorted candidates, random.Random(SEED)), so re-running
produces the same fixture. Writes MANIFEST.md and manifest.json next to the images.

Stdlib only on purpose: this script must not depend on src/, which it is used to test.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
PER_CLASS = 20
MAX_BYTES = 5 * 1024 * 1024

UUID_RE = re.compile(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})___", re.I)
AUG_RE = re.compile(r"(_flipLR|_90deg|_180deg|_270deg|_new30degFlipLR)$", re.I)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Verified on disk (BRIEF.md, Stage 1). Indexing that does not reproduce these is wrong.
EXPECTED_COUNTS = {
    ("train", "Grape___Black_rot"): 1888,
    ("train", "Grape___Esca_(Black_Measles)"): 1920,
    ("train", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"): 1722,
    ("train", "Grape___healthy"): 1692,
    ("test", "Grape___Black_rot"): 472,
    ("test", "Grape___Esca_(Black_Measles)"): 480,
    ("test", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"): 430,
    ("test", "Grape___healthy"): 423,
}
PARTS = ("train", "test")

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixture"


def index(root: Path) -> list[dict]:
    recs = []
    skipped = Counter()
    for part in PARTS:
        for cls_dir in sorted(p for p in (root / part).iterdir() if p.is_dir()):
            for f in sorted(cls_dir.iterdir()):
                if not f.is_file() or f.suffix.lower() not in IMG_EXT:
                    skipped[f.name] += 1
                    continue
                m = UUID_RE.match(f.name)
                assert m, f"filename without UUID: {f}"
                am = AUG_RE.search(f.stem)  # stem, not name: AUG_RE ends with $
                recs.append(dict(
                    path=f, part=part, cls_dir=cls_dir.name, leaf=m.group(1).lower(),
                    aug=bool(am), suffix=am.group(1) if am else "", size=f.stat().st_size,
                ))
    if skipped:
        print(f"skipped non-image entries: {dict(skipped)}")
    return recs


def audit(recs: list[dict]) -> dict:
    """Same quantities as notebook cell 11, computed independently here."""
    tr = {r["leaf"] for r in recs if r["part"] == "train"}
    te = {r["leaf"] for r in recs if r["part"] == "test"}
    test_orig = [r for r in recs if r["part"] == "test" and not r["aug"]]
    comp = [r for r in test_orig if r["leaf"] in tr]
    per_class = {}
    for c in sorted({r["cls_dir"] for r in recs}):
        t = {r["leaf"] for r in recs if r["cls_dir"] == c and r["part"] == "train"}
        s = {r["leaf"] for r in recs if r["cls_dir"] == c and r["part"] == "test"}
        per_class[c] = dict(test_leaves=len(s), also_in_train=len(t & s),
                            pct=round(100 * len(t & s) / max(len(s), 1), 1))
    return dict(
        n_images=len(recs),
        n_source_leaves=len({r["leaf"] for r in recs}),
        n_originals=sum(not r["aug"] for r in recs),
        n_augmented=sum(r["aug"] for r in recs),
        train_leaves=len(tr), test_leaves=len(te),
        test_leaves_also_in_train=len(tr & te),
        pct_test_leaves_also_in_train=round(100 * len(tr & te) / max(len(te), 1), 1),
        n_test_originals=len(test_orig),
        n_test_originals_leaf_in_train=len(comp),
        pct_test_originals_leaf_in_train=round(100 * len(comp) / max(len(test_orig), 1), 1),
        per_class=per_class,
    )


def select(recs: list[dict]) -> list[tuple[dict, str]]:
    """Return (record, role) pairs. Roles are documented in MANIFEST.md."""
    rng = random.Random(SEED)
    by_leaf: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_leaf[r["leaf"]].append(r)
    chosen: list[tuple[dict, str]] = []

    for cls in sorted({r["cls_dir"] for r in recs}):
        leaves = sorted(lf for lf, rs in by_leaf.items() if rs[0]["cls_dir"] == cls)
        used: set[str] = set()
        picks: list[tuple[dict, str]] = []

        def orig(lf, part):
            return [r for r in by_leaf[lf] if not r["aug"] and r["part"] == part]

        def augs(lf, part):
            return [r for r in by_leaf[lf] if r["aug"] and r["part"] == part]

        # 1. LEAK: two real cross-partition leaves from the supplied split, kept in
        #    their true partitions; the audit must find both.
        #    (a) original in test, augmented copy in train
        #    (b) original in train, augmented copy in test
        for o_part, a_part in (("test", "train"), ("train", "test")):
            cands = [lf for lf in leaves if lf not in used
                     and orig(lf, o_part) and augs(lf, a_part)]
            lf = rng.choice(cands)
            picks.append((orig(lf, o_part)[0], "leak"))
            picks.append((augs(lf, a_part)[0], "leak"))
            used.add(lf)

        # 2. GROUP: original and an augmented copy, both in train, no copy in test.
        #    Exercises leaf grouping in the source split.
        cands = [lf for lf in leaves if lf not in used and orig(lf, "train") and augs(lf, "train")
                 and not any(r["part"] == "test" for r in by_leaf[lf])]
        for lf in rng.sample(cands, 3):
            picks.append((orig(lf, "train")[0], "group"))
            picks.append((augs(lf, "train")[0], "group"))
            used.add(lf)

        # 3. CLEAN-TEST: original in test, no copy of the leaf anywhere in train.
        #    Healthy has none in the real data (100% leaf overlap), so none is picked.
        cands = [lf for lf in leaves if lf not in used and orig(lf, "test")
                 and not any(r["part"] == "train" for r in by_leaf[lf])]
        if cls == "Grape___healthy":
            assert not cands, "expected 100% healthy test-leaf overlap in the real data"
        else:
            lf = rng.choice(cands)
            picks.append((orig(lf, "test")[0], "clean-test"))
            used.add(lf)

        # 4. FILL: single originals, alternating train/test, to reach PER_CLASS images.
        #    Only the original is copied, so these leaves are single-image in the fixture.
        pools = {p: [lf for lf in leaves if lf not in used and orig(lf, p)] for p in PARTS}
        for p in PARTS:
            rng.shuffle(pools[p])
        i = 0
        while len(picks) < PER_CLASS:
            p = PARTS[i % 2]
            lf = pools[p].pop()
            if lf in used:
                continue
            picks.append((orig(lf, p)[0], "fill"))
            used.add(lf)
            i += 1
        chosen += picks
    return chosen


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    assert root.is_dir(), f"not a directory: {root}"
    assert FIXTURE.resolve() != root and REPO not in root.parents, "refusing: source inside repo"

    recs = index(root)

    # Indexing gate: exact per-folder counts.
    counts = Counter((r["part"], r["cls_dir"]) for r in recs)
    assert dict(counts) == EXPECTED_COUNTS, f"count mismatch: {dict(counts)}"
    print(f"indexed {len(recs):,} images; per-folder counts match BRIEF.md exactly")

    # The AUG_RE trap, shown on real names: matching the full name finds nothing.
    n_full = sum(bool(AUG_RE.search(r["path"].name)) for r in recs)
    print(f"augmented by stem: {sum(r['aug'] for r in recs):,}  |  by full name: {n_full:,}")
    assert n_full == 0 and sum(r["aug"] for r in recs) > 0

    # One leaf, one class, in the full dataset.
    leaf_cls: dict[str, set] = defaultdict(set)
    for r in recs:
        leaf_cls[r["leaf"]].add(r["cls_dir"])
    assert all(len(v) == 1 for v in leaf_cls.values()), "a leaf spans two classes"

    full = audit(recs)
    print("full-dataset audit, re-derived from filenames:")
    print(json.dumps(full, indent=2))

    chosen = select(recs)

    # Rebuild only the two partition folders inside the fixture.
    for p in PARTS:
        d = FIXTURE / p
        if d.exists():
            assert d.resolve().is_relative_to(FIXTURE.resolve())
            shutil.rmtree(d)
    for r, _ in chosen:
        dst = FIXTURE / r["part"] / r["cls_dir"] / r["path"].name
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert not dst.exists(), f"duplicate selection: {dst}"
        shutil.copyfile(r["path"], dst)

    fx = [r for r, _ in chosen]
    fx_audit = audit(fx)
    total_bytes = sum(r["size"] for r in fx)

    # Fixture requirements (BRIEF.md, Stage 1).
    assert total_bytes < MAX_BYTES, f"fixture is {total_bytes} bytes"
    for p in PARTS:
        o = sum(not r["aug"] for r in fx if r["part"] == p)
        a = sum(r["aug"] for r in fx if r["part"] == p)
        print(f"fixture {p}: {o} originals, {a} augmented")
        assert o > 0 and a > 0, f"fixture {p} needs both originals and augmented copies"
    leaked = sorted({r["leaf"] for r, role in chosen if role == "leak"})
    assert len(leaked) >= 2
    for cls in sorted({r["cls_dir"] for r in fx}):
        multi = [lf for lf in {r["leaf"] for r in fx if r["cls_dir"] == cls}
                 if {r["aug"] for r in fx if r["leaf"] == lf} == {True, False}]
        assert len(multi) >= 3, f"{cls}: only {len(multi)} leaves with original + augmented"

    write_manifest(root, chosen, fx_audit, full, total_bytes)
    print(f"\nfixture written: {len(fx)} images, {total_bytes / 1024:.0f} KiB")
    print(json.dumps(fx_audit, indent=2))


def write_manifest(root: Path, chosen, fx_audit: dict, full: dict, total_bytes: int) -> None:
    rows = [dict(partition=r["part"], class_dir=r["cls_dir"], filename=r["path"].name,
                 leaf=r["leaf"], augmented=r["aug"], suffix=r["suffix"], role=role,
                 bytes=r["size"]) for r, role in chosen]
    leak_leaves = sorted({x["leaf"] for x in rows if x["role"] == "leak"})
    (FIXTURE / "manifest.json").write_text(json.dumps(dict(
        source_root=str(root), seed=SEED, n_images=len(rows), bytes=total_bytes,
        leak_leaves=leak_leaves, expected_fixture_audit=fx_audit,
        full_dataset_audit=full, files=rows), indent=2) + "\n")

    L = ["# Test fixture — manifest", "",
         "Real images copied byte-for-byte from the supplied grape dataset by",
         "`tests/build_fixture.py` (seed 42). Filenames are preserved exactly. Each file sits",
         "in the same `{train,test}/Grape___<class>/` folder it occupies in the supplied split.",
         "Regenerate with:", "",
         "```bash",
         '.venv/bin/python tests/build_fixture.py "/Users/bryce/Downloads/Literature Review paper/Grape/grape_dataset"',
         "```", "",
         "## Roles", "",
         "| Role | Per class | What it exercises |", "|---|---|---|",
         "| `leak` | 2 leaves × 2 files | One leaf with its original in `test/` and an augmented copy in `train/`, and one the other way round. These are real cross-partition leaves from the supplied split, kept where they really are, so the audit must find them. |",
         "| `group` | 3 leaves × 2 files | Original and augmented copy of one leaf, both in `train/`. Grouped splitting must keep each pair together. |",
         "| `clean-test` | 1 leaf (not healthy) | Original in `test/` with no copy of that leaf in `train/` anywhere in the real data. Healthy has no such leaf, because its real test-leaf overlap is 100%. |",
         "| `fill` | the rest, to 20 per class | Single originals, alternating `train/` and `test/`. Only the original is copied. |",
         "",
         f"## Deliberate leaks ({len(leak_leaves)} leaves)", ""]
    for lf in leak_leaves:
        fs = [x for x in rows if x["leaf"] == lf]
        L.append(f"- `{lf}` ({fs[0]['class_dir']}): " + "; ".join(
            f"`{x['partition']}/{x['filename']}`" for x in fs))
    a = fx_audit
    L += ["", "## Expected audit on this fixture", "",
          "Stage 2's audit must reproduce these exactly. They are also in `manifest.json`.", "",
          "| Quantity | Value |", "|---|---|",
          f"| images | {a['n_images']} |", f"| source leaves | {a['n_source_leaves']} |",
          f"| originals / augmented | {a['n_originals']} / {a['n_augmented']} |",
          f"| train leaves / test leaves | {a['train_leaves']} / {a['test_leaves']} |",
          f"| test leaves also in train | {a['test_leaves_also_in_train']} ({a['pct_test_leaves_also_in_train']}%) |",
          f"| test originals whose leaf is in train | {a['n_test_originals_leaf_in_train']} of {a['n_test_originals']} ({a['pct_test_originals_leaf_in_train']}%) |",
          "", "Per class, test leaves also in train:", ""]
    for c, v in a["per_class"].items():
        L.append(f"- {c}: {v['also_in_train']} of {v['test_leaves']} ({v['pct']}%)")
    L += ["", f"Total size: {total_bytes:,} bytes.", "",
          "## Files", "", "| Partition | Class | Role | Aug | Filename |", "|---|---|---|---|---|"]
    for x in rows:
        L.append(f"| {x['partition']} | {x['class_dir']} | {x['role']} | "
                 f"{x['suffix'] or '—'} | `{x['filename']}` |")
    (FIXTURE / "MANIFEST.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
