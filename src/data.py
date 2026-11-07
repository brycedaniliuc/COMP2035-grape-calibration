"""Indexing, filename parsing, leaf grouping, the three split modes, and data loading.

Ported from notebook 01 (cells 9, 13, 15). Behaviour is unchanged except that problems
the notebook only warned about now raise (BRIEF.md: fail loudly):
  * a filename with no UUID  (notebook: treated as its own leaf, with a warning)
  * a leaf appearing under two classes  (notebook: silently overwritten)
Neither occurs in the supplied dataset.
"""

from __future__ import annotations

import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import Config

UUID_RE = re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})___", re.IGNORECASE)
AUG_RE = re.compile(r"(_flipLR|_90deg|_180deg|_270deg|_new30degFlipLR)$", re.IGNORECASE)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PARTS = ("train", "val", "test")

# The supplied dataset, verified on disk (BRIEF.md Stage 1): images per folder.
SUPPLIED_COUNTS = {
    ("train", "Grape___Black_rot"): 1888,
    ("train", "Grape___Esca_(Black_Measles)"): 1920,
    ("train", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"): 1722,
    ("train", "Grape___healthy"): 1692,
    ("test", "Grape___Black_rot"): 472,
    ("test", "Grape___Esca_(Black_Measles)"): 480,
    ("test", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"): 430,
    ("test", "Grape___healthy"): 423,
}


@dataclass(frozen=True)
class Record:
    path: str
    cls: str
    leaf: str
    aug: bool
    suffix: str
    partition: str  # top-level folder as supplied, lower-cased ("train" / "test")


def clean_class(name: str) -> str:
    """'Grape___Black_rot' -> 'Black_rot'."""
    return name.split("___")[-1] if "___" in name else name


def parse_filename(name: str) -> tuple[str, bool, str]:
    """Return (leaf UUID, is_augmented, suffix) for an image filename.

    AUG_RE ends with `$`, so it is matched against the stem: matching the full name
    (with `.JPG`) would classify every image as an original.
    """
    m = UUID_RE.match(name)
    if m is None:
        raise ValueError(f"filename has no source-leaf UUID: {name!r}")
    am = AUG_RE.search(Path(name).stem)
    return m.group(1).lower(), am is not None, am.group(1) if am else ""


def index_dataset(root: str | Path) -> list[Record]:
    """Index every image under root/<partition>/<class>/, in sorted order."""
    root = Path(root)
    assert root.is_dir(), f"data root not found: {root}"
    records = []
    for part_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cls_dir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            cls = clean_class(cls_dir.name)
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                leaf, aug, suffix = parse_filename(f.name)
                records.append(Record(str(f), cls, leaf, aug, suffix, part_dir.name.lower()))
    assert records, f"no images found under {root}"
    leaf_classes(records)  # raises if a leaf spans two classes
    return records


def leaf_classes(records: list[Record]) -> dict[str, str]:
    """Map leaf -> class. Raises if any leaf appears under more than one class."""
    out: dict[str, str] = {}
    for r in records:
        prev = out.setdefault(r.leaf, r.cls)
        if prev != r.cls:
            raise ValueError(f"leaf {r.leaf} appears under two classes: {prev}, {r.cls}")
    return out


def class_list(records: list[Record]) -> list[str]:
    return sorted({r.cls for r in records})


def index_summary(records: list[Record]) -> dict:
    return {
        "n_images": len(records),
        "n_source_leaves": len({r.leaf for r in records}),
        "n_originals": sum(not r.aug for r in records),
        "n_augmented": sum(r.aug for r in records),
        "augmentation_vocabulary": {(s or "(original)"): n for s, n in
                                    Counter(r.suffix for r in records).most_common()},
    }


# ---- splits ----------------------------------------------------------------

@dataclass
class Splits:
    train: list[Record] = field(default_factory=list)
    val: list[Record] = field(default_factory=list)
    test: list[Record] = field(default_factory=list)
    dropped_augmented: int = 0

    def __getitem__(self, name: str) -> list[Record]:
        return {"train": self.train, "val": self.val, "test": self.test}[name]

    def summary(self) -> dict:
        return {
            "split_sizes": {s: len(self[s]) for s in PARTS},
            "split_leaves": {s: len({r.leaf for r in self[s]}) for s in PARTS},
            "split_originals": {s: sum(not r.aug for r in self[s]) for s in PARTS},
            "dropped_augmented_from_eval": self.dropped_augmented,
        }


def make_splits(records: list[Record], cfg: Config) -> Splits:
    """Partition records. The sklearn calls are identical to notebook 01, so the
    `source` split reproduces the September split exactly for the same data and seed."""
    val_rel = cfg.val_frac / (cfg.val_frac + cfg.test_frac)
    sp = Splits()

    if cfg.split_mode == "source":
        lc = leaf_classes(records)
        leaves = sorted(lc)
        y = [lc[g] for g in leaves]
        g_tr, g_tmp, _, y_tmp = train_test_split(
            leaves, y, train_size=cfg.train_frac, stratify=y, random_state=cfg.seed)
        g_va, g_te, _, _ = train_test_split(
            g_tmp, y_tmp, train_size=val_rel, stratify=y_tmp, random_state=cfg.seed)
        s_tr, s_va = set(g_tr), set(g_va)
        for r in records:
            s = "train" if r.leaf in s_tr else ("val" if r.leaf in s_va else "test")
            if s != "train" and cfg.eval_on_originals_only and r.aug:
                sp.dropped_augmented += 1  # discarded, NOT moved to train
                continue
            sp[s].append(r)
        assert set(g_te) == {r.leaf for r in sp.test}, "test leaves lost during assignment"

    elif cfg.split_mode == "naive":
        y = [r.cls for r in records]
        i_tr, i_tmp, _, y_tmp = train_test_split(
            range(len(records)), y, train_size=cfg.train_frac, stratify=y,
            random_state=cfg.seed)
        i_va, i_te, _, _ = train_test_split(
            i_tmp, y_tmp, train_size=val_rel, stratify=y_tmp, random_state=cfg.seed)
        sp.train = [records[i] for i in i_tr]
        sp.val = [records[i] for i in i_va]
        sp.test = [records[i] for i in i_te]

    else:  # supplied
        supplied_train = [r for r in records if r.partition.startswith("train")]
        sp.test = [r for r in records if r.partition.startswith("test")]
        y = [r.cls for r in supplied_train]
        i_tr, i_va, _, _ = train_test_split(
            range(len(supplied_train)), y, train_size=1 - cfg.val_frac, stratify=y,
            random_state=cfg.seed)
        sp.train = [supplied_train[i] for i in i_tr]
        sp.val = [supplied_train[i] for i in i_va]

    check_splits(sp, records, cfg)
    return sp


def check_splits(sp: Splits, records: list[Record], cfg: Config) -> None:
    """Mandatory assertions. Leaf overlap is forbidden in `source` mode only: in
    `naive` and `supplied` mode it is the thing being measured."""
    classes = class_list(records)
    for s in PARTS:
        assert sp[s], f"{s} partition is empty"
        assert class_list(sp[s]) == classes, f"{s} partition is missing a class"
    paths = [r.path for s in PARTS for r in sp[s]]
    assert len(paths) == len(set(paths)), "an image appears in two partitions"
    assert len(paths) + sp.dropped_augmented == len(records), "images lost or invented"

    if cfg.split_mode == "source":
        tr, va, te = ({r.leaf for r in sp[s]} for s in PARTS)
        assert not (tr & va), "grouping failed: leaf in both train and val"
        assert not (tr & te), "grouping failed: leaf in both train and test"
        assert not (va & te), "grouping failed: leaf in both val and test"
        if cfg.eval_on_originals_only:
            assert all(not r.aug for r in sp.val + sp.test), \
                "augmented image found in an evaluation partition"
        # Augmented copies of eval leaves were dropped, never moved: train holds
        # exactly the images of train leaves.
        assert len(sp.train) == sum(r.leaf in tr for r in records)


def save_split_lists(sp: Splits, data_root: str | Path) -> dict[str, list[str]]:
    """Image paths per partition, relative to data_root, for the run's output."""
    return {s: [os.path.relpath(r.path, data_root) for r in sp[s]] for s in PARTS}


# ---- transforms and loaders --------------------------------------------------

def build_transforms(cfg: Config) -> tuple[transforms.Compose, transforms.Compose]:
    """Exactly those of Canoski et al., Section 5.2 (notebook 01, cell 15)."""
    train_tf = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.RandomCrop(cfg.img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


class LeafDataset(Dataset):
    def __init__(self, rows: list[Record], tf: transforms.Compose, cls2idx: dict[str, int]):
        self.rows, self.tf, self.cls2idx = rows, tf, cls2idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        r = self.rows[i]
        with Image.open(r.path) as im:
            img = im.convert("RGB")
        return self.tf(img), self.cls2idx[r.cls]


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(rows: list[Record], tf: transforms.Compose, cls2idx: dict[str, int],
                cfg: Config, device: torch.device, shuffle: bool) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    return DataLoader(LeafDataset(rows, tf, cls2idx), batch_size=cfg.batch_size,
                      shuffle=shuffle, num_workers=cfg.num_workers,
                      pin_memory=device.type == "cuda",
                      persistent_workers=cfg.num_workers > 0, drop_last=False,
                      generator=g, worker_init_fn=_seed_worker)


def check_supplied_dataset(root: str | Path) -> None:
    """Fail early: the full dataset must have exactly the verified per-folder counts.
    Run this before training, not after 26 minutes of it."""
    root = Path(root)
    got = {}
    for (part, cls_dir) in SUPPLIED_COUNTS:
        d = root / part / cls_dir
        assert d.is_dir(), f"missing folder: {d}"
        got[(part, cls_dir)] = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
    bad = {k: (v, SUPPLIED_COUNTS[k]) for k, v in got.items() if v != SUPPLIED_COUNTS[k]}
    assert not bad, f"dataset counts differ (found, expected): {bad}"
    print(f"dataset OK: {sum(got.values()):,} images in 8 folders, counts as verified")
