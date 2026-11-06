"""Dataset audit and leakage report for the split *as supplied* (notebook 01, cell 11).

The output dict has the same keys as results/*/audit.json.

    python -m src.audit --data-root "/path/to/grape_dataset"
"""

from __future__ import annotations

import argparse
import json

from src.data import Record, class_list, index_dataset, index_summary


def _train(r: Record) -> bool:
    return r.partition.startswith("train")


def _test(r: Record) -> bool:
    return r.partition.startswith("test")


def leaking_leaves(records: list[Record]) -> list[str]:
    """Leaves with at least one image in the supplied train/ and one in test/."""
    return sorted({r.leaf for r in records if _train(r)} & {r.leaf for r in records if _test(r)})


def supplied_split_audit(records: list[Record]) -> dict:
    tr = {r.leaf for r in records if _train(r)}
    te = {r.leaf for r in records if _test(r)}
    test_originals = [r for r in records if _test(r) and not r.aug]
    compromised = [r for r in test_originals if r.leaf in tr]

    audit = index_summary(records)
    audit.update({
        "supplied_train_leaves": len(tr),
        "supplied_test_leaves": len(te),
        "test_leaves_also_in_train": len(tr & te),
        "pct_test_leaves_also_in_train": round(100 * len(tr & te) / max(len(te), 1), 1),
        "n_test_originals": len(test_originals),
        "n_test_originals_leaf_in_train": len(compromised),
        "pct_test_originals_leaf_in_train":
            round(100 * len(compromised) / max(len(test_originals), 1), 1),
        "per_class_pct_test_leaves_also_in_train": {},
    })
    for c in class_list(records):
        t = {r.leaf for r in records if r.cls == c and _train(r)}
        s = {r.leaf for r in records if r.cls == c and _test(r)}
        audit["per_class_pct_test_leaves_also_in_train"][c] = \
            round(100 * len(t & s) / max(len(s), 1), 1)
    return audit


def print_audit(audit: dict, leaks: list[str], max_list: int = 20) -> None:
    print(f"images                 : {audit['n_images']:,}")
    print(f"source leaves          : {audit['n_source_leaves']:,}")
    print(f"original photographs   : {audit['n_originals']:,}")
    print(f"augmented copies       : {audit['n_augmented']:,}")
    print(f"source leaves in supplied train : {audit['supplied_train_leaves']:,}")
    print(f"source leaves in supplied test  : {audit['supplied_test_leaves']:,}")
    print(f"  of which also in train        : {audit['test_leaves_also_in_train']:,}"
          f"  ({audit['pct_test_leaves_also_in_train']}%)")
    print(f"original photographs in test    : {audit['n_test_originals']:,}")
    print(f"  whose leaf appears in train   : {audit['n_test_originals_leaf_in_train']:,}"
          f"  ({audit['pct_test_originals_leaf_in_train']}%)")
    print("leaf overlap by class:")
    for c, pct in audit["per_class_pct_test_leaves_also_in_train"].items():
        print(f"  {c:<38} {pct:5.1f}%")
    if len(leaks) <= max_list:
        print(f"leaking leaves ({len(leaks)}):")
        for lf in leaks:
            print(f"  {lf}")
    else:
        print(f"leaking leaves: {len(leaks):,} (too many to list)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", help="write the audit JSON here")
    a = ap.parse_args()
    records = index_dataset(a.data_root)
    audit = supplied_split_audit(records)
    print_audit(audit, leaking_leaves(records))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(audit, fh, indent=2)
        print(f"written -> {a.out}")


if __name__ == "__main__":
    main()
