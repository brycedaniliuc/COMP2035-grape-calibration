"""Bring the Colab outputs home in one command (RUNSHEET.md, step 9).

    .venv/bin/python tools/import_colab.py ~/Downloads/COMP2035_runs

1. Copies every output in the Drive folder into results/colab_<date>/, except checkpoints
   (*.pt, which stay in Drive). A file already there with different contents stops the
   import: nothing is ever overwritten.
2. Regenerates every number in the report (tools/make_tables.py) from the seed 42 source run,
   its sweep and its PlantSeg analysis, and the supplied run. Anything that is not a result
   (src/provenance.py) renders as [pending: ...].
3. Runs src.verify on every run, ensemble, PlantSeg and MC dropout output, and on the
   regenerated numbers, and prints one pass/fail table. Exit status 1 if anything failed.

<date> is the date of the seed 42 source run (from its run id), or --date.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.provenance import not_a_result  # noqa: E402
from tools.make_tables import build  # noqa: E402

PREFIXES = ("results_", "logits_", "splits_", "sweep_", "audit_", "ensemble_", "plantseg_",
            "mcdropout_")
STAMP = re.compile(r"_(\d{8})-\d{6}")


def outputs(drive: Path) -> list[Path]:
    """Everything worth keeping: run outputs and figure folders, never checkpoints."""
    keep = []
    for p in sorted(drive.iterdir()):
        if p.is_dir() and p.name.startswith("figures_"):
            keep.append(p)
        elif p.is_file() and p.name.startswith(PREFIXES) and p.suffix in (".json", ".npz"):
            keep.append(p)
    return keep


def copy(src: Path, dst: Path) -> str:
    if dst.exists():
        if src.is_dir():
            return "kept (folder exists)" if _same_tree(src, dst) else "CONFLICT"
        return "already here" if filecmp.cmp(src, dst, shallow=False) else "CONFLICT"
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "copied"


def _same_tree(a: Path, b: Path) -> bool:
    fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    return fa == fb and all(filecmp.cmp(a / f, b / f, shallow=False) for f in fa)


def seed42_source(d: Path) -> Path | None:
    """The newest source run trained with the split seed (42): the single model."""
    found = []
    for p in sorted(d.glob("results_*_source_*.json")):
        cfg = json.loads(p.read_text())["config"]
        if cfg.get("train_seed") is None and cfg.get("split_mode") == "source":
            found.append(p)
    return found[-1] if found else None


def newest(d: Path, pattern: str) -> Path | None:
    found = sorted(d.glob(pattern))
    return found[-1] if found else None


def verify(args: list[str]) -> tuple[str, str, int]:
    r = subprocess.run([sys.executable, "-m", "src.verify", *args], cwd=REPO,
                       capture_output=True, text=True)
    m = re.search(r"Tier 1: (\d+) passed, (\d+) failed, (\d+) skipped", r.stdout)
    summary = (f"{m.group(1)} passed, {m.group(2)} failed, {m.group(3)} skipped" if m
               else (r.stderr.strip().splitlines() or ["no output"])[-1][:90])
    fails = [line.strip() for line in r.stdout.splitlines() if " FAIL " in line]
    return summary, "\n      ".join(fails[:5]), r.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="the COMP2035_runs folder from Google Drive")
    ap.add_argument("--date", help="results/colab_<date>/ (default: the seed 42 run's date)")
    ap.add_argument("--results-root", default=str(REPO / "results"))
    ap.add_argument("--generated", default=str(REPO / "report" / "generated"))
    a = ap.parse_args(argv)
    drive = Path(a.drive).expanduser()
    assert drive.is_dir(), f"{drive} is not a folder: download COMP2035_runs from Drive first"
    found = outputs(drive)
    assert found, f"no Colab outputs in {drive}"

    first = seed42_source(drive)
    stamp = STAMP.search(first.name) if first else None
    date = a.date or (stamp.group(1) if stamp else None)
    assert date, "cannot tell the run date: give --date YYYYmmdd"
    dest = Path(a.results_root) / f"colab_{date}"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"1. copying {len(found)} outputs from {drive}\n   into {dest}")
    conflicts = []
    for p in found:
        status = copy(p, dest / p.name)
        if status == "CONFLICT":
            conflicts.append(p.name)
        print(f"   {status:<20} {p.name}")
    if conflicts:
        print(f"\nSTOP: {len(conflicts)} file(s) already in {dest} with different contents: "
              f"{conflicts}. Nothing was overwritten. Use another --date, or check by hand.")
        return 1

    base = seed42_source(dest)
    supplied = newest(dest, "results_*_supplied_*.json")
    sweep: Path | None = None
    plantseg: Path | None = None
    if base is not None:
        rid = json.loads(base.read_text())["run_id"]
        sweep = dest / f"sweep_{rid}.json" if (dest / f"sweep_{rid}.json").exists() else None
        plantseg = (dest / f"plantseg_{rid}.json" if (dest / f"plantseg_{rid}.json").exists()
                    else None)
    print("\n2. report numbers (tools/make_tables.py)")
    for what, chosen in (("single model", base), ("supplied", supplied), ("sweep", sweep),
                         ("plantseg", plantseg)):
        label = not_a_result(json.loads(chosen.read_text())) if chosen and what in (
            "single model", "supplied") else None
        print(f"   {what:<13} {chosen.name if chosen else '-- not run'}"
              f"{'   <- ' + label if label else ''}")
    build(base, supplied, sweep, Path(a.generated), plantseg=plantseg)

    print("\n3. verification (src.verify)")
    targets: list[tuple[str, list[str]]] = []
    for p in sorted(dest.glob("results_*.json")):
        rid = json.loads(p.read_text())["run_id"]
        req = ["--require-sweep"] if (dest / f"sweep_{rid}.json").exists() else []
        targets.append((p.name, ["--run", str(p), *req]))
    for pattern, flag in (("ensemble_*.json", "--ensemble"), ("plantseg_*.json", "--plantseg"),
                          ("mcdropout_*.json", "--mcdropout")):
        for p in sorted(dest.glob(pattern)):
            if flag == "--mcdropout" and p.name.startswith("mcdropout_train_"):
                continue
            targets.append((p.name, [flag, str(p)]))
    targets.append(("report numbers", ["--generated", a.generated]))
    rows, ok = [], True
    for name, args in targets:
        summary, fails, code = verify(args)
        ok &= code == 0
        rows.append((name, summary, "PASS" if code == 0 else "FAIL", fails))
    w = max(len(r[0]) for r in rows)
    print(f"\n   {'what':<{w}}  {'Tier 1':<34} result")
    for name, summary, result, fails in rows:
        print(f"   {name:<{w}}  {summary:<34} {result}")
        if fails:
            print(f"      {fails}")
    print(f"\n{'ALL PASS' if ok else 'SOMETHING FAILED: do not use the failed outputs'}"
          f" ({len(rows)} checked).")
    if ok:
        def short(p: Path) -> str:
            return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
        print(f"Next: git add {short(dest)} {short(Path(a.generated))} && "
              "git commit -m 'Import the Colab run' && git push")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
