"""Generate and validate the two Colab notebooks.

    .venv/bin/python tools/build_notebooks.py

Every cell's `source` keeps its trailing newlines (BRIEF.md: the nbformat newline rule).
Validation joins each code cell with "" (exactly as Jupyter does) and ast.parses it, then
runs nbformat.validate. Shell steps use subprocess, not `!` magics, so every code cell is
plain Python that ast can check.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import nbformat

REPO_URL = "https://github.com/brycedaniliuc/COMP2035-grape-calibration.git"
OUT = Path(__file__).resolve().parent.parent / "notebooks"


def _lines(src: str) -> list[str]:
    src = src.rstrip("\n")
    parts = src.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": _lines(src)}


def notebook(cells: list[dict]) -> dict:
    for i, c in enumerate(cells):  # nbformat 4.5 requires ids; stable so rebuilds diff cleanly
        c["id"] = f"cell-{i:02d}"
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}}}


SETTINGS = f'''# ---- settings: the only cell you normally edit ---------------------------------
REPO_URL = "{REPO_URL}"
REPO_REF = "main"                       # a branch, tag or commit hash
ZIP_PATH = "/content/drive/MyDrive/grape_dataset_for_colab.zip"   # must match Drive
DATA_ROOT = "/content/grape_dataset"    # the zip extracts to this folder
OUT = "/content/drive/MyDrive/COMP2035_runs"   # every output is written here, as it goes
'''

SETUP = '''import os, subprocess, sys, time
from pathlib import Path
T0 = time.time()

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")

REPO = Path("/content/COMP2035-grape-calibration")
if not REPO.exists():
    subprocess.run(["git", "clone", "--quiet", REPO_URL, str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", "--tags"], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", REPO_REF], check=True)
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
                        capture_output=True, text=True).stdout.strip()
os.chdir(REPO)
sys.path.insert(0, str(REPO))
print("code at", REPO_REF, commit)

# Colab already has torch, torchvision, numpy, scikit-learn, scikit-image and matplotlib.
# Do NOT pip install requirements.txt here: it pins the CPU build of torch.
import importlib.util
missing = [m for m in ("skimage", "sklearn", "nbformat") if importlib.util.find_spec(m) is None]
if missing:
    pkg = {"skimage": "scikit-image", "sklearn": "scikit-learn"}
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    *[pkg.get(m, m) for m in missing]], check=True)

from src.train import environment, select_device
import json
env = environment(select_device())
print(json.dumps(env, indent=1))
if env["device"] != "cuda":
    print("!! NO GPU: Runtime > Change runtime type > T4 GPU, then run all again")
os.makedirs(OUT, exist_ok=True)
print("outputs ->", OUT)
'''

DATA = '''# Fail early: unzip and check the exact per-folder counts BEFORE any training.
import shutil, zipfile
if not os.path.isdir(DATA_ROOT):
    assert os.path.exists(ZIP_PATH), (
        f"{ZIP_PATH} not found. Put grape_dataset_for_colab.zip at the top level of My Drive. "
        f"Files there: {sorted(os.listdir('/content/drive/MyDrive'))[:40]}")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall("/content/")
from src.data import check_supplied_dataset
check_supplied_dataset(DATA_ROOT)       # raises unless 7,222 train + 1,805 test, per class
print(f"elapsed {(time.time() - T0) / 60:.1f} min")
'''

NB1 = notebook([
    md("""# 01 · Baseline and calibration (ResNet-50, Colab T4)

Trains the baseline on the leak-free `source` split, then on the `supplied` split for the
leakage comparison. All code lives in `src/` of this repo; this notebook only calls it, so
Colab runs exactly the code the tests ran.

**Before running:** Runtime → Change runtime type → **T4 GPU**. Put
`grape_dataset_for_colab.zip` at the top level of My Drive. Then Runtime → Run all.

Every output goes to Google Drive (`OUT`) as it is produced, with the model, split and a
timestamp in its filename. See `HANDOVER.md` for timings and the files to check."""),
    code(SETTINGS),
    code(SETUP),
    code(DATA),
    md("## Dataset audit and leakage of the supplied split"),
    code('''from datetime import datetime
from src.audit import leaking_leaves, print_audit, supplied_split_audit
from src.data import index_dataset
records = index_dataset(DATA_ROOT)
audit = supplied_split_audit(records)
print_audit(audit, leaking_leaves(records))
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
with open(f"{OUT}/audit_{stamp}.json", "w") as fh:
    json.dump(audit, fh, indent=2)
print("written ->", f"{OUT}/audit_{stamp}.json")'''),
    md("""## Train on the `source` split (about 26 min on a T4)

The checkpoint is saved to Drive whenever validation loss improves, so an interrupted run
still leaves the best weights behind."""),
    code('''from src.config import Config
from src.train import run
cfg_source = Config(data_root=DATA_ROOT, out_dir=OUT, split_mode="source", model="resnet50")
res_source = run(cfg_source)
SOURCE_RESULTS = f"{OUT}/results_{res_source['run_id']}.json"
print("source run:", SOURCE_RESULTS, f"| elapsed {(time.time() - T0) / 60:.1f} min")'''),
    md("## Verify the source run (Tier 1 must pass; Tier 2 compares with September)"),
    code('''r = subprocess.run([sys.executable, "-m", "src.verify", "--run", SOURCE_RESULTS],
                   capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-3000:])
assert r.returncode == 0, "a Tier 1 check failed: stop and investigate before using this run"'''),
    md("## Train on the `supplied` split: the leakage comparison (about 18 min)"),
    code('''cfg_supplied = Config(data_root=DATA_ROOT, out_dir=OUT, split_mode="supplied",
                      model="resnet50")
res_supplied = run(cfg_supplied)
print("supplied run:", f"{OUT}/results_{res_supplied['run_id']}.json",
      f"| elapsed {(time.time() - T0) / 60:.1f} min")'''),
    md("## Everything this notebook wrote to Drive"),
    code('''for p in sorted(Path(OUT).iterdir()):
    print(f"{p.stat().st_size / 1e6:9.2f} MB  {p.name}")'''),
])

NB2 = notebook([
    md("""# 02 · Degradation study, coverage and threshold (Colab T4)

Loads the `source` checkpoint written by notebook 01 (no retraining). Applies blur,
brightening, darkening and occlusion at severities 0–5 at inference. It evaluates the fixed
temperature protocol (T = 1 primary; a fixed T grid; the LBFGS endpoint labelled as-fitted),
then derives coverage curves and the threshold, renders every figure, and verifies.

Run notebook 01 first. Runtime → Run all."""),
    code(SETTINGS + '''SOURCE_RESULTS = ""   # empty = the newest resnet50_source run in OUT
'''),
    code(SETUP),
    code(DATA),
    code('''if not SOURCE_RESULTS:  # the newest seed 42 baseline, never an ensemble member
    from src.ensemble import find_run
    found = find_run(OUT, seed=42)
    assert found, f"no finished resnet50 source run in {OUT}: run notebook 01 first"
    SOURCE_RESULTS = str(found)
print("using", SOURCE_RESULTS)'''),
    md("## Sweep: 4 corruptions × severities 0–5 on the 610 test originals, and on the 609 "
       "validation images that the threshold is chosen on"),
    code('''from src.evaluate import print_threshold, sweep
sw = sweep(SOURCE_RESULTS, data_root=DATA_ROOT)
print_threshold(sw["threshold"])
d = sw["degeneracy"]
print("validation errors:", d["n_val_errors"], "| NLL minimum at lower edge of T grid:",
      d["minimum_at_lower_edge"], "| LBFGS endpoint by max_iter:", d["lbfgs_endpoint_by_max_iter"])
print(f"elapsed {(time.time() - T0) / 60:.1f} min")'''),
    md("## Figures (paper PDF + slide PNG), written to Drive"),
    code('''from src.figures import corruption_grid, render, render_sweep
run_id = sw["run_id"]
fig_dir = f"{OUT}/figures_{run_id}"
paths = render_sweep(f"{OUT}/sweep_{run_id}.json", fig_dir, tag=run_id)
paths += render(corruption_grid(DATA_ROOT), "corruption_grid", fig_dir, tag=run_id)
print(len(paths), "files ->", fig_dir)'''),
    md("## Verify everything (Tier 1 must pass)"),
    code('''r = subprocess.run([sys.executable, "-m", "src.verify", "--run", SOURCE_RESULTS,
                    "--require-sweep"],
                   capture_output=True, text=True)
print(r.stdout[-8000:], r.stderr[-3000:])
assert r.returncode == 0, "a Tier 1 check failed: stop and investigate"
print(f"total elapsed {(time.time() - T0) / 60:.1f} min")'''),
    md("## Everything this notebook wrote to Drive"),
    code('''for p in sorted(Path(OUT).rglob("*")):
    if p.is_file() and run_id in p.name:
        print(f"{p.stat().st_size / 1e6:9.2f} MB  {p.relative_to(OUT)}")'''),
])


NB3 = notebook([
    md("""# 03 · Deep ensemble under corruption (upgrade, Colab T4)

Three ResNet-50 members on the same leak-free split, differing only in training seed:
**42 is the notebook 01 baseline, reused**; 43 and 44 are trained here (about 26 min each).
The ensemble predicts the mean softmax of the members. It goes through the same sweep as the
single model (4 corruptions × severities 0–5; Wilson and bootstrap intervals; coverage and
AURC for max softmax, entropy, margin, and the members' disagreement), and one figure
compares it with the single model at every severity.

**Needs:** notebook 01 finished (its seed 42 run in `OUT`). Notebook 02 is not needed; its
sweep is reused if it is there. **After a disconnect:** run all again. A member or sweep that
is already in Drive is reused, so only unfinished work is repeated."""),
    code(SETTINGS + '''MEMBER_SEEDS = (42, 43, 44)   # 42 = notebook 01's baseline, reused
'''),
    code(SETUP),
    code(DATA),
    md("## The seed 42 baseline from notebook 01"),
    code('''from src.ensemble import ensemble_sweep, ensure_member, ensure_sweep, find_run
BASE = find_run(OUT, seed=42)
assert BASE, f"no finished resnet50 source run in {OUT}: run notebook 01 first"
print("baseline:", BASE.name)'''),
    md("""## Members (about 26 min for each one trained)

A member already in Drive is reused. Each checkpoint is saved to Drive as it improves."""),
    code('''MEMBERS = []
for seed in MEMBER_SEEDS:
    MEMBERS.append(ensure_member(BASE, seed, data_root=DATA_ROOT))
    print(f"seed {seed}: {MEMBERS[-1].name} | elapsed {(time.time() - T0) / 60:.1f} min")'''),
    md("## Each member through the degradation sweep (about 6 min each, estimated)"),
    code('''for m in MEMBERS:
    ensure_sweep(m, data_root=DATA_ROOT)
print(f"elapsed {(time.time() - T0) / 60:.1f} min")'''),
    md("## The ensemble: mean softmax across members, in every condition"),
    code('''from src.evaluate import print_threshold
ENS = ensemble_sweep(MEMBERS, data_root=DATA_ROOT)
ENS_PATH = f"{OUT}/ensemble_{ENS['ens_id']}.json"
print_threshold(ENS["threshold"])'''),
    md("## Single model vs ensemble at every severity (paper PDF + slide PNG), to Drive"),
    code('''from src.figures import render_ensemble
paths = render_ensemble(ENS_PATH, f"{OUT}/figures_{ENS['ens_id']}", tag=ENS["ens_id"])
print(len(paths), "files ->", f"{OUT}/figures_{ENS['ens_id']}")'''),
    md("## Verify (Tier 1 must pass): every member, then the ensemble"),
    code('''for m in MEMBERS:
    r = subprocess.run([sys.executable, "-m", "src.verify", "--run", str(m), "--require-sweep"],
                       capture_output=True, text=True)
    print(m.name, "|", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
    assert r.returncode == 0, r.stdout[-6000:]
r = subprocess.run([sys.executable, "-m", "src.verify", "--ensemble", ENS_PATH],
                   capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-3000:])
assert r.returncode == 0, "a Tier 1 check failed: stop and investigate"
print(f"total elapsed {(time.time() - T0) / 60:.1f} min")'''),
    md("## Everything this notebook relies on or wrote, in Drive"),
    code('''ids = [ENS["ens_id"]] + [json.loads(m.read_text())["run_id"] for m in MEMBERS]
for p in sorted(Path(OUT).rglob("*")):
    if p.is_file() and any(i in p.name for i in ids):
        print(f"{p.stat().st_size / 1e6:9.2f} MB  {p.relative_to(OUT)}")'''),
])


def main() -> None:
    import json
    warnings.simplefilter("error")  # any nbformat validator warning fails the build
    for name, nb in (("01_baseline_calibration.ipynb", NB1),
                     ("02_degradation_study.ipynb", NB2),
                     ("03_deep_ensemble.ipynb", NB3)):
        path = OUT / name
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        # check the file as written: raw JSON, sources joined with ""
        raw = json.loads(path.read_text())
        for i, c in enumerate(raw["cells"]):
            assert isinstance(c["source"], list)
            assert all(s.endswith("\n") for s in c["source"][:-1]), f"{name} cell {i}"
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]))
        nbformat.validate(nbformat.reads(path.read_text(), as_version=4))
        n_code = sum(c["cell_type"] == "code" for c in raw["cells"])
        print(f"{name}: {len(raw['cells'])} cells, {n_code} code cells ast.parse OK "
              f"(joined with ''), nbformat.validate OK")


if __name__ == "__main__":
    main()
