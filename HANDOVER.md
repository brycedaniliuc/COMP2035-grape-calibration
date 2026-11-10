# HANDOVER — running the real experiment in Google Colab

**For the day itself, follow `RUNSHEET.md`:** one page of numbered steps. This file is the
reference behind it: what each cell does, what lands in Drive, and why.

This is the one step that needs a GPU, and the step where the September checkpoint and logits
were lost. Every output goes to Google Drive **as it is produced**, under run-tagged
filenames, so nothing depends on downloading from the Colab session.

## The notebooks: order, dependencies, and what is essential

Run them in this order. **01 and 02 are essential**: together they are the complete core
project (the leak-free baseline, the degradation study, coverage curves, the threshold, every
core figure). **03 and 04 are upgrades**, and **05 is optional**, run last and only if 01–04
went cleanly. Each upgrade is a separate
notebook that saves to Drive as it goes, so a disconnect costs at most the notebook it
happens in. If the session runs out after 02, the project is still complete. 03 and 04 do
not depend on each other: run them in either order, or only one.

Wall times say what they rest on: "September" is a measured T4 run, "measured locally" is a
CPU run of the same code on this machine, and "estimate" is neither.

| # | Notebook | Status | Needs in Drive | Wall time on a T4 (basis) |
|---|---|---|---|---|
| 01 | `01_baseline_calibration` | **essential** | the dataset zip | about 45 min (September: 25.9 + 17.8 min of training) |
| 02 | `02_degradation_study` | **essential** | 01's seed 42 source run | about 10 min (estimate: sweep about 6 min, see §2) |
| 03 | `03_deep_ensemble` | upgrade | 01's seed 42 source run (02 optional) | about 70 min (estimate: 2 × 26 min training + 3 sweeps) |
| 04 | `04_plantseg_field` | upgrade | 01's seed 42 run **and** 02's sweep (no PlantVillage zip needed) | about 5 min (estimate: a 1.7 GB download, then 334 images; measured locally: 95 s on CPU for the analysis) |
| 05 | `05_mc_dropout` | **optional, last** | 01's seed 42 run and 02's sweep (03 optional) | about 15 min (estimate: 5 head epochs, then features in 24 conditions) |

If a session dies in an upgrade notebook, re-run that notebook: anything it already finished
in Drive (a trained member, a sweep) is reused, not repeated.

## 0 · Before you start

1. **The code comes from GitHub.** The notebooks clone
   `https://github.com/brycedaniliuc/COMP2035-grape-calibration` (public) at `REPO_REF`, which
   is `main` by default. Leave it at `main`. There is no release tag yet; the old `v1.0` tag
   was deleted because it chose the threshold on the test images (DECISIONS.md D10).
2. **The dataset zip must be in Drive.** Upload
   `~/Downloads/Literature Review paper/Grape/grape_dataset_for_colab.zip` (160.6 MB) to the
   **top level of My Drive**, with exactly that name. The September runs used this same zip,
   extracted to `/content/grape_dataset`. The notebooks then check the exact count in each of
   the 8 class folders (7,222 train and 1,805 test images in total) **before any training**.
   A wrong or partial zip fails in the first minute, not at minute 26. Without this zip, build
   the same layout from the table in the README ("The dataset").
3. In Colab: **Runtime → Change runtime type → T4 GPU**.

## 1 · Notebook 01 — `notebooks/01_baseline_calibration.ipynb`

Open it from GitHub in Colab (File → Open notebook → GitHub). Run the cells in order
(Runtime → Run all works). Each wall time says what it rests on: a September measurement,
or an estimate marked as one.

| # | Cell | What it does | Expected wall time |
|---|---|---|---|
| 1 | settings | `REPO_REF`, `ZIP_PATH`, `OUT` (the default `OUT` is `MyDrive/COMP2035_runs`) | instant |
| 2 | setup | mounts Drive, clones the repo at `REPO_REF`, prints the commit, device and library versions | about 1 min (not yet measured on Colab) |
| 3 | data | unzips, then `check_supplied_dataset` (fail early) | about 1 min (not yet measured) |
| 4 | audit | the leakage audit, written to Drive | seconds |
| 5 | **train `source`** | ResNet-50, 5 + 25 epochs, patience 5 | **about 26 min** (September: 25.94 min on a T4; deterministic cuDNN may add some time, not measured) |
| 6 | verify | `src.verify` Tier 1 (must pass) and Tier 2 (vs September) | about 1 min |
| 7 | **train `supplied`** | the leakage comparison | **about 18 min** (September: 17.8 min) |
| 8 | list | prints every file written to Drive | seconds |

## 2 · Notebook 02 — `notebooks/02_degradation_study.ipynb`

| # | Cell | What it does | Expected wall time |
|---|---|---|---|
| 1–3 | settings, setup, data | as above. Leave `SOURCE_RESULTS` empty to pick the newest source run | about 2 min |
| 4 | locate | finds 01's seed 42 source run in `OUT`, never an ensemble member | instant |
| 5 | **sweep** | 4 corruptions × severities 0–5 on the 610 test originals and the 609 validation images; all temperature conditions; bootstrap CIs; threshold chosen on validation, checked on test | **about 6 min** (estimate, not measured on Colab; see the note below) |
| 6 | figures | 9 figures × (paper PDF + slide PNG) into `OUT/figures_<run_id>/` | about 1 min (not yet measured) |
| 7 | verify | Tier 1 must pass | about 1 min |
| 8 | list | prints every file of this run | seconds |

**Sweep timing, measured and estimated.** The sweep was smoke-run locally on the real dataset
(CPU, Apple M2 Pro, a one-epoch EfficientNet-B0 standing in for ResNet-50), before the
validation images were added to it. It took **92 min**, almost all of it CPU inference:
24 conditions × 610 images, about 0.38 s per image. The other parts were timed separately:
all 144 bootstrap CIs take about 0.5 min, and the corruptions take at most 12 ms per image.
On a T4, inference stops being the limit, and the limit becomes how fast Colab can load and
decode images. In September, Colab fed about 130 images/s during training (30 epochs of
about 6,900 images in 25.9 min). The sweep now degrades the validation images too, for
choosing the threshold: 24 × (610 + 609) = 29,256 images. At that rate they take about
4 min, plus the bootstrap, hence "about 6 min". If the cell is still running after 20 min,
check that cell 2 printed a CUDA device.

## 2b · Notebook 03 — `notebooks/03_deep_ensemble.ipynb` (upgrade)

Needs notebook 01's seed 42 source run in `OUT`. That run **is** ensemble member 1 (not
retrained). Members 2 and 3 are trained with the identical config and split, only the
training seed changed (`train_seed` 43 and 44; DECISIONS.md D13).

| # | Cell | What it does | Expected wall time |
|---|---|---|---|
| 1–3 | settings, setup, data | as above | about 2 min |
| 4 | baseline | finds 01's seed 42 run (`find_run`), never an ensemble member | instant |
| 5 | **members** | trains seeds 43 and 44, or reuses them if already in Drive | **about 52 min** (2 × September's 25.9 min; not measured with this code) |
| 6 | **sweeps** | each member through the full sweep (seed 42's from 02 is reused if present) | **about 12–18 min** (estimate: 2–3 × about 6 min) |
| 7 | ensemble | mean softmax across members, every condition; intervals, coverage, four scores, threshold | about 1 min (estimate) |
| 8 | figure | `ensemble_vs_single` (paper PDF + slide PNG) | seconds |
| 9 | verify | Tier 1 for each member, then the ensemble checks | about 3 min (estimate) |
| 10 | list | prints every file of the ensemble and its members | seconds |

**Check in Drive after 03:** for each of `resnet50_source_seed43_<stamp>` and
`..._seed44_<stamp>` the same four files as a notebook 01 run plus its three sweep files;
`ensemble_resnet50_source_ens3_<stamp>.json`, `ensemble_probs_<ens_id>.npz`,
`ensemble_val_probs_<ens_id>.npz`; `figures_<ens_id>/ensemble_vs_single_{paper,slide}_<ens_id>`.
Cell 9 must show Tier 1 with 0 failed for every member and for the ensemble.

Tested locally on the fixture by executing the notebook's own cells three times: a fresh run,
a re-run (nothing retrained), and a simulated disconnect during seed 44 (only seed 44
retrained). Not yet run on Colab.

## 2c · Notebook 04 — `notebooks/04_plantseg_field.ipynb` (upgrade)

Needs notebook 01's seed 42 run and notebook 02's sweep of it in `OUT`. It does **not** need
the PlantVillage zip. It downloads PlantSeg v6 (1.7 GB, Zenodo record 16665770, CC BY 4.0)
into the Colab session (`/content`), never to Drive, and refuses a file whose MD5 does not
match. **No PlantSeg image is written to Drive.**

| # | Cell | What it does | Expected wall time |
|---|---|---|---|
| 1–2 | settings, setup | as above | about 1 min |
| 3 | locate | the seed 42 run and its sweep (stops if 02 has not run) | instant |
| 4 | **download** | PlantSeg v6, then its MD5 (skipped if already in the session and correct) | about 1–2 min (estimate) |
| 5 | **analyses** | black rot leaves (accuracy, n = 25); the other three classes (out-of-distribution confidence only); the PlantVillage references; the threshold | about 1 min (measured locally: 95 s on CPU) |
| 6 | figures | `plantseg_confidence`, `plantseg_predicted_class` (paper PDF + slide PNG) | seconds |
| 7 | verify | Tier 1 must pass (5 checks; one fails if accuracy ever appears for the other classes) | seconds |
| 8 | list | prints every PlantSeg file in Drive | seconds |

**Check in Drive after 04:** `plantseg_<run_id>.json`, `plantseg_logits_<run_id>.npz`, and in
`figures_<run_id>/` the four `plantseg_*` files. Cell 7 must show Tier 1 with 0 failed.
Executed locally, cell by cell, on the real PlantSeg zip with a stand-in model; not yet run
on Colab.

## 2d · Notebook 05 — `notebooks/05_mc_dropout.ipynb` (optional, run last)

Needs notebook 01's seed 42 run and notebook 02's sweep, and the PlantVillage zip (it
retrains the head and degrades images). It keeps the backbone exactly as trained, puts
dropout (p = 0.5) before the final layer, retrains the head for 5 epochs, then makes 20
stochastic passes in every condition (D18). If notebook 03 ran, its figure compares all
three methods; otherwise two, and the title says so. A finished head or sweep is reused.

**Check in Drive after 05:** `mcdropout_<run_id>_mcdropout.json`, `mcdropout_head_*.pt`
(head only, small), `mcdropout_probs_*`, `mcdropout_val_probs_*`, `mcdropout_passes_*`,
`mcdropout_val_passes_*`, `mcdropout_train_*.json`, and `methods_comparison_*` in
`figures_<run_id>/`. The verify cell must show Tier 1 with 0 failed.

## 3 · Files that must be in Drive afterwards (`MyDrive/COMP2035_runs/`)

`<run_id>` is `<model>_<split>_<YYYYmmdd-HHMMSS>`. Nothing is ever overwritten: a run refuses
to start if its results file already exists.

From notebook 01:
- `audit_<stamp>.json`
- for **each** of `resnet50_source_<stamp>` and `resnet50_supplied_<stamp>`:
  - `best_<run_id>.pt` (the checkpoint, saved whenever validation loss improves; about 94 MB)
  - `logits_<run_id>.npz` (test and validation logits and labels, plus the LBFGS endpoint)
  - `splits_<run_id>.json` (every image path in train, val and test)
  - `results_<run_id>.json` (all metrics, the full config, library versions, the GPU name,
    the training history)

From notebook 02 (for the source run):
- `sweep_<run_id>.json` (every row, every temperature condition, the CIs, the threshold
  analysis and the degeneracy demonstration)
- `sweep_logits_<run_id>.npz` (test logits for all 24 conditions plus labels)
- `sweep_val_logits_<run_id>.npz` (validation logits for all 24 conditions plus labels; the
  threshold is chosen on these)
- `figures_<run_id>/`: 18 files. The figures are `accuracy_vs_severity`,
  `ece_vs_severity`, `ece_by_fixed_t`, `coverage_curves`, `confidence_hist`,
  `reliability_clean`, `temperature_degeneracy`, `aurc_by_score` and `corruption_grid`,
  each as `_paper_<run_id>.pdf` and `_slide_<run_id>.png`.

## 4 · Three numbers to check before trusting the run

1. **The split** reproduces September exactly: train / val / test = **6,315 / 609 / 610**,
   with **1,493** augmented copies of val/test leaves dropped. If this differs, the data or
   the code differs: stop.
2. **The audit** reproduces exactly: **9,027** images, **4,062** leaves, **87.3%** of test
   leaves also in train (`src.verify` Tier 2 checks the whole `audit.json`).
3. **The clean source result** is within ±0.01 of September (DECISIONS.md D2): accuracy
   1.0000, mean confidence 0.9222, ECE 0.0778. The LBFGS endpoint T is information only. It
   is not expected to match 0.237, because that value was an optimiser artefact (D1).

And `src.verify` must print **Tier 1: N passed, 0 failed**. The notebooks stop if it does not.

## 5 · If something goes wrong

- **The session dies mid-training.** The best checkpoint so far is already in Drive. Re-run
  notebook 01. A new run gets a new timestamp and never overwrites the old one.
- **A count check fails.** The zip is not the one described in §0. Do not edit the expected
  counts.
- **Tier 1 fails.** Do not use the run. Keep its files and investigate. Never adjust anything
  to make a number match.
- **No GPU.** Cell 2 prints `!! NO GPU`. Fix the runtime type and run all again.

## 6 · After the run: one command

Download the `COMP2035_runs` folder from Drive (right-click → Download), unzip it, then:

```
.venv/bin/python tools/import_colab.py ~/Downloads/COMP2035_runs
```

It does every step that used to be manual:
- **Copies** every output, except the `.pt` checkpoints, which stay in Drive, into
  `results/colab_<date>/`. It never overwrites: a different file already there stops the
  import.
- **Re-scores** any sweep written before the evaluation stored the exact score arrays it
  ranked on (DECISIONS.md D20). The logits are never changed; every AURC that moves is
  printed and recorded in the sweep JSON.
- **Regenerates** every number in the report (`tools/make_tables.py`) from the seed 42 run,
  its sweep, its PlantSeg analysis and the supplied run.
- **Runs `src.verify`** on every run, ensemble, PlantSeg and MC dropout output, and on the
  regenerated numbers.
- **Prints one pass/fail table.** It must end in `ALL PASS`, then prints the `git add ...`
  line to run.

Anything that is not a protocol result (D19) renders as `[pending: ...]` in the report, never
as a number.

To run the prediction + warning tool on the real model, download its `best_<run_id>.pt` from
Drive into the same folder as its `results_` file (it stays gitignored), then:
`.venv/bin/python -m src.predict --results results/colab_<date>/results_<run_id>.json
--sweep results/colab_<date>/sweep_<run_id>.json --images <folder> --out predictions.csv`
