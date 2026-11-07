# HANDOVER — running the real experiment in Google Colab

This is the one step that needs a GPU, and the step where the September checkpoint and logits
were lost. Follow it exactly. Every output goes to Google Drive **as it is produced**, under
run-tagged filenames, so nothing depends on downloading from the Colab session.

## 0 · Before you start

1. **The repo must be on GitHub.** The notebooks clone
   `https://github.com/brycedaniliuc/COMP2035-grape-calibration` (public). Until it is pushed,
   the first code cell fails at `git clone`.
2. **The dataset zip must be in Drive.** Upload
   `~/Downloads/Literature Review paper/Grape/grape_dataset_for_colab.zip` (160.6 MB) to the
   **top level of My Drive**, with exactly that name. The September runs used this same zip,
   extracted to `/content/grape_dataset`. The notebooks then check the exact per-folder counts
   (7,222 train and 1,805 test, per class) **before any training**. A wrong or partial zip
   fails in the first minute, not at minute 26.
3. In Colab: **Runtime → Change runtime type → T4 GPU**.

## 1 · Notebook 01 — `notebooks/01_baseline_calibration.ipynb`

Open it from GitHub in Colab (File → Open notebook → GitHub). Run the cells in order
(Runtime → Run all works). Wall times are labelled with their basis; none is a guess.

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
| 4 | locate | finds `results_resnet50_source_*.json` in `OUT` | instant |
| 5 | **sweep** | 4 corruptions × severities 0–5 on the 610 test originals; all temperature conditions; bootstrap CIs; threshold | **about 5 min** (estimate, not measured on Colab; see the note below) |
| 6 | figures | 9 figures × (paper PDF + slide PNG) into `OUT/figures_<run_id>/` | about 1 min (not yet measured) |
| 7 | verify | Tier 1 must pass | about 1 min |
| 8 | list | prints every file of this run | seconds |

**Sweep timing, measured and estimated.** The sweep was smoke-run locally on the real dataset
(CPU, Apple M2 Pro, a one-epoch EfficientNet-B0 standing in for ResNet-50). It took
**92 min**, almost all of it CPU inference: 24 conditions × 610 images, about 0.38 s per
image. The other parts were timed separately: all 144 bootstrap CIs take about 0.5 min, and
the corruptions take at most 12 ms per image. On a T4, inference stops being the limit, and
the limit becomes how fast Colab can load and decode images. In September, Colab fed about
130 images/s during training (30 epochs of about 6,900 images in 25.9 min). At that rate the
14,640 sweep images take about 2 min, plus the bootstrap and the validation pass, hence
"about 5 min". If the cell is still running after 20 min, check that cell 2 printed a CUDA
device.

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
- `sweep_logits_<run_id>.npz` (logits for all 24 conditions plus labels)
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

## 6 · After the run

Copy into `results/colab_<date>/` in the repo, for each run, **every file except the `.pt`
checkpoint**: `results_`, `logits_`, `splits_`, `sweep_` and `sweep_logits_` (`verify`
reads them side by side). Also copy `audit_<stamp>.json`. Copy the figures into `figures/`.
The checkpoints stay in Drive; `*.pt` is gitignored. Then repeat every check from the copied
files:
`.venv/bin/python -m src.verify --run results/colab_<date>/results_<run_id>.json`
