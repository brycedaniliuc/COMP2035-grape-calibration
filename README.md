# Confidence-Aware Grape Leaf Disease Recognition

**Bryce Daniliuc** · COMP2035 Topics in Computer Science · Adelaide University · 2026

Grape leaf disease classifiers reach near-perfect accuracy on PlantVillage-style laboratory
images. This project asks a narrower question: **when the model is wrong, does its
confidence say so?** The intended deliverable is a confidence threshold below which a
prediction is referred to a person instead of being acted on, and a statement of where that
threshold can and cannot be trusted.

---

## Status: what exists and what does not

| | Status | Where |
|---|---|---|
| Dataset audit and train/test leakage measurement | **Done** | `results/*/audit.json` |
| Leak-free split at the level of the source leaf | **Done** | `src/data.py` |
| ResNet-50 baseline, `source` and `supplied` splits (Sept 2026, Colab T4) | **Done** | `results/source/`, `results/supplied/` |
| ECE, reliability diagrams, temperature scaling on clean data | **Done** | same |
| Degradation sweep (blur, brighten, darken, occlusion × 5 severities) | **Built and tested on a fixture; not yet run on the real model** | `src/degrade.py`, `src/evaluate.py` |
| Coverage curves, three confidence scores, threshold derivation | **Built and tested; not yet run on the real model** | `src/evaluate.py` |
| Figures, independent verification | **Built and tested; not yet run on the real model** | `src/figures.py`, `src/verify.py` |
| The real degradation run | **Pending**: one Colab session | [`HANDOVER.md`](HANDOVER.md) |
| Prediction + confidence + warning tool | **Built and tested on the fixture**; its real threshold comes from the Colab run | `src/predict.py` |
| Deep ensemble (3 members) through the same degradation sweep, compared with the single model; mutual information as the disagreement score | **Built and tested on the fixture; not yet run on the real model** (notebook 03) | `src/ensemble.py` |
| MC dropout (dropout before the final layer, head retrained, 20 passes) through the same sweep; one figure comparing single model, ensemble and MC dropout | **Built and tested on the fixture; optional, not yet run on the real model** (notebook 05) | `src/mcdropout.py` |
| Field images (LDD in the submitted plan) | Not used. This repo runs the plan's §3.4 contingency instead: degraded held-out PlantVillage | `DECISIONS.md` D11 |
| Field photographs: PlantSeg v6 grape leaves (black rot accuracy, n = 25; the other three classes as out-of-distribution confidence only) | **Leaf filter done** (564 images tagged by eye against a stated rule); **analysis built and tested, real run pending** (notebook 04) | `results/plantseg/`, `src/plantseg.py`, D14–D16 |

No result of the degradation study appears in this README, because none exists yet.

---

## Findings so far

Every number below is read from a file in `results/`.

### 1 · The supplied train/test split leaks

The supplied dataset holds 9,027 images of **4,062** distinct leaves: 4,062 originals and
4,965 augmented copies (flips and rotations), named `<leaf-uuid>___<name><suffix>`. The
augmentation was done *before* the split, so copies of one leaf sit on both sides:

| Measure (supplied split) | Value |
|---|---|
| Test leaves that also appear in train | 1,344 of 1,539 (**87.3%**) |
| Test **originals** whose leaf appears in train | 621 of 816 (**76.1%**) |
| Per class: Black rot / Esca / Leaf blight / Healthy | 89.7% / 76.5% / 87.7% / **100.0%** |

Sources: `results/source/audit.json` (identical to `results/supplied/audit.json`). The same
numbers are re-derived from the filenames by separate code in `tests/build_fixture.py`
(`full_dataset_audit` in `tests/fixture/manifest.json`) and checked by
`tests/test_data.py::test_full_dataset_audit_matches_september_audit_json`.

### 2 · Even without leakage, the task is saturated

On the leak-free `source` split (split by leaf; 610 test originals, never seen in any form
during training), ResNet-50 classifies **every test image correctly**:

| | `source` (leak-free) | `supplied` (leaky) |
|---|---|---|
| Test images | 610 | 1,805 |
| Test accuracy | **1.0000** | 1.0000 |
| Mean confidence | 0.9222 | 0.9255 |
| ECE (15 bins) | 0.0778 | 0.0745 |

Sources: `results/source/results_resnet50_source.json`,
`results/supplied/results_resnet50_supplied.json`.

The model is **underconfident**, not overconfident: accuracy exceeds confidence. Label
smoothing of 0.1 over 4 classes makes the training target for the true class 0.925, close to
the observed mean confidence. All 610 confidences fall in three adjacent bins between 0.80
and 1.0, almost all in the top two (`results/source/reliability_before_resnet50_source.png`).
With so little spread, confidence has almost nothing to rank.

### 3 · Temperature scaling is undefined on a saturated task

At the kept checkpoint, validation accuracy was 1.0000 in both runs (the `history` field of
each results JSON). With no validation errors, validation NLL keeps falling as the
temperature *T* shrinks, so it has **no minimum**. The optimiser (LBFGS) stopped where its
iteration budget ran out: at 0.237 (`source`) and 0.051 (`supplied`). Those numbers are
optimiser endpoints, not fitted temperatures. The resulting "post-calibration" ECE of about
0 (1.9 × 10⁻⁶ for `source`) only means every confidence was pushed to 1. **Post-hoc
calibration needs errors to fit against, and a saturated benchmark supplies none.** This is
why the degradation study, where errors occur, is the real experiment.

---

## Method

- **Data.** The grape subset of PlantVillage (Hughes & Salathé 2015; Mohanty et al. 2016), as
  redistributed with offline augmentation in the Kaggle "New Plant Diseases Dataset". Four
  classes: black rot, esca, leaf blight, healthy.
- **Split (`source`).** Leaves (UUIDs) are split 70/15/15, stratified by class (seed 42).
  Validation and test contain originals only. Augmented copies of validation/test leaves are
  **discarded, never moved to train**. Asserted: no leaf in two partitions, one class per
  leaf.
- **Model and training.** ResNet-50 (ImageNet weights), following Canoski et al. (2026, §5.2):
  - two stages: head only, then full fine-tuning
  - AdamW with cosine annealing
  - class-balanced cross-entropy with label smoothing 0.1
  - batch 64, best checkpoint by validation loss

  The single deviation is the budget: 5 + 25 epochs with patience 5, against the paper's
  maximum of 100 with patience 20, so a run fits one free Colab session. Deterministic cuDNN
  is on for new runs.
- **Corruptions** (inference only, applied on uint8 pixels after resize and crop, before
  normalisation). Gaussian blur σ = 1, 2, 3, 4, 6 px and brightening (HSV V + 0.1 … 0.5) are
  ImageNet-C's own parameters (Hendrycks & Dietterich 2019, from the official
  `make_imagenet_c.py`). Darkening (V − 0.1 … 0.5) and occlusion (a black square covering
  5–40% of the image at a seeded random position) are **our own adaptations**, because
  ImageNet-C has neither. See `DECISIONS.md` D7 and `figures/corruption_grid_*`.
- **Temperature protocol.** No temperature is ever fitted on degraded data. There are four
  conditions, kept separate:
  1. validation NLL against T, plus the LBFGS endpoint for several iteration budgets, to show
     the degeneracy
  2. **T = 1** (no temperature) as the primary condition
  3. a fixed grid T ∈ {0.25, 0.5, 1, 2, 4}, each held fixed across all severities
  4. the LBFGS endpoint, held fixed and labelled as-fitted (degenerate) whenever validation
     has no errors
- **Statistics.** A Wilson 95% interval on every accuracy, and a percentile bootstrap 95%
  interval (1,000 resamples) on every ECE.
- **Confidence scores.** Max softmax, predictive entropy, and logit margin (top logit minus
  second).
- **Selective classification** (Geifman & El-Yaniv 2017). Coverage curves and the area
  under the risk–coverage curve (AURC) for each score at T = 1, and for max softmax and
  entropy at every temperature condition, because a temperature can reorder images. Margin
  divided by T ranks images exactly as margin does. AURC is reported at T = 1 only, to two
  decimals: at the LBFGS endpoint max softmax saturates at 1.0, so the ranking cannot order
  the tied predictions; the saturation counts are reported instead (DECISIONS.md D20).
- **Threshold.** Chosen on the **validation** images under every (corruption, severity)
  condition: the lowest score at which every condition reaches the target accuracy on the
  accepted predictions (primary target 95%). It is then applied, unchanged, to the **test**
  images under the same conditions. They played no part in choosing it, so the reported
  accuracy, coverage and referral cost are held out. A threshold chosen on the test images
  themselves meets the target in every condition by construction; that version is kept for
  comparison only, labelled optimistic. A threshold that only works by referring everything
  is flagged as such, not reported as a success. Every threshold is printed with a list of
  what it does not cover (`THRESHOLD_LIMITS` in `src/evaluate.py`).

---

## Reproducing

### Local (CPU): tests and the pipeline on the bundled 80-image fixture

```bash
git clone https://github.com/brycedaniliuc/COMP2035-grape-calibration.git
cd COMP2035-grape-calibration
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m pytest                      # all tests, including two end-to-end runs
.venv/bin/ruff check src tests tools            # static analysis
.venv/bin/mypy                                  # type checking (config in pyproject.toml)
```

The full pipeline on the fixture, in order (replace `<run_id>` with the id printed by the
first command):

```bash
.venv/bin/python -m src.train --data-root tests/fixture --model efficientnet_b0 \
    --batch-size 8 --head-epochs 1 --finetune-epochs 1 --num-workers 0 --out-dir outputs/fixture
.venv/bin/python -m src.evaluate --results outputs/fixture/results_<run_id>.json --num-workers 0
.venv/bin/python -m src.figures sweep --sweep outputs/fixture/sweep_<run_id>.json --out outputs/fixture/figures
.venv/bin/python -m src.verify --run outputs/fixture/results_<run_id>.json --determinism tests/fixture
```

`src.verify` re-derives every reported number with independent code and exits non-zero on
any mismatch. The fixture numbers are meaningless as results (n = 9 test images); the
fixture exists to exercise filename parsing, leaf grouping, the leakage audit and every code
path on real images.

### The dataset

The dataset is not in this repository (see the licence section). The code expects this
layout under `--data-root`, and `check_supplied_dataset` (`src/data.py`) refuses to run on
anything else:

| Folder | Images |
|---|---|
| `train/Grape___Black_rot/` | 1,888 |
| `train/Grape___Esca_(Black_Measles)/` | 1,920 |
| `train/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/` | 1,722 |
| `train/Grape___healthy/` | 1,692 |
| `test/Grape___Black_rot/` | 472 |
| `test/Grape___Esca_(Black_Measles)/` | 480 |
| `test/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/` | 430 |
| `test/Grape___healthy/` | 423 |

These are the grape folders of the Kaggle "New Plant Diseases Dataset". If your copy names
a top-level folder differently, rename it to `train` or `test`. The count check then
confirms you have the same 9,027 images.

### The prediction + confidence + warning tool

For any folder of images, it writes one CSV row per image: the predicted class, the
confidence (max softmax at T = 1), and `warning = yes` when the confidence is below the
threshold, meaning "refer this image to a person". The threshold comes from a sweep, where
it was chosen on validation images and checked on held-out test images:

```bash
.venv/bin/python -m src.predict --results outputs/fixture/results_<run_id>.json \
    --sweep outputs/fixture/sweep_<run_id>.json --images tests/fixture --out predictions.csv
```

Every run prints whether the held-out data support the threshold, what "no warning" does
and does not mean, and what the threshold does not cover. In particular, the warning cannot
recognise an image outside the model's scope (not a grape leaf, or another disease). A
threshold given by hand (`--threshold 0.9`) is accepted but labelled as carrying no accuracy
claim.

With the full dataset on disk, the leakage audit alone is:

```bash
.venv/bin/python -m src.audit --data-root /path/to/grape_dataset
```

### The report: numbers come from the data

`report/main.tex` is the final report (ACM sigconf, 8 pages), with the bibliography in
`report/refs.bib`. No result is typed into it. `tools/make_tables.py` reads the results JSON
and writes `report/generated/`: `numbers.tex` (macros such as `\testacc` and `\leakPct`)
and the results, audit, sweep and threshold tables. Anything not run yet appears as
`[pending: ...]`. `src.verify` fails if a generated file is stale: its source JSON or the
generator changed, it was edited by hand, or a pending source now exists.

```bash
.venv/bin/python tools/make_tables.py
.venv/bin/python -m src.verify --generated report/generated
cd report && latexmk -pdf main.tex
```

On this machine, `latexmk` needs TeX Live first on the PATH (`PATH=/Library/TeX/texbin:$PATH`),
because the MiKTeX `pdflatex` in `/usr/local/bin` cannot build its format file.

### The real experiment (Colab T4)

`notebooks/01_baseline_calibration.ipynb` (training) and
`notebooks/02_degradation_study.ipynb` (sweep, figures, verification). Both clone this repo
and import `src/`, so Colab runs exactly the code the tests ran. Step-by-step instructions,
timings, and the files to check are in [`HANDOVER.md`](HANDOVER.md). The September 2026
notebook that produced `results/` is kept, unchanged, in `notebooks/archive/`.

Tested locally with Python 3.13.5 and the versions pinned in `requirements.txt`. In Colab,
the preinstalled CUDA build of PyTorch is used instead of the CPU pin.

---

## Repository layout

```
src/            config, data (indexing, splits), audit, train, calibrate, degrade,
                evaluate (sweep, coverage, threshold), figures, verify,
                predict (prediction + confidence + warning), ensemble (deep ensemble),
                plantseg (field photographs)
tests/          pytest suite; fixture/ = 80 real images, MANIFEST.md, build_fixture.py
notebooks/      01-05 for Colab (RUNSHEET.md); archive/ holds the September 2026 notebook
tools/          build_notebooks.py (generates and validates the notebooks),
                make_tables.py (report tables and number macros from the results JSON),
                import_colab.py (brings the Colab outputs home and checks them),
                plantseg_contact_sheets.py (the PlantSeg leaf-filter sheets)
report/         the final report: main.tex (ACM sigconf skeleton), refs.bib, generated/
results/        the September 2026 runs: results JSON, audit, reliability diagrams
figures/        the corruption protocol figure (sweep figures follow the Colab run)
DECISIONS.md    every design choice not fixed by the submitted plan, with the reasoning
RUNSHEET.md     the Colab day, step by step
HANDOVER.md     the reference behind it: every cell, every file
```

---

## Data provenance and licence

The images come from PlantVillage (Hughes & Salathé, arXiv:1511.08060), via the Kaggle "New
Plant Diseases Dataset" (offline-augmented copy, labelled "Data files © Original Authors").
The dataset author's Hugging Face card (`mohanty/PlantVillage`) gives the licence as
**CC BY-SA 3.0**. The dataset itself is not in this repository. The 80 fixture images and the
corruption-grid figure are, under CC BY-SA 3.0 with attribution: see
`tests/fixture/IMAGES-LICENSE.md`.

The field photographs come from PlantSeg v6 (Wei et al. 2026; Zenodo record 16665770,
CC BY 4.0). No PlantSeg image is in this repository. `results/plantseg/leaf_filter.csv`
holds only each image's filename, PlantSeg's own labels, and the tags assigned here, and
notebook 04 downloads the images from Zenodo.

The code is MIT licensed (`LICENSE`).

## Citation

```bibtex
@misc{daniliuc2026grape,
  author = {Daniliuc, Bryce},
  title  = {Confidence-Aware Grape Leaf Disease Recognition},
  year   = {2026},
  note   = {COMP2035 Topics in Computer Science, Adelaide University},
  url    = {https://github.com/brycedaniliuc/COMP2035-grape-calibration}
}
```

## References

- Canoski, P., Spasev, V., Dimitrovski, I., Kitanovski, I. and Lameski, P. 2026. A
  dataset-centric benchmark of deep learning methods for grape leaf disease classification and
  detection. arXiv:2608.20608 (preprint).
- Geifman, Y. and El-Yaniv, R. 2017. Selective classification for deep neural networks.
  NeurIPS 30.
- Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. 2017. On calibration of modern neural
  networks. ICML.
- Hendrycks, D. and Dietterich, T. 2019. Benchmarking neural network robustness to common
  corruptions and perturbations. ICLR.
- Hughes, D. P. and Salathé, M. 2015. An open access repository of images on plant health to
  enable the development of mobile disease diagnostics. arXiv:1511.08060.
- Mohanty, S. P., Hughes, D. P. and Salathé, M. 2016. Using deep learning for image-based
  plant disease detection. Frontiers in Plant Science 7:1419.
- Wei, T., Chen, Z., Yu, X., Chapman, S., Melloy, P. and Huang, Z. 2026. A large-scale
  in-the-wild dataset for plant disease segmentation. Scientific Data 13, 205.
