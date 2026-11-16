# Confidence-Aware Grape Leaf Disease Recognition

**Bryce Daniliuc** · COMP2035 Topics in Computer Science · Adelaide University · 2026

Grape leaf disease classifiers reach near-perfect accuracy on PlantVillage-style laboratory
images. This project asks a narrower question: **when the model is wrong, does its
confidence say so?** The intended deliverable is a confidence threshold below which a
prediction is referred to a person instead of being acted on, and a statement of where that
threshold can and cannot be trusted.

---

## Status: what exists and what does not

The report uses the runs of 10–11 November 2026 (Colab, Tesla T4), stored in
`results/colab_20261110/`. The September 2026 baseline runs in `results/source/` and
`results/supplied/` came first; their weights and logits were not kept, so the November
runs retrained both splits from the same seeded split.

| | Status | Where |
|---|---|---|
| Dataset audit and train/test leakage measurement | **Done** | `results/colab_20261110/audit_*.json` |
| Leak-free split at the level of the source leaf | **Done** | `src/data.py` |
| ResNet-50 baseline, `source` and `supplied` splits | **Done** (Nov 2026; first run Sept 2026) | `results/colab_20261110/results_*.json` |
| ECE, reliability diagrams, temperature scaling on clean data | **Done** | same, and `figures_*` |
| Degradation sweep (blur, brighten, darken, occlusion × 5 severities) | **Done** (notebook 02) | `sweep_resnet50_source_20261110-083456.json` |
| Coverage curves, three confidence scores, threshold chosen on validation and checked on held-out test | **Done** | same |
| Figures, independent verification of every reported number | **Done**: all outputs pass `src.verify` | `figures_*`, `src/verify.py` |
| Deep ensemble (3 members, seeds 42–44) through the same sweep; mutual information as the disagreement score | **Done** (notebook 03) | `ensemble_resnet50_source_ens3_20261111-120456.json` |
| MC dropout (dropout p = 0.5 before the final layer, head retrained, 20 passes) through the same sweep | **Done** (notebook 05) | `mcdropout_resnet50_source_20261110-083456_mcdropout.json` |
| Field photographs: PlantSeg v6 grape leaves (black rot accuracy, n = 25; the other three diseases as out-of-distribution confidence only) | **Done** (notebook 04); leaf filter: 564 images tagged by eye against a stated rule | `plantseg_resnet50_source_20261110-083456.json`, `results/plantseg/`, D14–D16 |
| Prediction + confidence + warning tool | **Done**; it reads the threshold from a sweep file | `src/predict.py` |
| Field images (LDD in the submitted plan) | Not used. This repo runs the plan's §3.4 contingency instead: degraded held-out PlantVillage | `DECISIONS.md` D11 |
| Darkening and occlusion severities | **Our own design**, not ImageNet-C's; open for confirmation | `DECISIONS.md` D7 |

The files named without a folder are in `results/colab_20261110/`.

---

## Findings

Every number below is read from a file in `results/`, named beside it. Files named without
a folder are in `results/colab_20261110/`.

### 1 · The supplied train/test split leaks

The supplied dataset holds 9,027 images of **4,062** distinct leaves: 4,062 originals and
4,965 augmented copies (flips and rotations), named `<leaf-uuid>___<name><suffix>`. The
augmentation was done *before* the split, so copies of one leaf sit on both sides:

| Measure (supplied split) | Value |
|---|---|
| Test leaves that also appear in train | 1,344 of 1,539 (**87.3%**) |
| Test **originals** whose leaf appears in train | 621 of 816 (**76.1%**) |
| Per class: Black rot / Esca / Leaf blight / Healthy | 89.7% / 76.5% / 87.7% / **100.0%** |

Sources: `results/colab_20261110/audit_20261110-083456.json` (the same numbers as the
September `results/source/audit.json` and `results/supplied/audit.json`). The same
numbers are re-derived from the filenames by separate code in `tests/build_fixture.py`
(`full_dataset_audit` in `tests/fixture/manifest.json`) and checked by
`tests/test_data.py::test_full_dataset_audit_matches_september_audit_json`.

### 2 · Even without leakage, the task is saturated

On the leak-free `source` split (split by leaf; 610 test originals, never seen in any form
during training), ResNet-50 classifies **every test image correctly**:

| | `source` (leak-free) | `supplied` (leaky) |
|---|---|---|
| Test images | 610 | 1,805 |
| Test accuracy [Wilson 95% CI] | **1.0000** [0.994, 1.000] | 1.0000 [0.998, 1.000] |
| Mean confidence | 0.9221 | 0.9244 |
| ECE (15 bins) | 0.0779 | 0.0756 |

Sources: `results_resnet50_source_20261110-083456.json`,
`results_resnet50_supplied_20261110-090100.json`. The September runs gave the same
accuracy, with mean confidence 0.9222 and 0.9255 and ECE 0.0778 and 0.0745
(`results/source/results_resnet50_source.json`,
`results/supplied/results_resnet50_supplied.json`).

The model is **underconfident**, not overconfident: accuracy exceeds confidence. Label
smoothing of 0.1 over 4 classes makes the training target for the true class 0.925, close to
the observed mean confidence. All 610 confidences fall in three adjacent 15-bin intervals
between 0.80 and 1.0: 2 in 0.80–0.87, 503 in 0.87–0.93 and 105 in 0.93–1.0 (computed from
`logits_resnet50_source_20261110-083456.npz`; see `reliability_clean_*` in
`figures_resnet50_source_20261110-083456/`). With so little spread, confidence has almost
nothing to rank.

### 3 · Temperature scaling is undefined on a saturated task

At the kept checkpoint, validation accuracy was 1.0000 in both runs. With no validation
errors, validation NLL keeps falling as the temperature *T* shrinks, so it has **no
minimum**. The optimiser (LBFGS) stops where its iteration budget runs out: at 0.17
(`source`) and 0.241 (`supplied`) in the November runs, and at 0.237 and 0.051 in September.
Those numbers are optimiser endpoints, not fitted temperatures. The resulting
"post-calibration" ECE of about 0 (1.2 × 10⁻⁹ for `source`) only means every confidence was
pushed towards 1. **Post-hoc calibration needs errors to fit against, and a saturated
benchmark supplies none.** This is why the degradation study, where errors occur, is the
real experiment.

### 4 · Under corruption, confidence falls more slowly than accuracy

At severity 5 (`sweep_resnet50_source_20261110-083456.json`, T = 1):

| Corruption, severity 5 | Accuracy [95% CI] | Mean confidence | ECE |
|---|---|---|---|
| Gaussian blur, σ = 6 px | 0.4148 [0.376, 0.454] | 0.6410 | 0.2406 |
| Darkening, V − 0.5 | 0.6787 [0.641, 0.715] | 0.7818 | 0.1138 |
| Occlusion, 40% | 0.9869 [0.974, 0.993] | 0.8489 | 0.1415 |
| Brightening, V + 0.5 | 0.9934 [0.983, 0.997] | 0.9177 | 0.0820 |

At the LBFGS endpoint, max softmax is exactly 1.0 for more than one prediction in all 21
conditions (clean and 20 corrupted), for 28 to 609 of the 610 images. Under darkening at
severity 5, 231 predictions sit at exactly 1.0 and 19 of them are wrong (of 196 errors).
Max softmax cannot rank those, so no AURC is reported at that temperature (`DECISIONS.md`
D20).

### 5 · The threshold

For a 95% accuracy target, the threshold on max softmax (T = 1) chosen on degraded
validation images is **0.9132**. Applied unchanged to the held-out degraded test images, it
meets the target in **23 of 24** conditions: four corruptions at severities 0–5, where
severity 0 is the unaltered image. It accepts 58.1% of images on average and refers 41.9%,
and it refers no condition entirely. The miss is darkening at severity 5: 154 images
accepted (25.2%), accuracy 0.9286 [0.877, 0.960]. Source: the sweep file above.

The same procedure on the other two methods:

| Method | Threshold | Held-out conditions meeting 95% | Mean coverage |
|---|---|---|---|
| Single model | 0.9132 | 23 of 24 | 58.1% |
| Deep ensemble, 3 members | 0.8646 | 24 of 24 | 78.3% |
| MC dropout, 20 passes | 0.9028 | 23 of 24 (misses darkening s5) | 59.1% |

Sources: the sweep file, `ensemble_resnet50_source_ens3_20261111-120456.json`,
`mcdropout_resnet50_source_20261110-083456_mcdropout.json`.

### 6 · Field photographs (PlantSeg v6)

The single model on 334 PlantSeg grape leaf photographs, each photograph once
(`plantseg_resnet50_source_20261110-083456.json`):

- **Black rot**, the one PlantSeg class that is also a model class: 16 of 25 correct,
  accuracy **0.6400** [0.445, 0.798], mean confidence 0.6464. At the 0.9132 threshold, 12.0%
  of them are accepted without a warning. n = 25 is small; quote the interval.
- **Diseases the model was not trained on** (confidence only; the warning cannot recognise
  them as out of scope):

| Disease | Leaves | Mean confidence | Accepted without a warning |
|---|---|---|---|
| Downy mildew | 174 | 0.5524 | 2.3% |
| Leaf spot | 74 | 0.6666 | 6.8% |
| Leafroll | 61 | 0.6314 | 4.9% |

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
- **Deep ensemble.** Three `source`-split models trained with the same protocol and seeds
  42, 43 and 44; the prediction is the mean softmax at T = 1. Mutual information between
  members is the disagreement score (`DECISIONS.md` D13). The ensemble goes through the same
  sweep and threshold procedure as the single model.
- **MC dropout.** A dropout layer (p = 0.5) before the final linear layer of the seed-42
  model; only that head is retrained, for the protocol's 5 head epochs. The prediction is
  the mean softmax over 20 passes with dropout on, then the same sweep and threshold.
- **Field photographs.** PlantSeg v6 grape images, tagged by eye as leaf, berry or other
  against a rule fixed before tagging (`DECISIONS.md` D14–D16); only leaf photographs are
  used. Black rot is the only PlantSeg
  class that is also a model class, so it is the only one with an accuracy. The other
  diseases are reported as confidence only, against the threshold chosen above.

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

Five notebooks, run in the order given in [`RUNSHEET.md`](RUNSHEET.md):

| Notebook | Does |
|---|---|
| `01_baseline_calibration.ipynb` | trains ResNet-50 on the `source` and `supplied` splits |
| `02_degradation_study.ipynb` | the sweep, coverage curves, threshold, figures and verification |
| `03_deep_ensemble.ipynb` | trains members with seeds 43 and 44 and runs the ensemble through the sweep |
| `04_plantseg_field.ipynb` | downloads PlantSeg v6 from Zenodo and runs the field analysis |
| `05_mc_dropout.ipynb` | retrains the head with dropout and runs MC dropout through the sweep |

Each clones this repo and imports `src/`, so Colab runs exactly the code the tests ran.
Every cell, timing and output file is described in [`HANDOVER.md`](HANDOVER.md).
`tools/import_colab.py` copies the outputs home, regenerates the report numbers and runs
`src.verify` on everything. The September 2026 notebook that produced `results/source/` and
`results/supplied/` is kept, unchanged, in `notebooks/archive/`.

To check the stored November outputs without a GPU:

```bash
D=results/colab_20261110
.venv/bin/python -m src.verify --run $D/results_resnet50_source_20261110-083456.json
.venv/bin/python -m src.verify --ensemble $D/ensemble_resnet50_source_ens3_20261111-120456.json
.venv/bin/python -m src.verify --mcdropout $D/mcdropout_resnet50_source_20261110-083456_mcdropout.json
.venv/bin/python -m src.verify --plantseg $D/plantseg_resnet50_source_20261110-083456.json
.venv/bin/python -m src.verify --generated report/generated
```

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
report/         the final report: main.tex (ACM sigconf), refs.bib, generated/
results/        colab_20261110/ = the November 2026 runs the report uses (results, splits,
                logits, sweeps, ensemble, MC dropout, PlantSeg, figures_*/);
                source/, supplied/ = the September 2026 runs; plantseg/ = the leaf filter
figures/        the corruption protocol figure
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
