# Confidence-Aware Grape Leaf Disease Recognition

Code and results for COMP2035 Topics in Computer Science, Adelaide University, 2026.
Bryce Daniliuc (a1927102). Supervisor: Prof Volker Hessel. Co-supervisor: Shu Liang.

Deep learning models for classifying grape leaf disease report almost 100% accuracy, but
accuracy says nothing about how confident those predictions were. This project fine-tunes
ResNet-50 on the PlantVillage grape subset and asks whether the model can separate its
correct predictions from its incorrect ones once images are degraded. The deliverable is a
confidence threshold: a number below which a prediction is referred to a person instead of
acted on.

Four classes: black rot, esca, leaf blight, healthy.

Full method, results and discussion are in the report. This file covers what the code does,
what it found, and how to run it.

---

## Findings

Every number below comes from `results/colab_20261110/` and is re-derived by `src.verify`.
The source file for each is named.

**1 · The supplied split leaks.** The Kaggle release augments each leaf and then splits the
augmented images, so copies of the same physical leaf sit on both sides. 1,344 of 1,539 test
leaves — 87.3% — also appear in training, and 621 of the 816 unaugmented test images (76.1%)
are of leaves the model has already seen. A model evaluated on this partition is tested
largely on rotations and reflections of leaves it was trained on. Reported accuracy measures
recall of seen leaves, not generalisation to new ones. The healthy class leaks completely,
at 100.0%, because the augmentation equalises image counts across classes, not leaf counts.
(`src/audit.py`)

**2 · Removing the leak changed nothing.** Re-partitioning at the source-leaf level — 2,843
training leaves, 609 validation, 610 test, with augmented copies of held-out leaves discarded
rather than moved — still gives 1.0000 accuracy on the clean test set, 95% CI [0.994, 1.000].
That rules out leakage as the explanation and leaves a saturated benchmark. Mean confidence
is 0.9221 against accuracy 1.0000, so the model is underconfident, not overconfident; label
smoothing of 0.1 is the cause. ECE is 0.0779, and all of that is the model refusing to say
1.0 when it is always right. (`src/data.py`, `src/train.py`)

**3 · Temperature scaling could not be fitted.** With no validation errors to penalise, NLL
falls monotonically as temperature falls, and there is no minimum to find. LBFGS stopped at
T = 0.1699 after 100 iterations, driving validation NLL from 0.0824 to 3.02×10⁻⁶. Applying
that value pushed every prediction to full confidence: mean confidence 1.0000, ECE 0.0000.
A perfect calibration score, and completely worthless. The value is an optimiser artefact,
not a fitted parameter. (`src/calibrate.py`)

**4 · Detail loss is what breaks the model, not light loss.** Four corruptions at six
severities each. Blur takes accuracy from 1.0000 to 0.4148 and ECE from 0.0779 to 0.2406.
Darkening reaches 0.6787. Brightening and occlusion barely register — 0.9934 and 0.9869 at
severity 5. Under blur, confidence falls more slowly than accuracy does: the model stays
right about which predictions to doubt long after it stops being right about the class.
(`src/degrade.py`, `src/evaluate.py`)

**5 · The threshold.** Ranking by raw maximum softmax was sufficient; neither predictive
entropy nor logit margin improved on it. A threshold of 0.9132, chosen on degraded validation
images and never adjusted, held accuracy at or above 95% in twenty-three of the twenty-four
test conditions, referring 41.93% of predictions for human review. The one miss is darkening
at severity 5. A three-member deep ensemble met the target in all twenty-four and halved the
cost, referring 21.72%. MC dropout over 20 passes referred 40.90% and still missed one.
(`src/evaluate.py`, `src/ensemble.py`, `src/mcdropout.py`)

**6 · Field photographs.** On 25 black rot photographs from PlantSeg v6 the model scored
0.6400 [0.445, 0.798] with mean confidence 0.6464 — close to what the severest simulated blur
predicted (0.6410), and far below the 0.9221 of clean laboratory images. The confidence drop
is in the right direction and roughly the right size, which is the strongest evidence here
that the threshold would transfer. The sample is small and the comparison is indicative.
(`src/plantseg.py`)

---

## Method

ResNet-50, ImageNet-pretrained, 224 px. Batch 64. Head 5 epochs at LR 1e-3, then fine-tune
25 epochs at LR 1e-4, patience 5. Weight decay 1e-4, label smoothing 0.1, seed 42, AMP on.
This follows Canoski et al.'s protocol; the single deviation is the budget, 5 + 25 epochs
against their maximum of 100 with patience 20.

Splits are grouped by source leaf, recovered from the UUID in each filename. Augmented copies
of validation and test leaves are discarded, never moved into training.

Corruptions: Gaussian blur and brightness follow the ImageNet-C severity scales (Hendrycks &
Dietterich 2019). Darkening and occlusion are this project's own design and have not been
confirmed with the supervisor (see `DECISIONS.md`, D7).

ECE uses 15 equal-width bins on the maximum softmax. Accuracy intervals are Wilson 95%; ECE
intervals are percentile bootstrap over 1,000 resamples.

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

The full pipeline on the fixture, in order. Replace `<run_id>` with the id printed by the
first command.

```bash
.venv/bin/python -m src.train --data-root tests/fixture --model efficientnet_b0 \
    --batch-size 8 --head-epochs 1 --finetune-epochs 1 --num-workers 0 --out-dir outputs/fixture
.venv/bin/python -m src.evaluate --results outputs/fixture/results_<run_id>.json --num-workers 0
.venv/bin/python -m src.figures sweep --sweep outputs/fixture/sweep_<run_id>.json --out outputs/fixture/figures
.venv/bin/python -m src.verify --run outputs/fixture/results_<run_id>.json --determinism tests/fixture
```

`src.verify` re-derives every reported number with independent code and exits non-zero on any
mismatch. The fixture numbers are meaningless as results — nine test images — but the fixture
exercises filename parsing, leaf grouping, the leakage audit and every code path on real
images.

### The real experiment (Colab T4)

Notebooks `01`–`05` run the full study end to end. `RUNSHEET.md` is the order to run them in;
`HANDOVER.md` is the reference behind it, cell by cell. The dataset is not in this repository
and `check_supplied_dataset` (`src/data.py`) refuses to run on anything that is not the
expected layout.

### The prediction tool

`src/predict.py` takes an image, returns a class, a confidence and a warning when confidence
falls below the threshold. It is the deliverable in usable form.

---

## Repository layout

```
src/            config, data (indexing, splits), audit, train, calibrate, degrade,
                evaluate (sweep, coverage, threshold), figures, verify,
                predict (prediction + confidence + warning), ensemble, plantseg
tests/          pytest suite; fixture/ = 80 real images, MANIFEST.md, build_fixture.py
notebooks/      01-05 for Colab (RUNSHEET.md); archive/ holds the September 2026 notebook
tools/          build_notebooks.py, make_tables.py, import_colab.py,
                plantseg_contact_sheets.py
report/         the final report: main.tex (ACM sigconf), refs.bib, generated/
results/        colab_20261110/ = the November 2026 runs the report uses;
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
CC BY 4.0). No PlantSeg image is in this repository. `results/plantseg/leaf_filter.csv` holds
only each image's filename, PlantSeg's own labels, and the tags assigned here; notebook 04
downloads the images from Zenodo.

The code is MIT licensed (`LICENSE`).

---

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
- Lakshminarayanan, B., Pritzel, A. and Blundell, C. 2017. Simple and scalable predictive
  uncertainty estimation using deep ensembles. NeurIPS 30.
- Mohanty, S. P., Hughes, D. P. and Salathé, M. 2016. Using deep learning for image-based
  plant disease detection. Frontiers in Plant Science 7:1419.
- Wei, T., Chen, Z., Yu, X., Chapman, S., Melloy, P. and Huang, Z. 2026. A large-scale
  in-the-wild dataset for plant disease segmentation. Scientific Data 13, 205.
