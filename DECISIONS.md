# DECISIONS — not yet confirmed with the supervisor

Each item is a design choice that `CLAUDE.md` and the submitted paper do not fix. Each has a
recommendation, which stays a recommendation until the owner (and where needed the
supervisor) confirms it. Status is **OPEN** until then.

---

## D1 · Temperature scaling on a validation set with zero errors — RESOLVED

**RESOLVED 2026-11-07 (owner):** superseded by the AMENDED Stage 4 in BRIEF.md. Demonstrate the degeneracy (NLL vs T, LBFGS endpoint vs max_iter), use T = 1 as the primary condition, sweep a fixed T grid, and report the LBFGS endpoint only as the as-fitted (degenerate) value, never as a fitted temperature.

**Observation.** In both September runs the checkpoint that was kept had validation accuracy
1.0000 (`history` in both results JSONs). With no validation errors, validation NLL keeps
falling as T falls, so it has no minimum. The reported T (0.237 `source`, 0.051 `supplied`)
is wherever LBFGS stopped (`lr=0.1`, `max_iter=100`), not a fitted optimum. A different
`max_iter` would give a different T.

**Why it matters for Stage 4.** The promised protocol fits T on clean validation and holds it
fixed for the degraded test sets. With T ≈ 0.24, every degraded prediction gets sharpened
towards probability 1, including the wrong ones. That may well be the central finding.
It must be reported as a consequence of the degenerate fit, not hidden.

**Recommendation.** Keep the promised protocol: fit on clean validation, hold fixed. Record
the optimiser settings in the output JSON. Also report how T varies with `max_iter`, as
evidence for the degeneracy. Do not clip or regularise T silently. If the supervisor
prefers a bounded T, record that as a separate, labelled variant.

## D2 · What "reproduces" means for the Colab re-run (Tier 2) — RESOLVED

**RESOLVED 2026-11-07 (owner):** split sizes and audit numbers must match exactly. Accuracy, mean confidence and ECE must match within ±0.01 absolute. T is reported as information only, with no pass/fail. Deterministic cuDNN is on for the new run (`Config.deterministic=True`).

**Observation.** The split is seeded through `sklearn` and does reproduce exactly. Training
does not reproduce bit for bit. The notebook does not set
`torch.backends.cudnn.deterministic` or `torch.use_deterministic_algorithms`, uses AMP, and
does not seed DataLoader workers explicitly. A re-run will not reproduce 0.9222 / 0.0778 /
0.237 exactly, and T (see D1) is the least likely to match.

**Recommendation.** Tier 2 checks split sizes (6,315 / 609 / 610) and the audit numbers
exactly. It checks accuracy, mean confidence and ECE within a tolerance stated in advance,
and reports T without a pass/fail. Also turn on deterministic cuDNN for the new run, so
future Colab re-runs reproduce each other even if they cannot reproduce September.

## D3 · Class-weighted cross-entropy — RESOLVED

**RESOLVED 2026-11-07:** checked against Canoski et al. §5.2 (arXiv 2608.20608v1, PDF p. 50). The paper uses "weighted cross-entropy loss with label smoothing of 0.1" with balanced class weights from the training partition only, so it is **not a deviation**. CLAUDE.md is corrected to list it. The only protocol deviation is the epoch budget: 5 + 25 with patience 5, against the paper's maximum of 100 with patience 20.

**Observation.** The notebook weights the loss by inverse class frequency over the training
partition. `CLAUDE.md`'s training configuration does not mention this. The notebook says
the protocol follows Canoski et al., but this has not been checked against the paper.

**Recommendation.** Keep it, because it produced the only existing results. Check it against
the Canoski PDF during Stage 2 and correct `CLAUDE.md`'s configuration list either way.

## D4 · The prototype output promised in the paper — RESOLVED

**RESOLVED 2026-11-07 (owner):** approved. This is Stage 10 in BRIEF.md.
**Built 2026-11-09 (Stage 10):** `src/predict.py`. The threshold comes from the sweep (chosen
on validation, checked on held-out test, D10), or is given by hand and labelled as carrying no
accuracy claim. With no supported threshold, every image is flagged.

**Observation.** The abstract and §1 promise a returned "disease prediction, a confidence
indicator, an explicit warning for uncertain cases". `BRIEF.md` has no stage that builds
this.

**Recommendation.** After Stage 5, add a small `predict(image) -> (label, confidence, warning)`
built on the derived threshold, or on the documented absence of one. It costs about an hour
and closes a promise the marker can check.

## D5 · Repo visibility, and whether `CLAUDE.md` is committed — RESOLVED

**RESOLVED 2026-11-07 (owner):** the repo is public. CLAUDE.md and BRIEF.md stay out of git permanently (both gitignored). The README carries the owner's name and course code, not the student ID.

**Observation.** The notebooks clone this repo, so a private repo needs a token in Colab and
a public one does not. `CLAUDE.md` carries a student ID and supervisor names. `BRIEF.md`
describes the working process.

**Current state.** `BRIEF.md` is gitignored. `CLAUDE.md` is deliberately **left uncommitted**,
so it can still be kept out of history if the repo goes public. Adding it later is easy;
removing it from history is not.

**Recommendation.** The owner decides. It must be decided before anything is pushed (Gate 8).

## D6 · Fixture leaks are real, not relocated — OPEN (low stakes)

**Observation.** `BRIEF.md` asks for leaves whose copies are "deliberately placed" to create a
leak. The fixture instead *selects* 8 leaves (2 per class) that really do span `train/` and
`test/` in the supplied split. Each file is kept in the folder it actually occupies.

**Why.** Moving a file into the other partition would make the fixture disagree with the real
dataset, and a test could then pass on a layout that never occurs. The audit still has
known leaks to find, and they are named in `tests/fixture/MANIFEST.md`.

**Recommendation.** Keep it. If an artificially planted leak is wanted as well, it can be added
under a separate role.

## D7 · Degradation severities — OPEN (two of four are our own design)

**Verified from source.** These come from hendrycks/robustness,
`ImageNet-C/create_c/make_imagenet_c.py` (commit e0377d4), by Hendrycks & Dietterich
(ICLR 2019). The paper itself gives no numbers and defers to this code.
- **Gaussian blur**: sigma = 1, 2, 3, 4, 6 px. This is `gaussian_blur`, verbatim
  (`skimage.filters.gaussian`, on [0, 1]).
- **Brightness +**: HSV V += 0.1, 0.2, 0.3, 0.4, 0.5, clipped. This is `brightness`,
  verbatim. ImageNet-C only brightens.
- **Pipeline**: ImageNet-C applies its corruptions after `Resize(256)` → `CenterCrop(224)`
  and casts with `np.uint8` (truncation). We do the same, so the sigmas are on the same pixel
  scale. **Difference:** ImageNet-C then saves JPEG at quality 85. We corrupt in memory and
  skip that second lossy step.

**Our adaptations (not from ImageNet-C):**
- **Brightness −** (darkening): HSV V −= 0.1 … 0.5, the mirror of `brightness`. The brief
  asks for exposure shift "in both directions" and ImageNet-C has no darkening. Severity 5
  is close to black (see `figures/corruption_grid_*`), which is harsh by construction.
- **Occlusion**: one solid black square covering 5, 10, 20, 30, 40 % of the 224×224 crop,
  at a uniformly random position seeded per (image, corruption, severity) with `zlib.crc32`.
  ImageNet-C has no occlusion corruption. The nearest is `spatter` (mud blobs), which is not
  a controlled occluder.

**Recommendation.** Keep these. Report the two adapted series as our own design in the
Method section, not as ImageNet-C.

## D8 · Redistributing PlantVillage images in a public repo — RESOLVED (owner accepted CC BY-SA 3.0, 2026-11-09)

**Ruling.** The owner accepted the recommendation below: the fixture images and the
corruption-grid figure are published under CC BY-SA 3.0, with attribution, as stated in
`tests/fixture/IMAGES-LICENSE.md` and the README.

`tests/fixture/` (80 images) and `figures/corruption_grid_*` contain PlantVillage images.

**Verified 2026-11-07 (fetched sources):**
- The dataset author's Hugging Face card, https://huggingface.co/datasets/mohanty/PlantVillage
  (account `mohanty`), gives `license: cc-by-sa-3.0`, both on the page and in the raw README
  YAML. This is the only explicit licence found for the images.
- The Kaggle copy we actually use ("New Plant Diseases Dataset", vipoooool / Samir Bhattarai)
  is labelled "Data files © Original Authors". It is an offline-augmented derivative of
  https://github.com/spMohanty/PlantVillage-Dataset.
- The GitHub repos (spMohanty, salathegroup) show no licence file. The GitHub API returned
  404 for both.

**Recommendation.** Commit the fixture. Code is MIT. The images keep CC BY-SA 3.0, with
attribution to Hughes & Salathé (PlantVillage) and Mohanty, Hughes & Salathé (2016), stated
in `tests/fixture/IMAGES-LICENSE.md` and the README. Share-alike applies to those images
(and the grid figure), not to the code. Not verified: that the Hugging Face label covers
every image. It is the author's own statement, which is the best available evidence.

## D9 · Student ID in git history — RESOLVED (owner approved the rewrite, 2026-11-09)

**Found 2026-11-07 by scanning every commit.** The repo is public (D5), and before any push
the student ID was in every commit, in three places: the commit author and committer email
(the owner's university student-number address, name `brycebryce`); the header of the
archived September notebook (name with the ID, plus both supervisors' names); and an
earlier version of this entry, which quoted the ID while describing the problem.

**Done, before the first push:**
- This repo's commit identity is `Bryce Daniliuc <210462737+brycedaniliuc@users.noreply.github.com>`,
  the GitHub no-reply address for account `brycedaniliuc` (id from `gh api user`). It is set
  in the repo-local config; the global git config is unchanged.
- History rewritten with `git filter-branch`. Every commit now has that author and
  committer. The archived notebook's header line is reduced to course and name, and the
  supervisor line is removed. Only that markdown cell changed; every code cell is
  byte-identical, so it is still the notebook that produced `results/`. Commit messages and
  dates are unchanged.
- Checked after the rewrite: no tree, message, author or committer on `main` contains the
  ID. Each rewritten commit differs from its original only in those lines. 71 tests pass,
  and the archived notebook passes `nbformat.validate`.
- A local backup branch, `backup/pre-id-rewrite`, holds the original history. It is never
  pushed, and is deleted once the pushed repo has been checked.

## D10 · Where the threshold is chosen — CHANGED IN STAGE 9 (owner to confirm)

**Found in the Stage 9 review.** The threshold was chosen on the 610 test images and its
accuracy reported on the same 610 images, so "every condition reaches 95%" held by
construction. On real images (the CPU smoke run, a one-epoch EfficientNet-B0 standing in for
ResNet-50), I chose the threshold on one random half of the test images and applied it to the
other half, 20 times. In 9 of the 20 splits at least one condition fell below 95% on the
held-out half, down to 0.842.

**Now.** The sweep also degrades the validation images (all 24 conditions). The threshold is
chosen there and applied, unchanged, to the test images. The held-out test numbers are the
result. The test-chosen threshold is still computed, labelled optimistic, for comparison
only. Cost: the sweep reads 29,256 images instead of 14,640 (about 4 more minutes on Colab,
estimated). `src.verify` re-derives both the validation choice and the test numbers.

**Recommendation.** Keep it. Choosing on validation and reporting on test is the standard
protocol, and the brief asks for a threshold derived "defensibly". The validation images
were used during training only for early stopping and checkpoint selection, never seen as
training examples.

## D11 · Field images: LDD not used — RECORDED (the submitted plan's own contingency)

The submitted plan (§3.1) named LDD (Rossi et al. 2022) as the harder evaluation set, and
§3.4 gave the contingency: "if the LDD does not work", degraded held-out PlantVillage serves
as the proxy. Per the owner's assessment (in the owner's notes; not re-checked in this
review), LDD has no public download URL and no licence. This repo implements the
contingency. PlantSeg (Wei et al. 2026) is the owner's candidate substitute for real field
images; it is an upgrade after Stage 9 and is not done.

## D12 · Reference [19] (Wei et al. 2026, PlantSeg) is listed but never cited — OPEN (owner)

The submitted paper's reference list has 20 entries; [19] (Wei et al., "A Large-Scale
In-the-wild Dataset for Plant Disease Segmentation", Scientific Data 13, 205) is never cited
in its text. It is kept in `report/refs.bib` (key `wei2026plantseg`). In the final report,
either cite it (the natural place is Limitations and Future Work, as the field-image dataset
for PlantSeg, D11) or remove it.

Other differences between the submitted reference list and the verified records (2026-11-09),
noted above each entry in `report/refs.bib`:
- [2] The CSIRO page's title is now "Protecting wine grapes from mildew", not "Grapevine
  mildew" as cited. Same URL.
- [4] The publisher prints two authors as "Okumus" and "Cinar", without diacritics.
- [9] The article prints its first author as "S. Hernandez", not "Sergio".
- [13], [17], [18] Volume, issue or pages that the submitted list lacked were added from
  Crossref. The LNCS volume number of [17] could not be verified and is left out.
- [1] Canoski et al. and [15] Noyan are arXiv preprints with no published version listed;
  both entries say "Preprint".

## D13 · Deep ensemble design (upgrade 1) — DONE in code, owner to confirm

- **Members share one split.** `Config.seed` sets the split, so a member trained with seed 43
  would get a new split: another member's test leaves would land in its training set, and
  the members would not share a test set. So members keep `seed = 42` and set a separate
  `train_seed` (43, 44), which changes only head initialisation, shuffling and augmentation.
  With `train_seed` unset (the single-model path) training uses `seed` exactly as before:
  a fixture run under the new code gave bit-identical logits and results to the old code.
- **Seed 42 is the notebook 01 baseline, reused**, so the ensemble costs two trainings.
- **Prediction:** mean softmax across members, at T = 1. No temperature is fitted for the
  ensemble, so ensemble and single model are compared on the same footing (T = 1, the
  primary condition, D1).
- **Same sweep as the single model:** all 24 conditions; Wilson and bootstrap intervals;
  coverage and AURC for max softmax, entropy and margin; the threshold chosen on degraded
  validation and checked on held-out test (D10). Margin on the averaged distribution is
  log(p1 / p2), which equals the logit margin for a single model.
- **The fourth score (disagreement):** the variance across members of each class
  probability, averaged over the classes; ranked high-agreement-first
  (`neg_member_variance`). The brief's "mean predictive entropy of the averaged
  distribution" is the entropy score applied to the ensemble's mean, so it is already score
  two. Each row also reports its mean over images. Choosing the variance rather than mutual
  information as the disagreement score is a design choice: owner to confirm.
- **Reuse after a disconnect:** notebook 03 reuses any finished member or sweep in Drive. A
  member whose results JSON is missing (a disconnect mid-training) is trained again. Tested.

## D14 · PlantSeg (upgrade 2): version, licence, and what its grape classes are — phase A done, phase B OPEN (owner decides)

**Exact source.** Zenodo record 16665770, DOI 10.5281/zenodo.16665770: version 6 of 7 of
concept 13293890 (published 2025-07-31). "A Large-Scale In-the-wild Dataset for Plant
Disease Segmentation", by Wei, Chen, Yu, Chapman, Melloy and Huang (the authors of
reference [19], `wei2026plantseg`). The file is `Plantseg.zip`, 1,691,076,634 bytes; its MD5
`23385d877bc84bab51b54630055c75b8` matches the record. Downloaded 2026-11-09 to
`~/code/external_data/plantseg_zenodo_16665770/`, **outside the repository**. No PlantSeg
image is, or will be, committed.

**Licence: CC BY 4.0, not CC BY-NC-ND 4.0.** The Zenodo record lists **CC BY 4.0** for
version 6. The brief said CC BY-NC-ND 4.0; that is the licence of versions 1–4 (2024),
and the newer version 7 (2025-11-25, `plantseg.zip`, 1.06 GB) is CC BY-NC 4.0. Each image
also has its own licence in the metadata: of the 564 grape images, 459 are CC0 and 105 are
CC-BY. Attribution is required either way.

**Metadata file name.** Version 6 has no `PlantSeg-Meta.csv`. Its one metadata file is
`Plantseg/Metadata.csv` (11,458 rows; columns Name, Index, Plant, Disease, Resolution, Label
file, Mask ratio, URL, License, Split). Images are in `Plantseg/images/{train,val,test}/`.

**The grape images (`Plant == "Grape"`): 564, in four classes.**

| PlantSeg disease | Images | train / val / test (PlantSeg's split) | Matches our classes? |
|---|---|---|---|
| grape downy mildew | 280 | 194 / 30 / 56 | no |
| grape black rot | 122 | 85 / 13 / 24 | **by name**: Black_rot |
| grape leaf spot | 91 | 63 / 10 / 18 | **unverified**: possibly Leaf_blight (Isariopsis Leaf Spot) |
| grapevine leafroll disease | 71 | 49 / 8 / 14 | no (a virus) |

There is no esca and no healthy class. 20 other rows contain "grape" only in their URL, and
are other plants.

**A visual look (thumbnails of every image in the two possibly matching classes, not a
count for the report):**
- Most of the 122 black rot images show **berries or clusters**; roughly a fifth show
  leaves. Our model was trained only on PlantVillage leaves.
- All 91 leaf spot images show leaves in the field, with a mix of spot types. Nothing in
  the metadata says that "leaf spot" is Isariopsis leaf spot.

**So neither branch of the brief applies cleanly.** One class matches by name but is mostly
another organ, one may match, and two do not match at all.

**Recommendation for phase B (owner decides before anything is built):**
- Run the classifier over all 564 grape images and report the **confidence distribution**
  for each PlantSeg class: out-of-distribution confidence, never accuracy.
- Also report **which of our four classes it predicts** for each PlantSeg class. For black
  rot this one number is checkable ("of PlantSeg's 122 black rot images, mostly berries, the
  model called x% Black_rot"). Label it as agreement on a mostly non-leaf subset, not
  field accuracy.
- Leave leaf spot out of any agreement number unless its identity is confirmed.

Inference only: nothing is trained on PlantSeg, and PlantSeg's own split is not used.

## D15 · PlantSeg leaf filter (phase B1): the rule, fixed before any image was tagged

**Why filter.** The classifier was trained only on PlantVillage photographs of single grape
leaves. A PlantSeg image of berries tests nothing about it, so phase B uses only the images
whose subject is a leaf. PlantSeg's own metadata says nothing about organ, so every one of
the 564 grape images (`Plant == "Grape"`, D14) is tagged by looking at it.

**How.** `tools/plantseg_contact_sheets.py` lays the images out in contact sheets of 25
(5 × 5, each image about 280 px), class by class, in filename order. Every image gets a
stable id: BR (black rot), DM (downy mildew), LS (leaf spot) or LR (leafroll), plus a
number. Each image is tagged from the sheet against the rule below, never from its filename
or metadata. The tags go to `results/plantseg/leaf_filter.csv`, with a reason wherever the
rule needed a judgement. The sheets are kept outside the repository (they are PlantSeg
images) so the owner can spot-check every tag. Tagging was done by visual inspection during
the coding work, not by the owner. The owner's spot-check is the second pair of eyes.

**The rule.** Applied in this order; the first that fits decides.
1. **other**: a collage or multi-panel composite, a drawing, diagram or text graphic, or an
   image with no identifiable grape organ.
2. **leaf**: the main subject is grape leaf blade(s). Leaf tissue is most of the plant
   material in focus, and the symptoms shown are on leaves. A leaf held in a hand, on a
   table or on the vine all count. A wide view counts only if leaves fill most of the frame
   and single leaves can be made out.
3. **berry**: the main subject is berries or bunches. Fruit is most of the plant material in
   focus, and the symptoms shown are on the fruit.
4. **other**: everything else. That includes shoots, stems, tendrils, trunks and flowers;
   distant whole-vine or row views where no organ dominates; and images where leaves and
   berries are both prominent, or where it is unclear which carries the symptoms.

**Conservative by design.** When unsure, the tag is "other", never "leaf". The leaf subset
is meant to be clean, not large.
