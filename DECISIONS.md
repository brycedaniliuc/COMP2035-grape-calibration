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
