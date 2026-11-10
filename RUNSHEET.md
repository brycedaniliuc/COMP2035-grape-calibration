# RUNSHEET — the Colab day, step by step

Follow the steps in order. Nothing has to be remembered. Times are for a Colab T4: the
training times were measured in September; everything else is an estimate.

| Notebook | What | Status | Time |
|---|---|---|---|
| 01 | train the model | **ESSENTIAL** | about 45 min |
| 02 | degradation study | **ESSENTIAL** | about 10 min |
| 03 | deep ensemble | upgrade | about 70 min |
| 04 | PlantSeg field photos | upgrade | about 5 min |
| 05 | MC dropout | optional, run last | about 15 min |

**Out of time?** Cut from the bottom: skip 05, then 04, then 03. **01 + 02 alone is the
complete project.** 03 and 04 do not depend on each other.

---

## Before you start (once, about 5 min)

1. On the laptop, open Finder at `~/Downloads/Literature Review paper/Grape/`.
2. In a browser, open **drive.google.com** → **My Drive** (the top level, not inside a
   folder). Drag `grape_dataset_for_colab.zip` (160.6 MB) into it. Wait until the upload
   says it is complete. Keep the name exactly `grape_dataset_for_colab.zip`.
3. Open **colab.research.google.com**.

## Running a notebook (the same 5 clicks every time)

4. **File → Open notebook → GitHub** tab. In the box, type
   `brycedaniliuc/COMP2035-grape-calibration` and press Enter. Click the notebook named in
   the step below.
5. **Runtime → Change runtime type → T4 GPU → Save.**
6. **Runtime → Run all.**
7. A pop-up asks to access Google Drive: **Connect to Google Drive** → choose your account →
   **Continue / Allow** (tick every box if asked).
8. Leave the tab open and visible until the last cell finishes. A cell has finished when its
   spinner stops. Then check the numbers below. Do not edit any cell.

---

## 01 — `01_baseline_calibration.ipynb` — ESSENTIAL — about 45 min

Check, from top to bottom:
- cell "setup": shows `"device": "cuda"` and `"device_name": "Tesla T4"` (or another GPU).
  It must **not** show `!! NO GPU`.
- cell "data": `dataset OK: 9,027 images in 8 folders, counts as verified`
- cell "audit": `images : 9,027`, `source leaves : 4,062`, and **87.3%** of test leaves also in
  train
- cell "train source": the split is **train 6315, val 609, test 610**
- cell "verify": `Tier 1: N passed, 0 failed`. In the Tier 2 lines:
  - **split** and **audit**: must say PASS
  - **accuracy** (1.0000), **mean confidence** (0.9222) and **ECE** (0.0778): each within
    0.01. If one says FAIL, the notebook still continues; note the number and carry on (it
    is a finding, not an error)
- the last cell lists the files written to Drive, `COMP2035_runs`

## 02 — `02_degradation_study.ipynb` — ESSENTIAL — about 10 min

- cell "sweep": 24 lines (4 corruptions × severities 0–5), then
  `T identical across all 24 rows` and `clean row vs training run: accuracy ... == ...`
- a threshold block is printed (`threshold chosen on degraded validation images`)
- cell "verify": `Tier 1: N passed, 0 failed`

**After 02 the project is complete.** Everything below is an upgrade.

## 03 — `03_deep_ensemble.ipynb` — upgrade — about 70 min

- cell "members": `seed 42: ...`, then `seed 43: training` and `seed 44: training` (each
  about 26 min)
- cell "ensemble": a table of 24 rows (single model vs ensemble)
- cell "verify": every member line shows `Tier 1: N passed, 0 failed`, and so does the
  ensemble

## 04 — `04_plantseg_field.ipynb` — upgrade — about 5 min

- cell "download": `PlantSeg v6 OK: 1.69 GB, md5 23385d877bc84bab51b54630055c75b8`
- cell "analyses": black rot leaves `n = 25`; three out-of-distribution groups (n = 174, 74,
  61); **no accuracy** is printed for those three, and that is correct
- cell "verify": `Tier 1: 5 passed, 0 failed`

## 05 — `05_mc_dropout.ipynb` — OPTIONAL, only if 01–04 went cleanly — about 15 min

- cell "MC dropout": 5 lines `MC dropout head epoch ...`, then `MC dropout ...: 20 passes`
- cell "verify": `Tier 1: N passed, 0 failed`

---

## Back on the laptop (about 10 min)

9. In **drive.google.com**, right-click the folder **COMP2035_runs** → **Download**. Drive
   zips it. Double-click the zip in `~/Downloads` to unzip, so the folder
   `~/Downloads/COMP2035_runs` exists. (If Drive split it into several zips, unzip them all
   into that one folder.)
10. Open Terminal, then run these two commands:
    ```
    cd ~/code/COMP2035-grape-calibration
    .venv/bin/python tools/import_colab.py ~/Downloads/COMP2035_runs
    ```
    It copies everything into `results/colab_<date>/`, regenerates every number in the
    report, checks them all, and prints one table. **It must end in `ALL PASS`.**
11. It prints one last line beginning `Next: git add ...`. Copy and run that line: it saves
    the results to GitHub.

---

## When something goes wrong

**1. "...grape_dataset_for_colab.zip not found. Put grape_dataset_for_colab.zip at the top
level of My Drive."** (cell "data", red error)
- **Cause:** the zip is not in My Drive, is inside a folder, has another name, or is still
  uploading.
- **Fix:** check in drive.google.com that `grape_dataset_for_colab.zip` sits directly in My
  Drive and has finished uploading. Then **Runtime → Run all** again.
- If the error says **"dataset counts differ"** instead, the zip is the wrong file or only
  partly uploaded. Upload it again from the folder in step 1.

**2. `!! NO GPU`** (cell "setup"), **or Colab says "Cannot connect to GPU backend" / "usage
limits"**
- **Cause:** the runtime is not a GPU, or the free GPU quota is used up for now.
- **Fix:** **Runtime → Change runtime type → T4 GPU → Save → Runtime → Run all.** If Colab
  refuses a GPU because of usage limits, stop and try again in a few hours. **Do not run on
  the CPU:** training would take many hours.

**3. "Runtime disconnected"** or the session closes part-way
- **Cause:** an idle tab, a lost connection, or Colab's time limit.
- **Fix:** **Reconnect**, then **Runtime → Run all** again in the same notebook. Nothing
  finished is lost: every output was already saved to Drive.
  - In 03, 04 and 05, finished work is reused.
  - In 01, an unfinished training run is simply started again (it is about 26 min).
- **To prevent it:** keep the Colab tab visible, and the laptop awake and plugged in.

**Anything else, or any `Tier 1: ... N failed` with N > 0:** stop. Do not import that
notebook's outputs. Keep everything in Drive as it is and ask for help, with a screenshot of
the red message. Every notebook is safe to run again later.
