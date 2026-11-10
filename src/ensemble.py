"""Deep ensemble (upgrade 1; Lakshminarayanan et al. 2017), compared with the single model
under the same corruptions as in Ovadia et al. (2019).

Members are runs on the SAME `source` split (seed 42) with the same protocol, differing only
in training seed (`Config.train_seed`): 42 is the notebook 01 baseline, reused rather than
retrained; 43 and 44 are trained here. A new split per member would put one member's test
leaves into another's training set.

The ensemble's prediction is the mean softmax across members (T = 1). It goes through the
same degradation sweep as the single model: each member's sweep logits (test and validation,
all 24 conditions) are averaged in probability space, and every row gets the same
statistics: accuracy with a Wilson interval, ECE with a bootstrap interval, coverage and AURC
for max softmax, entropy and margin. Margin on the ensemble is log(p1 / p2) of the averaged
distribution, which is what the logit margin is for a single model.

A fourth score is the ensemble's own: disagreement between members, the variance across
members of each class probability, averaged over the classes (DECISIONS.md D13). Each row
also reports the mean predictive entropy of the averaged distribution. The threshold is
chosen on degraded validation images and checked on the held-out test images (D10).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import Config
from src.degrade import CORRUPTIONS, SEVERITIES
from src.evaluate import (
    evaluate_logits,
    reported_aurc,
    rescore,
    risk_coverage,
    sweep,
    threshold_analysis,
)
from src.provenance import not_a_result
from src.train import _json_safe, run

MEMBER_SEEDS = (42, 43, 44)
# Config fields allowed to differ between members: the training seed, and where things are.
NOT_PROTOCOL = {"train_seed", "out_dir", "data_root", "num_workers"}


def training_seed(res: dict) -> int:
    cfg = res["config"]
    return cfg["seed"] if cfg.get("train_seed") is None else cfg["train_seed"]


def protocol(res: dict) -> dict:
    return {k: v for k, v in res["config"].items() if k not in NOT_PROTOCOL}


def find_run(out_dir: str | Path, seed: int, model: str = "resnet50",
             split: str = "source") -> Path | None:
    """The newest finished run (results JSON and checkpoint present) with this training seed.
    Seed 42 finds the notebook 01 baseline, never an ensemble member."""
    found = []
    for p in sorted(Path(out_dir).glob(f"results_{model}_{split}_*.json")):
        res = json.loads(p.read_text())
        if training_seed(res) == seed and (p.parent / res["artefacts"]["checkpoint"]).exists():
            found.append(p)
    return found[-1] if found else None


def ensure_member(base_results: str | Path, seed: int, data_root: str | None = None) -> Path:
    """The run for this training seed: reused if one exists in the base run's folder, else
    trained now with the base run's exact config and only `train_seed` changed. It is written
    next to the base run (in Colab: Drive), reading the data from `data_root` if given."""
    base_results = Path(base_results)
    base = json.loads(base_results.read_text())
    if training_seed(base) == seed:
        return base_results
    found = find_run(base_results.parent, seed, base["config"]["model"],
                     base["config"]["split_mode"])
    if found is not None:
        assert protocol(json.loads(found.read_text())) == protocol(base), \
            f"{found.name} has a different protocol from the base run"
        print(f"seed {seed}: reusing {found.name}")
        return found
    print(f"seed {seed}: training")
    where = {"out_dir": str(base_results.parent), **({"data_root": data_root} if data_root
                                                       else {})}
    res = run(Config(**{**base["config"], "train_seed": seed, **where}))
    return base_results.parent / f"results_{res['run_id']}.json"


def ensure_sweep(results_path: str | Path, data_root: str | None = None) -> Path:
    """The run's sweep (test and validation logits for every condition), run if missing."""
    results_path = Path(results_path)
    run_id = json.loads(results_path.read_text())["run_id"]
    d = results_path.parent
    files = [d / f"sweep_{run_id}.json", d / f"sweep_logits_{run_id}.npz",
             d / f"sweep_val_logits_{run_id}.npz"]
    if all(f.exists() for f in files):
        print(f"{run_id}: reusing its sweep")
        rescore(files[0])  # a sweep from before D20 gets its ranked score arrays
    else:
        sweep(results_path, data_root=data_root)
    return files[0]


def softmax64(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-300, None))).sum(-1)


def combine(sample_logits: list[np.ndarray]) -> dict[str, np.ndarray]:
    """S samples of logits (ensemble members, or MC dropout passes), per image:
    mean: the mean softmax, the prediction.
    total_entropy: entropy of the mean, the total predictive uncertainty.
    expected_entropy: mean of the samples' own entropies, the aleatoric part.
    mutual_info: total minus expected, the epistemic part: how much the samples disagree
        (D13). Never negative in exact arithmetic; float rounding is clipped at 0.
    variance: variance across samples of each class probability, mean over the classes."""
    p = np.stack([softmax64(z) for z in sample_logits])  # samples x images x classes
    mean = p.mean(0)
    total = entropy(mean)
    expected = entropy(p).mean(0)
    return {"mean": mean, "total_entropy": total, "expected_entropy": expected,
            "mutual_info": np.clip(total - expected, 0, None), "variance": p.var(0).mean(1)}


UNCERTAINTY = ("total_entropy", "expected_entropy", "mutual_info", "variance")
UNCERTAINTY_NOTE = ("per image: total_entropy = H[mean softmax]; expected_entropy = mean of the "
                    "samples' entropies (aleatoric); mutual_info = total - expected (epistemic, "
                    "the disagreement score, D13); variance = variance across samples of each "
                    "class probability, mean over classes. Fourth score: neg_mutual_info.")


def pseudo_logits(probs: dict, val_probs: dict) -> tuple[dict, dict]:
    """Test and validation stores as log probabilities, for threshold_analysis."""
    def conv(st: dict) -> dict:
        return {k: (log_probs(v) if k != "labels" else v) for k, v in st.items() if "__" not in k}
    return conv(probs), conv(val_probs)


def log_probs(p: np.ndarray) -> np.ndarray:
    """Pseudo-logits: softmax(log p) = p, so every single-model statistic applies as is."""
    return np.log(np.clip(p, 1e-300, None))


def evaluate_samples(test: dict[str, list[np.ndarray]], val: dict[str, list[np.ndarray]],
                     y: np.ndarray, classes: list[str], n_bins: int, seed: int,
                     parameter: dict[str, float]) -> tuple[list[dict], dict, dict]:
    """The single model's sweep statistics for a method that averages S samples, in every
    condition: accuracy (Wilson), ECE (bootstrap) and coverage at T = 1 on the mean softmax;
    AURC for max softmax, entropy, margin and the fourth score, mutual information
    (`neg_mutual_info`, higher = samples agree more); the threshold chosen on the validation
    images and checked on test (by the caller, through pseudo_logits). Returns (rows, test
    store, validation store)."""
    probs: dict[str, np.ndarray] = {"labels": y}
    val_probs: dict[str, np.ndarray] = {}
    rows = []
    for key in test:
        c = combine(test[key])
        probs[key] = c["mean"]
        for u in UNCERTAINTY:
            probs[f"{key}__{u}"] = c[u]
        val_probs[key] = combine(val[key])["mean"]
        corruption, severity = key.rsplit("_s", 1)
        ranked: dict[str, np.ndarray] = {}
        row = {"corruption": corruption, "severity": int(severity), "parameter": parameter[key],
               **evaluate_logits(log_probs(c["mean"]), y, classes, {"T=1": 1.0}, n_bins, seed,
                                 ranked)}
        probs.update({f"{key}__{k}": v for k, v in ranked.items()})  # D20
        correct = (c["mean"].argmax(1) == y).astype(float)
        row["scores"]["neg_mutual_info"] = risk_coverage(-c["mutual_info"], correct)
        row["uncertainty"] = {f"mean_{u}": float(c[u].mean()) for u in UNCERTAINTY}
        rows.append(row)
    s0 = [probs[k] for k in test if k.endswith("_s0")]
    assert all(np.array_equal(s0[0], x) for x in s0[1:]), "severity 0 differs by corruption"
    assert all(np.allclose(probs[k].sum(1), 1) for k in test)
    return rows, probs, val_probs


def _aurc(row: dict) -> str:
    x = reported_aurc(row, "scores", "max_softmax")
    return "sat." if x is None else f"{x:.2f}"


def ensemble_sweep(member_results: Sequence[str | Path], stamp: str | None = None,
                   data_root: str | None = None) -> dict:
    paths = [Path(p) for p in member_results]
    res = [json.loads(p.read_text()) for p in paths]
    assert len(res) >= 2, "an ensemble needs at least two members"
    seeds = [training_seed(r) for r in res]
    assert len(set(seeds)) == len(seeds), f"duplicate training seeds: {seeds}"
    assert all(protocol(r) == protocol(res[0]) for r in res), "members differ in protocol"
    splits = [json.loads((p.parent / r["artefacts"]["splits"]).read_text())
              for p, r in zip(paths, res, strict=True)]
    assert all(s == splits[0] for s in splits), "members do not share one split"
    out_dir = paths[0].parent
    sw = [json.loads(ensure_sweep(p, data_root).read_text()) for p in paths]
    test = [dict(np.load(out_dir / f"sweep_logits_{r['run_id']}.npz")) for r in res]
    val = [dict(np.load(out_dir / f"sweep_val_logits_{r['run_id']}.npz")) for r in res]
    for store in (test, val):
        assert all(set(s) == set(store[0]) for s in store), "members' sweeps differ"
        assert all(np.array_equal(s["labels"], store[0]["labels"]) for s in store)

    cfg = Config(**{k: v for k, v in res[0]["config"].items() if k in Config().to_dict()})
    classes = res[0]["classes"]
    y, y_val = test[0]["labels"], val[0]["labels"]
    keys = [f"{c}_s{s}" for c in CORRUPTIONS for s in SEVERITIES]
    member_rows = {k: [next(r for r in s["rows"] if f"{r['corruption']}_s{r['severity']}" == k)
                       for s in sw] for k in keys}
    rows, probs, val_probs = evaluate_samples(
        {k: [t[k] for t in test] for k in keys}, {k: [v[k] for v in val] for k in keys}, y,
        classes, cfg.n_bins, cfg.seed, {k: member_rows[k][0]["parameter"] for k in keys})
    val_probs["labels"] = y_val
    for row, k in zip(rows, keys, strict=True):
        row["members"] = [{"run_id": r["run_id"], "training_seed": s, "accuracy": m["accuracy"],
                           "ece_T1": m["temperature"]["T=1"]["ece"]}
                          for r, s, m in zip(res, seeds, member_rows[k], strict=True)]

    base = next((i for i, s in enumerate(seeds) if s == 42), 0)
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    ens_id = f"{cfg.model}_{cfg.split_mode}_ens{len(res)}_{stamp}"
    out = {"ens_id": ens_id, "n_members": len(res), "classes": classes,
           "members": [{"run_id": r["run_id"], "training_seed": s, "results": p.name}
                       for r, s, p in zip(res, seeds, paths, strict=True)],
           "single_model": {"run_id": res[base]["run_id"],
                            "sweep": f"sweep_{res[base]['run_id']}.json"},
           "n_test": int(y.size), "n_val": int(y_val.size),
           "prediction": "mean softmax across members, T = 1",
           "uncertainty": UNCERTAINTY_NOTE, "not_a_result": not_a_result(res[0]),
           "rows": rows, "threshold": threshold_analysis(*pseudo_logits(probs, val_probs))}
    (out_dir / f"ensemble_{ens_id}.json").write_text(
        json.dumps(_json_safe(out), indent=1, allow_nan=False))
    np.savez(out_dir / f"ensemble_probs_{ens_id}.npz", **probs)
    np.savez(out_dir / f"ensemble_val_probs_{ens_id}.npz", **val_probs)
    single = sw[base]
    print(f"ensemble {ens_id}: members {seeds}; single model = seed {seeds[base]}")
    print(f"{'condition':<18}{'acc single':>11}{'ensemble':>10}{'ECE single':>12}{'ensemble':>10}"
          f"{'AURC single':>13}{'ensemble':>10}")
    for r in rows:
        s = next(x for x in single["rows"]
                 if (x["corruption"], x["severity"]) == (r["corruption"], r["severity"]))
        print(f"{r['corruption'] + ' s' + str(r['severity']):<18}{s['accuracy']:>11.4f}"
              f"{r['accuracy']:>10.4f}{s['temperature']['T=1']['ece']:>12.4f}"
              f"{r['temperature']['T=1']['ece']:>10.4f}"
              f"{_aurc(s):>13}{_aurc(r):>10}")  # D20: "sat." where the score saturates
    print(f"written -> {out_dir / f'ensemble_{ens_id}.json'}")
    return out
