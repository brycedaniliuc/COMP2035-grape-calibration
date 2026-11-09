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
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import Config
from src.degrade import CORRUPTIONS, SEVERITIES
from src.evaluate import evaluate_logits, risk_coverage, sweep, threshold_analysis
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
    else:
        sweep(results_path, data_root=data_root)
    return files[0]


def softmax64(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def combine(member_logits: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(mean softmax, disagreement): disagreement[i] is the variance across members of each
    class probability for image i, averaged over the classes."""
    p = np.stack([softmax64(z) for z in member_logits])  # members x images x classes
    return p.mean(0), p.var(0).mean(1)


def log_probs(p: np.ndarray) -> np.ndarray:
    """Pseudo-logits: softmax(log p) = p, so every single-model statistic applies as is."""
    return np.log(np.clip(p, 1e-300, None))


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-300, None))).sum(1)


def ensemble_sweep(member_results: list[str | Path], stamp: str | None = None,
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
    probs: dict[str, np.ndarray] = {"labels": y}
    val_probs: dict[str, np.ndarray] = {"labels": y_val}
    rows = []
    for key in keys:
        p, disagree = combine([t[key] for t in test])
        probs[key], probs[f"{key}__disagreement"] = p, disagree
        val_probs[key] = combine([v[key] for v in val])[0]
        corruption, severity = key.rsplit("_s", 1)
        member_rows = [next(r for r in s["rows"] if f"{r['corruption']}_s{r['severity']}" == key)
                       for s in sw]
        row = {"corruption": corruption, "severity": int(severity),
               "parameter": member_rows[0]["parameter"],
               **evaluate_logits(log_probs(p), y, classes, {"T=1": 1.0}, cfg.n_bins,
                                 cfg.seed)}
        correct = (p.argmax(1) == y).astype(float)
        row["scores"]["neg_member_variance"] = risk_coverage(-disagree, correct)
        row["disagreement"] = {"mean_member_variance": float(disagree.mean()),
                               "mean_predictive_entropy": float(entropy(p).mean())}
        row["members"] = [{"run_id": r["run_id"], "training_seed": s, "accuracy": m["accuracy"],
                           "ece_T1": m["temperature"]["T=1"]["ece"]}
                          for r, s, m in zip(res, seeds, member_rows, strict=True)]
        rows.append(row)
    s0 = [probs[f"{c}_s0"] for c in CORRUPTIONS]
    assert all(np.array_equal(s0[0], x) for x in s0[1:]), "severity 0 differs by corruption"
    assert all(np.allclose(probs[k].sum(1), 1) for k in keys)

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
           "disagreement": "variance across members of each class probability, mean over "
                           "classes (score neg_member_variance: higher = members agree more)",
           "rows": rows, "threshold": threshold_analysis(
               {k: (log_probs(v) if k != "labels" else v) for k, v in probs.items()
                if not k.endswith("__disagreement")},
               {k: (log_probs(v) if k != "labels" else v) for k, v in val_probs.items()})}
    (out_dir / f"ensemble_{ens_id}.json").write_text(
        json.dumps(_json_safe(out), indent=1, allow_nan=False))
    np.savez(out_dir / f"ensemble_probs_{ens_id}.npz", **probs)  # type: ignore[arg-type]
    np.savez(out_dir / f"ensemble_val_probs_{ens_id}.npz", **val_probs)  # type: ignore[arg-type]
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
              f"{s['scores']['max_softmax']['aurc']:>13.4f}{r['scores']['max_softmax']['aurc']:>10.4f}")
    print(f"written -> {out_dir / f'ensemble_{ens_id}.json'}")
    return out
