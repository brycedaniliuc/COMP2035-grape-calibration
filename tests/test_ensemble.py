"""The deep ensemble (upgrade 1): members share one split, the mean softmax and the members'
disagreement are right by hand, and the whole chain verifies on the fixture."""

import json

import numpy as np
import pytest

from src.config import Config
from src.ensemble import combine, ensemble_sweep, ensure_member, find_run
from src.train import parse_config, run
from src.verify import ensemble_checks


def test_mean_softmax_and_disagreement_known_answer():
    """Member A: logits (0, 0) -> (0.5, 0.5). Member B: (log 3, 0) -> (0.75, 0.25).
    Mean (0.625, 0.375). Variance across members of each class probability: 0.015625 for
    both classes, so the mean over classes is 0.015625."""
    p, var = combine([np.array([[0.0, 0.0]]), np.array([[np.log(3.0), 0.0]])])
    assert np.allclose(p, [[0.625, 0.375]]) and np.allclose(var, [0.015625])


def test_default_config_trains_with_the_split_seed():
    """The single-model path: no train_seed, so training uses `seed` exactly as before."""
    cfg = Config()
    assert cfg.train_seed is None and cfg.training_seed == cfg.seed == 42
    assert Config(train_seed=43).training_seed == 43 and Config(train_seed=43).seed == 42
    assert parse_config([]).train_seed is None
    assert parse_config(["--train-seed", "44"]).train_seed == 44


@pytest.fixture(scope="module")
def members(fixture_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("ens")
    cfg = Config(data_root=str(fixture_root), model="efficientnet_b0", pretrained=False,
                 batch_size=8, head_epochs=1, finetune_epochs=0, num_workers=0, out_dir=str(out))
    base = out / f"results_{run(cfg)['run_id']}.json"
    return out, base, [ensure_member(base, s) for s in (42, 43, 44)]


def test_members_share_the_split_and_seed_42_is_the_baseline(members):
    out, base, paths = members
    assert paths[0] == base and find_run(out, 42, "efficientnet_b0") == base
    res = [json.loads(p.read_text()) for p in paths]
    assert [r["config"]["train_seed"] for r in res] == [None, 43, 44]
    assert all("_seed" in r["run_id"] for r in res[1:])
    splits = [(out / r["artefacts"]["splits"]).read_text() for r in res]
    assert splits[0] == splits[1] == splits[2]
    # a re-run reuses the finished members: no new runs appear
    n = len(list(out.glob("results_*.json")))
    assert [ensure_member(base, s) for s in (42, 43, 44)] == paths
    assert len(list(out.glob("results_*.json"))) == n


def test_ensemble_sweep_and_verification(members):
    out, _, paths = members
    ens = ensemble_sweep(paths)
    assert len(ens["rows"]) == 24 and ens["n_members"] == 3
    for r in ens["rows"]:
        assert set(r["scores"]) == {"max_softmax", "neg_entropy", "margin",
                                    "neg_member_variance"}
        assert r["disagreement"]["mean_member_variance"] >= 0
        assert [m["training_seed"] for m in r["members"]] == [42, 43, 44]
    assert set(ens["threshold"]["by_score"]) == {"max_softmax", "neg_entropy", "margin"}
    checks = ensemble_checks(out / f"ensemble_{ens['ens_id']}.json")
    assert checks and all(c[2] == "PASS" for c in checks), [c for c in checks if c[2] != "PASS"]


def test_a_member_on_another_split_is_refused(members, fixture_root):
    out, _, paths = members
    cfg = Config(**{**json.loads(paths[0].read_text())["config"], "seed": 7})  # new split
    other = out / f"results_{run(cfg)['run_id']}.json"
    with pytest.raises(AssertionError, match="split|protocol"):
        ensemble_sweep([paths[0], other])
