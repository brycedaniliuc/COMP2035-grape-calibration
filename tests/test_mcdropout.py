"""MC dropout (D18): the passes by hand, and the whole method on the stand-in Drive folder."""

import json

import numpy as np
import torch
from torch import nn

from src.mcdropout import N_PASSES, P_DROP, mc_dropout, passes
from src.verify import mcdropout_checks


def test_passes_drop_features_and_repeat_exactly():
    torch.manual_seed(0)
    lin = nn.Linear(200, 4)
    f = torch.ones(3, 200)
    a, b = passes(f, lin), passes(f, lin)
    assert a.shape == (N_PASSES, 3, 4) and np.array_equal(a, b)  # seeded masks: reproducible
    assert not np.array_equal(a[0], a[1])  # dropout active: passes differ
    # the expectation of a dropped-and-rescaled input is the input: averaged over many
    # masks, the mean pass approaches the deterministic output
    many = np.stack([passes(f, lin, n=400)]).mean(axis=(0, 1))
    with torch.no_grad():
        assert np.allclose(many, lin(f).numpy(), atol=0.15)
    assert 0 < P_DROP < 1


def test_mc_dropout_on_the_stand_in_folder(colab_drive):
    mc = json.loads(next(p for p in colab_drive.glob("mcdropout_*.json")
                         if not p.name.startswith("mcdropout_train_")).read_text())
    assert mc["n_passes"] == N_PASSES and mc["p_drop"] == P_DROP and len(mc["rows"]) == 24
    assert mc["not_a_result"].startswith("NOT A RESULT")  # fixture + stand-in model
    for r in mc["rows"]:
        assert set(r["scores"]) == {"max_softmax", "neg_entropy", "margin", "neg_mutual_info"}
        u = r["uncertainty"]
        assert u["mean_variance"] >= 0 and u["mean_mutual_info"] >= 0
    assert set(mc["threshold"]["by_score"]) == {"max_softmax", "neg_entropy", "margin"}
    checks = mcdropout_checks(colab_drive / f"mcdropout_{mc['mc_id']}.json")
    assert all(c[2] == "PASS" for c in checks), [c for c in checks if c[2] != "PASS"]
    assert any("with the ensemble" in c[3] for c in checks)  # all three methods in the figure
    # finished work is reused, not repeated (a re-run after a disconnect)
    again = mc_dropout(colab_drive / f"results_{mc['base_run_id']}.json")
    assert again["rows"] == mc["rows"]


def _make_pre_d20(json_path, store_path) -> None:
    """Turn a result into one written before D20: no saturation statistic on any curve, and
    no ranked score arrays beside the logits or probabilities."""
    res = json.loads(json_path.read_text())
    for row in res["rows"]:
        curves = list(row["scores"].values())
        for cond in row["temperature"].values():
            curves += list(cond["coverage"].values())
        for c in curves:
            c.pop("saturation", None)
    res.pop("rescored", None)
    res.pop("saturation_added", None)
    json_path.write_text(json.dumps(res))
    store = dict(np.load(store_path))
    np.savez(store_path, **{k: v for k, v in store.items() if "__" not in k})


def test_a_cached_result_from_before_d20_is_caught_before_any_gpu_time(colab_drive, tmp_path,
                                                                        monkeypatch, capsys):
    """Notebook 05 (D20): a stale MC dropout result is recomputed, a stale single-model sweep
    is re-scored from its logits, and a stale ensemble stops it with the file and the fix,
    all in the preflight, before the head is trained or a pass is run."""
    import shutil

    import pytest

    import src.mcdropout as M
    from src.ensemble import StaleResult, find_ensemble, find_run
    from src.evaluate import pre_d20
    d = tmp_path / "drive"
    shutil.copytree(colab_drive, d)
    base = find_run(d, 42, "efficientnet_b0")  # the seed 42 run, not an ensemble member
    assert base is not None
    rid = json.loads(base.read_text())["run_id"]
    mid = M.mc_id(rid)
    want = json.loads((d / f"mcdropout_{mid}.json").read_text())

    # 1. stale MC dropout result: recomputed (the head is reused), identical to the original
    _make_pre_d20(d / f"mcdropout_{mid}.json", d / f"mcdropout_probs_{mid}.npz")
    assert pre_d20(json.loads((d / f"mcdropout_{mid}.json").read_text()))
    M.mc_dropout(base)
    assert "Recomputing it; the trained head is reused" in capsys.readouterr().out
    again = json.loads((d / f"mcdropout_{mid}.json").read_text())
    assert again["rows"] == want["rows"] and pre_d20(again) is None

    # 2. stale single-model sweep: re-scored from its saved logits (CPU), then current
    _make_pre_d20(d / f"sweep_{rid}.json", d / f"sweep_logits_{rid}.npz")
    assert M.preflight(base) is True
    assert pre_d20(json.loads((d / f"sweep_{rid}.json").read_text()),
                   np.load(d / f"sweep_logits_{rid}.npz").files) is None

    # 3. stale ensemble: stops before any GPU time, naming the file and the fix
    def no_gpu(*a, **k):
        raise AssertionError("GPU work started before the stale input was caught")
    monkeypatch.setattr(M, "train_head", no_gpu)
    monkeypatch.setattr(M, "load_base", no_gpu)
    ens = find_ensemble(d, rid)
    assert ens is not None
    eid = ens.stem.removeprefix("ensemble_")
    _make_pre_d20(ens, d / f"ensemble_probs_{eid}.npz")
    (d / f"mcdropout_{mid}.json").unlink()  # even with nothing to reuse, nothing starts
    with pytest.raises(StaleResult, match=f"ensemble_{eid}.json: .*before D20.*notebook 03"):
        M.mc_dropout(base)


def test_notebook_03_brings_cached_member_sweeps_up_to_date_before_training(colab_drive,
                                                                            tmp_path):
    import shutil

    from src.ensemble import MEMBER_SEEDS, check_cached, find_run
    from src.evaluate import pre_d20
    d = tmp_path / "drive"
    shutil.copytree(colab_drive, d)
    base = find_run(d, 42, "efficientnet_b0")
    assert base is not None
    members = [find_run(d, s, "efficientnet_b0") for s in MEMBER_SEEDS]
    rid = json.loads(members[1].read_text())["run_id"]  # type: ignore[union-attr]
    _make_pre_d20(d / f"sweep_{rid}.json", d / f"sweep_logits_{rid}.npz")
    check_cached(base, MEMBER_SEEDS)
    sw = json.loads((d / f"sweep_{rid}.json").read_text())
    assert pre_d20(sw, np.load(d / f"sweep_logits_{rid}.npz").files) is None
    assert "rescored" in sw  # recorded, never silent
