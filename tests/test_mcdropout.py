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
