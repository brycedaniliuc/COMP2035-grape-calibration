"""End-to-end baseline run on the fixture, twice, to prove it runs and is deterministic.

Uses random initialisation (pretrained=False) so the test needs no network access.
"""

import json

import numpy as np

from src.config import Config
from src.train import run

VOLATILE = {"run_id", "minutes", "artefacts"}


def test_pipeline_runs_and_is_bit_identical(fixture_root, tmp_path):
    def once(tag: str) -> tuple[dict, dict]:
        cfg = Config(data_root=str(fixture_root), model="efficientnet_b0", pretrained=False,
                     batch_size=8, head_epochs=1, finetune_epochs=1, num_workers=0,
                     out_dir=str(tmp_path / tag))
        res = run(cfg, stamp=tag)
        npz = np.load(tmp_path / tag / res["artefacts"]["logits"])
        return res, {k: npz[k] for k in npz.files}

    r1, l1 = once("a")
    r2, l2 = once("b")

    for k in l1:
        assert np.array_equal(l1[k], l2[k]), f"{k} differs between identical runs"
    strip = lambda r: {k: v for k, v in r.items() if k not in VOLATILE | {"config"}}  # noqa: E731
    assert json.dumps(strip(r1), sort_keys=True) == json.dumps(strip(r2), sort_keys=True)

    # outputs exist and are valid JSON with no NaN
    saved = json.loads((tmp_path / "a" / "results_efficientnet_b0_source_a.json").read_text())
    assert saved["split_sizes"]["test"] == len(l1["test_labels"])
    assert set(saved["config"]) == set(Config().to_dict())


def test_sweep_on_fixture_checkpoint(fixture_root, tmp_path):
    """Train once, then sweep: severity 0 reproduces the clean run, T is held fixed
    across all 24 rows (both asserted inside sweep), and the threshold analysis exists."""
    from src.degrade import CORRUPTIONS, SEVERITIES
    from src.evaluate import sweep

    cfg = Config(data_root=str(fixture_root), model="efficientnet_b0", pretrained=False,
                 batch_size=8, head_epochs=1, finetune_epochs=1, num_workers=0,
                 out_dir=str(tmp_path))
    res = run(cfg, stamp="s")
    sw = sweep(tmp_path / f"results_{res['run_id']}.json")
    assert len(sw["rows"]) == len(CORRUPTIONS) * len(SEVERITIES)
    clean = [r for r in sw["rows"] if r["severity"] == 0]
    assert all(r["accuracy"] == res["test_accuracy"] for r in clean)
    th = sw["threshold"]
    assert set(th["by_score"]) == {"max_softmax", "neg_entropy", "margin"}
    assert th["limits"]
    for by_t in th["by_score"].values():
        for r in by_t.values():
            assert set(r) == {"validation", "test", "in_sample_test"}
            if r["validation"]["found"]:  # applied unchanged to the held-out test images
                assert r["test"]["threshold"] == r["validation"]["threshold"]
    for row in sw["rows"]:
        for cond in row["temperature"].values():
            assert set(cond["coverage"]) == {"max_softmax", "neg_entropy"}
    assert sw["clean_check"]["max_abs_logit_diff_test"] == 0.0
    npz = np.load(tmp_path / f"sweep_logits_{res['run_id']}.npz")
    assert len(npz.files) == 1 + len(CORRUPTIONS) * len(SEVERITIES)
    val = np.load(tmp_path / f"sweep_val_logits_{res['run_id']}.npz")
    assert set(val.files) == set(npz.files) and val["labels"].size == sw["n_val"]

    # report tables and macros: a fixture run is not a result (D19), so by default every
    # number is pending; only the test flag lets it through, labelled NOT A RESULT
    from src.verify import generated_checks
    from tools.make_tables import build
    assert sw["not_a_result"].startswith("NOT A RESULT")
    src = tmp_path / f"results_{res['run_id']}.json"
    sweep_path = tmp_path / f"sweep_{res['run_id']}.json"
    gen = tmp_path / "generated"
    build(src, None, sweep_path, gen)
    assert all(c[2] == "PASS" for c in generated_checks(gen))
    assert "\\pending{threshold}" in (gen / "table_threshold.tex").read_text()
    assert "% excluded: source" in (gen / "numbers.tex").read_text()
    forced = tmp_path / "forced"
    build(src, None, sweep_path, forced, allow_not_a_result=True)
    assert all(c[2] == "PASS" for c in generated_checks(forced))
    assert "of 24" in (forced / "table_threshold.tex").read_text()
    assert "NOT A RESULT" in (forced / "table_threshold.tex").read_text()
    assert "\\notaresult{" in (forced / "numbers.tex").read_text()
