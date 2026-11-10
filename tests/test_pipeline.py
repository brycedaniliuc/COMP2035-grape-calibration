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
    logit_keys = [k for k in npz.files if "__" not in k]
    assert len(logit_keys) == 1 + len(CORRUPTIONS) * len(SEVERITIES)
    # beside the logits, the exact arrays every curve was ranked on (D20): max softmax and
    # entropy at every T, and the T = 1 block with margin
    assert len(npz.files) - len(logit_keys) == len(CORRUPTIONS) * len(SEVERITIES) * (
        2 * len(sw["temperatures"]) + 3)
    val = np.load(tmp_path / f"sweep_val_logits_{res['run_id']}.npz")
    assert set(val.files) == set(logit_keys) and val["labels"].size == sw["n_val"]

    # verify ranks on those arrays and the AURC agreement is exact; without them (an output
    # from before D20) the check fails rather than recomputing scores and guessing
    from src.verify import tier1
    run_path = tmp_path / f"results_{res['run_id']}.json"
    aurc = {c[0]: c for c in tier1(run_path, require_sweep=True)}
    assert aurc["sweep: AURC at every T re-derived"][2] == "PASS"
    assert aurc["sweep: stored scores = float64 softmax, entropy, margin"][2] == "PASS"
    old = tmp_path / "old"
    old.mkdir()
    for f in tmp_path.glob(f"*{res['run_id']}*"):
        if f.is_file():
            (old / f.name).write_bytes(f.read_bytes())
    np.savez(old / f"sweep_logits_{res['run_id']}.npz", **{k: npz[k] for k in logit_keys})
    aurc = {c[0]: c for c in tier1(old / run_path.name, require_sweep=True)}
    assert aurc["sweep: AURC at every T re-derived"][2] == "FAIL"
    assert "no stored score arrays" in aurc["sweep: AURC at every T re-derived"][3]

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
    # AURC is reported at the precision it is stable to (D20): two decimals, never more
    from src.evaluate import report_aurc
    lines = [ln for ln in (forced / "table_sweep.tex").read_text().splitlines()
             if ln.endswith("\\\\") and "&" in ln and not ln.startswith(("Corruption", " &"))]
    assert lines and all(len(ln[:-2].rsplit("&", 1)[1].strip().split(".")[1]) == 2
                         for ln in lines)
    want = report_aurc(clean[0]["scores"]["max_softmax"]["aurc"])
    assert lines[0][:-2].rsplit("&", 1)[1].strip() == f"{want:.2f}"
