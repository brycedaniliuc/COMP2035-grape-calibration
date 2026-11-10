"""tools/make_tables.py and the staleness check in src.verify."""

import json
import shutil
from pathlib import Path

from src import verify
from tools.make_tables import REPO, SEPTEMBER, build, run_numbers


def _status(gen: Path) -> dict[str, tuple[str, str]]:
    return {name: (status, detail) for name, _, status, detail in verify.generated_checks(gen)}


def test_committed_report_numbers_are_current():
    """report/generated must match the JSON in results/ and the current generator.
    If this fails: .venv/bin/python tools/make_tables.py"""
    st = _status(REPO / "report" / "generated")
    assert len(st) == 8 and all(s == "PASS" for s, _ in st.values()), st


def test_september_runs_are_read_correctly():
    src = run_numbers(json.loads(SEPTEMBER["source"].read_text()))
    sup = run_numbers(json.loads(SEPTEMBER["supplied"].read_text()))
    assert (src["n"], src["k"], sup["n"], sup["k"]) == (610, 610, 1805, 1805)
    assert round(src["acc_ci"][0], 5) == 0.99374  # Wilson lower bound at k = n = 610
    # zero validation errors at the kept checkpoint in both runs: not fitted temperatures
    assert src["degenerate"] and sup["degenerate"]


def test_staleness_is_detected(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    src = Path(shutil.copy(SEPTEMBER["source"], runs / "src.json"))
    sup = Path(shutil.copy(SEPTEMBER["supplied"], runs / "sup.json"))
    gen = tmp_path / "gen"
    monkeypatch.setattr(verify, "REPO", tmp_path)  # pending globs resolve under tmp_path
    build(src, sup, None, gen, pending_globs={"sweep": "colab_*/sweep_*.json"})
    st = _status(gen)
    assert all(s == "PASS" for s, _ in st.values()), st
    assert "\\pending{sweep}" in (gen / "table_sweep.tex").read_text()
    assert "\\newcommand{\\testacc}{1.0000\\xspace}" in (gen / "numbers.tex").read_text()

    # a hand edit of a generated number
    t = gen / "table_results.tex"
    t.write_text(t.read_text().replace("0.9222", "0.9300"))
    assert _status(gen)["generated table_results.tex is current"] == ("FAIL", "edited by hand")

    # the source JSON changes (a re-run): every file built from it is stale
    build(src, sup, None, gen, pending_globs={"sweep": "colab_*/sweep_*.json"})
    d = json.loads(src.read_text())
    d["mean_confidence"] = 0.5
    src.write_text(json.dumps(d))
    st = _status(gen)
    assert st["generated table_results.tex is current"][0] == "FAIL"
    assert st["generated table_sweep.tex is current"][0] == "PASS"  # not built from it

    # the pending sweep appears
    build(src, sup, None, gen, pending_globs={"sweep": "colab_*/sweep_*.json"})
    (tmp_path / "colab_x").mkdir()
    (tmp_path / "colab_x" / "sweep_a.json").write_text("{}")
    st = _status(gen)
    assert st["generated table_sweep.tex is current"] == (
        "FAIL", "pending source now exists: colab_*/sweep_*.json")
    assert st["generated table_results.tex is current"][0] == "PASS"

    # a file not written by the tool
    (gen / "table_extra.tex").write_text("\\begin{tabular}{l}x\\end{tabular}\n")
    assert _status(gen)["generated table_extra.tex is current"][0] == "FAIL"
