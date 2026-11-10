"""tools/import_colab.py on the stand-in Drive folder: one command copies, regenerates the
report numbers and verifies; nothing that is not a result becomes a paper number."""

import re

from tools.import_colab import main


def test_import_copies_regenerates_and_verifies(colab_drive, tmp_path, capsys):
    results, gen = tmp_path / "results", tmp_path / "generated"
    code = main([str(colab_drive), "--results-root", str(results), "--generated", str(gen)])
    out = capsys.readouterr().out
    assert code == 0, out
    dest = results / "colab_20261111"
    assert not list(dest.rglob("*.pt"))  # checkpoints stay in Drive
    assert list(dest.glob("figures_*")) and list(dest.glob("mcdropout_passes_*.npz"))
    assert "ALL PASS" in out and " FAIL" not in out.split("3. verification")[1]
    for target in ("ensemble_", "plantseg_", "mcdropout_", "report numbers", "results_"):
        assert target in out
    # stand-in data never becomes a paper number: every macro is pending, and the header
    # says why each source was excluded
    numbers = (gen / "numbers.tex").read_text()
    values = re.findall(r"\\newcommand\{\\\w+\}\{(.*?)\\xspace\}", numbers)
    assert values and all(v.startswith("\\pending{") for v in values)
    assert "% excluded: source" in numbers and "NOT A RESULT" in numbers


def test_import_never_overwrites(colab_drive, tmp_path, capsys):
    results, gen = tmp_path / "results", tmp_path / "generated"
    assert main([str(colab_drive), "--results-root", str(results), "--generated", str(gen)]) == 0
    assert main([str(colab_drive), "--results-root", str(results), "--generated", str(gen)]) == 0
    assert "already here" in capsys.readouterr().out  # a second import is harmless
    target = next((results / "colab_20261111").glob("splits_*.json"))
    target.write_text("{}")  # the copy now differs from Drive
    assert main([str(colab_drive), "--results-root", str(results), "--generated", str(gen)]) == 1
    assert "STOP" in capsys.readouterr().out and target.read_text() == "{}"
