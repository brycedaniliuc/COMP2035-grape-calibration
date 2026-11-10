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


def test_a_sweep_from_before_d20_is_rescored_once(colab_drive, tmp_path, capsys):
    """A Drive folder whose sweeps predate the stored score arrays (D20): the import
    re-scores them from the saved logits (never changed), verification then passes exactly,
    and a second import recognises the re-scored files as the Drive files they came from."""
    import json
    import shutil

    import numpy as np
    old = tmp_path / "drive"
    shutil.copytree(colab_drive, old, ignore=shutil.ignore_patterns("*.pt"))
    for p in old.glob("sweep_logits_*.npz"):
        z = dict(np.load(p))
        np.savez(p, **{k: v for k, v in z.items() if "__" not in k})
    originals = {p.name: dict(np.load(p)) for p in old.glob("sweep_logits_*.npz")}
    results, gen = tmp_path / "results", tmp_path / "generated"
    code = main([str(old), "--results-root", str(results), "--generated", str(gen)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "re-scored sweep_" in out and "logits unchanged" in out and "ALL PASS" in out
    dest = results / "colab_20261111"
    for name, z in originals.items():
        after = dict(np.load(dest / name))
        assert all(np.array_equal(after[k], v) for k, v in z.items())  # logits untouched
        assert len(after) > len(z)  # ranked arrays added
        rec = json.loads((dest / name.replace("sweep_logits_", "sweep_")
                          .replace(".npz", ".json")).read_text())["rescored"]
        assert set(rec) >= {"originals_sha256", "curves_changed", "machine", "date"}
    assert main([str(old), "--results-root", str(results), "--generated", str(gen)]) == 0
    out = capsys.readouterr().out
    assert "already here (re-scored, D20)" in out and "CONFLICT" not in out
    assert "already re-scored on" in out
