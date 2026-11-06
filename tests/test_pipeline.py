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
