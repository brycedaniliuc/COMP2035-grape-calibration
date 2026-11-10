import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixture"


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return FIXTURE


@pytest.fixture(scope="session")
def manifest() -> dict:
    return json.loads((FIXTURE / "manifest.json").read_text())


@pytest.fixture(scope="session")
def colab_drive(tmp_path_factory) -> Path:
    """Everything notebooks 01-05 leave in Drive, built on the 80-image fixture with a small
    stand-in model (so every output is labelled NOT A RESULT). PlantSeg gets a stand-in zip
    with the real filter's image names, filled with fixture images."""
    import zipfile

    from src import figures as F
    from src.config import Config
    from src.data import index_dataset
    from src.ensemble import ensemble_sweep, ensure_member
    from src.evaluate import sweep
    from src.mcdropout import mc_dropout
    from src.plantseg import FILTER, SPLIT_DIR, plantseg_eval, selection
    from src.train import run

    d = tmp_path_factory.mktemp("COMP2035_runs")
    cfg = Config(data_root=str(FIXTURE), model="efficientnet_b0", pretrained=False,
                 batch_size=8, head_epochs=1, finetune_epochs=0, num_workers=0, out_dir=str(d))
    res = run(cfg, stamp="20261111-090000")
    base = d / f"results_{res['run_id']}.json"
    (d / "audit_20261111-085900.json").write_text(json.dumps(res["audit"]))
    run(Config(**{**cfg.to_dict(), "split_mode": "supplied"}), stamp="20261111-091000")
    sweep(base)
    members = [ensure_member(base, s) for s in (42, 43, 44)]
    ens = ensemble_sweep(members, stamp="20261111-093000")
    images = [r.path for r in index_dataset(FIXTURE)]
    rows = [r for g in selection(FILTER).values() for r in g]
    zpath = tmp_path_factory.mktemp("plantseg") / "Plantseg.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for i, r in enumerate(rows):
            z.write(images[i % len(images)],
                    f"Plantseg/images/{SPLIT_DIR[r['plantseg_split']]}/{r['image']}")
    assert len(rows) == 334
    plantseg_eval(base, zpath, check_md5=False)
    mc = mc_dropout(base)
    rid = res["run_id"]
    F.render_sweep(d / f"sweep_{rid}.json", d / f"figures_{rid}", tag=rid)
    F.render_ensemble(d / f"ensemble_{ens['ens_id']}.json", d / f"figures_{ens['ens_id']}",
                      tag=ens["ens_id"])
    F.render_plantseg(d / f"plantseg_{rid}.json", d / f"figures_{rid}", tag=rid)
    F.render_methods(d / f"mcdropout_{mc['mc_id']}.json", d / f"figures_{rid}", tag=rid)
    return d
