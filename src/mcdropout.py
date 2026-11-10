"""MC dropout (upgrade 3; Gal & Ghahramani 2016), the last confidence method of §3.2.

The seed 42 run's backbone is kept exactly as trained: frozen, BatchNorm in evaluation mode.
A dropout layer (p = 0.5) goes before the final linear layer, and that head is retrained for
the protocol's 5 head epochs (AdamW, cosine schedule, class-weighted cross-entropy with label
smoothing 0.1, training augmentation), keeping the epoch with the lowest validation loss.

Prediction: 20 stochastic forward passes with dropout active; the mean softmax is the
prediction and the spread between passes is the uncertainty (predictive variance, and mutual
information as the fourth score, D13). Because the backbone is deterministic, each image's
features are computed once per condition and the 20 seeded dropout masks are applied to
them: exactly the same as 20 full passes, and reproducible.

It goes through the same sweep as the single model and the ensemble: 4 corruptions x
severities 0-5 on the test and validation images, Wilson and bootstrap intervals, coverage
and AURC for every score, and the threshold chosen on validation (D10). D18 records the
choices (p, the number of passes, head-only retraining).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import Config
from src.data import build_transforms, class_list, index_dataset, make_loader, make_splits
from src.degrade import CORRUPTIONS, SEVERITIES, DegradedDataset, severity_parameter
from src.ensemble import UNCERTAINTY_NOTE, evaluate_samples, find_run, pseudo_logits
from src.evaluate import threshold_analysis
from src.provenance import not_a_result
from src.train import (
    _json_safe,
    build_model,
    class_weights,
    configure_determinism,
    select_device,
    set_seed,
)

P_DROP = 0.5
N_PASSES = 20
MASK_SEED = 20261110  # dropout masks: pass s uses MASK_SEED + s, the same in every condition


def mc_id(base_run_id: str) -> str:
    return f"{base_run_id}_mcdropout"


def split_model(model: nn.Module, name: str) -> tuple[nn.Module, nn.Linear]:
    """(feature extractor, final linear layer). The extractor ends where the linear began."""
    if name == "resnet50":
        linear = model.fc
        model.fc = nn.Identity()
    elif name == "efficientnet_b0":
        classifier = model.classifier
        assert isinstance(classifier, nn.Sequential)
        linear = classifier[1]
        classifier[1] = nn.Identity()
    else:
        raise ValueError(f"unknown model: {name}")
    assert isinstance(linear, nn.Linear)
    return model, linear


def load_base(results_path: Path, device: torch.device) -> tuple[dict, nn.Module, nn.Linear]:
    res = json.loads(results_path.read_text())
    model, _ = build_model(res["config"]["model"], len(res["classes"]), pretrained=False)
    model.load_state_dict(torch.load(results_path.parent / res["artefacts"]["checkpoint"],
                                     map_location=device, weights_only=True))
    feats, linear = split_model(model, res["config"]["model"])
    feats = feats.to(device).eval()
    for q in feats.parameters():
        q.requires_grad_(False)
    return res, feats, linear.to(device)


def config_of(res: dict, data_root: str | None) -> Config:
    keep = {k: v for k, v in res["config"].items() if k in Config().to_dict()}
    return Config(**{**keep, **({"data_root": data_root} if data_root else {})})


def train_head(results_path: str | Path, data_root: str | None = None) -> Path:
    """Retrain the head with dropout. Reused if this run's head is already finished."""
    results_path = Path(results_path)
    d = results_path.parent
    device = select_device()
    res, feats, linear = load_base(results_path, device)
    mid = mc_id(res["run_id"])
    done = d / f"mcdropout_train_{mid}.json"
    if done.exists():
        print(f"{mid}: reusing the trained head")
        return done
    cfg = config_of(res, data_root)
    set_seed(cfg.training_seed)
    configure_determinism(cfg.deterministic)
    records = index_dataset(cfg.data_root)
    classes = class_list(records)
    assert classes == res["classes"]
    splits = make_splits(records, cfg)
    assert splits.summary()["split_sizes"] == res["split_sizes"], "split differs from the run"
    cls2idx = {c: i for i, c in enumerate(classes)}
    train_tf, eval_tf = build_transforms(cfg)
    train_dl = make_loader(splits.train, train_tf, cls2idx, cfg, device, shuffle=True)
    val_dl = make_loader(splits.val, eval_tf, cls2idx, cfg, device, shuffle=False)

    head = nn.Sequential(nn.Dropout(P_DROP), nn.Linear(linear.in_features, linear.out_features))
    head[1].load_state_dict(linear.state_dict())  # start from the trained head
    head = head.to(device)
    weights = (class_weights([r.cls for r in splits.train], classes).to(device)
               if cfg.class_weighted_loss else None)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.head_epochs,
                                                       eta_min=cfg.lr_min_stage1)
    amp = cfg.use_amp and device.type == "cuda"
    ckpt = d / f"mcdropout_head_{mid}.pt"

    def features(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device.type, enabled=amp):
            return feats(x.to(device, non_blocking=True)).float()

    best, history, t0 = float("inf"), [], time.time()
    for epoch in range(1, cfg.head_epochs + 1):
        head.train()
        tr_loss, seen = 0.0, 0
        for x, y in train_dl:
            y = y.to(device)
            loss = criterion(head(features(x)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss, seen = tr_loss + loss.item() * y.size(0), seen + y.size(0)
        sched.step()
        head.eval()  # dropout off: the deterministic validation loss picks the epoch
        va_loss, va_ok, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                y = y.to(device)
                z = head(features(x))
                va_loss += criterion(z, y).item() * y.size(0)
                va_ok, n = va_ok + int((z.argmax(1) == y).sum()), n + y.size(0)
        row = {"epoch": epoch, "train_loss": tr_loss / seen, "val_loss": va_loss / n,
               "val_acc": va_ok / n}
        history.append(row)
        print(f"MC dropout head epoch {epoch}: {row}", flush=True)
        if row["val_loss"] < best:  # saved to Drive as it improves
            best = row["val_loss"]
            torch.save(head.state_dict(), ckpt)
    done.write_text(json.dumps(_json_safe({
        "mc_id": mid, "base_run_id": res["run_id"], "p_drop": P_DROP,
        "head_epochs": cfg.head_epochs, "best_val_loss": best, "history": history,
        "minutes": round((time.time() - t0) / 60, 2), "head": ckpt.name}), indent=1))
    return done


@torch.no_grad()
def passes(feats_x: torch.Tensor, linear: nn.Linear, n: int = N_PASSES) -> np.ndarray:
    """n stochastic passes of the head on fixed features: (passes, images, classes)."""
    out = []
    for s in range(n):
        g = torch.Generator().manual_seed(MASK_SEED + s)
        keep = (torch.rand(feats_x.shape, generator=g) >= P_DROP).float() / (1 - P_DROP)
        out.append(linear(feats_x * keep).numpy())
    return np.stack(out).astype(np.float32)


def mc_dropout(results_path: str | Path, data_root: str | None = None) -> dict:
    """Train the head if needed, then the MC dropout sweep. Reused if already finished."""
    results_path = Path(results_path)
    d = results_path.parent
    res0 = json.loads(results_path.read_text())
    mid = mc_id(res0["run_id"])
    out_path = d / f"mcdropout_{mid}.json"
    if out_path.exists():
        print(f"{mid}: reusing the finished MC dropout sweep")
        return json.loads(out_path.read_text())
    trained = json.loads(train_head(results_path, data_root).read_text())
    device = select_device()
    res, feats, _ = load_base(results_path, device)
    state = torch.load(d / trained["head"], map_location="cpu", weights_only=True)
    k, n_in = state["1.weight"].shape  # the head's own size: no width assumed
    linear = nn.Linear(n_in, k)
    linear.load_state_dict({"weight": state["1.weight"], "bias": state["1.bias"]})
    linear.eval()
    cfg = config_of(res, data_root)
    set_seed(cfg.seed)
    configure_determinism(cfg.deterministic)
    records = index_dataset(cfg.data_root)
    classes = class_list(records)
    splits = make_splits(records, cfg)
    assert splits.summary()["split_sizes"] == res["split_sizes"], "split differs from the run"
    cls2idx = {c: i for i, c in enumerate(classes)}
    amp = cfg.use_amp and device.type == "cuda"

    def features(rows: list, corruption: str, severity: int) -> tuple[torch.Tensor, np.ndarray]:
        dl = DataLoader(DegradedDataset(rows, cls2idx, cfg, corruption, severity),
                        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=device.type == "cuda")
        fs, ys = [], []
        for x, y in dl:
            with torch.no_grad(), torch.autocast(device.type, enabled=amp):
                fs.append(feats(x.to(device)).float().cpu())
            ys.append(y)
        return torch.cat(fs), torch.cat(ys).numpy()

    keys = [f"{c}_s{s}" for c in CORRUPTIONS for s in SEVERITIES]
    test_s, val_s, y, y_val, det = {}, {}, None, None, None
    for c in CORRUPTIONS:
        for s in SEVERITIES:
            k = f"{c}_s{s}"
            ft, y = features(splits.test, c, s)
            fv, y_val = features(splits.val, c, s)
            test_s[k], val_s[k] = passes(ft, linear), passes(fv, linear)
            if k == keys[0]:
                with torch.no_grad():
                    det = linear(ft).numpy()  # dropout off: the deterministic prediction
        print(f"  MC dropout, {c}: severities 0-5 done", flush=True)
    assert y is not None and y_val is not None and det is not None
    rows, probs, val_probs = evaluate_samples(
        {k: list(v) for k, v in test_s.items()}, {k: list(v) for k, v in val_s.items()}, y,
        classes, cfg.n_bins, cfg.seed,
        {k: severity_parameter(k.rsplit("_s", 1)[0], int(k.rsplit("_s", 1)[1])) for k in keys})
    val_probs["labels"] = y_val
    out = {"mc_id": mid, "not_a_result": not_a_result(res), "classes": classes,
           "base_run_id": res["run_id"], "p_drop": P_DROP, "n_passes": N_PASSES,
           "mask_seed": MASK_SEED, "training": trained,
           "single_model": {"run_id": res["run_id"], "sweep": f"sweep_{res['run_id']}.json"},
           "n_test": int(y.size), "n_val": int(y_val.size),
           "prediction": f"mean softmax over {N_PASSES} dropout passes, T = 1",
           "deterministic_clean_accuracy": float((det.argmax(1) == y).mean()),
           "uncertainty": UNCERTAINTY_NOTE, "rows": rows,
           "threshold": threshold_analysis(*pseudo_logits(probs, val_probs))}
    np.savez(d / f"mcdropout_probs_{mid}.npz", **probs)
    np.savez(d / f"mcdropout_val_probs_{mid}.npz", **val_probs)
    np.savez(d / f"mcdropout_passes_{mid}.npz", labels=y, **test_s)  # type: ignore[arg-type]
    np.savez(d / f"mcdropout_val_passes_{mid}.npz", labels=y_val, **val_s)  # type: ignore[arg-type]
    out_path.write_text(json.dumps(_json_safe(out), indent=1, allow_nan=False))
    print(f"MC dropout {mid}: {N_PASSES} passes, p = {P_DROP}; clean accuracy "
          f"{rows[0]['accuracy']:.4f}, mean confidence "
          f"{rows[0]['temperature']['T=1']['mean_confidence']:.4f}")
    print(f"written -> {out_path}")
    return out


def find_mc(out_dir: str | Path, model: str = "resnet50") -> Path | None:
    base = find_run(out_dir, 42, model)
    if base is None:
        return None
    p = Path(out_dir) / f"mcdropout_{mc_id(json.loads(base.read_text())['run_id'])}.json"
    return p if p.exists() else None
