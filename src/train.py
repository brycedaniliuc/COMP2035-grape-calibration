"""Training loop, checkpointing and the end-to-end baseline run (notebook 01, cells 17-29).

    python -m src.train --data-root tests/fixture --model efficientnet_b0 \
        --batch-size 8 --head-epochs 1 --finetune-epochs 1 --num-workers 0 --out-dir outputs/fixture

Every Config field is a command-line flag (underscores become hyphens).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

# Must be set before CUDA initialises (select_device() calls get_device_name), or
# deterministic cuBLAS silently does not apply. Harmless on CPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import models

from src.audit import leaking_leaves, print_audit, supplied_split_audit
from src.calibrate import calibration_summary, lbfgs_temperature, softmax_outputs
from src.config import Config
from src.data import (
    build_transforms,
    class_list,
    index_dataset,
    make_loader,
    make_splits,
    save_split_lists,
)


def select_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
    print(f"device: {device.type} ({name}) | torch {torch.__version__}")
    return device


def environment(device: torch.device) -> dict:
    """Library versions and hardware, recorded with every run."""
    from importlib.metadata import version
    return {"device": device.type,
            "device_name": (torch.cuda.get_device_name(0) if device.type == "cuda"
                            else platform.processor()),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "scikit-learn": version("scikit-learn"), "scikit-image": version("scikit-image"),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "platform": platform.platform()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    """Deterministic kernels where they exist. On CUDA some ops (e.g. adaptive average
    pooling backward) have no deterministic version, so those warn rather than fail."""
    if not enabled:
        return
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "set before CUDA initialises"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=torch.cuda.is_available())


def build_model(name: str, n_classes: int, pretrained: bool) -> tuple[nn.Module, tuple[str]]:
    """Return the model and the parameter-name prefix of its classifier head."""
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m, ("fc",)
    if name == "efficientnet_b0":
        e = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        head = e.classifier[1]
        assert isinstance(head, nn.Linear)
        e.classifier[1] = nn.Linear(head.in_features, n_classes)
        return e, ("classifier",)
    raise ValueError(f"unknown model: {name}")


def class_weights(train_classes: list[str], classes: list[str]) -> torch.Tensor:
    """Inverse-frequency weights from the training partition only."""
    counts = collections.Counter(train_classes)
    n, k = len(train_classes), len(classes)
    assert all(counts[c] > 0 for c in classes), "a class is missing from train"
    return torch.tensor([n / (k * counts[c]) for c in classes], dtype=torch.float32)


class Trainer:
    def __init__(self, model: nn.Module, head: tuple[str], criterion: nn.Module,
                 cfg: Config, device: torch.device):
        self.model, self.head, self.criterion = model, head, criterion
        self.cfg, self.device = cfg, device
        self.amp = cfg.use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp)
        assert self.head_params() and self.backbone_params(), "head prefix matched nothing"

    def head_params(self) -> list[nn.Parameter]:
        return [p for n, p in self.model.named_parameters() if n.startswith(self.head)]

    def backbone_params(self) -> list[nn.Parameter]:
        return [p for n, p in self.model.named_parameters() if not n.startswith(self.head)]

    def run_epoch(self, loader: DataLoader, optimiser: torch.optim.Optimizer | None
                  ) -> tuple[float, float]:
        train = optimiser is not None
        self.model.train(train)
        tot_loss, correct, seen = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, enabled=self.amp):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                if optimiser is not None:
                    optimiser.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimiser)
                    self.scaler.update()
                tot_loss += loss.item() * y.size(0)
                correct += int((logits.argmax(1) == y).sum().item())
                seen += y.size(0)
        assert seen > 0, "empty loader"
        return tot_loss / seen, correct / seen

    def fit(self, train_dl: DataLoader, val_dl: DataLoader, ckpt: Path) -> dict:
        """Two-stage schedule. Best checkpoint by validation loss; early stopping in
        stage 2 only, measured against the best loss from either stage."""
        cfg = self.cfg
        history: list[dict] = []
        best_val, bad, stopped_early = math.inf, 0, False
        t0 = time.time()

        def epoch(stage: int, ep: int, n: int, opt: torch.optim.Optimizer,
                  sch: torch.optim.lr_scheduler.LRScheduler) -> bool:
            nonlocal best_val, bad
            tl, ta = self.run_epoch(train_dl, opt)
            vl, va = self.run_epoch(val_dl, None)
            sch.step()
            history.append({"stage": stage, "epoch": ep, "train_loss": tl, "train_acc": ta,
                            "val_loss": vl, "val_acc": va})
            improved = vl < best_val
            if improved:
                best_val, bad = vl, 0
                torch.save(self.model.state_dict(), ckpt)
            elif stage == 2:
                bad += 1
            print(f"  [{ep}/{n}] train {tl:.4f}/{ta:.4f}  val {vl:.4f}/{va:.4f}"
                  f"{'  *' if improved else ''}")
            return improved

        for p in self.backbone_params():
            p.requires_grad_(False)
        opt: torch.optim.Optimizer = torch.optim.AdamW(
            self.head_params(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
        sch: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.head_epochs, eta_min=cfg.lr_min_stage1)
        print("stage 1 - backbone frozen")
        for ep in range(1, cfg.head_epochs + 1):
            epoch(1, ep, cfg.head_epochs, opt, sch)

        if cfg.finetune_epochs > 0:
            for p in self.model.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(
                [{"params": self.backbone_params(), "lr": cfg.lr_backbone},
                 {"params": self.head_params(), "lr": cfg.lr_head}],
                weight_decay=cfg.weight_decay)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.finetune_epochs, eta_min=cfg.lr_min_stage2)
            print("stage 2 - full fine-tune")
            for ep in range(1, cfg.finetune_epochs + 1):
                epoch(2, ep, cfg.finetune_epochs, opt, sch)
                if bad >= cfg.patience:
                    print(f"  early stop - no validation improvement for {cfg.patience} epochs")
                    stopped_early = True
                    break

        self.model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        return {"history": history, "epochs_run": len(history), "stopped_early": stopped_early,
                "best_val_loss": best_val, "minutes": round((time.time() - t0) / 60, 2)}


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device
                   ) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    out, ys = [], []
    for x, y in loader:
        out.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def _json_safe(obj: object) -> object:
    """NaN -> None so the JSON is valid; anything else non-finite is an error."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        assert math.isfinite(obj), f"non-finite value in results: {obj}"
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    return obj


def run(cfg: Config, stamp: str | None = None) -> dict:
    """Index, audit, split, train, evaluate, calibrate; write every output. Returns results."""
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{cfg.model}_{cfg.split_mode}_{stamp}"
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / f"results_{run_id}.json").exists(), f"run {run_id} already exists"
    print(f"run id: {run_id}")
    print(json.dumps(cfg.to_dict(), indent=2))

    device = select_device()
    set_seed(cfg.seed)
    configure_determinism(cfg.deterministic)

    records = index_dataset(cfg.data_root)
    classes = class_list(records)
    cls2idx = {c: i for i, c in enumerate(classes)}
    audit = supplied_split_audit(records)
    print_audit(audit, leaking_leaves(records))

    set_seed(cfg.seed)
    splits = make_splits(records, cfg)
    print(json.dumps(splits.summary(), indent=2))

    train_tf, eval_tf = build_transforms(cfg)
    train_dl = make_loader(splits.train, train_tf, cls2idx, cfg, device, shuffle=True)
    val_dl = make_loader(splits.val, eval_tf, cls2idx, cfg, device, shuffle=False)
    test_dl = make_loader(splits.test, eval_tf, cls2idx, cfg, device, shuffle=False)

    model, head = build_model(cfg.model, len(classes), cfg.pretrained)
    model = model.to(device)
    weights = (class_weights([r.cls for r in splits.train], classes).to(device)
               if cfg.class_weighted_loss else None)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

    ckpt = out / f"best_{run_id}.pt"
    fit = Trainer(model, head, criterion, cfg, device).fit(train_dl, val_dl, ckpt)

    test_logits, test_y = collect_logits(model, test_dl, device)
    val_logits, val_y = collect_logits(model, val_dl, device)
    assert test_y.tolist() == [cls2idx[r.cls] for r in splits.test], "test label order drifted"

    t_fit = lbfgs_temperature(val_logits, val_y, cfg.temperature_lr, cfg.temperature_max_iter)
    cal = calibration_summary(test_logits, test_y, t_fit["t_lbfgs_endpoint"], cfg.n_bins)
    _, preds, _, correct = softmax_outputs(test_logits, test_y)
    cm = np.zeros((len(classes), len(classes)), int)
    for t_, p_ in zip(test_y, preds, strict=True):
        cm[t_, p_] += 1

    logits_path = out / f"logits_{run_id}.npz"
    np.savez(logits_path, test_logits=test_logits, test_labels=test_y,
             val_logits=val_logits, val_labels=val_y,
             t_lbfgs_endpoint=np.array([t_fit["t_lbfgs_endpoint"]]))
    splits_path = out / f"splits_{run_id}.json"
    splits_path.write_text(json.dumps(save_split_lists(splits, cfg.data_root), indent=1))

    results = {
        "run_id": run_id,
        "config": cfg.to_dict(),
        "environment": environment(device),
        "classes": classes,
        "audit": audit,
        **splits.summary(),
        **{k: v for k, v in fit.items() if k != "history"},
        "test_accuracy": cal["accuracy"],
        "per_class_accuracy": {c: float(correct[test_y == i].mean())
                               for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
        "mean_confidence": cal["mean_confidence"], "ece": cal["ece"],
        "reliability_bins": cal["reliability_bins"],
        # LBFGS endpoint over T on validation NLL. Not a fitted temperature when
        # t_lbfgs["degenerate"] is true (zero validation errors): see DECISIONS.md D1.
        "t_lbfgs": t_fit,
        "mean_confidence_at_t_lbfgs": cal["mean_confidence_at_t"],
        "ece_at_t_lbfgs": cal["ece_at_t"],
        "reliability_bins_at_t_lbfgs": cal["reliability_bins_at_t"],
        "history": fit["history"],
        "artefacts": {"checkpoint": ckpt.name, "logits": logits_path.name,
                      "splits": splits_path.name},
    }
    assert set(results["config"]) == {f.name for f in fields(Config)}, "config not fully recorded"
    res_path = out / f"results_{run_id}.json"
    res_path.write_text(json.dumps(_json_safe(results), indent=2, allow_nan=False))

    bar = "=" * 62
    print(f"{bar}\n HEADLINE NUMBERS\n{bar}")
    print(f" split mode            {cfg.split_mode}")
    print(f" model                 {cfg.model}")
    print(f" test images           {len(test_y):,}")
    print(f" test accuracy         {cal['accuracy']:.4f}")
    print(f" mean confidence       {cal['mean_confidence']:.4f}")
    print(f" ECE                   {cal['ece']:.4f}")
    print(f" T, LBFGS endpoint     {t_fit['t_lbfgs_endpoint']:.3f}"
          f"{'  (DEGENERATE: zero validation errors)' if t_fit['degenerate'] else ''}")
    print(f" ECE at that T         {cal['ece_at_t']:.4f}")
    print(f"{bar}\nwritten -> {res_path}")
    return results


def parse_config(argv: list[str] | None = None) -> Config:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    for f in fields(Config):
        d = getattr(defaults, f.name)
        if isinstance(d, bool):
            ap.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name,
                            type=lambda s: {"true": True, "false": False}[s.lower()],
                            default=d, metavar="{true,false}")
        else:
            ap.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, type=type(d), default=d)
    return Config(**vars(ap.parse_args(argv)))


if __name__ == "__main__":
    run(parse_config())
