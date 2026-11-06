"""Run configuration. Values only: no logic beyond checking the values are legal.

Defaults are the protocol of the September 2026 runs (notebook 01, after Canoski et al.
2026), with two additions recorded in DECISIONS.md: `deterministic` (D2) and the
temperature optimiser settings (D1), which were previously hard-coded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SPLIT_MODES = ("source", "naive", "supplied")
MODELS = ("resnet50", "efficientnet_b0")


@dataclass(frozen=True)
class Config:
    # ---- data ------------------------------------------------------------
    data_root: str = "/content/grape_dataset"  # must contain train/ and test/

    # ---- partitioning ----------------------------------------------------
    # "source"   : pool everything, split by source leaf, evaluate on originals only
    # "naive"    : pool everything, split by image, ignore leaf identity
    # "supplied" : use the train/ and test/ folders exactly as provided
    split_mode: str = "source"
    eval_on_originals_only: bool = True  # applies to "source" only
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    # ---- model -----------------------------------------------------------
    model: str = "resnet50"
    pretrained: bool = True  # ImageNet weights
    img_size: int = 224
    resize: int = 256

    # ---- training: Canoski et al. 2026, Section 5.2 (shortened budget) ----
    batch_size: int = 64
    head_epochs: int = 5  # stage 1, backbone frozen
    finetune_epochs: int = 25  # stage 2 (paper: 100)
    patience: int = 5  # (paper: 20)
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    lr_min_stage1: float = 1e-5
    lr_min_stage2: float = 1e-6
    label_smoothing: float = 0.1
    class_weighted_loss: bool = True  # inverse frequency over train; see D3
    num_workers: int = 2
    seed: int = 42
    use_amp: bool = True  # only takes effect on CUDA
    deterministic: bool = True  # see D2

    # ---- calibration -----------------------------------------------------
    n_bins: int = 15
    temperature_lr: float = 0.1  # LBFGS, as in notebook 01; see D1
    temperature_max_iter: int = 100

    # ---- output ----------------------------------------------------------
    out_dir: str = "outputs"

    def __post_init__(self) -> None:
        assert self.split_mode in SPLIT_MODES, f"bad split_mode: {self.split_mode}"
        assert self.model in MODELS, f"bad model: {self.model}"
        assert abs(self.train_frac + self.val_frac + self.test_frac - 1.0) < 1e-9, \
            "fractions must sum to 1"
        assert self.head_epochs >= 1 and self.finetune_epochs >= 0
        assert self.batch_size >= 1 and self.n_bins >= 1 and self.num_workers >= 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
