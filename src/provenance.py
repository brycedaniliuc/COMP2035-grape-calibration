"""Is this a result? Every output is labelled from the run's own record, so a fixture or
stand-in number can never pass for a real one.

A run is a result only if it is the protocol run on the full dataset: ResNet-50 with ImageNet
weights, 5 + 25 epochs, trained on the 9,027-image dataset. Anything else (the 80-image test
fixture, the CPU smoke runs with a stand-in model) carries a "NOT A RESULT: ..." label in its
JSON and on its figures, and tools/make_tables.py will not turn it into paper numbers.
"""

from __future__ import annotations

FULL_DATASET_IMAGES = 9027
PROTOCOL_EPOCHS = (5, 25)


def not_a_result(res: dict) -> str | None:
    """None for a real result, otherwise the reasons it is not one."""
    cfg = res.get("config", {})
    if "MODEL" in cfg:  # the September 2026 runs, in the notebook's own format
        model, pretrained = cfg["MODEL"], True
        epochs = (cfg.get("HEAD_EPOCHS"), cfg.get("FINETUNE_EPOCHS"))
    else:
        model, pretrained = cfg.get("model"), cfg.get("pretrained")
        epochs = (cfg.get("head_epochs"), cfg.get("finetune_epochs"))
    n = (res.get("audit") or {}).get("n_images")
    why = []
    if n != FULL_DATASET_IMAGES:
        why.append(f"trained on {n} images, not the full {FULL_DATASET_IMAGES} "
                   "(test fixture or partial data)")
    if model != "resnet50":
        why.append(f"stand-in model {model}, not ResNet-50")
    if not pretrained:
        why.append("no ImageNet weights")
    if epochs != PROTOCOL_EPOCHS:
        why.append(f"{epochs[0]} + {epochs[1]} epochs, not the protocol's 5 + 25")
    return "NOT A RESULT: " + "; ".join(why) if why else None
