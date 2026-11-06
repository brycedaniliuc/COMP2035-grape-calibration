"""Expected calibration error, reliability bins and temperature scaling.

Ported from notebook 01 (cells 21, 23, 27). ECE uses equal-width bins on the max softmax
probability; the first bin is closed at both ends, every other bin is (lo, hi].
Temperature is T = exp(log_T), so T stays positive, fitted by LBFGS on validation NLL.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def softmax_outputs(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs, preds, confs, correct) for logits / temperature."""
    assert temperature > 0, f"temperature must be positive, got {temperature}"
    probs = F.softmax(torch.as_tensor(logits, dtype=torch.float32) / temperature, 1).numpy()
    preds = probs.argmax(1)
    confs = probs.max(1)
    correct = (preds == labels).astype(float)
    return probs, preds, confs, correct


def ece_and_bins(confs: np.ndarray, correct: np.ndarray, n_bins: int = 15
                 ) -> tuple[float, list[dict]]:
    """ECE = sum over bins of (n_bin / N) * |acc_bin - conf_bin|.

    Returns the ECE and one row per bin: lo, hi, n, acc, conf (acc and conf are NaN for
    an empty bin).
    """
    confs = np.asarray(confs, dtype=float)
    correct = np.asarray(correct, dtype=float)
    assert confs.ndim == 1 and confs.shape == correct.shape and confs.size > 0
    assert np.all((confs >= 0) & (confs <= 1)), "confidences must lie in [0, 1]"

    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows, weight = 0.0, [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if not m.any():
            rows.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "acc": math.nan, "conf": math.nan})
            continue
        acc, conf = float(correct[m].mean()), float(confs[m].mean())
        w = float(m.mean())
        ece += w * abs(acc - conf)
        weight += w
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "acc": acc, "conf": conf})
    assert sum(r["n"] for r in rows) == confs.size, "a prediction fell in no bin"
    assert abs(weight - 1.0) < 1e-9, f"bin weights sum to {weight}, not 1"
    return ece, rows


def lbfgs_temperature(val_logits: np.ndarray, val_labels: np.ndarray,
                    lr: float = 0.1, max_iter: int = 100) -> dict:
    """Run LBFGS on validation NLL over T. Same optimiser and settings as notebook 01.

    The output is the LBFGS *endpoint*. It is a fitted temperature only when the fit is
    not degenerate.

    If the validation set has no errors, NLL has no minimum (it keeps falling as T -> 0)
    and T is wherever LBFGS stops. This is reported as `degenerate`, not hidden (D1).
    """
    logits = torch.as_tensor(val_logits, dtype=torch.float32)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_T) keeps T positive
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nll(logits / torch.exp(log_t), labels)
        loss.backward()
        return loss

    with torch.no_grad():
        nll_before = float(nll(logits, labels))
    opt.step(closure)
    t = float(torch.exp(log_t).detach())
    with torch.no_grad():
        nll_after = float(nll(logits / t, labels))
    assert math.isfinite(t) and t > 0, f"temperature fit failed: T = {t}"
    n_err = int((logits.argmax(1) != labels).sum())
    return {"t_lbfgs_endpoint": t, "val_nll_before": nll_before, "val_nll_after": nll_after,
            "n_val": int(labels.numel()), "n_val_errors": n_err, "degenerate": n_err == 0,
            "lr": lr, "max_iter": max_iter}


def calibration_summary(logits: np.ndarray, labels: np.ndarray, temperature: float,
                        n_bins: int) -> dict:
    """Accuracy, mean confidence, ECE and bins, before and after temperature scaling."""
    _, preds, confs, correct = softmax_outputs(logits, labels)
    _, preds_t, confs_t, correct_t = softmax_outputs(logits, labels, temperature)
    acc, acc_t = float(correct.mean()), float(correct_t.mean())
    assert abs(acc - acc_t) < 1e-9, "temperature scaling must not change accuracy"
    assert np.array_equal(preds, preds_t), "temperature scaling changed a prediction"
    ece, bins = ece_and_bins(confs, correct, n_bins)
    ece_t, bins_t = ece_and_bins(confs_t, correct_t, n_bins)
    return {"accuracy": acc, "mean_confidence": float(confs.mean()), "ece": ece,
            "reliability_bins": bins, "t": temperature,
            "mean_confidence_at_t": float(confs_t.mean()), "ece_at_t": ece_t,
            "reliability_bins_at_t": bins_t}
