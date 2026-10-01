"""Resuming a training run from one of its checkpoints (``--checkpoint X --resume``).

Model weights, optimizers and LR scheduler are in every checkpoint. What is not, in
checkpoints written before 28 Sep 2026, is the loss's own state: the factor
normalisation computes ``active_weights`` once at ``factor_warmup_epochs`` (and
``--grad_weighting`` re-calibrates some of them). Without it, a run resumed after the
warm-up would never pass ``epoch == factor_warmup_epochs`` again and would train on
``recon`` alone for the rest of the run. So for those checkpoints the weights are
recovered from the original run's train log, and the resume refuses to start if they
cannot be.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import torch.nn as nn

_ACTIVE = re.compile(r"Active weights: (\{.*\})\s*$")
_GRADW = re.compile(r"Gradient-norm weights .*?: (\{.*\})\s*$")
_EPOCH = re.compile(r"Train epoch (\d+):")


def criterion_state(criterion) -> dict:
    """What has to go into a checkpoint for the loss to resume exactly."""
    state = {
        "active_weights": dict(getattr(criterion, "active_weights", {}) or {}),
        "factor_weights_applied": bool(getattr(criterion, "_factor_weights_applied", False)),
    }
    if isinstance(criterion, nn.Module):
        state["module_state"] = criterion.state_dict()   # DLW / homoscedastic log-vars
    return state


def _weights_from_train_log(run_dir: Path, start_epoch: int) -> dict | None:
    """Replay the weight updates logged before ``start_epoch``."""
    logs = sorted(run_dir.glob("train_*.log"))
    if not logs:
        return None
    weights = None
    for line in logs[-1].read_text(errors="ignore").splitlines():
        m = _EPOCH.search(line)
        if m and int(m.group(1)) >= start_epoch:
            break
        m = _ACTIVE.search(line)
        if m:
            weights = {k: float(v) for k, v in ast.literal_eval(m.group(1)).items()}
            continue
        m = _GRADW.search(line)
        if m and weights is not None:
            weights.update({k: float(v) for k, v in ast.literal_eval(m.group(1)).items()})
    return weights


def restore_criterion_state(criterion, checkpoint: dict, checkpoint_path: str,
                            start_epoch: int, logger) -> None:
    saved = checkpoint.get("criterion_state")
    if saved and set(saved["active_weights"]) != set(criterion.active_weights):
        raise RuntimeError(f"[resume] loss terms {sorted(criterion.active_weights)} but the "
                           f"checkpoint trained {sorted(saved['active_weights'])}")
    if saved:
        criterion.active_weights = dict(saved["active_weights"])
        criterion._factor_weights_applied = saved["factor_weights_applied"]
        if saved.get("module_state") and isinstance(criterion, nn.Module):
            criterion.load_state_dict(saved["module_state"])
        logger.info(f"[resume] loss state restored from the checkpoint: {criterion.active_weights}")
        return

    if isinstance(criterion, nn.Module) and any(True for _ in criterion.parameters()):
        raise RuntimeError("[resume] this loss has learnable parameters and the checkpoint "
                           "predates criterion_state: it cannot be resumed exactly")
    if not getattr(criterion, "use_factor_weights", False):
        logger.info("[resume] fixed loss weights, nothing to restore")
        return
    if start_epoch <= criterion.factor_warmup_epochs:
        logger.info("[resume] before the factor warm-up: weights will be computed as usual")
        return

    run_dir = Path(checkpoint_path).resolve().parent.parent
    weights = _weights_from_train_log(run_dir, start_epoch)
    if weights is None:
        raise RuntimeError(f"[resume] no 'Active weights' line before epoch {start_epoch} in "
                           f"{run_dir}/train_*.log: cannot recover the factor weights")
    # Zero-weight terms are dropped from loss_dict, so the key sets are the active
    # terms: they must be the same, or the resumed run optimises a different loss.
    if set(criterion.active_weights) != set(weights):
        raise RuntimeError(f"[resume] loss terms {sorted(criterion.active_weights)} but the "
                           f"checkpoint's run trained {sorted(weights)}: set loss_dict as "
                           "in its experiment_config.log")
    criterion.active_weights = {k: weights[k] for k in criterion.active_weights}
    criterion._factor_weights_applied = True
    logger.info(f"[resume] factor weights recovered from {run_dir.name}'s train log: "
                f"{criterion.active_weights}")
