"""Per-term gradient logging for the multi-term SSP losses.

The factor normalisation of `FixedWeightSSPLoss` equalises loss *values*, which
says nothing about how hard each term actually pulls on the weights. Two failure
modes have already cost full runs here and are invisible in the loss curves:

* a term with a value but no useful gradient (`max_pos`, `matched_extrema_pos`,
  report sections 8 and 14.4b);
* a term whose gradient is large enough to dominate the global norm, so that
  `clip_grad_norm_` shrinks *every* term's step, including the reconstruction's
  (a candidate explanation for E1's +3.7% RMSE_int).

For each logged batch this writes one row per term to ``grad_terms.csv``:

    epoch, step, term, weight, value, weighted_value,
    grad_norm         ||d(weighted term)/d theta||, the term's pull on the weights
    share             grad_norm / total_grad_norm
    cos_total         cosine with the full gradient: < 0 means the term is overruled
    cos_recon         cosine with the recon term's gradient: < 0 means they conflict
    total_grad_norm   ||d loss / d theta||, before clipping
    clip_coef         min(1, clip_max_norm / total_grad_norm), what clipping applies

Cost: one extra backward pass per active term on each logged batch, so log a few
batches every few epochs, not every step.
"""
from __future__ import annotations

import csv
import math
import os

import torch

FIELDS = ["epoch", "step", "term", "weight", "value", "weighted_value", "grad_norm",
          "share", "cos_total", "cos_recon", "total_grad_norm", "clip_coef"]


class TermGradientLogger:
    def __init__(self, out_dir, params, every_epochs=10, batches_per_epoch=2,
                 clip_max_norm=0.0, extra_epochs=()):
        """
        Args:
            out_dir: run directory; the CSV is ``out_dir/grad_terms.csv``.
            params: the parameters the main optimiser steps (not the aux quantiles).
            every_epochs: log on epochs 0, every_epochs, 2*every_epochs, ...
            batches_per_epoch: how many leading batches of a logged epoch to log.
            clip_max_norm: the value passed to ``clip_grad_norm_``, to report the
                coefficient it will apply (0 = no clipping).
            extra_epochs: epochs to log in addition, e.g. the ones just after the
                factor weights are applied.
        """
        self.path = os.path.join(out_dir, "grad_terms.csv")
        self.params = [p for p in params if p.requires_grad]
        self.every_epochs = max(1, int(every_epochs))
        self.batches_per_epoch = int(batches_per_epoch)
        self.clip_max_norm = float(clip_max_norm or 0.0)
        self.extra_epochs = {int(e) for e in extra_epochs}
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(FIELDS)

    def wants(self, epoch, batch_idx):
        due = epoch % self.every_epochs == 0 or epoch in self.extra_epochs
        return due and batch_idx < self.batches_per_epoch

    def _flat_grad(self, scalar):
        if not (torch.is_tensor(scalar) and scalar.requires_grad):
            return None
        grads = torch.autograd.grad(scalar, self.params, retain_graph=True,
                                    allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                          for g, p in zip(grads, self.params)])

    @staticmethod
    def _cos(a, b):
        if a is None or b is None:
            return float("nan")
        den = a.norm() * b.norm()
        return float(torch.dot(a, b) / den) if den > 0 else float("nan")

    def log(self, epoch, step, criterion, out_criterion):
        """Call after ``criterion(out_net, d)`` and before ``loss.backward()``."""
        total = self._flat_grad(out_criterion["loss"])
        if total is None:
            return
        total_norm = float(total.norm())
        clip = (min(1.0, self.clip_max_norm / (total_norm + 1e-6))
                if self.clip_max_norm > 0 else 1.0)

        # {name: (weight, raw value, contribution to out["loss"] with its graph)}
        terms = dict(getattr(criterion, "last_weighted_terms", {}) or {})
        recon = terms.get("recon")
        g_recon = self._flat_grad(recon[2]) if recon is not None else None

        rows = []
        for name, (weight, value, contrib) in terms.items():
            g = g_recon if name == "recon" else self._flat_grad(contrib)
            gn = float(g.norm()) if g is not None else 0.0
            rows.append([epoch, step, name, weight, float(value), float(contrib), gn,
                         gn / total_norm if total_norm > 0 else math.nan,
                         self._cos(g, total), self._cos(g, g_recon),
                         total_norm, clip])
            del g
        rows.append([epoch, step, "TOTAL", math.nan, float(out_criterion["loss"]),
                     float(out_criterion["loss"]), total_norm, 1.0, 1.0,
                     self._cos(total, g_recon), total_norm, clip])
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
