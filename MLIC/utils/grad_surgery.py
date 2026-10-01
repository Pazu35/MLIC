"""Gradient surgery for the auxiliary (structure) loss terms.

E1b (report section 14.7) showed that `local_extrema_pos` trains, but its gradient
is anti-aligned with the reconstruction gradient for the whole run (cosine -0.4 to
-0.6) and 2-3x larger, and that this costs +5% RMSE and +15% ECS on validation.
Two remedies, each behind its own switch in `train.py`:

* :func:`projected_backward` (``--grad_projection``): PCGrad-style. The gradient of
  the auxiliary terms is computed separately from the rest of the loss ("anchor":
  the value terms and the rate), and when the two conflict (negative dot product)
  the auxiliary gradient loses its component along the anchor gradient. What is
  left cannot increase the anchor loss to first order.
* gradient-norm weighting (``--grad_weighting``), implemented in
  :meth:`FixedWeightSSPLoss.apply_gradient_weights`: the auxiliary weights are set
  so each term's gradient norm is a fixed fraction of recon's, instead of matching
  loss *values* as the factor normalisation does.
"""
from __future__ import annotations

import math

import torch

from MLIC.MLIC.loss.rd_loss import STRUCTURE_LOSS_TERMS

AUX_TERMS = tuple(STRUCTURE_LOSS_TERMS)


def split_loss(criterion, out_criterion, aux_names=AUX_TERMS):
    """``(anchor, aux)`` with ``anchor + aux == out_criterion["loss"]``.

    ``aux`` is the sum of the weighted auxiliary contributions the criterion
    recorded in ``last_weighted_terms`` for this forward pass; ``anchor`` is the sum
    of every other recorded contribution (value terms and rate), checked against
    ``out["loss"]`` so that nothing the criterion adds is ever dropped. ``aux`` is None when no auxiliary term has
    a non-zero weight (e.g. before the factor warm-up), in which case the caller
    should do an ordinary backward.
    """
    terms = getattr(criterion, "last_weighted_terms", {}) or {}
    parts = [contrib for name, (_, _, contrib) in terms.items() if name in aux_names]
    if not parts:
        return out_criterion["loss"], None
    aux = sum(parts)
    # Built from the other recorded contributions rather than as loss - aux, so
    # the anchor backward never traverses the structure terms' graph.
    anchor = sum(contrib for name, (_, _, contrib) in terms.items()
                 if name not in aux_names)
    total = float(out_criterion["loss"])
    if abs(float(anchor) + float(aux) - total) > 1e-4 * max(1.0, abs(total)):
        raise RuntimeError("last_weighted_terms does not add up to out['loss']; "
                           "cannot split it for gradient projection")
    return anchor, aux


def projected_backward(params, anchor, aux, scale=1.0):
    """Accumulate ``grad(anchor) + proj(grad(aux))`` into ``p.grad``.

    The projection removes from the auxiliary gradient its component along the
    anchor gradient when, and only when, the two point in opposite directions
    (global dot product over all parameters, as in PCGrad). ``scale`` is applied to
    both, for gradient accumulation.

    Returns a dict with the cosine between the two gradients before projection,
    whether it was projected, and the norms, for logging.
    """
    g_a = torch.autograd.grad(anchor * scale, params, retain_graph=True,
                              allow_unused=True)
    g_x = torch.autograd.grad(aux * scale, params, allow_unused=True)

    dot = na2 = nx2 = 0.0
    for a, x in zip(g_a, g_x):
        if a is not None:
            na2 += float((a * a).sum())
            if x is not None:
                dot += float((a * x).sum())
        if x is not None:
            nx2 += float((x * x).sum())

    conflict = dot < 0 and na2 > 0
    coef = dot / na2 if conflict else 0.0
    for p, a, x in zip(params, g_a, g_x):
        if a is None and x is None:
            continue
        g = torch.zeros_like(p) if a is None else a.clone()
        if x is not None:
            g += x if a is None or not conflict else x - coef * a
        p.grad = g if p.grad is None else p.grad + g

    nx2_after = nx2 - (dot * dot / na2 if conflict else 0.0)
    return dict(
        cos=dot / math.sqrt(na2 * nx2) if na2 > 0 and nx2 > 0 else float("nan"),
        projected=conflict,
        anchor_norm=math.sqrt(na2),
        aux_norm=math.sqrt(nx2),
        aux_norm_after=math.sqrt(max(nx2_after, 0.0)),
    )
