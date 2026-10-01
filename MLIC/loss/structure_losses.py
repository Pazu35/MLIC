"""Differentiable structure losses for SSP profiles.

Written to replace two terms that could never have worked:

* ``max_pos`` was ``|argmax(pred) - argmax(target)|``. ``torch.argmax`` returns
  integer indices, so the term carried **no gradient at all** — adding it to the
  distortion sum added a constant. Replacement: :func:`soft_max_pos_loss`.
* ``extrema_pos`` was the squared difference of the *centroids* of the two
  extremum masks, one scalar per profile. Any rearrangement preserving the mean
  index is invisible to it. Replacement: :func:`matched_extrema_pos_loss`, which
  scores every reference extremum against the prediction *at that depth*.

Plus one new term aimed at the quantity that actually correlates with the
transmission-loss error across the checkpoint sweep (Spearman rho = -0.82 against
`tl_grid_mae`, the largest in the metric set): whether the reconstruction
reproduces the *prominence* of each reference extremum at all —
:func:`prominence_recall_loss`.

Three properties every term here has, and the old ones did not.

1. **A gradient.** Verified by `manuscrit/experiment/pretreatment_runs/scripts/
   gradient_audit.py`, which is what caught the original problem.
2. **A per-extremum penalty**, not an aggregate that can cancel.
3. **A magnitude near 1.** The positional terms divide by ``pos_scale_m`` so they
   are dimensionless; the recall term is bounded in [0, 1] by construction. The
   old `extrema_pos` was an MSE in *index squared*, whose natural scale is ~10^4
   times the reconstruction loss, and `FixedWeightSSPLoss`'s factor normalisation
   (which equalises loss *values*) answered that with a multiplier of 5e-5 —
   burying the gradient along with it.

All functions take ``(N, C, H, W)`` with depth on ``C``, and expect **physical**
profiles in m/s. Pass ``norm_mean``/``norm_std`` through :func:`denormalize` first
if the network works on per-level normalised data: with `mean_std_along_depth`
normalisation the argmax over depth of the *normalised* profile is the largest
anomaly relative to that level's climatology, which is not the sound-speed
maximum, and not what the evaluation metrics look at.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

EPS = 1e-6


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Per-level affine inverse, so the structure terms see m/s."""
    return x * std + mean


def half_window_levels(depth: torch.Tensor, scale_m: float, min_half: int = 1) -> int:
    """Half-window in levels spanning about ``scale_m`` metres on this grid.

    Every window here is specified in metres and converted once, because a window
    fixed in *levels* is a different physical scale at every depth on the native
    axis -- 1.08 m per level at the surface against 21.6 m at the bottom. That is
    the same defect that makes the reported `f1_score` incomparable between two
    vertical samplings, and there is no reason to rebuild it in the loss.
    """
    dz = float(torch.median(torch.diff(depth.flatten())).abs())
    return max(int(min_half), int(round(scale_m / max(dz, 1e-6))))


def _window_max(x: torch.Tensor, half: int) -> torch.Tensor:
    """Running maximum over a centred window of ``2*half+1`` levels, along dim 1.

    `max_pool1d` pads with -inf, so the window is simply truncated at the edges,
    which is what a topographic relief wants. Differentiable: the gradient goes to
    whichever level held the maximum.
    """
    n, c, h, w = x.shape
    y = x.permute(0, 2, 3, 1).reshape(-1, 1, c)
    y = F.max_pool1d(y, kernel_size=2 * half + 1, stride=1, padding=half)
    return y.reshape(n, h, w, c).permute(0, 3, 1, 2)


def _window_min(x: torch.Tensor, half: int) -> torch.Tensor:
    return -_window_max(-x, half)


def _window_sum(x: torch.Tensor, half: int) -> torch.Tensor:
    """Sum over a centred window along dim 1, in O(1) extra memory via cumsum."""
    c = x.shape[1]
    cs = torch.cumsum(x, dim=1)
    cs = torch.cat([torch.zeros_like(cs[:, :1]), cs], dim=1)   # cs[i] = sum(x[:i])
    idx = torch.arange(c, device=x.device)
    hi = (idx + half + 1).clamp(max=c)
    lo = (idx - half).clamp(min=0)
    return cs.index_select(1, hi) - cs.index_select(1, lo)


def _soft_peak_depth(
    x: torch.Tensor, z: torch.Tensor, half: int, beta: float, sign: int
) -> torch.Tensor:
    """Softmax-weighted depth of the extremum inside the window centred on each level.

    ``sign=+1`` locates a maximum, ``-1`` a minimum. The result is a field of the
    same shape as ``x``: entry ``[n, k, h, w]`` is where the codec *thinks* the
    extremum near level ``k`` sits, in metres. Differentiable in ``x``.
    """
    s = sign * x
    s = s - s.amax(dim=1, keepdim=True)          # stabilise; beta*s <= 0
    e = torch.exp(beta * s)
    return _window_sum(e * z, half) / (_window_sum(e, half) + EPS)


def _multiscale_relief(x: torch.Tensor, halves: "tuple[int, ...]", sign: int) -> torch.Tensor:
    """Largest relief of ``x`` visible within any of the given half-windows.

    Topographic prominence is the drop to the lowest saddle before a higher point,
    which has no cheap vectorised form. A single window is a poor stand-in: it
    cannot see a feature wider than itself, so a broad sound-channel maximum scores
    as flat and is dropped by the prominence gate. Taking the maximum relief over a
    ladder of windows recovers the wide features without losing the narrow ones,
    and converges to the true prominence as the widest window grows.
    """
    out = None
    for half in halves:
        relief = (x - _window_min(x, half)) if sign > 0 else (_window_max(x, half) - x)
        out = relief if out is None else torch.maximum(out, relief)
    return out


class ReferenceExtrema(NamedTuple):
    """Where the extrema of the *target* are, and how prominent. No gradient."""
    max_mask: torch.Tensor
    min_mask: torch.Tensor
    prom_max: torch.Tensor
    prom_min: torch.Tensor


def reference_extrema(
    target: torch.Tensor,
    halves: "tuple[int, ...]",
    min_prominence_frac: float = 0.05,
) -> ReferenceExtrema:
    """Prominent extrema of the target, by sign change plus a multi-scale relief gate.

    The gate mirrors the `weighted_f1` evaluation metric, which keeps extrema whose
    prominence exceeds a fraction of the profile's range. Without it every term
    here would spend most of its weight on the numerical wiggle section 9 of the
    diagnostics shows no rate-constrained codec was going to reproduce.

    ``halves`` are half-windows in *levels*, produced from metre scales by
    :func:`half_window_levels`.
    """
    with torch.no_grad():
        d = torch.diff(target, dim=1)
        up, dn = d[:, :-1], d[:, 1:]
        pad = torch.zeros_like(target[:, :1])
        is_max = torch.cat([pad, ((up > 0) & (dn <= 0)).to(target.dtype), pad], dim=1)
        is_min = torch.cat([pad, ((up < 0) & (dn >= 0)).to(target.dtype), pad], dim=1)

        prom_max = _multiscale_relief(target, halves, +1)
        prom_min = _multiscale_relief(target, halves, -1)

        span = target.amax(dim=1, keepdim=True) - target.amin(dim=1, keepdim=True)
        thr = float(min_prominence_frac) * span
        is_max = is_max * (prom_max >= thr).to(target.dtype)
        is_min = is_min * (prom_min >= thr).to(target.dtype)
        return ReferenceExtrema(is_max, is_min,
                                prom_max.clamp_min(0.0), prom_min.clamp_min(0.0))


def _depth_view(depth: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return depth.to(device=like.device, dtype=like.dtype).view(1, -1, 1, 1)


# --------------------------------------------------------------------------
# the losses
# --------------------------------------------------------------------------
def soft_max_pos_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    depth: torch.Tensor,
    beta: float = 10.0,
    pos_scale_m: float = 50.0,
) -> torch.Tensor:
    """Depth of the profile maximum, softly located, in units of ``pos_scale_m``.

    The differentiable replacement for ``max_pos``. ``beta`` sets how sharply the
    softmax picks the maximum out of the profile: too low and it reports the
    centre of mass, too high and the gradient vanishes everywhere but one level.
    10 is a reasonable start on profiles whose range is O(10) m/s.
    """
    z = _depth_view(depth, pred)

    def soft_z(x: torch.Tensor) -> torch.Tensor:
        s = x - x.amax(dim=1, keepdim=True)
        e = torch.exp(beta * s)
        return (e * z).sum(dim=1) / (e.sum(dim=1) + EPS)

    return F.mse_loss(soft_z(pred) / pos_scale_m, soft_z(target) / pos_scale_m)


def matched_extrema_pos_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    depth: torch.Tensor,
    search_m: float = 30.0,
    beta: float = 10.0,
    pos_scale_m: float = 50.0,
    min_prominence_frac: float = 0.05,
    prominence_scales_m: "tuple[float, ...]" = (10.0, 25.0, 50.0, 100.0),
    ref: Optional[ReferenceExtrema] = None,
) -> torch.Tensor:
    """Prominence-weighted depth error of every reference extremum, individually.

    The differentiable replacement for ``extrema_pos``. For each extremum of the
    target, the extremum of the *prediction* within ``search_m`` metres is located
    by a local soft-argmax, and the distance between the two is penalised. Because
    the window is local and the sum runs over extrema rather than over their mean,
    moving one extremum up and another down no longer cancels -- which is exactly
    the degeneracy that made the old centroid term inert.

    The target's position is located the *same* soft way, not taken as the level
    index, so the soft-argmax's own bias cancels and a perfect reconstruction
    scores exactly zero.

    ``search_m`` is the matching tolerance: it is the loss-side analogue of the
    evaluation metric's ``f1_score_dist_<d>m``, and setting it to the tolerance you
    intend to report is the sane default.
    """
    z = _depth_view(depth, pred)
    half = half_window_levels(depth, search_m)
    halves = tuple(half_window_levels(depth, m) for m in prominence_scales_m)
    ref = ref or reference_extrema(target, halves, min_prominence_frac)

    loss = pred.new_zeros(())
    weight = pred.new_zeros(())
    for mask, prom, sign in ((ref.max_mask, ref.prom_max, +1),
                             (ref.min_mask, ref.prom_min, -1)):
        z_pred = _soft_peak_depth(pred, z, half, beta, sign)
        with torch.no_grad():
            z_ref = _soft_peak_depth(target, z, half, beta, sign)
        w = prom * mask
        loss = loss + (w * ((z_pred - z_ref) / pos_scale_m).pow(2)).sum()
        weight = weight + w.sum()
    return loss / (weight + EPS)


def metre_windows(depth: torch.Tensor, capture_m: float, min_half: int = 1):
    """Per-level windows of every level within ``capture_m`` metres, as a padded index.

    Returns ``(idx, valid)``, both ``(C, L)``: row ``k`` lists the levels ``j`` with
    ``|z_j - z_k| <= capture_m`` (always at least ``min_half`` neighbours each
    side), padded with ``k`` itself and flagged invalid. Unlike
    :func:`half_window_levels`, which turns metres into one level count through the
    *median* spacing, the window here really is in metres at every depth: on the
    native axis 40 m is about +/-20 levels at the surface and +/-2 at the bottom.
    """
    z = depth.flatten().to(torch.float64)
    c = z.numel()
    k = torch.arange(c, device=z.device)
    inside = (z.view(-1, 1) - z.view(1, -1)).abs() <= float(capture_m)
    inside |= (k.view(-1, 1) - k.view(1, -1)).abs() <= int(min_half)
    width = int(inside.sum(dim=1).max())
    # stable sort puts the in-window levels first, in depth order
    order = torch.sort((~inside).to(torch.int8), dim=1, stable=True).indices[:, :width]
    valid = torch.gather(inside, 1, order)
    idx = torch.where(valid, order, k.view(-1, 1).expand(-1, width))
    return idx, valid


def local_extrema_pos_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    depth: torch.Tensor,
    capture_m: float = 40.0,
    local_beta: float = 4.0,
    pos_scale_m: float = 50.0,
    min_prominence_frac: float = 0.05,
    prominence_scales_m: "tuple[float, ...]" = (10.0, 25.0, 50.0, 100.0),
    ref: Optional[ReferenceExtrema] = None,
) -> torch.Tensor:
    """Prominence-weighted depth error of every reference extremum, with a live gradient.

    What :func:`matched_extrema_pos_loss` was meant to be. That term has two defects
    (report section 14.4b), each fatal on its own:

    1. its softmax is stabilised on the *profile-wide* extremum with ``beta`` in
       1/(m/s), so any extremum a few m/s below the global one underflows under
       ``EPS`` and gets exactly zero gradient -- it only ever trained the global
       maximum and minimum;
    2. its window is ``search_m`` converted through the median spacing (+/-1 level
       for 10 m on the native axis) and the softmax is near-hard, so the gradient
       dies once the predicted extremum is more than a few metres off -- i.e. it
       only reaches extrema already counted as hits at the reported tolerance.

    Here, for each reference extremum at level ``k``:

    * the window is every level within ``capture_m`` **metres** of ``z_k``
      (:func:`metre_windows`), set wider than the tolerance to be improved;
    * the softmax is stabilised on the window's own maximum, so every extremum
      contributes, whatever its value relative to the rest of the profile;
    * the temperature is relative: logits are ``local_beta * (x - max) / prom_ref``,
      with ``prom_ref`` the reference extremum's prominence. A level half a
      prominence below the peak keeps weight ``exp(-local_beta / 2)``, so the
      soft-argmax stays soft and its gradient does not vanish with distance.

    The target is located the same way with the same window and temperature, so a
    perfect reconstruction scores exactly zero. Only the gated extrema are
    gathered, so memory scales with (number of extrema) x (window length), not with
    the full field. Dimensionless via ``pos_scale_m``; bounded by
    ``(capture_m / pos_scale_m)^2`` per extremum.
    """
    n, c, h, w = target.shape
    halves = tuple(half_window_levels(depth, m) for m in prominence_scales_m)
    ref = ref or reference_extrema(target, halves, min_prominence_frac)
    idx, valid = metre_windows(depth.to(pred.device), capture_m)
    zc = depth.to(device=pred.device, dtype=pred.dtype).flatten()

    def flat(x):
        return x.permute(0, 2, 3, 1).reshape(-1, c)

    p_flat, t_flat = flat(pred), flat(target)
    loss = pred.new_zeros(())
    weight = pred.new_zeros(())
    for mask, prom, sign in ((ref.max_mask, ref.prom_max, +1),
                             (ref.min_mask, ref.prom_min, -1)):
        prof, lev = flat(mask).nonzero(as_tuple=True)
        if prof.numel() == 0:
            continue
        pr = flat(prom)[prof, lev].clamp_min(1e-3)            # (E,), no grad
        win = idx[lev]                                        # (E, L)
        ok = valid[lev]
        zw = zc[win]                                          # (E, L) metres

        def soft_depth(x_flat):
            s = sign * x_flat[prof.view(-1, 1), win]          # (E, L)
            s = s.masked_fill(~ok, float("-inf"))
            s = s - s.amax(dim=1, keepdim=True).detach()      # local stabilisation
            a = torch.softmax(local_beta * s / pr.view(-1, 1), dim=1)
            return (a * zw).sum(dim=1)

        z_pred = soft_depth(p_flat)
        with torch.no_grad():
            z_ref = soft_depth(t_flat)
        loss = loss + (pr * ((z_pred - z_ref) / pos_scale_m).pow(2)).sum()
        weight = weight + pr.sum()
    return loss / (weight + EPS)


def prominence_recall_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    depth: torch.Tensor,
    min_prominence_frac: float = 0.05,
    prominence_scales_m: "tuple[float, ...]" = (10.0, 25.0, 50.0, 100.0),
    ref: Optional[ReferenceExtrema] = None,
) -> torch.Tensor:
    """Prominence mass of the reference extrema the prediction fails to reproduce.

    For each extremum of the target, compare its prominence with the prominence the
    prediction shows *at the same depth*, and charge the shortfall, weighted by how
    prominent the reference extremum is. Bounded in [0, 1]: 0 when every reference
    extremum is reproduced with at least its own prominence, 1 when the
    reconstruction is flat. This is the trainable counterpart of the `weighted_f1`
    recall term.

    Only the **deficit** is penalised. Over-production is left to the reconstruction
    and derivative terms: across the 15 evaluated checkpoints every model
    *under*-produces prominent extrema (`count_bias` -0.026 to -0.055 per profile)
    and the ones that under-produce most are the ones whose transmission loss is
    worst -- Spearman rho = -0.82 against `tl_grid_mae`, the largest correlation in
    the metric set. Penalising overshoot too would push toward a ringing this codec
    does not have.
    """
    halves = tuple(half_window_levels(depth, m) for m in prominence_scales_m)
    ref = ref or reference_extrema(target, halves, min_prominence_frac)
    prom_pred_max = _multiscale_relief(pred, halves, +1)
    prom_pred_min = _multiscale_relief(pred, halves, -1)

    loss = pred.new_zeros(())
    weight = pred.new_zeros(())
    for mask, prom_ref, prom_pred in ((ref.max_mask, ref.prom_max, prom_pred_max),
                                      (ref.min_mask, ref.prom_min, prom_pred_min)):
        w = prom_ref * mask
        deficit = torch.relu(1.0 - prom_pred / (prom_ref + EPS))
        loss = loss + (w * deficit.pow(2)).sum()
        weight = weight + w.sum()
    return loss / (weight + EPS)


def structure_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    depth: torch.Tensor,
    which: "set[str]",
    search_m: float = 30.0,
    beta: float = 10.0,
    pos_scale_m: float = 50.0,
    min_prominence_frac: float = 0.05,
    prominence_scales_m: "tuple[float, ...]" = (10.0, 25.0, 50.0, 100.0),
    capture_m: float = 40.0,
    local_beta: float = 4.0,
) -> "dict[str, torch.Tensor]":
    """Compute the requested structure terms, sharing one reference detection.

    Detecting the target's extrema is the expensive half and does not depend on
    which term wants them, so the two extremum-based losses share a single pass.
    """
    out: dict = {}
    if "soft_max_pos" in which:
        out["soft_max_pos"] = soft_max_pos_loss(
            pred, target, depth, beta=beta, pos_scale_m=pos_scale_m)

    needs_ref = {"matched_extrema_pos", "local_extrema_pos", "prominence_recall"} & set(which)
    if not needs_ref:
        return out

    halves = tuple(half_window_levels(depth, m) for m in prominence_scales_m)
    ref = reference_extrema(target, halves, min_prominence_frac)
    if "matched_extrema_pos" in which:
        out["matched_extrema_pos"] = matched_extrema_pos_loss(
            pred, target, depth, search_m=search_m, beta=beta,
            pos_scale_m=pos_scale_m, min_prominence_frac=min_prominence_frac,
            prominence_scales_m=prominence_scales_m, ref=ref)
    if "local_extrema_pos" in which:
        out["local_extrema_pos"] = local_extrema_pos_loss(
            pred, target, depth, capture_m=capture_m, local_beta=local_beta,
            pos_scale_m=pos_scale_m, min_prominence_frac=min_prominence_frac,
            prominence_scales_m=prominence_scales_m, ref=ref)
    if "prominence_recall" in which:
        out["prominence_recall"] = prominence_recall_loss(
            pred, target, depth, min_prominence_frac=min_prominence_frac,
            prominence_scales_m=prominence_scales_m, ref=ref)
    return out


__all__ = [
    "denormalize",
    "half_window_levels",
    "reference_extrema",
    "soft_max_pos_loss",
    "matched_extrema_pos_loss",
    "metre_windows",
    "local_extrema_pos_loss",
    "prominence_recall_loss",
    "structure_losses",
]
