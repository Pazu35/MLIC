import math
import warnings
from contextlib import contextmanager

import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
import numpy as np
from FASCINATION.src import differentiable_fonc as DF
from MLIC.MLIC.loss import structure_losses as SL

SIGNIFICANT_DEPTH_M = 762.0

# Terms that are computed and logged but cannot train anything. `max_pos` is
# `|argmax(pred) - argmax(target)|`: torch.argmax returns integer indices, the
# result has no grad_fn, and adding it to the distortion sum adds a constant.
# Four runs (A1, A3, A4, CX1) carried it at a non-zero weight and none of them
# improved on it -- each finished *worse* on max_pos than the matched run that
# never optimised it. Requesting one now raises, naming the replacement.
INERT_LOSS_TERMS = {"max_pos": "soft_max_pos"}

# Terms that have a gradient but a degenerate one, kept for backward
# compatibility with existing configs and warned about once.
DEGENERATE_LOSS_TERMS = {
    "extrema_pos": ("it penalises the *centroid* of the extremum mask, so any "
                    "rearrangement preserving the mean index is invisible to it; "
                    "use local_extrema_pos"),
    "matched_extrema_pos": ("its softmax is stabilised on the profile-wide extremum, "
                            "so only the global max/min get a gradient, and its "
                            "window is +/-1 level on the native axis, so it only "
                            "reaches extrema already within tolerance (report "
                            "section 14.4b); use local_extrema_pos"),
}

STRUCTURE_LOSS_TERMS = ("soft_max_pos", "matched_extrema_pos", "local_extrema_pos",
                        "prominence_recall")


def check_loss_dict(loss_dict, allow_inert=False):
    """Refuse a configuration that asks for a term which cannot train.

    Raising here rather than warning is deliberate: the alternative is another
    2000-epoch run whose headline term contributed nothing, which is what the
    first four attempts were.
    """
    offenders = {k: r for k, r in INERT_LOSS_TERMS.items()
                 if float(loss_dict.get(k, 0.0) or 0.0) > 0}
    if offenders and not allow_inert:
        lines = [f"  '{k}' has no gradient -- use '{r}' instead"
                 for k, r in offenders.items()]
        raise ValueError(
            "loss_dict requests inert term(s):\n" + "\n".join(lines)
            + "\nThese are still computed and logged for continuity, but they "
              "cannot train the network. Pass allow_inert_terms=True to "
              "reproduce an old run deliberately."
        )
    for k, why in DEGENERATE_LOSS_TERMS.items():
        if float(loss_dict.get(k, 0.0) or 0.0) > 0:
            warnings.warn(f"loss term '{k}' is degenerate: {why}", RuntimeWarning,
                          stacklevel=2)


def rate_reference_bits_per_profile(target, native_depth_levels=None,
                                    bits_per_level=None):
    """Bits per water column in the field the compression ratio is quoted against.

    `cr_treshold` is a floor on the compression ratio, imposed as
    `ReLU(bpe - bits_per_profile_reference / cr_treshold)`. The reference used to
    be `target.nelement() * element_size * 8 / (N*H*W)`, which is
    `model_levels * element_size * 8` -- the size of the field **on the model's
    own depth grid**. That makes the bit budget proportional to how many levels
    the model chose to use: at `cr_treshold=10000` and float32 the budget is
    0.5024 bits/profile for a 157-level model, 0.3072 for 96 and 0.2048 for 64.
    Three models asked for "compression ratio 10000" therefore train at three
    different rates, and the measured group-C runs landed at 0.504, 0.239 and
    0.142 bits/profile -- so their reconstruction errors are not comparable, and
    the grid ablation they were meant to be became a rate-distortion sweep.

    With `native_depth_levels` set, the reference is the **native** field
    instead, so `cr_treshold` targets `cr_native` and every model gets the same
    bit budget per water column whatever grid it runs on. For a model already on
    the native grid the two are identical, so no run done so far changes except
    the reduced-grid ones, which is the point.
    """
    levels = native_depth_levels or int(target.shape[1])
    # `bits_per_level` defaults to the training tensor's own precision, which is what
    # this always did. It is exposed because `test_metrics.py` builds `cr_native`
    # against a float32 reference (it casts the truth to float32 before scoring), so
    # on a float64 run the loss would target a budget twice the one the metric
    # reports. Every run to date is float32 and the two agree; pass 32 explicitly if
    # that ever stops being true.
    width = float(bits_per_level) if bits_per_level else target.element_size() * 8
    return float(levels) * width


class StructureLossMixin:
    """Depth-aware structure terms, shared by every weighting scheme.

    Holds the depth axis and, optionally, the normalisation statistics. The
    structure terms are defined on **physical** profiles in m/s: with
    `mean_std_along_depth` normalisation the argmax over depth of the normalised
    profile is the largest anomaly relative to that level's climatology, which is
    neither the sound-speed maximum nor what any evaluation metric looks at.
    """

    def _init_structure(self, depth_array=None, norm_stats=None,
                        structure_params=None, allow_inert_terms=False,
                        native_depth_levels=None,
                        rate_reference_bits_per_level=None):
        self.allow_inert_terms = bool(allow_inert_terms)
        # Number of levels in the field the compression ratio is *quoted against* --
        # the native axis, not whatever reduced grid this model happens to use. See
        # _rate_reference_bits_per_profile.
        self.native_depth_levels = (int(native_depth_levels)
                                    if native_depth_levels else None)
        self.rate_reference_bits_per_level = rate_reference_bits_per_level
        sp = dict(search_m=30.0, beta=10.0, pos_scale_m=50.0,
                  min_prominence_frac=0.05,
                  prominence_scales_m=(10.0, 25.0, 50.0, 100.0),
                  capture_m=40.0, local_beta=4.0)
        sp.update(structure_params or {})
        self.structure_params = sp

        if depth_array is not None:
            z = torch.as_tensor(np.asarray(depth_array, dtype=np.float64).ravel(),
                                dtype=torch.float32)
        else:
            z = torch.zeros(0)
        self.register_buffer("_depth_m", z)

        offset, scale = _affine_from_norm_stats(norm_stats)
        self.register_buffer("_norm_offset", offset if offset is not None else torch.zeros(0))
        self.register_buffer("_norm_scale", scale if scale is not None else torch.zeros(0))

    def _rate_reference_bits_per_profile(self, target):
        """See :func:`rate_reference_bits_per_profile`."""
        return rate_reference_bits_per_profile(
            target, self.native_depth_levels,
            getattr(self, "rate_reference_bits_per_level", None))

    @contextmanager
    def physical_inputs(self):
        """Declare that ``pred``/``target`` are already in m/s for the enclosed calls.

        Training feeds the criterion normalised tensors and :meth:`_physical`
        converts them for the structure terms. ``test_one_epoch`` de-normalises
        before calling the criterion, so without this the structure terms saw
        ``(x * std + mean) * std + mean`` and logged numbers unrelated to the
        training objective. Only the structure terms are affected; the value terms
        never pass through :meth:`_physical`.
        """
        previous = getattr(self, "_inputs_are_physical", False)
        self._inputs_are_physical = True
        try:
            yield self
        finally:
            self._inputs_are_physical = previous

    def _physical(self, x):
        """Undo the per-level normalisation, when the stats were supplied."""
        if self._norm_scale.numel() == 0 or getattr(self, "_inputs_are_physical", False):
            return x
        scale = self._norm_scale.to(device=x.device, dtype=x.dtype)
        offset = self._norm_offset.to(device=x.device, dtype=x.dtype)
        return x * scale + offset

    def _structure_terms(self, pred, target, wanted):
        wanted = {n for n in wanted if n in STRUCTURE_LOSS_TERMS}
        if not wanted:
            return {}
        if self._depth_m.numel() == 0:
            raise ValueError(
                f"structure losses {sorted(wanted)} need depth_array; pass it to "
                "the loss constructor (train.py already has dm.depth_array)."
            )
        return SL.structure_losses(
            self._physical(pred), self._physical(target),
            self._depth_m.to(pred.device), wanted, **self.structure_params)


def _affine_from_norm_stats(norm_stats):
    """(offset, scale) as (1, C, 1, 1) tensors so that ``x * scale + offset`` is m/s."""
    if not norm_stats:
        return None, None
    method = norm_stats.get("method")
    params = norm_stats.get("params", {})

    def t(key):
        return torch.as_tensor(np.asarray(params[key], dtype=np.float64),
                               dtype=torch.float32).reshape(1, -1, 1, 1)

    try:
        if method == "mean_std_along_depth":
            return t("mean_along_depth"), t("std_along_depth")
        if method == "min_max_along_depth":
            lo, hi = t("x_min_along_depth"), t("x_max_along_depth")
            return lo, hi - lo
        if method == "mean_std":
            return (torch.tensor(float(params["mean"])).reshape(1, 1, 1, 1),
                    torch.tensor(float(params["std"])).reshape(1, 1, 1, 1))
        if method == "min_max":
            lo = float(params["x_min"])
            return (torch.tensor(lo).reshape(1, 1, 1, 1),
                    torch.tensor(float(params["x_max"]) - lo).reshape(1, 1, 1, 1))
    except (KeyError, TypeError, ValueError):
        return None, None
    return None, None


def resolve_significant_depth_idx(depth_array, significant_depth=SIGNIFICANT_DEPTH_M):
    """Index of the deepest level still within `significant_depth` metres.

    The weighted losses emphasise the upper ocean by depth index, so the cutoff has to be
    resolved against the grid actually in use: a hardcoded index 60 means 762 m on the
    uniform grid but only 354 m on the non-uniform one, silently changing which part of
    the water column is emphasised. Falls back to 60 when no grid is supplied, which is
    what that index meant on the uniform grid.
    """
    if depth_array is None:
        return 60
    z = np.asarray(depth_array, dtype=np.float64).ravel()
    return int(np.clip(np.searchsorted(z, significant_depth, side="right") - 1, 0, len(z) - 1))


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics='mse', cr_treshold=None,
                 native_depth_levels=None, rate_reference_bits_per_level=None,
                 **kwargs):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metrics = metrics
        self.cr_treshold = cr_treshold
        self.native_depth_levels = (int(native_depth_levels)
                                    if native_depth_levels else None)
        self.rate_reference_bits_per_level = rate_reference_bits_per_level
    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        bpe = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.cr_treshold is None:
            out["bpp_loss"] = bpe
        else:
            bpe_original = rate_reference_bits_per_profile(
                target, getattr(self, "native_depth_levels", None),
                getattr(self, "rate_reference_bits_per_level", None))
            bpe_treshold = bpe_original / self.cr_treshold
            out["bpp_loss"] = nn.ReLU()(bpe - bpe_treshold)
            
        if self.metrics == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == 'ms-ssim':
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]

        return out



def differentiable_extrema_mask(signal, sharpness=10.0, mode="minmax"):
    d1 = signal[..., 1:] - signal[..., :-1]
    if mode == "minmax":
        d1_left = d1[..., :-1]
        d1_right = d1[..., 1:]
        prod = d1_left * d1_right
        mask = torch.sigmoid(-sharpness * prod)
    elif mode == "inflection":
        d2 = d1[..., 1:] - d1[..., :-1]
        d2_left = d2[..., :-1]
        d2_right = d2[..., 1:]
        prod = d2_left * d2_right
        mask = torch.sigmoid(-sharpness * prod)
    else:
        raise ValueError("mode must be 'minmax' or 'inflection'")
    mask = F.pad(mask, (1, 1), mode="constant", value=0.0)
    return mask

class HeteroscedasticSSPLoss(nn.Module):
    def __init__(self, sharpness=10.0, lambda_minmax=1.0, lambda_inflection=1.0, binary_scale=True):
        super().__init__()
        self.sharpness = sharpness
        self.lambda_minmax = lambda_minmax
        self.lambda_inflection = lambda_inflection
        self.binary_scale = binary_scale

    def forward(self, pred, log_var, target, use_pred_mask=False):
        mse = (pred - target) ** 2
        precision = torch.exp(-log_var)
        scale = 255 ** 2 if self.binary_scale else 1.0
        loss_recon = (precision * mse * scale + log_var).mean()

        dp = pred[..., 1:] - pred[..., :-1]
        dt = target[..., 1:] - target[..., :-1]
        loss_deriv = F.mse_loss(dp, dt)

        base = pred if use_pred_mask else target

        mask_minmax = differentiable_extrema_mask(base, sharpness=self.sharpness, mode="minmax")
        mse_minmax = mse * mask_minmax
        prec_minmax = precision * mask_minmax
        loss_minmax = ((prec_minmax * mse_minmax * scale) + log_var * mask_minmax).sum() / mask_minmax.sum().clamp(min=1)

        mask_infl = differentiable_extrema_mask(base, sharpness=self.sharpness, mode="inflection")
        mse_infl = mse * mask_infl
        prec_infl = precision * mask_infl
        loss_infl = ((prec_infl * mse_infl * scale) + log_var * mask_infl).sum() / mask_infl.sum().clamp(min=1)

        total_loss = loss_recon + loss_deriv \
                     + self.lambda_minmax * loss_minmax \
                     + self.lambda_inflection * loss_infl

        return total_loss, {
            "recon": loss_recon.item(),
            "deriv": loss_deriv.item(),
            "minmax": loss_minmax.item(),
            "inflection": loss_infl.item(),
            "mask_minmax_mean": mask_minmax.mean().item(),
            "mask_inflection_mean": mask_infl.mean().item(),
        }
    

def diff_mask(x, eps=1e-6):
    """Soft differentiable mask for extrema and inflection points."""
    dx = x[:, 1:, :] - x[:, :-1, :]          # slope
    d2x = dx[:, 1:, :] - dx[:, :-1, :]       # curvature

    dx = F.pad(dx, (1,0))
    d2x = F.pad(d2x, (1,1))

    # Min/max indicator: slope sign change
    minmax = torch.sigmoid(-50.0 * dx * torch.roll(dx, 1, dims=1))
    # Inflection indicator: curvature flip
    inflection = torch.sigmoid(50.0 * (-d2x.abs()))

    mask = minmax + inflection
    return mask / (mask.max(dim=1, keepdim=True)[0] + eps)



class HomoscedasticSSPLoss(StructureLossMixin, nn.Module):
    def __init__(self, 
                 loss_dict={"recon": 1.0,
                            "weighted_recon": 1.0,
                            "deriv": 1.0,
                            "lsd": 1.0,
                            "max_pos": 1.0,
                            "max_value": 1.0,
                            "extrema_pos": 1.0,
                            "extrema_value": 1.0}, 
                extrema_method="both",
                lmbda=1e-2,
                use_smoothl1=True,
                lambda_deriv=0.1,
                lambda_extrema=0.5,
                cr_treshold=None,
                depth_array=None,
                norm_stats=None,
                structure_params=None,
                allow_inert_terms=False,
                native_depth_levels=None,
                rate_reference_bits_per_level=None,
                **kwargs):
        super().__init__()
        self.lmbda = lmbda
        check_loss_dict(loss_dict, allow_inert=allow_inert_terms)
        self.loss_dict = loss_dict
        self.extrema_method = extrema_method
        self.use_smoothl1 = use_smoothl1
        # Previously absent: `cr_treshold` landed in **kwargs and was silently
        # dropped, so this method ignored the rate constraint entirely while
        # accepting the key in loss_params. Same failure mode as `max_pos`.
        self.cr_treshold = cr_treshold
        self.max_significant_depth_idx = resolve_significant_depth_idx(depth_array)
        self._init_structure(depth_array=depth_array, norm_stats=norm_stats,
                             structure_params=structure_params,
                             allow_inert_terms=allow_inert_terms,
                             native_depth_levels=native_depth_levels,
                             rate_reference_bits_per_level=rate_reference_bits_per_level)

        # self.lambda_deriv = lambda_deriv
        # self.lambda_extrema = lambda_extrema

        # log variances (homoscedastic uncertainty)
        self.log_vars = nn.Parameter(torch.zeros(len(self.loss_dict)))  # 4 tasks: recon, deriv, extrema_pos, extrema_value
        self.loss_weights = loss_dict.copy()

    def forward(self, output, target):
        """
        Args:
            pred: (batch, length)
            target: (batch, length)
        """

        N, _, H, W = target.size()
        out = {}
        losses_dict = {}
        num_pixels = N * H * W

        recon_loss, weighted_recon_loss, deriv_loss, max_pos_loss, max_value_loss, extrema_pos_loss, extrema_value_loss = 0, 0, 0, 0, 0, 0, 0

        # === Bitrate term (bpp loss) ===
        bpe = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.cr_treshold is None:
            out["bpp_loss"] = bpe
        else:
            bpe_original = self._rate_reference_bits_per_profile(target)
            bpe_treshold = bpe_original / self.cr_treshold
            out["bpp_loss"] = nn.ReLU()(bpe - bpe_treshold)

        # === Distortion term ===
        pred = output["x_hat"]

        # Base reconstruction loss
        if "recon" in self.loss_dict and self.loss_dict["recon"] > 0:
            if self.use_smoothl1:
                recon_loss = F.smooth_l1_loss(pred, target, reduction="mean")
            else:
                recon_loss = torch.mean((pred - target) ** 2)

            losses_dict["recon"] = recon_loss

            # # Heavier weight on first 30 points
        # weights = torch.ones_like(recon_loss)
        # weights[:, :30] *= 2.0
        if "weighted_recon" in self.loss_dict and self.loss_dict["weighted_recon"] > 0:
            weighted_recon_loss = weighted_mse_loss(pred, target, max_significant_depth_idx=self.max_significant_depth_idx, decay_factor=0.1, use_smoothl1=self.use_smoothl1)
            losses_dict["weighted_recon"] = weighted_recon_loss

        if "deriv" in self.loss_dict and self.loss_dict["deriv"] > 0:
            # Derivative-aware loss
            dp = pred[:, 1:] - pred[:, :-1]
            dt = target[:, 1:] - target[:, :-1]
            deriv_loss = F.mse_loss(dp, dt)
            losses_dict["deriv"] = deriv_loss

        if "lsd" in self.loss_dict and self.loss_dict["lsd"] > 0:
            lsd_loss = spectral_loss(pred, target, axis=1, normalize=True, use_smoothl1=self.use_smoothl1)
            losses_dict["lsd"] = lsd_loss

        if "max_pos" in self.loss_dict and self.loss_dict["max_pos"] > 0:
            # Unreachable unless allow_inert_terms=True: argmax has no gradient,
            # so this only ever added a constant to the distortion sum.
            with torch.no_grad():
                max_pos_loss = torch.abs(
                    torch.argmax(pred, dim=1) - torch.argmax(target, dim=1)
                ).float().mean()
            losses_dict["max_pos"] = max_pos_loss

        if "max_value" in self.loss_dict and self.loss_dict["max_value"] > 0:
            max_value_loss = F.mse_loss(torch.max(pred, dim=1)[0], torch.max(target, dim=1)[0])
            max_value_loss = max_value_loss.float().mean()
            losses_dict["max_value"] = max_value_loss


        if ("extrema_pos" in self.loss_dict and self.loss_dict["extrema_pos"] > 0) or ("extrema_value" in self.loss_dict and self.loss_dict["extrema_value"] > 0):
        # Extremum-aware loss (using diff_mask on target)
            extrema_pos_loss, extrema_value_loss = position_and_value_loss(pred, target, dim=1, tau=10, mode=self.extrema_method)
            losses_dict["extrema_pos"] = extrema_pos_loss
            losses_dict["extrema_value"] = extrema_value_loss

            if "extrema_pos" not in self.loss_dict or self.loss_dict["extrema_pos"] == 0:
                extrema_pos_loss = 0  
                del losses_dict["extrema_pos"]
            if "extrema_value" not in self.loss_dict or self.loss_dict["extrema_value"] == 0:
                extrema_value_loss = 0
                del losses_dict["extrema_value"]
        # Extremum-aware term
        # mask = diff_mask(target.mean(dim=(1,3)))  # collapse lat/lon, keep z-profile
        # extrema_loss = ((x_hat.mean(dim=(1,3)) - target.mean(dim=(1,3))) ** 2 * mask).mean()


        # Homoscedastic weighting
        losses = list(losses_dict.values())
        distortion = 0
        for i in range(len(losses)):
            precision = torch.exp(-self.log_vars[i])
            distortion += precision * losses[i] + self.log_vars[i]
            self.loss_weights[list(losses_dict.keys())[i]] = precision.item()

        out["recon_loss"] = recon_loss
        out["weighted_recon_loss"] = weighted_recon_loss
        out["deriv_loss"] = deriv_loss
        out["max_pos_loss"] = max_pos_loss
        out["max_value_loss"] = max_value_loss
        out["extrema_pos_loss"] = extrema_pos_loss
        out["extrema_value_loss"] = extrema_value_loss
        out["ms_ssim_loss"] = None

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]

        return out 
    #, {
        #     "recon": recon_loss.item(),
        #     "weighted_recon": weighted_recon_loss.item(),
        #     "deriv": deriv_loss.item(),
        #     "extrema_pos": extrema_pos_loss.item(),
        #     "extrema_value": extrema_value_loss.item(),
        #     "log_vars": self.log_vars.data.cpu().numpy()
        # }



class DynamicLossWeightingSSPLoss(StructureLossMixin, nn.Module):
    """
    Dynamic Loss Weighting (DLW) for SSP compression.
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)
    but with dynamic weight updates based on loss ratios.
    
    The weights are updated each step based on:
    - Loss magnitude ratios (to balance different scales)
    - Rate of change of losses (to focus on harder tasks)
    """
    
    def __init__(
        self,
        loss_dict={
            "recon": 1.0,
            "weighted_recon": 1.0,
            "deriv": 1.0,
            "lsd": 1.0,
            "weighted_deriv": 1.0,
            "curvature_recon": 1.0,
            "soft_peak": 1.0,  # NEW: soft peak localization
            "wasserstein_peak": 1.0,  # NEW: Wasserstein peak alignment
            "max_pos": 1.0,
            "max_value": 1.0,
            "extrema_pos": 1.0,
            "extrema_value": 1.0
        },
        curvature_beta=10.0,
        peak_beta=20.0,  # sharpness for soft peak and Wasserstein
        extrema_method="both",
        lmbda=1e-2,
        use_smoothl1=True,
        cr_treshold=10000.0,
        # DLW specific parameters
        dlw_method="gradnorm",  # "gradnorm", "uncertainty", "dwa", "ruw"
        alpha=1.5,  # GradNorm: restoring force strength
        temperature=2.0,  # DWA: temperature for softmax
        ema_decay=0.9,  # EMA decay for loss history
        device="cuda",
        depth_array=None,
        norm_stats=None,
        structure_params=None,
        allow_inert_terms=False,
        native_depth_levels=None,
        rate_reference_bits_per_level=None,
        **kwargs
    ):
        super().__init__()
        self.lmbda = lmbda
        check_loss_dict(loss_dict, allow_inert=allow_inert_terms)
        self.loss_dict = loss_dict
        self.extrema_method = extrema_method
        self.use_smoothl1 = use_smoothl1
        self.max_significant_depth_idx = resolve_significant_depth_idx(depth_array)
        self._init_structure(depth_array=depth_array, norm_stats=norm_stats,
                             structure_params=structure_params,
                             allow_inert_terms=allow_inert_terms,
                             native_depth_levels=native_depth_levels,
                             rate_reference_bits_per_level=rate_reference_bits_per_level)
        self.cr_treshold = cr_treshold
        self.dlw_method = dlw_method
        self.alpha = alpha
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.device = device
        self.curvature_beta = curvature_beta
        self.peak_beta = peak_beta
        
        # Number of active losses
        self.n_tasks = len([k for k, v in loss_dict.items() if v > 0])
        self.active_loss_names = [k for k, v in loss_dict.items() if v > 0]
        
        if dlw_method == "uncertainty":
            # Learnable log-variances (homoscedastic uncertainty)
            self.log_vars = nn.Parameter(torch.zeros(self.n_tasks).to(device))
        
        elif dlw_method == "gradnorm":
            # Learnable weights for GradNorm
            self.weights = nn.Parameter(torch.ones(self.n_tasks).to(device))
            self.register_buffer('initial_losses', torch.zeros(self.n_tasks).to(device))
            self.register_buffer('losses_initialized', torch.tensor(False).to(device))
        
        elif dlw_method == "dwa":
            # Dynamic Weight Average: uses loss history
            self.register_buffer('loss_history', torch.zeros(2, self.n_tasks).to(device))  # [t-1, t-2]
            self.register_buffer('step_count', torch.tensor(0).to(device))
        
        elif dlw_method == "ruw":
            # Random Uncertainty Weighting
            self.register_buffer('ema_losses', torch.ones(self.n_tasks).to(device))
            self.register_buffer('step_count', torch.tensor(0).to(device))
        
        # For monitoring
        self.register_buffer('current_weights', torch.ones(self.n_tasks).to(device))
        self.register_buffer('loss_ema', torch.zeros(self.n_tasks).to(device))

    def _compute_individual_losses(self, pred, target):
        """Compute all individual loss components."""
        losses = {}
        
        if "recon" in self.active_loss_names:
            if self.use_smoothl1:
                losses["recon"] = F.smooth_l1_loss(pred, target, reduction="mean")
            else:
                losses["recon"] = torch.mean((pred - target) ** 2)
        
        if "weighted_recon" in self.active_loss_names:
            losses["weighted_recon"] = weighted_mse_loss(
                pred, target,
                max_significant_depth_idx=self.max_significant_depth_idx,
                decay_factor=0.1,
                use_smoothl1=self.use_smoothl1
            )
        
        if "deriv" in self.active_loss_names:
            dp = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            dt = target[:, 1:, :, :] - target[:, :-1, :, :]
            losses["deriv"] = F.mse_loss(dp, dt)

        if "lsd" in self.active_loss_names:
            losses["lsd"] = spectral_loss(
                pred,
                target,
                axis=1,
                normalize=True,
                use_smoothl1=self.use_smoothl1,
            )
        
        if "weighted_deriv" in self.active_loss_names:
            losses["weighted_deriv"] = weighted_deriv_loss(
                pred, target,
                max_significant_depth_idx=self.max_significant_depth_idx,
                decay_factor=0.1,
                use_smoothl1=self.use_smoothl1
            )

        if "curvature_recon" in self.active_loss_names:
            losses["curvature_recon"] = curvature_weighted_loss(
                pred, target,
                depth_dim=1,
                beta=self.curvature_beta,
                use_smoothl1=self.use_smoothl1
            )
        
        if "soft_peak" in self.active_loss_names:
            losses["soft_peak"] = soft_peak_localization_loss(
                pred, target,
                depth_dim=1,
                beta=self.peak_beta
            )
        
        if "wasserstein_peak" in self.active_loss_names:
            losses["wasserstein_peak"] = wasserstein_peak_alignment_loss(
                pred, target,
                depth_dim=1,
                beta=self.peak_beta
            )

        if "max_pos" in self.active_loss_names:
            # Unreachable unless allow_inert_terms=True -- argmax has no gradient.
            with torch.no_grad():
                losses["max_pos"] = torch.abs(
                    torch.argmax(pred, dim=1) - torch.argmax(target, dim=1)
                ).float().mean()
        
        if "max_value" in self.active_loss_names:
            losses["max_value"] = F.mse_loss(
                torch.max(pred, dim=1)[0],
                torch.max(target, dim=1)[0]
            )
        
        if "extrema_pos" in self.active_loss_names or "extrema_value" in self.active_loss_names:
            extrema_pos, extrema_val = position_and_value_loss(
                pred, target, dim=1, tau=10, mode=self.extrema_method
            )
            if "extrema_pos" in self.active_loss_names:
                losses["extrema_pos"] = extrema_pos
            if "extrema_value" in self.active_loss_names:
                losses["extrema_value"] = extrema_val

        losses.update(self._structure_terms(pred, target, self.active_loss_names))

        return losses

    def _uncertainty_weighting(self, losses_tensor):
        """
        Homoscedastic uncertainty weighting (Kendall et al.).
        L = sum_i (1/(2*sigma_i^2) * L_i + log(sigma_i))
        """
        precisions = torch.exp(-self.log_vars)
        weighted_losses = precisions * losses_tensor + self.log_vars
        self.current_weights = precisions.detach()
        return weighted_losses.sum()

    def _dwa_weighting(self, losses_tensor):
        """
        Dynamic Weight Average (Liu et al., 2019).
        Weights based on relative loss descent rate.
        """
        if self.step_count < 2:
            # Not enough history, use uniform weights
            weights = torch.ones_like(losses_tensor)
        else:
            # w_i(t) = softmax(L_i(t-1) / L_i(t-2) / T)
            ratios = self.loss_history[0] / (self.loss_history[1] + 1e-8)
            weights = F.softmax(ratios / self.temperature, dim=0) * self.n_tasks
        
        self.current_weights = weights.detach()
        return (weights * losses_tensor).sum()

    def _ruw_weighting(self, losses_tensor):
        """
        Random Uncertainty Weighting (RUW).
        Sample weights from distribution based on loss magnitudes.
        """
        if self.training:
            # Sample random weights from log-normal distribution
            # Variance inversely proportional to loss magnitude
            log_weights = torch.randn(self.n_tasks, device=losses_tensor.device)
            normalized_losses = losses_tensor / (self.ema_losses + 1e-8)
            weights = torch.exp(log_weights) * (1.0 / (normalized_losses.detach() + 1e-8))
            weights = weights / weights.sum() * self.n_tasks
        else:
            weights = torch.ones_like(losses_tensor)
        
        self.current_weights = weights.detach()
        return (weights * losses_tensor).sum()

    def _gradnorm_weighting(self, losses_tensor):
        """
        GradNorm-style weighting (Chen et al., 2018).
        Note: Full GradNorm requires gradient computation, this is simplified.
        """
        # Initialize with first batch losses
        if not self.losses_initialized:
            self.initial_losses = losses_tensor.detach().clone()
            self.losses_initialized = torch.tensor(True)
        
        # Compute inverse training rates
        loss_ratios = losses_tensor / (self.initial_losses + 1e-8)
        inverse_rates = loss_ratios / (loss_ratios.mean() + 1e-8)
        
        # Target weights: higher weight for slower-improving tasks
        target_weights = inverse_rates ** self.alpha
        target_weights = target_weights / target_weights.sum() * self.n_tasks
        
        # Use learned weights normalized
        weights = F.softmax(self.weights, dim=0) * self.n_tasks
        
        self.current_weights = weights.detach()
        return (weights * losses_tensor).sum()

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # === Bitrate term ===
        bpe = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.cr_treshold is None:
            out["bpp_loss"] = bpe
        else:
            bpe_original = self._rate_reference_bits_per_profile(target)
            bpe_treshold = bpe_original / self.cr_treshold
            out["bpp_loss"] = nn.ReLU()(bpe - bpe_treshold)

        # === Compute individual losses ===
        pred = output["x_hat"]
        losses_dict = self._compute_individual_losses(pred, target)
        
        # Stack losses in order
        losses_tensor = torch.stack([losses_dict[name] for name in self.active_loss_names])
        
        # Update EMA for monitoring
        self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * losses_tensor.detach()

        # === Apply DLW method ===
        if self.dlw_method == "uncertainty":
            distortion = self._uncertainty_weighting(losses_tensor)
        
        elif self.dlw_method == "dwa":
            distortion = self._dwa_weighting(losses_tensor)
            # Update history
            self.loss_history[1] = self.loss_history[0].clone()
            self.loss_history[0] = losses_tensor.detach()
            self.step_count += 1
        
        elif self.dlw_method == "ruw":
            distortion = self._ruw_weighting(losses_tensor)
            self.ema_losses = self.ema_decay * self.ema_losses + (1 - self.ema_decay) * losses_tensor.detach()
            self.step_count += 1
        
        elif self.dlw_method == "gradnorm":
            distortion = self._gradnorm_weighting(losses_tensor)
        
        else:
            # Fallback: fixed weights from loss_dict
            weights = torch.tensor(
                [self.loss_dict[name] for name in self.active_loss_names],
                device=losses_tensor.device
            )
            distortion = (weights * losses_tensor).sum()
            self.current_weights = weights

        # Store individual losses for monitoring
        for name in self.active_loss_names:
            out[f"{name}_loss"] = losses_dict[name]
        
        out["ms_ssim_loss"] = None
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        
        # Store weights for logging
        out["dlw_weights"] = {
            name: self.current_weights[i].item() 
            for i, name in enumerate(self.active_loss_names)
        }

        return out

    def get_weights_dict(self):
        """Return current weights as a dictionary for logging."""
        return {
            name: self.current_weights[i].item()
            for i, name in enumerate(self.active_loss_names)
        }
    

class FixedWeightSSPLoss(StructureLossMixin, nn.Module):
    def __init__(
        self, 
        loss_dict={"recon": 1.0,
                    "weighted_recon": 1.0, 
                    "deriv": 10.0,
                    "lsd": 1.0,
                    "curvature_recon": 1.0,
                    "soft_peak": 1.0,  # soft peak localization
                    "wasserstein_peak": 1.0,  # Wasserstein peak alignment
                    "max_pos": 0.0,          # inert: no gradient, see INERT_LOSS_TERMS
                    "max_value": 1.0,
                    "extrema_pos": 0.0,      # degenerate: centroid, see DEGENERATE_LOSS_TERMS
                    "extrema_value": 1.0,
                    "soft_max_pos": 0.0,          # differentiable replacement for max_pos
                    "matched_extrema_pos": 0.0,   # degenerate: see DEGENERATE_LOSS_TERMS
                    "local_extrema_pos": 0.0,     # working replacement for extrema_pos
                    "prominence_recall": 0.0},    # new: prominence-mass recall
        extrema_method="both", 
        lmbda=1e-2, 
        use_smoothl1=True,
        cr_treshold=10000.0,
        recon_treshold=None,
        curvature_beta=10.0,
        peak_beta=20.0,  # sharpness for soft peak and Wasserstein
        # Poids fixes pour chaque loss
        auto_normalize=False,
        use_factor_weights=False,
        factor_warmup_epochs=150,
        device="cuda",
        dtype="float32",  # Normalise automatiquement pour équilibrer les magnitudes
        depth_array=None,
        deriv_target_smooth_sigma_m=None,
        deriv_target_smooth_max_depth_m=100.0,
        deriv_target_smooth_taper_m=100.0,
        deriv_smooth_applies_to=("deriv", "weighted_deriv"),
        norm_stats=None,
        structure_params=None,
        allow_inert_terms=False,
        native_depth_levels=None,
        rate_reference_bits_per_level=None,
        **kwargs
    ):
        super().__init__()
        self.lmbda = lmbda

        self.extrema_method = extrema_method
        self.use_smoothl1 = use_smoothl1
        self.max_significant_depth_idx = resolve_significant_depth_idx(depth_array)
        self.recon_treshold = recon_treshold
        self.cr_treshold = cr_treshold if recon_treshold is None else None  # If recon_treshold is set, ignore cr_treshold
        if self.recon_treshold is not None:
            loss_dict = {k:0.0 for k in loss_dict.keys()}
            loss_dict["recon"] = 1.0  # Only use recon loss for factor weighting if recon_treshold is set
        check_loss_dict(loss_dict, allow_inert=allow_inert_terms)
        # Terms with a zero weight are not computed at all: they cost a forward
        # pass (the structure terms and lsd are the expensive ones) and, being
        # summed with weight 0, a backward pass too, for a number that could only
        # ever be logged. `recon` is kept under factor weighting, where it is both
        # the warm-up objective and the reference magnitude.
        self.loss_dict_config = dict(loss_dict)
        loss_dict = {k: v for k, v in loss_dict.items()
                     if float(v or 0.0) > 0 or (k == "recon" and use_factor_weights)}
        self.loss_dict = loss_dict
        self.loss_weights = loss_dict.copy()
        self.last_weighted_terms = {}
        self._diagnostic_terms = False
        self._init_structure(depth_array=depth_array, norm_stats=norm_stats,
                             structure_params=structure_params,
                             allow_inert_terms=allow_inert_terms,
                             native_depth_levels=native_depth_levels,
                             rate_reference_bits_per_level=rate_reference_bits_per_level)
        self.curvature_beta = curvature_beta
        self.peak_beta = peak_beta
        self.auto_normalize = auto_normalize
        self.use_factor_weights = use_factor_weights
        self.factor_warmup_epochs = factor_warmup_epochs
        self._factor_weights_applied = False
        self.device = device
        self.dtype = dtype
        
        # Active weights: used during forward pass
        # Before factor weights are applied, only recon=1.0, others=0
        if self.use_factor_weights:
            self.active_weights = {k: (1.0 if k == "recon" else 0.0) for k in loss_dict.keys()}
        else:
            self.active_weights = loss_dict.copy()
        
        # Smoothing of the *target* of the derivative terms. The reconstruction target
        # is deliberately left raw: section 9 of the diagnostics finds that smoothing
        # the field target moves it away from what the model already produces above
        # 100 m, while the derivative target is the one carrying wiggle the model
        # never fits. Off by default (sigma=None) so existing runs are unchanged.
        self.deriv_target_smooth_sigma_m = deriv_target_smooth_sigma_m
        self.deriv_target_smooth_max_depth_m = deriv_target_smooth_max_depth_m
        self.deriv_target_smooth_taper_m = deriv_target_smooth_taper_m
        self.deriv_smooth_applies_to = tuple(deriv_smooth_applies_to)
        smooth_op = None
        if deriv_target_smooth_sigma_m is not None:
            if depth_array is None:
                raise ValueError("deriv_target_smooth_sigma_m needs depth_array to build "
                                 "a metre-domain kernel on the real depth axis")
            smooth_op = depth_smoothing_matrix(
                depth_array, deriv_target_smooth_sigma_m,
                max_depth_m=deriv_target_smooth_max_depth_m,
                taper_m=deriv_target_smooth_taper_m,
                dtype=torch.float64 if dtype == "float64" else torch.float32)
        self.register_buffer('deriv_smooth_op', smooth_op if smooth_op is not None
                             else torch.zeros(0))

        # Pour stocker les magnitudes moyennes (si auto_normalize)
        self.register_buffer('loss_magnitudes', torch.ones(len(loss_dict)))
        self.register_buffer('magnitude_count', torch.tensor(0))

    @contextmanager
    def diagnostic_terms(self):
        """Also compute the zero-weight terms of the configured loss_dict.

        Training skips them (a forward and, summed at weight 0, a backward pass for
        a number that is only logged). Validation and test run under no_grad, so
        there they are cheap and give the matched-baseline value of a term a run
        does not train -- e.g. local_extrema_pos on a run without it. They are
        reported as `<name>_loss` and never enter out["loss"].
        """
        previous = self._diagnostic_terms
        self._diagnostic_terms = True
        try:
            yield self
        finally:
            self._diagnostic_terms = previous

    def _smooth_depth(self, x):
        """Apply the metre-domain smoother along the depth axis of (N, C, H, W)."""
        if self.deriv_smooth_op.numel() == 0:
            return x
        op = self.deriv_smooth_op.to(device=x.device, dtype=x.dtype)
        return torch.einsum("cd,ndhw->nchw", op, x)

    def update_magnitudes(self, losses_dict):
        """Met à jour les estimations de magnitude (EMA)"""
        if not self.auto_normalize or not self.training:
            return
        
        magnitudes = torch.tensor([
            losses_dict[name].detach().item() 
            for name in self.loss_dict.keys()
        ], device=self.loss_magnitudes.device, dtype=self.loss_magnitudes.dtype)
        
        # Exponential moving average
        alpha = 0.01  # Taux d'apprentissage pour l'EMA
        self.loss_magnitudes = (1 - alpha) * self.loss_magnitudes + alpha * magnitudes
        self.magnitude_count += 1

    def apply_factor_weights(self, model, dataloader, device, logger=None):
        """
        Compute and apply factor-based weight normalization.
        
        The factor weights are computed such that if loss_dict["weighted_recon"] = 5.0,
        then the effective weight on weighted_recon will be 5x the weight on recon.
        
        This is achieved by normalizing each loss by its baseline magnitude,
        then applying the user-specified factors.
        """
        if self._factor_weights_applied:
            return
        
        model.eval()
        losses_accum = {name: [] for name in self.loss_dict.keys()}
        
        if logger:
            logger.info("Computing factor-based weights on validation batch...")
        
        with torch.no_grad():
            # Use a few batches to estimate loss magnitudes
            for i, d in enumerate(dataloader):
                if i >= 10:  # Use first 10 batches
                    break
                d = d.to(device)
                out_net = model(d, torch.zeros(len(d),dtype=int, device=device), torch.zeros(len(d), device=device))
                
                # Temporarily compute all losses
                pred = out_net["x_hat"]
                target = d
                
                if "recon" in self.loss_dict:
                    if self.use_smoothl1:
                        recon_loss = F.smooth_l1_loss(pred, target, reduction="mean")
                    else:
                        recon_loss = torch.mean((pred - target) ** 2)

                    
                    losses_accum["recon"].append(recon_loss.item())
                
                if "weighted_recon" in self.loss_dict:
                    weighted_recon_loss = weighted_mse_loss(
                        pred, target, 
                        max_significant_depth_idx=self.max_significant_depth_idx, 
                        decay_factor=0.1, 
                        use_smoothl1=self.use_smoothl1
                    )
                    losses_accum["weighted_recon"].append(weighted_recon_loss.item())
                
                if "deriv" in self.loss_dict:
                    dp = pred[:, 1:, :, :] - pred[:, :-1, :, :]
                    dt = target[:, 1:, :, :] - target[:, :-1, :, :]
                    deriv_loss = F.mse_loss(dp, dt)
                    losses_accum["deriv"].append(deriv_loss.item())
                
                if "weighted_deriv" in self.loss_dict:
                    weighted_deriv = weighted_deriv_loss(
                        pred, target,
                        max_significant_depth_idx=self.max_significant_depth_idx,
                        decay_factor=0.1,
                        use_smoothl1=self.use_smoothl1
                    )
                    losses_accum["weighted_deriv"].append(weighted_deriv.item())

                if "curvature_recon" in self.loss_dict:
                    curvature_loss = curvature_weighted_loss(
                        pred, target,
                        depth_dim=1,
                        beta=self.curvature_beta,
                        use_smoothl1=self.use_smoothl1
                    )
                    losses_accum["curvature_recon"].append(curvature_loss.item())
                
                if "soft_peak" in self.loss_dict:
                    soft_peak_loss = soft_peak_localization_loss(
                        pred, target,
                        depth_dim=1,
                        beta=self.peak_beta
                    )
                    losses_accum["soft_peak"].append(soft_peak_loss.item())
                
                if "wasserstein_peak" in self.loss_dict:
                    wasserstein_loss = wasserstein_peak_alignment_loss(
                        pred, target,
                        depth_dim=1,
                        beta=self.peak_beta
                    )
                    losses_accum["wasserstein_peak"].append(wasserstein_loss.item())
                
                if "max_pos" in self.loss_dict:
                    # Diagnostic only -- see the note in forward().
                    max_pos_loss = torch.abs(
                        torch.argmax(pred, dim=1) - torch.argmax(target, dim=1)
                    ).float().mean()
                    losses_accum["max_pos"].append(max_pos_loss.item())
                
                if "max_value" in self.loss_dict:
                    max_value_loss = F.mse_loss(
                        torch.max(pred, dim=1)[0], 
                        torch.max(target, dim=1)[0]
                    )
                    losses_accum["max_value"].append(max_value_loss.item())
                
                if "extrema_pos" in self.loss_dict or "extrema_value" in self.loss_dict:
                    extrema_pos_loss, extrema_value_loss = position_and_value_loss(
                        pred, target, dim=1, tau=10, mode=self.extrema_method
                    )
                    if "extrema_pos" in self.loss_dict:
                        losses_accum["extrema_pos"].append(extrema_pos_loss.item())
                    if "extrema_value" in self.loss_dict:
                        losses_accum["extrema_value"].append(extrema_value_loss.item())

                # The structure terms have to be calibrated here too, or their
                # magnitude is estimated as 1.0 and the factor weight is wrong.
                for _name, _val in self._structure_terms(
                        pred, target, self.loss_dict).items():
                    losses_accum[_name].append(_val.item())
        
        # Compute mean magnitudes
        mean_magnitudes = {}
        for name, values in losses_accum.items():
            if len(values) > 0:
                mean_magnitudes[name] = sum(values) / len(values)
            else:
                mean_magnitudes[name] = 1.0
        
        # Compute normalization factors relative to recon loss
        recon_mag = mean_magnitudes.get("recon", 1.0)
        if recon_mag == 0:
            recon_mag = 1.0
        
        norm_factors = {}
        for name, mag in mean_magnitudes.items():
            if mag > 0:
                norm_factors[name] = recon_mag / mag
            else:
                norm_factors[name] = 1.0
        
        # Apply factor weights: active_weight = loss_dict[name] * norm_factor
        # This ensures that if loss_dict["weighted_recon"] = 5.0, 
        # the effective weight is 5x the normalized recon weight
        for name in self.loss_dict.keys():
            self.active_weights[name] = self.loss_dict[name] * norm_factors.get(name, 1.0)
        
        self._factor_weights_applied = True
        
        if logger:
            logger.info("Factor-based weights applied:")
            logger.info(f"  Mean magnitudes: {mean_magnitudes}")
            logger.info(f"  Normalization factors: {norm_factors}")
            logger.info(f"  Active weights: {self.active_weights}")
        else:
            print("Factor-based weights applied:")
            print(f"  Mean magnitudes: {mean_magnitudes}")
            print(f"  Normalization factors: {norm_factors}")
            print(f"  Active weights: {self.active_weights}")
        
        model.train()

    def apply_gradient_weights(self, model, dataloader, device, names=None,
                               ratio=0.2, n_batches=4, logger=None):
        """Weight the auxiliary terms by gradient norm instead of by loss value.

        The factor normalisation equalises loss *values*, which set
        `local_extrema_pos` to 0.62 in E1b and gave it a gradient 2-3x recon's
        (report section 14.7). Here each term in ``names`` gets

            active_weight = loss_dict[name] * ratio * |grad recon| / |grad name|

        measured on ``n_batches`` training batches, so with ``loss_dict[name] = 1``
        its gradient norm is ``ratio`` times recon's (recon at its active weight).
        Terms not in ``names`` keep their factor weights. Call after
        :meth:`apply_factor_weights`; can be called again later to re-calibrate.
        """
        names = [n for n in (names or STRUCTURE_LOSS_TERMS)
                 if float(self.loss_dict.get(n, 0.0) or 0.0) > 0]
        if not names or "recon" not in self.loss_dict:
            return {}
        params = [p for p in model.parameters() if p.requires_grad]

        def gnorm(loss):
            if not (torch.is_tensor(loss) and loss.requires_grad):
                return 0.0
            gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return math.sqrt(sum(float((g * g).sum()) for g in gs if g is not None))

        was_training = model.training
        model.train()
        ratios = {n: [] for n in names}
        for i, d in enumerate(dataloader):
            if i >= n_batches:
                break
            d = d.to(device)
            out_net = model(d, torch.zeros(len(d), dtype=int, device=device),
                            torch.zeros(len(d), device=device))
            out = self(out_net, d)
            g_recon = self.active_weights.get("recon", 1.0) * gnorm(out["recon_loss"])
            for n in names:
                g = gnorm(out[f"{n}_loss"])
                if g > 0:
                    ratios[n].append(g_recon / g)
            del out, out_net
        model.train(was_training)

        applied = {}
        for n, r in ratios.items():
            if r:
                self.active_weights[n] = (float(self.loss_dict[n]) * float(ratio)
                                          * sum(r) / len(r))
                applied[n] = self.active_weights[n]
        msg = (f"Gradient-norm weights (target |grad| = {ratio} x recon's, "
               f"{n_batches} batches): {applied}")
        (logger.info if logger else print)(msg)
        return applied

    def forward(self, output, target):
        """
        Args:
            output: dict contenant 'x_hat' (batch, C, H, W) et 'likelihoods'
            target: (batch, C, H, W)
        """
        N, _, H, W = target.size()
        out = {}
        losses_dict = {}
        num_pixels = N * H * W
        # Terms to compute: the trained ones, plus -- inside diagnostic_terms(),
        # i.e. validation/test -- the zero-weight ones, for logging only. Only
        # self.loss_dict enters the weighted sum below.
        wanted = (self.loss_dict_config if self._diagnostic_terms else self.loss_dict)

        # === Bitrate term (bpp loss) ===
        bpe_original = self._rate_reference_bits_per_profile(target)


        bpe = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.cr_treshold is None:
            out["bpp_loss"] = bpe
        else:
            bpe_treshold = bpe_original / self.cr_treshold
            out["bpp_loss"] = nn.ReLU()(bpe - bpe_treshold)

        # === Distortion term ===
        pred = output["x_hat"]

        # Calcul de toutes les losses individuelles
        if "recon" in wanted:
            if self.use_smoothl1:
                recon_loss = F.smooth_l1_loss(pred, target, reduction="mean")
            else:
                recon_loss = torch.mean((pred - target) ** 2)

            
            if self.recon_treshold is not None:
                recon_loss = nn.ReLU()(recon_loss - self.recon_treshold)

            losses_dict["recon"] = recon_loss
        
        if "weighted_recon" in wanted:
            weighted_recon_loss = weighted_mse_loss(
                pred, target, 
                max_significant_depth_idx=self.max_significant_depth_idx, 
                decay_factor=0.1, 
                use_smoothl1=self.use_smoothl1
            )
            losses_dict["weighted_recon"] = weighted_recon_loss
        
        if "deriv" in wanted:
            deriv_target = (self._smooth_depth(target)
                            if "deriv" in self.deriv_smooth_applies_to else target)
            dp = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            dt = deriv_target[:, 1:, :, :] - deriv_target[:, :-1, :, :]
            deriv_loss = F.mse_loss(dp, dt)
            losses_dict["deriv"] = deriv_loss

        if "lsd" in wanted:
            lsd_loss = spectral_loss(
                pred,
                target,
                axis=1,
                normalize=True,
                use_smoothl1=self.use_smoothl1,
            )
            losses_dict["lsd"] = lsd_loss
        
        if "weighted_deriv" in wanted:
            wderiv_target = (self._smooth_depth(target)
                             if "weighted_deriv" in self.deriv_smooth_applies_to else target)
            weighted_deriv = weighted_deriv_loss(
                pred, wderiv_target,
                max_significant_depth_idx=self.max_significant_depth_idx,
                decay_factor=0.1,
                use_smoothl1=self.use_smoothl1
            )
            losses_dict["weighted_deriv"] = weighted_deriv
        
        if "curvature_recon" in wanted:
            curvature_loss = curvature_weighted_loss(
                pred, target,
                depth_dim=1,
                beta=self.curvature_beta,
                use_smoothl1=self.use_smoothl1
            )
            losses_dict["curvature_recon"] = curvature_loss
        
        if "soft_peak" in wanted:
            soft_peak_loss = soft_peak_localization_loss(
                pred, target,
                depth_dim=1,
                beta=self.peak_beta
            )
            losses_dict["soft_peak"] = soft_peak_loss
        
        if "wasserstein_peak" in wanted:
            wasserstein_loss = wasserstein_peak_alignment_loss(
                pred, target,
                depth_dim=1,
                beta=self.peak_beta
            )
            losses_dict["wasserstein_peak"] = wasserstein_loss
        
        if "max_pos" in wanted:
            # Diagnostic only. argmax is not differentiable, so this cannot train
            # anything; computed under no_grad and kept in the logs so the column
            # stays comparable with the runs that reported it. check_loss_dict
            # refuses a non-zero weight on it.
            with torch.no_grad():
                losses_dict["max_pos"] = torch.abs(
                    torch.argmax(pred, dim=1) - torch.argmax(target, dim=1)
                ).float().mean()
        
        if "max_value" in wanted:
            max_value_loss = F.mse_loss(
                torch.max(pred, dim=1)[0], 
                torch.max(target, dim=1)[0]
            )
            losses_dict["max_value"] = max_value_loss
        
        if "extrema_pos" in wanted or "extrema_value" in wanted:
            extrema_pos_loss, extrema_value_loss = position_and_value_loss(
                pred, target, dim=1, tau=10, mode=self.extrema_method
            )
            if "extrema_pos" in wanted:
                losses_dict["extrema_pos"] = extrema_pos_loss
            if "extrema_value" in wanted:
                losses_dict["extrema_value"] = extrema_value_loss

        losses_dict.update(self._structure_terms(pred, target, wanted))

        # Met à jour les magnitudes moyennes
        self.update_magnitudes(losses_dict)

        # === Weighted sum avec normalisation optionnelle ===
        # A term at weight 0 (factor warm-up) stays out of the sum, so the backward
        # pass does not traverse its graph. `last_weighted_terms` keeps each term's
        # contribution to out["loss"] with its graph, for grad_monitor.
        distortion = 0
        weighted_terms = {}
        for i, name in enumerate(self.loss_dict.keys()):
            loss = losses_dict[name]
            
            # Use active_weights when factor weights are enabled
            if self.use_factor_weights:
                weight = self.active_weights[name]
                term = weight * loss
            elif self.auto_normalize and self.magnitude_count > 100:
                # Normalise par la magnitude moyenne pour équilibrer
                weight = self.loss_dict[name]
                term = weight * (loss / (self.loss_magnitudes[i] + 1e-8))
                self.loss_weights[name] = weight / (self.loss_magnitudes[i].item() + 1e-8)
            else:
                weight = self.loss_dict[name]
                term = weight * loss
            if float(weight) == 0.0:
                continue
            distortion += term
            weighted_terms[name] = (float(weight), loss.detach(), self.lmbda * term)
        weighted_terms["bpp"] = (1.0, out["bpp_loss"].detach(), out["bpp_loss"])
        self.last_weighted_terms = weighted_terms

        # Stockage pour monitoring
        for name in losses_dict:
            out[f"{name}_loss"] = losses_dict[name]
        
        out["ms_ssim_loss"] = None
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        
        # # Ajoute info pour monitoring
        # out["loss_weights"] = self.loss_weights
        # out["loss_magnitudes"] = self.loss_magnitudes.clone()

        return out






# ============================================
# UTILITAIRE: Estimer les poids optimaux
# ============================================

def estimate_optimal_weights(net, criterion, dataloader, device, method="inverse_mean"):
    """
    Estime les poids optimaux pour équilibrer les losses
    
    Args:
        method: "inverse_mean" → poids = 1/mean_loss
                "inverse_std" → poids = 1/std_loss
                "snr" → poids = mean²/std² (signal-to-noise ratio)
    """
    net.eval()
    losses_accum = {name: [] for name in criterion.loss_dict.keys()}
    
    print("Estimation des poids sur le dataset...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 100:  # Limite pour aller vite
                break
            
            output = net(batch.to(device))
            result = criterion(output, batch.to(device))
            
            for name in criterion.loss_dict.keys():
                losses_accum[name].append(result[f"{name}_loss"].item())
    
    # Calcule les statistiques
    stats = {}
    for name, values in losses_accum.items():
        stats[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values)
        }
    
    # Calcule les poids selon la méthode
    weights = {}
    
    if method == "inverse_mean":
        # Poids inversement proportionnel à la magnitude
        for name in criterion.loss_dict.keys():
            weights[name] = 1.0 / (stats[name]['mean'] + 1e-8)
    
    elif method == "inverse_std":
        # Poids inversement proportionnel à la variance
        for name in criterion.loss_dict.keys():
            weights[name] = 1.0 / (stats[name]['std'] + 1e-8)
    
    elif method == "snr":
        # Signal-to-noise ratio 
        for name in criterion.loss_dict.keys():
            mean = stats[name]['mean']
            std = stats[name]['std']
            weights[name] = (mean ** 2) / (std ** 2 + 1e-8)
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    # Normalise pour que la somme = nombre de losses
    total = sum(weights.values())
    n = len(weights)
    weights = {k: v * n / total for k, v in weights.items()}
    
    # Affiche les résultats
    print("\n" + "="*60)
    print(f"Statistiques des losses et poids estimés ({method}):")
    print("="*60)
    for name in criterion.loss_dict.keys():
        print(f"{name:20s} | mean={stats[name]['mean']:.6f} | "
              f"std={stats[name]['std']:.6f} | weight={weights[name]:.4f}")
    print("="*60 + "\n")
    
    return weights, stats




def fourier_loss(outputs, inputs, depth_dim=1):
    outputs_fft = torch.fft.fft(outputs, dim=depth_dim)
    inputs_fft = torch.fft.fft(inputs, dim=depth_dim)
    return torch.mean((torch.abs(outputs_fft) - torch.abs(inputs_fft)) ** 2)



def error_treshold_based_mse_loss(inputs, outputs, max_value_threshold=3.0):
    mask = (torch.abs(inputs - outputs) > max_value_threshold).float()
    diff = (inputs - outputs)**2
    # Only keep differences for masked elements
    masked_diff = diff * mask
    sse = masked_diff.sum()
    n = mask.sum()
    mse = sse / (n + 1e-8)
    return mse



def weighted_mse_loss(outputs, inputs, max_significant_depth_idx = 10, decay_factor = 1000, use_smoothl1=False):  #decay_factor = 0.1

    max_significant_depth_idx = max(0, min(max_significant_depth_idx, inputs.shape[1]-1))  # Ensure it's within bounds
    signal_length = inputs.shape[1]

    weights = torch.ones(signal_length, device=inputs.device, dtype=inputs.dtype)


    weights[:max_significant_depth_idx] = 1.0  # Strong emphasis on the first points
    weights[max_significant_depth_idx+1:] = torch.exp(-decay_factor * torch.arange(max_significant_depth_idx+1, signal_length))

    # Reshape to match inputs shape
    weights = weights.view(1, 1, -1, 1, 1)  # Shape: [1, 1, signal_length, 1, 1]

    if use_smoothl1:
        weighted_loss = F.smooth_l1_loss(weights * outputs, weights * inputs, reduction='mean')
    else:
        weighted_loss =  torch.mean(weights * (outputs - inputs) ** 2)
    return weighted_loss



def depth_smoothing_matrix(depth_array, sigma_m, max_depth_m=None, taper_m=100.0,
                           device=None, dtype=torch.float32):
    """Gaussian smoothing along depth, of constant width in **metres**, optionally
    restricted to the upper ocean.

    Built for smoothing the *target* of the derivative term. The diagnostics behind
    this (notebook section 9) show the reconstruction's dc/dz sitting closer to a
    10-15 m smoothed reference than to the raw one above ~100 m, while below ~600 m
    the raw target is already the better one -- so the operator is the identity
    below ``max_depth_m``, with a linear taper of width ``taper_m`` so the loss has
    no kink there.

    A matrix rather than a convolution because the depth axis is not uniform: the
    kernel weights each source level by the depth interval it represents, so the
    same physical scale is removed at every depth.
    """
    z = np.asarray(depth_array, dtype=np.float64).ravel()
    dz = np.gradient(z)
    w = np.exp(-0.5 * ((z[:, None] - z[None, :]) / float(sigma_m)) ** 2) * dz[None, :]
    w /= w.sum(axis=1, keepdims=True)

    if max_depth_m is not None:
        alpha = np.clip((float(max_depth_m) + float(taper_m) - z) / max(float(taper_m), 1e-9), 0.0, 1.0)
        w = alpha[:, None] * w + (1.0 - alpha)[:, None] * np.eye(z.size)

    t = torch.as_tensor(w, dtype=dtype)
    return t if device is None else t.to(device)


def weighted_deriv_loss(outputs, inputs, max_significant_depth_idx=10, decay_factor=1000, use_smoothl1=False):
    """Weighted derivative loss with emphasis on first depth indices.
    
    Computes derivative along depth dimension (dim=1) and applies 
    exponentially decaying weights similar to weighted_mse_loss.
    """
    # Compute derivatives along depth dimension
    dp = outputs[:, 1:, :, :] - outputs[:, :-1, :, :]
    dt = inputs[:, 1:, :, :] - inputs[:, :-1, :, :]
    
    # derivative signal length is one less than input
    deriv_length = dp.shape[1]
    max_significant_depth_idx = max(0, min(max_significant_depth_idx, deriv_length - 1))  # Ensure it's within bounds
    
    weights = torch.ones(deriv_length, device=inputs.device, dtype=inputs.dtype)
    
    # Adjust max_significant_depth_idx for derivative (shifted by 1)
    max_idx = min(max_significant_depth_idx, deriv_length)
    
    weights[:max_idx] = 1.0  # Strong emphasis on the first points
    if max_idx < deriv_length:
        weights[max_idx:] = torch.exp(-decay_factor * torch.arange(0, deriv_length - max_idx, device=inputs.device, dtype=inputs.dtype))
    
    # Reshape to match derivative shape (B, C-1, H, W) -> weights shape (1, C-1, 1, 1)
    weights = weights.view(1, -1, 1, 1)
    
    if use_smoothl1:
        weighted_loss = F.smooth_l1_loss(weights * dp, weights * dt, reduction='mean')
    else:
        weighted_loss = torch.mean(weights * (dp - dt) ** 2)
    
    return weighted_loss


def power_spectrum(x, axis=-1, n=None, normalize=True):
    """Compute a differentiable one-sided power spectrum for real-valued tensors."""
    X = torch.fft.rfft(x, n=n, dim=axis)
    ps = X.real.pow(2) + X.imag.pow(2)
    if normalize:
        length = x.shape[axis] if n is None else n
        ps = ps / max(1, int(length))
    return ps


def spectral_loss(outputs, inputs, axis=1, n=None, normalize=True, use_smoothl1=False):
    """Loss between output and target power spectra along the chosen axis."""
    outputs_ps = power_spectrum(outputs, axis=axis, n=n, normalize=normalize)
    inputs_ps = power_spectrum(inputs, axis=axis, n=n, normalize=normalize)

    if use_smoothl1:
        return F.smooth_l1_loss(outputs_ps, inputs_ps, reduction='mean')
    return F.mse_loss(outputs_ps, inputs_ps)



def max_position_and_value_loss(inputs,outputs, depth_dim=1):

        inputs_max_value, inputs_max_pos = torch.max(inputs, dim=depth_dim)
        outputs_max_value, outputs_max_pos = torch.max(outputs, dim=depth_dim)

        max_position_loss =  nn.MSELoss()(inputs_max_pos.float(), outputs_max_pos.float()) 
        max_value_loss =  nn.MSELoss()(inputs_max_value, outputs_max_value) 

        return max_position_loss, max_value_loss


# def min_max_position_and_value_loss(inputs,outputs, depth_dim=1, tau = 10):

#     signal_length = inputs.shape[1]
#     min_max_inputs_mask = DF.differentiable_min_max_search(inputs,dim=depth_dim,tau=tau)
#     min_max_outputs_mask = DF.differentiable_min_max_search(outputs, dim=depth_dim, tau=tau)
#     signal_shape = [1] * inputs.dim()
#     signal_shape[depth_dim] = -1
#     index_tensor = torch.arange(0, signal_length, device=inputs.device, dtype=inputs.dtype).view(*signal_shape) 
#     truth_inflex_pos = (min_max_inputs_mask * index_tensor).sum(dim=depth_dim)/min_max_inputs_mask.sum(dim=depth_dim)
#     pred_inflex_pos = (min_max_outputs_mask * index_tensor).sum(dim=depth_dim)/min_max_outputs_mask.sum(dim=depth_dim)

#     min_max_pos_loss = nn.MSELoss()(pred_inflex_pos, truth_inflex_pos)
#     min_max_value_loss = nn.MSELoss(reduction="none")(outputs,inputs)*min_max_inputs_mask
#     min_max_value_loss = min_max_value_loss.mean()

#     return min_max_pos_loss, min_max_value_loss

def position_and_value_loss(inputs, outputs, dim=1, tau=10, mode="minmax"):
    """
    Compute position and value loss for extrema (min/max) or inflection points.
    Args:
        inputs: ground truth tensor (B, L, ...)
        outputs: predicted tensor (B, L, ...)
        dim: dimension along which to detect
        tau: softness for differentiable sign
        mode: "minmax" or "inflection"
    Returns:
        pos_loss: MSE of detected positions
        value_loss: MSE of values at detected points
    """
    signal_length = inputs.shape[dim]

    if mode == "minmax":
        mask_inputs = DF.differentiable_min_max_search(inputs, dim=dim, tau=tau)
        mask_outputs = DF.differentiable_min_max_search(outputs, dim=dim, tau=tau)
    elif mode == "inflection":
        mask_inputs = DF.differentiable_inflection_search(inputs, dim=dim, tau=tau)
        mask_outputs = DF.differentiable_inflection_search(outputs, dim=dim, tau=tau)
    elif mode == "both":
        mask_inputs_minmax = DF.differentiable_min_max_search(inputs, dim=dim, tau=tau)
        mask_outputs_minmax = DF.differentiable_min_max_search(outputs, dim=dim, tau=tau)
        mask_inputs_infl = DF.differentiable_inflection_search(inputs, dim=dim, tau=tau)
        mask_outputs_infl = DF.differentiable_inflection_search(outputs, dim=dim, tau=tau)
        mask_inputs = torch.clamp(mask_inputs_minmax + mask_inputs_infl, 0, 1)
        mask_outputs = torch.clamp(mask_outputs_minmax + mask_outputs_infl, 0, 1)
    else:
        raise ValueError("mode must be 'minmax' or 'inflection'")

    # index tensor along chosen dim
    shape = [1] * inputs.dim()
    shape[dim] = -1
    index_tensor = torch.arange(
        0, signal_length, device=inputs.device, dtype=inputs.dtype
    ).view(*shape)

    # expected index (soft position)
    truth_pos = (mask_inputs * index_tensor).sum(dim=dim) / (mask_inputs.sum(dim=dim) + 1e-6)
    pred_pos = (mask_outputs * index_tensor).sum(dim=dim) / (mask_outputs.sum(dim=dim) + 1e-6)

    pos_loss = nn.MSELoss()(pred_pos, truth_pos)

    # value loss (only at GT mask positions)
    value_loss = (
        nn.MSELoss(reduction="none")(outputs, inputs) * mask_inputs
    ).sum() / (mask_inputs.sum() + 1e-6)

    return pos_loss, value_loss


def gradient_mse_loss(inputs, outputs, depth_tens, depth_dim=1):
    assert len(depth_tens)>1, "Depth tensor must have more than one element"
    coordinates = (depth_tens,)
    ssp_gradient_inputs = torch.gradient(input = inputs, spacing = coordinates, dim=depth_dim)[0]
    ssp_gradient_outputs = torch.gradient(input = outputs, spacing = coordinates, dim=depth_dim)[0]

    gradient_loss =  nn.MSELoss()(ssp_gradient_inputs, ssp_gradient_outputs) 
    return gradient_loss



def f1_score(min_max_idx_truth, min_max_idx_ae, dim=1):
    # Define the kernel based on the shape of the truth tensor
    kernel_shape = [1] * (min_max_idx_truth.ndim - 1)
    kernel_shape[0] = 10  # Set the size of the kernel along the specified axis
    kernel = torch.ones(kernel_shape, device=min_max_idx_truth.device, dtype=min_max_idx_truth.dtype)

    if min_max_idx_truth.ndim == 2:
        # Expand the truth tensor with the kernel for 2D inputs
        truth_expanded = F.conv1d(min_max_idx_truth.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding='same').squeeze()
        ae_expanded = F.conv1d(min_max_idx_ae.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding='same').squeeze()
    elif min_max_idx_truth.ndim == 4:
        # Expand the truth tensor with the kernel for 4D inputs
        truth_expanded = F.conv3d(min_max_idx_truth.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding='same').squeeze()
        ae_expanded = F.conv3d(min_max_idx_ae.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding='same').squeeze()
    else:
        raise ValueError("Unsupported input dimensions")

    # Compute the true positives
    true_positives = (truth_expanded > 0) & (min_max_idx_ae > 0)
    num_true_positives = torch.sum(true_positives).item()

    # Compute the false positives
    false_positives = (truth_expanded == 0) & (min_max_idx_ae > 0)
    num_false_positives = torch.sum(false_positives).item()

    # Compute the true negatives
    true_negatives = (min_max_idx_truth == 0) & (min_max_idx_ae == 0)
    num_true_negatives = torch.sum(true_negatives).item()

    # Compute the false negatives
    false_negatives = (min_max_idx_truth > 0) & (ae_expanded == 0)
    num_false_negatives = torch.sum(false_negatives).item()

    precision_score = num_true_positives / (num_true_positives + num_false_positives)
    recall_score = num_true_positives / (num_true_positives + num_false_negatives)
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)

    return f1_score


def ratio_exceeding_abs_error(inputs, outputs, threshold=3):
    abs_error = torch.abs(inputs - outputs)
    exceeding_mask = abs_error > threshold
    percentage_exceeding = torch.sum(exceeding_mask).item() / exceeding_mask.numel()
    return percentage_exceeding

def max_abs_error(inputs, outputs):
    abs_error = torch.abs(inputs - outputs)
    max_error = torch.max(abs_error).item()
    return max_error




def soft_peak_localization_loss(
    pred,
    target,
    depth_dim=1,
    beta=20.0,          # peak sharpness
    position_weight=1.0,
    amplitude_weight=1.0,
):
    """
    Differentiable peak localization loss for SSP profiles.
    
    Computes a soft peak position and amplitude loss using softmax-weighted
    coordinates. This encourages the model to preserve peak locations and values.
    
    Args:
        pred: tensor of shape (B, C, H, W) - predicted SSP profiles
        target: tensor of shape (B, C, H, W) - ground truth SSP profiles
        depth_dim: dimension along which to detect peaks (default=1 for depth)
        beta: sharpness of peak detection (higher = sharper peaks)
        position_weight: weight for position loss component
        amplitude_weight: weight for amplitude loss component
    
    Returns:
        Scalar loss value (weighted sum of position and amplitude losses)
    """
    B, C, H, W = pred.shape
    device = pred.device
    dtype = pred.dtype
    
    # Create coordinate grid along depth dimension [0, 1]
    L = pred.shape[depth_dim]  # depth length
    x = torch.linspace(0, 1, L, device=device, dtype=dtype)
    
    # Reshape x for broadcasting: (1, C, 1, 1) for depth_dim=1
    shape = [1] * pred.dim()
    shape[depth_dim] = L
    x = x.view(*shape)
    
    # Softmax peak probability along depth dimension
    p_pred = F.softmax(beta * pred, dim=depth_dim)
    p_target = F.softmax(beta * target, dim=depth_dim)
    
    # Soft peak positions (expected value of position)
    peak_pos_pred = torch.sum(p_pred * x, dim=depth_dim)
    peak_pos_target = torch.sum(p_target * x, dim=depth_dim)
    
    # Peak amplitude (soft-weighted values)
    peak_amp_pred = torch.sum(p_pred * pred, dim=depth_dim)
    peak_amp_target = torch.sum(p_target * target, dim=depth_dim)
    
    # Position loss
    pos_loss = F.mse_loss(peak_pos_pred, peak_pos_target)
    
    # Amplitude loss
    amp_loss = F.mse_loss(peak_amp_pred, peak_amp_target)
    
    return position_weight * pos_loss + amplitude_weight * amp_loss


def wasserstein_peak_alignment_loss(
    pred,
    target,
    depth_dim=1,
    beta=20.0,          # sharpness of peak detection
    eps=1e-8,
):
    """
    Wasserstein-1 distance between soft peak distributions.
    
    Computes the Earth Mover's Distance (EMD) between softmax-weighted
    distributions along the depth dimension. This provides a robust
    measure of peak alignment that is smooth and differentiable.
    
    Args:
        pred: tensor of shape (B, C, H, W) - predicted SSP profiles
        target: tensor of shape (B, C, H, W) - ground truth SSP profiles
        depth_dim: dimension along which to compute distributions (default=1 for depth)
        beta: sharpness of peak detection (higher = sharper distributions)
        eps: small constant for numerical stability
    
    Returns:
        Scalar loss value (mean Wasserstein-1 distance)
    """
    # Soft peak distributions via softmax
    p_pred = F.softmax(beta * pred, dim=depth_dim)
    p_target = F.softmax(beta * target, dim=depth_dim)
    
    # Cumulative Distribution Functions (CDFs)
    cdf_pred = torch.cumsum(p_pred, dim=depth_dim)
    cdf_target = torch.cumsum(p_target, dim=depth_dim)
    
    # Wasserstein-1 distance = integral of |CDF_pred - CDF_target|
    w_distance = torch.mean(torch.abs(cdf_pred - cdf_target))
    
    return w_distance


def curvature_weighted_loss(
    pred,
    target,
    depth_dim=1,
    beta=10.0,             # sharpness of extrema focus
    use_smoothl1=True,
    eps=1e-8,
):
    """
    Curvature-weighted loss that emphasizes regions with high curvature (extrema).
    
    Computes squared error weighted by the local curvature magnitude,
    giving more importance to min/max and inflection regions.
    
    Args:
        pred: tensor of shape (B, C, H, W) - predicted SSP profiles
        target: tensor of shape (B, C, H, W) - ground truth SSP profiles
        depth_dim: dimension along which to compute curvature (default=1 for depth)
        beta: sharpness of curvature weighting (higher = sharper focus on extrema)
        use_smoothl1: if True, use smooth L1 loss instead of MSE
        eps: small constant for numerical stability
    
    Returns:
        Scalar loss value
    """
    # First derivative along depth dimension
    d1_pred = torch.diff(pred, dim=depth_dim)
    d1_target = torch.diff(target, dim=depth_dim)
    
    # Second derivative (curvature proxy)
    d2_pred = torch.diff(d1_pred, dim=depth_dim)
    d2_target = torch.diff(d1_target, dim=depth_dim)
    
    # Pad to match original depth dimension size
    # Padding format: (left, right) for last dim, then second-to-last, etc.
    if depth_dim == 1:
        d2_target = F.pad(d2_target, (0, 0, 0, 0, 1, 1))  # pad depth dim
        d2_pred = F.pad(d2_pred, (0, 0, 0, 0, 1, 1))
    else:
        raise ValueError(f"Unsupported depth_dim={depth_dim}, expected 1")
    
    # Soft extrema weighting based on target curvature
    # Large curvature -> large weight (focus on extrema regions)
    weights = torch.tanh(beta * torch.abs(d2_target))
    
    # Normalize weights to avoid scale issues
    weights = weights / (weights.mean() + eps)
    
    # Compute weighted loss
    if use_smoothl1:
        se = F.smooth_l1_loss(pred, target, reduction='none')
    else:
        se = (pred - target) ** 2
    
    weighted = weights * se
    return weighted.mean()
