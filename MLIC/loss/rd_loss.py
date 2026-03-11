import math
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
import numpy as np
from FASCINATION.src import differentiable_fonc as DF

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics='mse', cr_treshold=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metrics = metrics
        self.cr_treshold = cr_treshold
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
            bpe_original = target.nelement()*target.element_size()*8 / num_pixels
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



class HomoscedasticSSPLoss(nn.Module):
    def __init__(self, 
                 loss_dict={"recon": 1.0,
                            "weighted_recon": 1.0,
                            "deriv": 1.0,
                            "max_pos": 1.0,
                            "max_value": 1.0,
                            "extrema_pos": 1.0,
                            "extrema_value": 1.0}, 
                extrema_method="both",
                lmbda=1e-2,
                use_smoothl1=True,
                lambda_deriv=0.1,
                lambda_extrema=0.5,
                **kwargs):
        super().__init__()
        self.lmbda = lmbda
        self.loss_dict = loss_dict
        self.extrema_method = extrema_method
        self.use_smoothl1 = use_smoothl1

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
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

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
            weighted_recon_loss = weighted_mse_loss(pred, target, max_significant_depth_idx=60, decay_factor=0.1, use_smoothl1=self.use_smoothl1)
            losses_dict["weighted_recon"] = weighted_recon_loss

        if "deriv" in self.loss_dict and self.loss_dict["deriv"] > 0:
            # Derivative-aware loss
            dp = pred[:, 1:] - pred[:, :-1]
            dt = target[:, 1:] - target[:, :-1]
            deriv_loss = F.mse_loss(dp, dt)
            losses_dict["deriv"] = deriv_loss

        if "max_pos" in self.loss_dict and self.loss_dict["max_pos"] > 0:
            max_pos_loss =torch.abs(torch.argmax(pred,dim=1) - torch.argmax(target,dim=1))
            max_pos_loss = max_pos_loss.float().mean()
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



class DynamicLossWeightingSSPLoss(nn.Module):
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
        **kwargs
    ):
        super().__init__()
        self.lmbda = lmbda
        self.loss_dict = loss_dict
        self.extrema_method = extrema_method
        self.use_smoothl1 = use_smoothl1
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
                max_significant_depth_idx=60,
                decay_factor=0.1,
                use_smoothl1=self.use_smoothl1
            )
        
        if "deriv" in self.active_loss_names:
            dp = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            dt = target[:, 1:, :, :] - target[:, :-1, :, :]
            losses["deriv"] = F.mse_loss(dp, dt)
        
        if "weighted_deriv" in self.active_loss_names:
            losses["weighted_deriv"] = weighted_deriv_loss(
                pred, target,
                max_significant_depth_idx=60,
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
            bpe_original = target.nelement() * target.element_size() * 8 / num_pixels
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
    

class FixedWeightSSPLoss(nn.Module):
    def __init__(
        self, 
        loss_dict={"recon": 1.0,
                    "weighted_recon": 1.0, 
                    "deriv": 10.0,
                    "curvature_recon": 1.0,
                    "soft_peak": 1.0,  # soft peak localization
                    "wasserstein_peak": 1.0,  # Wasserstein peak alignment
                    "max_pos": 0.00001,
                    "max_value": 1.0,
                    "extrema_pos": 0.00001, 
                    "extrema_value": 1.0}, 
        extrema_method="both", 
        lmbda=1e-2, 
        use_smoothl1=True,
        cr_treshold=10000.0,
        curvature_beta=10.0,
        peak_beta=20.0,  # sharpness for soft peak and Wasserstein
        # Poids fixes pour chaque loss
        auto_normalize=False,
        use_factor_weights=False,
        factor_warmup_epochs=150,
        device="cuda",
        dtype="float32",  # Normalise automatiquement pour équilibrer les magnitudes
        **kwargs
    ):
        super().__init__()
        self.lmbda = lmbda
        self.loss_dict = loss_dict
        self.loss_weights = loss_dict.copy()
        self.extrema_method = extrema_method
        self.use_smoothl1 = use_smoothl1
        self.cr_treshold = cr_treshold
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
        
        # Pour stocker les magnitudes moyennes (si auto_normalize)
        self.register_buffer('loss_magnitudes', torch.ones(len(loss_dict)))
        self.register_buffer('magnitude_count', torch.tensor(0))

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
                        max_significant_depth_idx=60, 
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
                        max_significant_depth_idx=60,
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

        # === Bitrate term (bpp loss) ===
        bpe_original = target.nelement()*target.element_size()*8 / num_pixels


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
        if "recon" in self.loss_dict:
            if self.use_smoothl1:
                recon_loss = F.smooth_l1_loss(pred, target, reduction="mean")
            else:
                recon_loss = torch.mean((pred - target) ** 2)
            losses_dict["recon"] = recon_loss
        
        if "weighted_recon" in self.loss_dict:
            weighted_recon_loss = weighted_mse_loss(
                pred, target, 
                max_significant_depth_idx=60, 
                decay_factor=0.1, 
                use_smoothl1=self.use_smoothl1
            )
            losses_dict["weighted_recon"] = weighted_recon_loss
        
        if "deriv" in self.loss_dict:
            dp = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            dt = target[:, 1:, :, :] - target[:, :-1, :, :]
            deriv_loss = F.mse_loss(dp, dt)
            losses_dict["deriv"] = deriv_loss
        
        if "weighted_deriv" in self.loss_dict:
            weighted_deriv = weighted_deriv_loss(
                pred, target,
                max_significant_depth_idx=60,
                decay_factor=0.1,
                use_smoothl1=self.use_smoothl1
            )
            losses_dict["weighted_deriv"] = weighted_deriv
        
        if "curvature_recon" in self.loss_dict:
            curvature_loss = curvature_weighted_loss(
                pred, target,
                depth_dim=1,
                beta=self.curvature_beta,
                use_smoothl1=self.use_smoothl1
            )
            losses_dict["curvature_recon"] = curvature_loss
        
        if "soft_peak" in self.loss_dict:
            soft_peak_loss = soft_peak_localization_loss(
                pred, target,
                depth_dim=1,
                beta=self.peak_beta
            )
            losses_dict["soft_peak"] = soft_peak_loss
        
        if "wasserstein_peak" in self.loss_dict:
            wasserstein_loss = wasserstein_peak_alignment_loss(
                pred, target,
                depth_dim=1,
                beta=self.peak_beta
            )
            losses_dict["wasserstein_peak"] = wasserstein_loss
        
        if "max_pos" in self.loss_dict:
            max_pos_loss = torch.abs(
                torch.argmax(pred, dim=1) - torch.argmax(target, dim=1)
            )
            max_pos_loss = max_pos_loss.float().mean()
            losses_dict["max_pos"] = max_pos_loss
        
        if "max_value" in self.loss_dict:
            max_value_loss = F.mse_loss(
                torch.max(pred, dim=1)[0], 
                torch.max(target, dim=1)[0]
            )
            losses_dict["max_value"] = max_value_loss
        
        if "extrema_pos" in self.loss_dict or "extrema_value" in self.loss_dict:
            extrema_pos_loss, extrema_value_loss = position_and_value_loss(
                pred, target, dim=1, tau=10, mode=self.extrema_method
            )
            if "extrema_pos" in self.loss_dict:
                losses_dict["extrema_pos"] = extrema_pos_loss
            if "extrema_value" in self.loss_dict:
                losses_dict["extrema_value"] = extrema_value_loss

        # Met à jour les magnitudes moyennes
        self.update_magnitudes(losses_dict)

        # === Weighted sum avec normalisation optionnelle ===
        distortion = 0
        for i, name in enumerate(self.loss_dict.keys()):
            loss = losses_dict[name]
            
            # Use active_weights when factor weights are enabled
            if self.use_factor_weights:
                weight = self.active_weights[name]
                distortion += weight * loss
            elif self.auto_normalize and self.magnitude_count > 100:
                # Normalise par la magnitude moyenne pour équilibrer
                weight = self.loss_dict[name]
                normalized_loss = loss / (self.loss_magnitudes[i] + 1e-8)
                distortion += weight * normalized_loss
                self.loss_weights[name] = weight / (self.loss_magnitudes[i].item() + 1e-8)
            else:
                weight = self.loss_dict[name]
                distortion += weight * loss

        # Stockage pour monitoring
        for name in self.loss_dict.keys():
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

    signal_length = inputs.shape[1]

    weights = torch.ones(signal_length, device=inputs.device, dtype=inputs.dtype)

    #max_significant_depth_idx = torch.searchsorted(depth_tens, significant_depth, right=False)

    weights[:max_significant_depth_idx] = 1.0  # Strong emphasis on the first points
    weights[max_significant_depth_idx+1:] = torch.exp(-decay_factor * torch.arange(max_significant_depth_idx+1, signal_length))

    # Reshape to match inputs shape
    weights = weights.view(1, 1, -1, 1, 1)  # Shape: [1, 1, signal_length, 1, 1]

    if use_smoothl1:
        weighted_loss = F.smooth_l1_loss(weights * outputs, weights * inputs, reduction='mean')
    else:
        weighted_loss =  torch.mean(weights * (outputs - inputs) ** 2)
    return weighted_loss



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
