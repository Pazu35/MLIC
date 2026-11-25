"""Check of the target-RMSE hinge (19 Nov 2026).
PYTHONPATH=/Odyssey/private/o23gauvr/code:/Odyssey/private/o23gauvr/code/MLIC/MLIC python loss/test_target_rmse.py"""
import numpy as np
import torch
from loss.rd_loss import FixedWeightSSPLoss

D, eps = 5, 0.3
z, sig = np.array([0., 10., 30., 60., 100.]), np.array([2., 1.5, 1., .5, .2])
ns = {"method": "mean_std_along_depth", "params": {"mean_along_depth": np.full((1, D, 1, 1), 1500.),
                                                   "std_along_depth": sig.reshape(1, -1, 1, 1)}}
c = FixedWeightSSPLoss(loss_dict={"recon": 1.0}, lmbda=1.0, recon_treshold=eps ** 2, depth_array=z,
                       norm_stats=ns, use_factor_weights=False, native_depth_levels=D)
torch.manual_seed(0)
t = torch.randn(2, D, 64, 64)
p = t + 0.4 * torch.randn(2, D, 64, 64)
lik = {"y": torch.full((2, 8, 4, 4), 0.5), "z": torch.full((2, 4, 1, 1), 0.5)}
w = np.r_[(z[1] - z[0]) / 2, (z[2:] - z[:-2]) / 2, (z[-1] - z[-2]) / 2] / (z[-1] - z[0])
ref = max(((((p - t).numpy() * sig.reshape(1, -1, 1, 1)) ** 2).mean(axis=(0, 2, 3)) * w).sum() - eps ** 2, 0)
assert abs(float(c({"x_hat": p, "likelihoods": lik}, t)["recon_loss"]) - ref) < 1e-5  # normalised inputs
s = torch.tensor(sig).view(1, -1, 1, 1).float()
with c.physical_inputs():  # validation path: inputs already in m/s
    assert abs(float(c({"x_hat": p * s + 1500, "likelihoods": lik}, t * s + 1500)["recon_loss"]) - ref) < 1e-3
print("ok")
